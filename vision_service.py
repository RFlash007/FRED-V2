import asyncio
import base64
import io
import time
from PIL import Image
import numpy as np
from config import config, ollama_manager
from ollietec_theme import apply_theme
from ollie_print import olliePrint_simple
from persona_service import persona_service

apply_theme()


class VisionService:
    def __init__(self):
        self.model = config.VISION_MODEL
        
        # Initialize Qwen dimensions - optimized for Pi camera native resolution
        self.qwen_max_pixels = 3584 * 3584  # 12.8 MP - Qwen 2.5-VL 7B limit
        self.pi_native_width = 3280  # Pi camera native width
        self.pi_native_height = 2464  # Pi camera native height
        self.pi_native_pixels = self.pi_native_width * self.pi_native_height  # 8.1 MP - optimal
        
        # Scene state
        self.current_scene_description = "No visual input processed yet"
        self.last_scene_description = "No previous scene data"
        self.scene_timestamp = None
        
        # Processing control
        self.vision_processing_lock = None  # Created lazily when needed
        self.currently_processing_vision = False
        
        # CENTRALIZED CONNECTION: Remove direct client storage
        # Use ollama_manager.chat_concurrent_safe() for all calls
        
        # Processing task
        self.processing_task = None
        
        olliePrint_simple(f"Vision service ready - {self.model}")
    
    def set_pi_connection_status(self, connected: bool):
        """Called when Pi connects/disconnects"""
        self.pi_connected = connected
        if connected and not self.is_processing and self.enabled:
            self.start_continuous_processing()
        elif not connected and self.is_processing:
            self.stop_continuous_processing()
    
    # store_latest_frame method removed - using on-demand capture instead
    
    def start_continuous_processing(self):
        """Start the continuous vision processing loop"""
        if self.is_processing or not self.enabled:
            return
        
        self.is_processing = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        olliePrint_simple("Vision processing started")
    
    def stop_continuous_processing(self):
        """Stop the continuous vision processing"""
        self.is_processing = False
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
        olliePrint_simple("Vision processing stopped")
    
    async def _processing_loop(self):
        """Main processing loop - request-based fresh image capture"""
        while self.is_processing and self.pi_connected:
            try:
                if time.time() - self.last_processing_time >= self.processing_interval:
                    # Request fresh image capture from Pi
                    await self._request_fresh_capture()
                    self.last_processing_time = time.time()
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                olliePrint_simple(f"Vision processing error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _request_fresh_capture(self):
        """Request a fresh image capture from Pi via data channel"""
        
        # Create lock if it doesn't exist
        if self.vision_processing_lock is None:
            self.vision_processing_lock = asyncio.Lock()
        
        # Check if already processing
        if self.currently_processing_vision:
            olliePrint_simple("⏳ [VISION] Already processing image, skipping request", 'warning')
            return
            
        try:
            # Import here to avoid circular imports
            from webrtc_server import send_capture_request_to_pi
            
            olliePrint_simple("🎯 [VISION] Requesting fresh image capture from Pi...")
            success = await send_capture_request_to_pi()
            
            if not success:
                olliePrint_simple("❌ [VISION] Failed to request image capture from Pi", 'warning')
                
        except Exception as e:
            olliePrint_simple(f"Vision capture request error: {e}")
    
    async def process_fresh_image(self, image_data, format_type='jpeg'):
        """Process a fresh image received from Pi"""
        
        # Create lock if it doesn't exist
        if self.vision_processing_lock is None:
            self.vision_processing_lock = asyncio.Lock()
        
        # Use lock to prevent overlapping processing
        async with self.vision_processing_lock:
            if self.currently_processing_vision:
                olliePrint_simple("⏳ [VISION] Already processing, dropping duplicate image", 'warning')
                return
                
            self.currently_processing_vision = True
            
            try:
                await self._process_image_internal(image_data, format_type)
            finally:
                self.currently_processing_vision = False
    
    async def _process_image_internal(self, image_data, format_type='jpeg'):
        """Internal image processing method"""
        try:
            import base64
            from PIL import Image
            import io
            
            # Decode the image data
            if isinstance(image_data, str):
                # Base64 encoded image
                image_bytes = base64.b64decode(image_data)
            else:
                # Raw bytes
                image_bytes = image_data
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            olliePrint_simple(f"📸 [VISION] Processing fresh {image.size[0]}x{image.size[1]} image from Pi")
            
            # Convert PIL image to frame-like object for existing processing
            frame_array = np.array(image.convert('RGB'))
            
            # Store for enrollment tool access
            self.last_processed_image_array = frame_array
            
            # Process face recognition
            self.last_recognized_faces = await asyncio.to_thread(
                persona_service.recognize_faces, frame_array
            )
            
            # Convert image to base64 for vision model
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(config.VISION_FRAME_QUALITY * 100))
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create vision prompt and analyze
            prompt = self._create_vision_prompt()
            
            system_prompt = config.VISION_SYSTEM_PROMPT

            olliePrint_simple(f"🤖 [QWEN] Calling {self.model} for vision analysis...")
            
            # Call Qwen 2.5-VL 7B with proper error handling (no timeout - model needs time)
            try:
                # CENTRALIZED CONNECTION: Use connection manager instead of direct client
                response = await asyncio.to_thread(
                    ollama_manager.chat_concurrent_safe,
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64]
                    }]
                )
                
                olliePrint_simple("✅ [QWEN] Model response received", 'success')
                
            except Exception as model_error:
                olliePrint_simple(f"❌ [QWEN] Model call failed: {model_error}", 'error')
                self.current_scene_description = f"Vision processing failed: {model_error}"
                return
            
            # Get raw model response - NO JSON PARSING
            raw_response = response['message']['content']
            
            # Update scene descriptions
            self.last_scene_description = self.current_scene_description
            
            # Use raw response directly
            if config.VISION_MAX_DESCRIPTION_LENGTH > 0:
                self.current_scene_description = raw_response[:config.VISION_MAX_DESCRIPTION_LENGTH]
            else:
                self.current_scene_description = raw_response
            
            # Always show fresh vision analysis
            olliePrint_simple(f"🔍 [QWEN OUTPUT] {self.current_scene_description}")
            olliePrint_simple("✅ [VISION] Fresh image processing COMPLETE", 'success')
            
        except Exception as e:
            olliePrint_simple(f"Fresh image processing error: {e}", 'error')
            import traceback
            traceback.print_exc()
    
    # _process_current_frame method removed - replaced by process_fresh_image for on-demand processing
    
    # _format_structured_response method removed - using raw Qwen output only
    
    def _create_vision_prompt(self):
        """Optimized prompt incorporating advanced techniques for F.R.E.D.'s contextual assistance"""
        
        # Prepare the people summary from persona recognition
        people_summary = "No one visible"
        if self.last_recognized_faces:
            names = [face['name'] for face in self.last_recognized_faces]
            if len(names) == 1:
                people_summary = names[0]
            else:
                name_counts = {}
                for name in names:
                    name_counts[name] = name_counts.get(name, 0) + 1
                
                parts = []
                for name, count in name_counts.items():
                    if name == "An unknown person":
                        parts.append(f"{count} unknown")
                    else:
                        parts.append(name)
                
                people_summary = ", ".join(parts)

        # Use configurable base prompt with dynamic people context
        base_prompt = config.VISION_USER_PROMPT + f"\n\nPeople visible: {people_summary}"

        # Add change detection with enhanced context
        if self.last_scene_description:
            change_prompt = f"""

**CHANGE ANALYSIS:**
Previous: {self.last_scene_description[:150]}...

Add these fields to JSON if significant changes:
- "changes": "what's different from previous scene"
- "change_significance": "minor/moderate/major"

Detect: new people, objects, activities, user focus shifts, environmental changes."""
            return base_prompt + change_prompt
        else:
            return base_prompt
    
    # _frame_to_base64 method removed - no longer processing WebRTC frames directly
    
    def get_current_visual_context(self):
        """Get the current visual context for injection into conversations"""
        if not self.current_scene_description:
            return "No visual context available."
        
        return self.current_scene_description
    
    def is_vision_available(self):
        """Check if vision processing is available and working"""
        return self.enabled and self.pi_connected and bool(self.current_scene_description)

# Global vision service instance
vision_service = VisionService() 