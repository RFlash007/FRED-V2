import asyncio
import base64
import io
import time
from PIL import Image
import ollama
import numpy as np
from config import config
from ollietec_theme import apply_theme
from ollie_print import olliePrint_simple

apply_theme()


class VisionService:
    def __init__(self):
        self.processing_interval = config.VISION_PROCESSING_INTERVAL
        self.model = config.VISION_MODEL
        self.enabled = config.VISION_ENABLED
        self.is_processing = False
        self.pi_connected = False
        
        # Scene tracking
        self.current_frame = None
        self.last_scene_description = ""
        self.current_scene_description = ""
        self.last_processing_time = 0
        
        # Ollama client
        self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        
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
    
    def store_latest_frame(self, frame):
        """Store the latest frame from WebRTC"""
        try:
            self.current_frame = frame
            # Reduced logging frequency - only log every 10th frame
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 1
            
            if self._frame_count % 10 == 0:
                olliePrint_simple(f"Frame stored ({frame.width}x{frame.height})")
        except Exception as e:
            olliePrint_simple(f"Frame storage error: {e}")
    
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
        """Main processing loop - runs every N seconds when Pi is connected"""
        while self.is_processing and self.pi_connected:
            try:
                if time.time() - self.last_processing_time >= self.processing_interval:
                    # Request fresh frame from Pi
                    await self._request_and_process_frame()
                    self.last_processing_time = time.time()
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                olliePrint_simple(f"Vision processing error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _request_and_process_frame(self):
        """Request a fresh frame from Pi and process it"""
        try:
            # Import here to avoid circular imports
            from webrtc_server import request_frame_from_client
            
            # Request frame from the first connected Pi client
            frame = await request_frame_from_client("127.0.0.1")
            
            if frame:
                self.current_frame = frame
                await self._process_current_frame()
                
        except Exception as e:
            olliePrint_simple(f"Frame request error: {e}")
    
    async def _process_current_frame(self):
        """Process the current frame with Qwen 2.5-VL 7B"""
        if not self.current_frame:
            return
        
        try:
            # Convert frame to base64
            image_b64 = self._frame_to_base64(self.current_frame)
            
            # Create detailed prompt with change detection
            prompt = self._create_vision_prompt()
            
            # Call Qwen 2.5-VL 7B
            response = await asyncio.to_thread(
                self.ollama_client.chat,
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }]
            )
            
            # Update scene descriptions
            self.last_scene_description = self.current_scene_description
            new_description = response['message']['content']
            
            # Limit description length if configured
            if config.VISION_MAX_DESCRIPTION_LENGTH > 0:
                new_description = new_description[:config.VISION_MAX_DESCRIPTION_LENGTH]
            
            self.current_scene_description = new_description
            
            # Only show significant scene changes
            if self.last_scene_description != self.current_scene_description:
                olliePrint_simple(f"Scene: {self.current_scene_description}")
            
        except ollama.ResponseError as e:
            olliePrint_simple(f"Vision model error: {e}")
            self.current_scene_description = "Vision processing temporarily unavailable."
        except Exception as e:
            olliePrint_simple(f"Frame processing error: {e}")
    
    def _create_vision_prompt(self):
        """Create detailed prompt for scene analysis"""
        base_prompt = """Describe this scene concisely, focusing on key objects/people, their positions/actions, environmental factors, text content, spatial relationships, and notable colors/materials.

"""
        
        if self.last_scene_description:
            return base_prompt + f"""Previous scene description: {self.last_scene_description}

IMPORTANT: If anything has changed from the previous scene, clearly state what has changed at the beginning of your response."""
        else:
            return base_prompt + "This is the first frame being analyzed."
    
    def _frame_to_base64(self, frame):
        """Convert WebRTC frame to base64"""
        try:
            # Convert frame to PIL Image
            frame_array = frame.to_ndarray(format="rgb24")
            image = Image.fromarray(frame_array)
            
            # Compress for efficiency
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(config.VISION_FRAME_QUALITY * 100))
            
            # Encode to base64
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            olliePrint_simple(f"Frame conversion error: {e}")
            raise
    
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