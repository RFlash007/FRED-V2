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
from persona_service import persona_service

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
        self.last_recognized_faces = []
        
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
            
            system_prompt = (
                "You are F.R.E.D.'s visual perception analyzing a FRESH image capture. "
                "This is real-time visual context for immediate user assistance. "
                "Output ONLY actionable insights F.R.E.D. needs right now. Be precise, practical, concise."
            )

            # Call Qwen 2.5-VL 7B
            response = await asyncio.to_thread(
                self.ollama_client.chat,
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
            
            # Parse and update scene description
            raw_response = response['message']['content']
            
            try:
                import json
                if raw_response.strip().startswith('{'):
                    parsed_response = json.loads(raw_response)
                    formatted_description = self._format_structured_response(parsed_response)
                else:
                    formatted_description = raw_response
            except json.JSONDecodeError:
                formatted_description = raw_response
            
            # Update scene descriptions
            self.last_scene_description = self.current_scene_description
            
            if config.VISION_MAX_DESCRIPTION_LENGTH > 0:
                formatted_description = formatted_description[:config.VISION_MAX_DESCRIPTION_LENGTH]
            
            self.current_scene_description = formatted_description
            
            # Always show fresh vision analysis
            olliePrint_simple(f"🔍 [FRESH VISION] {self.current_scene_description}")
            
        except Exception as e:
            olliePrint_simple(f"Fresh image processing error: {e}", 'error')
            import traceback
            traceback.print_exc()
    
    # _process_current_frame method removed - replaced by process_fresh_image for on-demand processing
    
    def _format_structured_response(self, parsed_json):
        """Convert structured JSON to readable format for F.R.E.D.'s context"""
        try:
            parts = []
            
            if 'summary' in parsed_json:
                parts.append(f"Scene: {parsed_json['summary']}")
            
            if 'people_activity' in parsed_json:
                parts.append(f"Person: {parsed_json['people_activity']}")
            
            if 'assistance_context' in parsed_json:
                parts.append(f"Context: {parsed_json['assistance_context']}")
            
            if 'affordances' in parsed_json and parsed_json['affordances']:
                affordances_str = ", ".join(parsed_json['affordances'][:3])  # Limit to 3 for brevity
                parts.append(f"Available: {affordances_str}")
            
            if 'changes' in parsed_json and parsed_json['changes']:
                change_level = parsed_json.get('change_significance', 'change')
                parts.append(f"Change ({change_level}): {parsed_json['changes']}")
            
            if 'uncertainty' in parsed_json and parsed_json['uncertainty']:
                parts.append(f"Uncertain: {parsed_json['uncertainty']}")
            
            if 'confidence' in parsed_json:
                conf = parsed_json['confidence']
                if conf < 70:
                    parts.append(f"Confidence: {conf}%")
            
            return " | ".join(parts) if parts else str(parsed_json)
            
        except Exception:
            return str(parsed_json)
    
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

        # Enhanced structured prompt with key techniques restored
        base_prompt = f"""Analyze for F.R.E.D.'s contextual assistance. Think step-by-step:

1. **OBSERVE:** Scene layout, objects, lighting, spatial context
2. **PEOPLE:** {people_summary} - activity, posture, emotional state, focus
3. **AFFORDANCES:** What actions are possible? What tools/objects enable tasks?
4. **ASSISTANCE:** What might user need help with? Opportunities to assist?

Output JSON:
{{
  "summary": "~15 words scene description",
  "people_activity": "What {people_summary} is doing/feeling - be specific",
  "assistance_context": "Concrete help opportunities F.R.E.D. could offer",
  "affordances": ["actionable", "objects", "within", "reach"],
  "confidence": 85,
  "uncertainty": "specific aspects unclear (if any)"
}}

**CRITICAL:** If uncertain about details, state specifics in "uncertainty" field.
Focus: Enable F.R.E.D.'s proactive, contextual assistance."""

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