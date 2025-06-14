import asyncio
import base64
import io
import time
import logging
from PIL import Image
import ollama
import numpy as np
from config import config

logger = logging.getLogger(__name__)

class VisionService:
    def __init__(self):
        self.processing_interval = config.VISION_PROCESSING_INTERVAL
        self.model = config.VISION_MODEL
        self.enabled = config.VISION_ENABLED
        self.is_processing = False
        self.pi_connected = False  # Only process when Pi is connected
        
        # Scene tracking
        self.current_frame = None
        self.last_scene_description = ""
        self.current_scene_description = ""
        self.last_processing_time = 0
        
        # Ollama client
        self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        
        # Processing task
        self.processing_task = None
        
        logger.info(f"Vision service initialized - Model: {self.model}, Interval: {self.processing_interval}s")
    
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
            print(f"📸 Frame received and stored (size: {frame.width}x{frame.height})")
        except Exception as e:
            logger.error(f"Error storing frame: {e}")
            print(f"❌ Frame storage error: {e}")
    
    def start_continuous_processing(self):
        """Start the continuous vision processing loop"""
        if self.is_processing or not self.enabled:
            return
        
        self.is_processing = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Vision processing started - Pi glasses connected")
    
    def stop_continuous_processing(self):
        """Stop the continuous vision processing"""
        self.is_processing = False
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
        logger.info("Vision processing stopped - Pi glasses disconnected")
    
    async def _processing_loop(self):
        """Main processing loop - runs every N seconds when Pi is connected"""
        print("🔄 Vision processing loop started (on-demand mode)")
        while self.is_processing and self.pi_connected:
            try:
                if time.time() - self.last_processing_time >= self.processing_interval:
                    print(f"🎯 Time for vision processing (interval: {self.processing_interval}s)")
                    print(f"🔗 Connecting to Ollama at: {config.OLLAMA_BASE_URL}")
                    print(f"🤖 Using vision model: {self.model}")
                    
                    # Request fresh frame from Pi
                    await self._request_and_process_frame()
                    self.last_processing_time = time.time()
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                logger.info("Vision processing loop cancelled")
                break
            except Exception as e:
                print(f"❌ Vision processing error: {e}")
                logger.error(f"Vision processing error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _request_and_process_frame(self):
        """Request a fresh frame from Pi and process it"""
        try:
            # Import here to avoid circular imports
            from webrtc_server import request_frame_from_client
            
            # Request frame from the first connected Pi client
            # In the future, this could be extended to handle multiple Pi clients
            frame = await request_frame_from_client("127.0.0.1")  # Assuming localhost for now
            
            if frame:
                # Store the fresh frame
                self.current_frame = frame
                print(f"📸 Fresh frame received for processing (size: {frame.width}x{frame.height})")
                
                # Process it
                await self._process_current_frame()
            else:
                print("❌ No frame received from Pi client")
                
        except Exception as e:
            print(f"❌ Error requesting frame from Pi: {e}")
            logger.error(f"Error requesting frame from Pi: {e}")
    
    async def _process_current_frame(self):
        """Process the current frame with Gemma3:4b"""
        if not self.current_frame:
            return
        
        try:
            print("🖼️ Converting frame to base64...")
            # Convert frame to base64
            image_b64 = self._frame_to_base64(self.current_frame)
            print(f"✅ Frame converted, base64 length: {len(image_b64)} chars")
            
            # Create detailed prompt with change detection
            prompt = self._create_vision_prompt()
            print(f"📝 Vision prompt created: {len(prompt)} chars")
            
            print("🧠 Sending to Ollama for vision analysis...")
            # Call Gemma3:4b
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
            
            print("🎯 === SCENE ANALYSIS RESULT ===")
            print(f"📊 {self.current_scene_description}")
            print("=================================")
            
            logger.debug(f"Vision update: {self.current_scene_description[:100]}...")
            
        except ollama.ResponseError as e:
            print(f"❌ Ollama vision model error: {e}")
            print("💡 Make sure 'ollama serve' is running and 'gemma3:4b' model is installed")
            logger.error(f"Ollama vision model error: {e}")
            self.current_scene_description = "Vision processing temporarily unavailable."
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            logger.error(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_vision_prompt(self):
        """Create detailed prompt for scene analysis"""
        base_prompt = """You are receiving a 896x896 pixel image optimized for your vision processing. Describe this scene in comprehensive detail, including:
- All objects, people, and their specific positions/locations
- Current activities or actions taking place
- Environmental context (lighting, setting, atmosphere)
- Any text, screens, or displays visible
- Spatial relationships between elements
- Colors, materials, and notable details

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
            logger.error(f"Frame conversion error: {e}")
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