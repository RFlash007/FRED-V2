#!/usr/bin/env python3
"""
F.R.E.D. Pi Glasses Client - Consolidated Local STT & Communication
Vault-Tec Approved™ | OLLIE-TEC Advanced Wasteland Computing Division

This is the single startup file for F.R.E.D. Pi glasses that handles:
- Local speech-to-text processing (Whisper tiny.en)
- Video streaming via camera
- Communication with F.R.E.D. mainframe
- Audio playback from F.R.E.D. responses

Post-apocalyptic engineering at its finest!
"""

import os
# Shut up libcamera: only errors
os.environ.setdefault('LIBCAMERA_LOG_LEVELS', '*:3')

import asyncio
import argparse
import requests
import json
import time
import sys
import base64
import tempfile
import subprocess
import threading
import queue
import numpy as np
import logging
import json
from typing import Optional, Callable
import vosk

# Import Ollie-Tec theming
from ollietec_theme import apply_theme, banner

# Apply theming to all prints
apply_theme()

logger = logging.getLogger(__name__)

class FREDPiSTTService:
    """
    Vault-Tec Speech Recognition Module
    Optimized for Pi with Vosk small English model
    """
    
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.is_initialized = False
        
        # Audio processing configuration (optimized for Pi)
        self.sample_rate = 16000
        self.channels = 1
        
        # Speech detection settings
        self.speech_buffer = []
        self.last_speech_time = 0
        self.silence_duration = 1.0  # Slightly longer for Vosk
        self.silence_threshold = 0.002
        
        # Processing control
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_processing = False
        self.processing_thread = None
        self.transcription_callback: Optional[Callable] = None
        
        # Wake word detection
        self.wake_words = [
            "fred", "hey fred", "okay fred", 
            "hi fred", "excuse me fred", "fred are you there"
        ]
        self.stop_words = [
            "goodbye", "bye fred", "stop listening", 
            "that's all", "thank you fred", "sleep now"
        ]
        
        # Performance monitoring
        self._transcription_count = 0
        self._start_time = time.time()
        
    def initialize(self):
        """Initialize the Vosk model optimized for Pi"""
        if self.is_initialized:
            return True
            
        try:
            print("[PIP-BOY STT] Initializing voice recognition systems...")
            print("🔧 Loading Vosk small English model...")
            
            # Set Vosk log level to reduce noise
            vosk.SetLogLevel(-1)
            
            # Look for Vosk model in common locations
            model_paths = [
                "models/vosk-model-small-en-us-0.15",
                "../models/vosk-model-small-en-us-0.15",
                "./vosk-model-small-en-us-0.15",
                "/home/raspberry/FRED-V2/pi_client/models/vosk-model-small-en-us-0.15"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                print("❌ [CRITICAL] Vosk model not found!")
                print("💡 [SOLUTION] Run: bash install_vosk_model.sh")
                return False
            
            # Initialize Vosk model
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            
            print("✅ [PIP-BOY STT] Voice recognition ONLINE")
            print(f"📊 Model: Vosk small English (optimized for Pi)")
            print(f"📍 Path: {model_path}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"[CRITICAL] STT initialization failed: {e}")
            print(f"❌ [PIP-BOY STT] Voice recognition FAILED: {e}")
            print("💡 [SOLUTION] Run: bash install_vosk_model.sh")
            return False
    
    def start_processing(self, callback: Callable):
        """Start the audio processing with wake word detection"""
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        print("🎤 [PIP-BOY STT] Starting audio processing...")
        self.transcription_callback = callback
        self.is_processing = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_audio_loop, 
            daemon=True
        )
        self.processing_thread.start()
        
        print("👂 [PIP-BOY STT] Listening for wake word...")
        logger.info("Pi STT processing started - waiting for wake word")
        return True
    
    def stop_processing(self):
        """Stop audio processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("🔇 [PIP-BOY STT] Voice recognition offline")
        logger.info("Pi STT processing stopped")
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk for processing"""
        if self.is_processing and not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def _process_audio_loop(self):
        """Main audio processing loop with wake word detection"""
        while self.is_processing:
            try:
                if self.audio_queue.empty():
                    time.sleep(0.1)
                    continue
                
                audio_chunk = self.audio_queue.get()
                
                # Calculate audio level for voice activity detection
                audio_level = np.abs(audio_chunk).mean()
                
                # Skip if too quiet
                if audio_level < self.silence_threshold:
                    # Check if we should process buffered speech
                    if self.is_listening and self.speech_buffer:
                        if time.time() - self.last_speech_time > self.silence_duration:
                            self._process_complete_utterance()
                    continue
                
                # Process audio with Whisper
                try:
                    text = self._transcribe_audio(audio_chunk)
                    if text and len(text.strip()) > 0:
                        self._handle_transcribed_text(text.strip().lower())
                        
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                time.sleep(0.1)
    
    def _transcribe_audio(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio using optimized Whisper settings"""
        try:
            # Ensure proper audio format
            if len(audio_chunk) < 1600:  # Less than 0.1 seconds
                return ""
            
            # Normalize audio to float32 [-1, 1] for Whisper
            if audio_chunk.dtype == np.int16:
                audio_data = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_data = audio_chunk.flatten().astype(np.float32)
            
            # Optimized transcription settings for Pi
            segments, info = self.model.transcribe(
                audio_data,
                language="en",
                beam_size=1,                    # Fastest beam size
                temperature=0.0,                # Deterministic
                condition_on_previous_text=False, # Faster processing
                vad_filter=True,                # Built-in VAD
                vad_parameters={"min_silence_duration_ms": 300},
                initial_prompt="Voice command to FRED AI assistant.",
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.5
            )
            
            # Combine segments efficiently
            text = ""
            for segment in segments:
                text += segment.text
            
            # Performance monitoring
            self._transcription_count += 1
            if self._transcription_count % 50 == 0:
                elapsed = time.time() - self._start_time
                avg_time = elapsed / self._transcription_count
                print(f"📊 [PERFORMANCE] {self._transcription_count} transcriptions, avg: {avg_time:.2f}s")
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def _handle_transcribed_text(self, text: str):
        """Handle transcribed text with wake word detection"""
        print(f"🎙️ [DETECTED] '{text}'")
        
        # Check for wake words when not listening
        if not self.is_listening:
            if any(wake_word in text for wake_word in self.wake_words):
                print(f"👋 [WAKE] Wake word detected! Listening...")
                self.is_listening = True
                self.speech_buffer = []
                self.last_speech_time = time.time()
                return
        
        # Process speech while listening
        if self.is_listening:
            # Check for stop words
            if any(stop_word in text for stop_word in self.stop_words):
                print(f"💤 [SLEEP] Stop word detected")
                if self.transcription_callback:
                    self.transcription_callback("goodbye")
                self.is_listening = False
                self.speech_buffer = []
                return
            
            # Add to speech buffer if meaningful
            if len(text.split()) > 1:  # More than one word
                print(f"📝 [BUFFER] Adding: '{text}'")
                self.last_speech_time = time.time()
                self.speech_buffer.append(text)
    
    def _process_complete_utterance(self):
        """Process complete buffered utterance"""
        if not self.speech_buffer:
            return
            
        complete_text = " ".join(self.speech_buffer)
        print(f"🗣️ [COMPLETE] Processing: '{complete_text}'")
        
        # Clear buffer
        self.speech_buffer = []
        
        # Send to callback
        if self.transcription_callback:
            self.transcription_callback(complete_text)
        
        # Resume listening
        print("👂 [READY] Listening for next command...")


class FREDPiClient:
    """
    F.R.E.D. Pi Glasses Client - Vault-Tec Integration Module
    Handles communication with F.R.E.D. mainframe and local processing
    """
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.stt_service = FREDPiSTTService()
        self.is_running = False
        
        # Camera setup for vision
        self.camera = None
        self.last_image_time = 0
        self.vision_interval = 5  # Send image every 5 seconds
        
    async def start(self):
        """Start the F.R.E.D. Pi client"""
        print(banner("F.R.E.D. Pi Glasses v2.0"))
        print("[VAULT-TEC] Initializing post-apocalyptic AI interface...")
        
        # Initialize camera
        await self._init_camera()
        
        # Start STT service
        if not self.stt_service.start_processing(self._handle_transcription):
            print("❌ [CRITICAL] STT initialization failed")
            return False
        
        self.is_running = True
        print("[VAULT-NET] F.R.E.D. Pi Glasses ONLINE!")
        print("[PIP-BOY] Ready for wasteland operations...")
        
        # Main loop
        await self._run_main_loop()
        
    async def _init_camera(self):
        """Initialize Picamera2 for vision"""
        try:
            print("📸 [VISION] Initializing Pip-Boy camera systems...")
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Configure for optimal quality ([using native resolution for Qwen 2.5-VL][[memory:1877597798655505118]])
            sensor_modes = self.camera.sensor_modes
            max_mode = max(sensor_modes, key=lambda x: x['size'][0] * x['size'][1])
            max_res = max_mode['size']
            print(f"🎯 [VISION] Using native resolution {max_res} = {(max_res[0] * max_res[1] / 1_000_000):.1f} MP")
            
            config = self.camera.create_still_configuration(
                main={"size": max_res, "format": "RGB888"},
                controls={
                    "FrameRate": 1,     # Low FPS for still captures
                    "Brightness": 0.1,
                    "Contrast": 1.1,
                    "Saturation": 1.0,
                },
                buffer_count=1
            )
            
            self.camera.configure(config)
            self.camera.start()
            print("✅ [VISION] Camera systems ONLINE")
            
        except Exception as e:
            print(f"❌ [VISION] Camera initialization failed: {e}")
            self.camera = None
    
    async def _run_main_loop(self):
        """Main operational loop"""
        while self.is_running:
            try:
                # Capture and send vision periodically
                current_time = time.time()
                if (self.camera and 
                    current_time - self.last_image_time > self.vision_interval):
                    await self._capture_and_send_vision()
                    self.last_image_time = current_time
                
                # Keep running
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Field operative terminating connection")
                break
            except Exception as e:
                print(f"[ERROR] Main loop error: {e}")
                await asyncio.sleep(1)
        
        # Cleanup
        await self._cleanup()
    
    async def _capture_and_send_vision(self):
        """Capture image and send to F.R.E.D. for analysis"""
        try:
            if not self.camera:
                return
                
            # Capture image
            image_array = self.camera.capture_array("main")
            
            # Convert to base64 ([not cropping as user prefers][[memory:1146910764787162391]])
            import cv2
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR), 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to F.R.E.D.
            await self._send_to_fred({
                'type': 'vision',
                'image': image_b64,
                'timestamp': time.time()
            })
            
            print("📸 [VISION] Environmental scan transmitted to F.R.E.D.")
            
        except Exception as e:
            print(f"❌ [VISION] Image capture failed: {e}")
    
    def _handle_transcription(self, text: str):
        """Handle transcribed speech from STT service"""
        print(f"🎤 [COMMAND] '{text}'")
        
        # Send to F.R.E.D. for processing
        asyncio.create_task(self._send_to_fred({
            'type': 'speech',
            'text': text,
            'timestamp': time.time()
        }))
    
    async def _send_to_fred(self, data):
        """Send data to F.R.E.D. server"""
        try:
            headers = {
                'Authorization': 'Bearer fred_pi_glasses_2024',
                'Content-Type': 'application/json'
            }
            
            # Use async HTTP request
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.server_url}/pi_data',
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        await self._handle_fred_response(result)
                    else:
                        print(f"❌ [COMM] Server error: {response.status}")
                        
        except asyncio.TimeoutError:
            print("⏰ [COMM] F.R.E.D. server timeout")
        except Exception as e:
            print(f"❌ [COMM] Communication error: {e}")
    
    async def _handle_fred_response(self, response):
        """Handle response from F.R.E.D."""
        try:
            # Handle text responses ([display on Pi terminal as requested][[memory:4525952157057302528]])
            if 'text' in response:
                print(f"\n[F.R.E.D.] {response['text']}")
            
            # Handle audio responses  
            if 'audio_base64' in response:
                audio_b64 = response['audio_base64']
                audio_format = response.get('audio_format', 'wav')
                self._play_audio_from_base64(audio_b64, audio_format)
                
        except Exception as e:
            print(f"❌ [RESPONSE] Error handling F.R.E.D. response: {e}")
    
    def _play_audio_from_base64(self, audio_b64, format_type='wav'):
        """Decode base64 audio and play it on the Pi"""
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(audio_b64)
            print(f"[AUDIO] F.R.E.D. transmission received ({len(audio_data)} bytes)")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Play using aplay (ALSA) - most reliable on Pi
            try:
                subprocess.run(['aplay', temp_file_path], check=True, capture_output=True)
                print("[SUCCESS] F.R.E.D. voice transmission complete")
            except subprocess.CalledProcessError:
                # Fallback to paplay (PulseAudio)
                try:
                    subprocess.run(['paplay', temp_file_path], check=True, capture_output=True)
                    print("[SUCCESS] F.R.E.D. voice transmission complete (PulseAudio)")
                except subprocess.CalledProcessError:
                    # Last resort: mpv
                    try:
                        subprocess.run(['mpv', '--no-video', temp_file_path], check=True, capture_output=True)
                        print("[SUCCESS] F.R.E.D. voice transmission complete (mpv)")
                    except subprocess.CalledProcessError:
                        print(f"[CRITICAL] All audio protocols failed - check Pip-Boy speakers")
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_err:
                print(f"[WARNING] Failed to purge audio cache: {cleanup_err}")
                
        except Exception as e:
            print(f"[CRITICAL] Audio playback system failure: {e}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        print("[CLEANUP] Shutting down Pip-Boy systems...")
        
        # Stop STT
        self.stt_service.stop_processing()
        
        # Stop camera
        if self.camera:
            try:
                self.camera.stop()
                print("📸 [VISION] Camera systems offline")
            except Exception as e:
                print(f"⚠️ [VISION] Camera cleanup error: {e}")
        
        self.is_running = False
        print("[VAULT-TEC] All systems offline. Stay safe in the wasteland!")


def get_server_url(provided_url=None):
    """Auto-discover F.R.E.D. server or use provided URL"""
    if provided_url:
        # Validate provided URL
        if not provided_url.startswith(('http://', 'https://')):
            provided_url = f'http://{provided_url}'
        
        print(f"🔗 Using provided F.R.E.D. server: {provided_url}")
        return provided_url
    
    print("🔍 Auto-discovering F.R.E.D. mainframe...")
    
    # Try common local addresses
    local_addresses = [
        'http://localhost:8000',
        'http://127.0.0.1:8000',
        'http://localhost:5000',
        'http://127.0.0.1:5000'
    ]
    
    for url in local_addresses:
        try:
            response = requests.get(f'{url}/health', timeout=2)
            if response.status_code == 200:
                print(f"✅ Found F.R.E.D. mainframe at: {url}")
                return url
        except requests.exceptions.RequestException:
            continue
    
    # Check tunnel info
    try:
        with open('../tunnel_info.json', 'r') as f:
            tunnel_info = json.load(f)
            tunnel_url = tunnel_info.get('public_url')
            if tunnel_url:
                print(f"🌐 Found tunnel URL: {tunnel_url}")
                return tunnel_url
    except FileNotFoundError:
        pass
    
    print("❌ [CRITICAL] No F.R.E.D. server found!")
    print("\n[VAULT-TEC] Troubleshooting protocols:")
    print("1. Start F.R.E.D. server: python app.py")
    print("2. Use local URL: --server http://localhost:8000")
    print("3. Use ngrok tunnel: --server https://xxx.ngrok.io")
    print("4. Or use ngrok URL if connecting remotely")
    
    raise Exception("❌ No F.R.E.D. server found. Please specify --server URL")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="F.R.E.D. Pi Glasses Client with Local STT")
    parser.add_argument('--server', help='F.R.E.D. server URL (auto-discovery if not provided)')
    args = parser.parse_args()
    
    try:
        # Get server URL
        server_url = get_server_url(args.server)
        
        # Create and start client
        client = FREDPiClient(server_url)
        await client.start()
        
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Field operative terminating connection")
    except Exception as e:
        print(f"\n[CRITICAL] System failure: {e}")
        print("\n[VAULT-TEC] Troubleshooting protocols:")
        print("1. Verify F.R.E.D. mainframe is operational")
        print("2. Check wasteland communication network")
        print("3. Try manual server specification with --server")
        sys.exit(1)


    def start_processing(self, callback: Callable):
        """Start the audio processing with wake word detection"""
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        print("🎤 [PIP-BOY STT] Starting audio processing...")
        self.transcription_callback = callback
        self.is_processing = True
        
        # Start processing thread for audio capture
        self.processing_thread = threading.Thread(
            target=self._audio_capture_loop, 
            daemon=True
        )
        self.processing_thread.start()
        
        print("👂 [PIP-BOY STT] Listening for wake word...")
        return True
    
    def stop_processing(self):
        """Stop audio processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("🔇 [PIP-BOY STT] Voice recognition offline")
    
    def _audio_capture_loop(self):
        """Audio capture and processing loop using sounddevice"""
        try:
            import sounddevice as sd
            print("🎤 [AUDIO] Starting audio capture...")
            
            def audio_callback(indata, frames, time, status):
                if self.is_processing:
                    # Convert to float32 and add to queue
                    audio_data = indata.flatten().astype(np.float32)
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio_data)
            
            # Start audio stream
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=audio_callback,
                dtype=np.float32
            ):
                print("✅ [AUDIO] Audio capture ONLINE")
                
                while self.is_processing:
                    try:
                        if not self.audio_queue.empty():
                            audio_chunk = self.audio_queue.get()
                            self._process_audio_chunk(audio_chunk)
                        else:
                            time.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        time.sleep(0.1)
                        
        except ImportError:
            print("❌ [AUDIO] sounddevice not available")
        except Exception as e:
            print(f"❌ [AUDIO] Audio capture failed: {e}")
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process individual audio chunk"""
        try:
            # Calculate audio level for voice activity detection
            audio_level = np.abs(audio_chunk).mean()
            
            # Skip if too quiet
            if audio_level < self.silence_threshold:
                if self.is_listening and self.speech_buffer:
                    if time.time() - self.last_speech_time > self.silence_duration:
                        self._process_complete_utterance()
                return
            
            # Transcribe audio
            text = self._transcribe_audio(audio_chunk)
            if text and len(text.strip()) > 0:
                self._handle_transcribed_text(text.strip().lower())
                
        except Exception as e:
            logger.error(f"Audio chunk processing error: {e}")
    
    def _transcribe_audio(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio using Vosk"""
        try:
            if len(audio_chunk) < 1600:  # Less than 0.1 seconds
                return ""
            
            # Convert to int16 format that Vosk expects
            if audio_chunk.dtype == np.float32:
                # Convert float32 [-1, 1] to int16
                audio_data = (audio_chunk * 32767).astype(np.int16)
            else:
                audio_data = audio_chunk.astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_data.tobytes()
            
            # Process with Vosk
            if self.recognizer.AcceptWaveform(audio_bytes):
                # Final result
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '')
            else:
                # Partial result
                result = json.loads(self.recognizer.PartialResult())
                text = result.get('partial', '')
            
            self._transcription_count += 1
            if self._transcription_count % 100 == 0:
                elapsed = time.time() - self._start_time
                avg_time = elapsed / self._transcription_count
                print(f"📊 [PERFORMANCE] {self._transcription_count} transcriptions, avg: {avg_time:.3f}s")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def _handle_transcribed_text(self, text: str):
        """Handle transcribed text with wake word detection"""
        print(f"🎙️ [DETECTED] '{text}'")
        
        # Check for wake words when not listening
        if not self.is_listening:
            if any(wake_word in text for wake_word in self.wake_words):
                print(f"👋 [WAKE] Wake word detected! Listening...")
                self.is_listening = True
                self.speech_buffer = []
                self.last_speech_time = time.time()
                return
        
        # Process speech while listening
        if self.is_listening:
            # Check for stop words
            if any(stop_word in text for stop_word in self.stop_words):
                print(f"💤 [SLEEP] Stop word detected")
                if self.transcription_callback:
                    self.transcription_callback("goodbye")
                self.is_listening = False
                self.speech_buffer = []
                return
            
            # Add to speech buffer if meaningful
            if len(text.split()) > 1:  # More than one word
                print(f"📝 [BUFFER] Adding: '{text}'")
                self.last_speech_time = time.time()
                self.speech_buffer.append(text)
    
    def _process_complete_utterance(self):
        """Process complete buffered utterance"""
        if not self.speech_buffer:
            return
            
        complete_text = " ".join(self.speech_buffer)
        print(f"🗣️ [COMPLETE] Processing: '{complete_text}'")
        
        # Clear buffer
        self.speech_buffer = []
        
        # Send to callback
        if self.transcription_callback:
            self.transcription_callback(complete_text)
        
        # Resume listening
        print("👂 [READY] Listening for next command...")


class FREDPiClient:
    """F.R.E.D. Pi Glasses Client - Main interface"""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.stt_service = FREDPiSTTService()
        self.is_running = False
        
        # Camera setup
        self.camera = None
        self.last_image_time = 0
        self.vision_interval = 10  # Send image every 10 seconds
        
    async def start(self):
        """Start the F.R.E.D. Pi client"""
        print(banner("F.R.E.D. Pi Glasses v2.0"))
        print("[VAULT-TEC] Initializing post-apocalyptic AI interface...")
        
        # Initialize camera
        await self._init_camera()
        
        # Start STT service
        if not self.stt_service.start_processing(self._handle_transcription):
            print("❌ [CRITICAL] STT initialization failed")
            return False
        
        self.is_running = True
        print("[VAULT-NET] F.R.E.D. Pi Glasses ONLINE!")
        print("[PIP-BOY] Ready for wasteland operations...")
        
        # Main loop
        await self._run_main_loop()
        
    async def _init_camera(self):
        """Initialize Picamera2 for vision"""
        try:
            print("📸 [VISION] Initializing Pip-Boy camera systems...")
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Configure for optimal quality
            sensor_modes = self.camera.sensor_modes
            max_mode = max(sensor_modes, key=lambda x: x['size'][0] * x['size'][1])
            max_res = max_mode['size']
            print(f"🎯 [VISION] Using native resolution {max_res}")
            
            config = self.camera.create_still_configuration(
                main={"size": max_res, "format": "RGB888"},
                controls={"FrameRate": 1, "Brightness": 0.1, "Contrast": 1.1},
                buffer_count=1
            )
            
            self.camera.configure(config)
            self.camera.start()
            print("✅ [VISION] Camera systems ONLINE")
            
        except Exception as e:
            print(f"❌ [VISION] Camera initialization failed: {e}")
            self.camera = None
    
    async def _run_main_loop(self):
        """Main operational loop"""
        while self.is_running:
            try:
                # Capture and send vision periodically
                current_time = time.time()
                if (self.camera and 
                    current_time - self.last_image_time > self.vision_interval):
                    await self._capture_and_send_vision()
                    self.last_image_time = current_time
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Field operative terminating connection")
                break
            except Exception as e:
                print(f"[ERROR] Main loop error: {e}")
                await asyncio.sleep(1)
        
        await self._cleanup()
    
    async def _capture_and_send_vision(self):
        """Capture image and send to F.R.E.D."""
        try:
            if not self.camera:
                return
                
            # Capture image
            image_array = self.camera.capture_array("main")
            
            # Convert to base64
            import cv2
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR), 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to F.R.E.D.
            await self._send_to_fred({
                'type': 'vision',
                'image': image_b64,
                'timestamp': time.time()
            })
            
            print("📸 [VISION] Environmental scan transmitted to F.R.E.D.")
            
        except Exception as e:
            print(f"❌ [VISION] Image capture failed: {e}")
    
    def _handle_transcription(self, text: str):
        """Handle transcribed speech from STT service"""
        print(f"🎤 [COMMAND] '{text}'")
        
        # Send to F.R.E.D. for processing
        asyncio.create_task(self._send_to_fred({
            'type': 'speech',
            'text': text,
            'timestamp': time.time()
        }))
    
    async def _send_to_fred(self, data):
        """Send data to F.R.E.D. server"""
        try:
            headers = {
                'Authorization': 'Bearer fred_pi_glasses_2024',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f'{self.server_url}/pi_data',
                json=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                await self._handle_fred_response(result)
            else:
                print(f"❌ [COMM] Server error: {response.status_code}")
                        
        except Exception as e:
            print(f"❌ [COMM] Communication error: {e}")
    
    async def _handle_fred_response(self, response):
        """Handle response from F.R.E.D."""
        try:
            # Handle text responses
            if 'text' in response:
                print(f"\n[F.R.E.D.] {response['text']}")
            
            # Handle audio responses  
            if 'audio_base64' in response:
                audio_b64 = response['audio_base64']
                audio_format = response.get('audio_format', 'wav')
                self._play_audio_from_base64(audio_b64, audio_format)
                
        except Exception as e:
            print(f"❌ [RESPONSE] Error handling F.R.E.D. response: {e}")
    
    def _play_audio_from_base64(self, audio_b64, format_type='wav'):
        """Play audio from base64 data"""
        try:
            audio_data = base64.b64decode(audio_b64)
            print(f"[AUDIO] F.R.E.D. transmission received ({len(audio_data)} bytes)")
            
            with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                subprocess.run(['aplay', temp_file_path], check=True, capture_output=True)
                print("[SUCCESS] F.R.E.D. voice transmission complete")
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(['paplay', temp_file_path], check=True, capture_output=True)
                    print("[SUCCESS] F.R.E.D. voice transmission complete (PulseAudio)")
                except subprocess.CalledProcessError:
                    print("[CRITICAL] Audio playback failed")
            
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
                
        except Exception as e:
            print(f"[CRITICAL] Audio playback system failure: {e}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        print("[CLEANUP] Shutting down Pip-Boy systems...")
        
        self.stt_service.stop_processing()
        
        if self.camera:
            try:
                self.camera.stop()
                print("📸 [VISION] Camera systems offline")
            except Exception as e:
                print(f"⚠️ [VISION] Camera cleanup error: {e}")
        
        self.is_running = False
        print("[VAULT-TEC] All systems offline. Stay safe in the wasteland!")


def get_server_url(provided_url=None):
    """Auto-discover F.R.E.D. server or use provided URL"""
    if provided_url:
        if not provided_url.startswith(('http://', 'https://')):
            provided_url = f'http://{provided_url}'
        print(f"🔗 Using provided F.R.E.D. server: {provided_url}")
        return provided_url
    
    print("🔍 Auto-discovering F.R.E.D. mainframe...")
    
    # Try common local addresses
    local_addresses = [
        'http://localhost:8000',
        'http://127.0.0.1:8000',
        'http://localhost:5000',
        'http://127.0.0.1:5000'
    ]
    
    for url in local_addresses:
        try:
            response = requests.get(f'{url}/health', timeout=2)
            if response.status_code == 200:
                print(f"✅ Found F.R.E.D. mainframe at: {url}")
                return url
        except:
            continue
    
    # Check tunnel info
    try:
        with open('../tunnel_info.json', 'r') as f:
            tunnel_info = json.load(f)
            tunnel_url = tunnel_info.get('public_url')
            if tunnel_url:
                print(f"🌐 Found tunnel URL: {tunnel_url}")
                return tunnel_url
    except:
        pass
    
    print("❌ [CRITICAL] No F.R.E.D. server found!")
    print("\n[VAULT-TEC] Troubleshooting protocols:")
    print("1. Start F.R.E.D. server: python app.py")
    print("2. Use local URL: --server http://localhost:8000")
    print("3. Use ngrok tunnel: --server https://xxx.ngrok.io")
    
    raise Exception("❌ No F.R.E.D. server found. Please specify --server URL")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="F.R.E.D. Pi Glasses Client with Local STT")
    parser.add_argument('--server', help='F.R.E.D. server URL (auto-discovery if not provided)')
    args = parser.parse_args()
    
    try:
        server_url = get_server_url(args.server)
        client = FREDPiClient(server_url)
        await client.start()
        
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Field operative terminating connection")
    except Exception as e:
        print(f"\n[CRITICAL] System failure: {e}")
        print("\n[VAULT-TEC] Troubleshooting protocols:")
        print("1. Verify F.R.E.D. mainframe is operational")
        print("2. Check wasteland communication network")
        print("3. Try manual server specification with --server")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main()) 