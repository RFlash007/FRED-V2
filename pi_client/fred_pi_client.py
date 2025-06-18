#!/usr/bin/env python3
"""
F.R.E.D. Pi Glasses Client - Consolidated Local STT & Communication
Vault-Tec Approved™ | OLLIE-TEC Advanced Wasteland Computing Division

This is the single startup file for F.R.E.D. Pi glasses that handles:
- Local speech-to-text processing (Vosk small English model)
- WebRTC communication with F.R.E.D. mainframe
- Camera vision capture
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
from typing import Optional, Callable
import vosk

# Import Ollie-Tec theming
from ollietec_theme import apply_theme, banner

# WebRTC imports
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

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
    """F.R.E.D. Pi Glasses Client - WebRTC interface with local STT"""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.stt_service = FREDPiSTTService()
        self.is_running = False
        
        # WebRTC setup
        self.pc = None
        self.data_channel = None
        
        # Camera setup  
        self.camera = None
        
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
        
        # Setup WebRTC connection
        await self._setup_webrtc()
        
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

    async def _setup_webrtc(self):
        """Setup WebRTC connection to F.R.E.D. server"""
        try:
            print("🔗 [WEBRTC] Establishing secure link to F.R.E.D. mainframe...")
            
            # Configure ICE servers
            try:
                ice_servers = [
                    RTCIceServer("stun:stun.l.google.com:19302"),
                    RTCIceServer("stun:stun1.l.google.com:19302")
                ]
                config = RTCConfiguration(iceServers=ice_servers)
                self.pc = RTCPeerConnection(configuration=config)
            except Exception:
                print("🔧 Using legacy WebRTC configuration")
                self.pc = RTCPeerConnection()
            
            # Set up data channel for communication
            self.data_channel = self.pc.createDataChannel('chat')
            
            @self.data_channel.on('open')
            def on_open():
                print('[VAULT-NET] Secure connection established with F.R.E.D. mainframe!')
                print('[PIP-BOY] Audio/visual sensors ONLINE - ready for wasteland operations...')
            
            @self.data_channel.on('message')
            def on_message(message):
                if message.startswith('[HEARTBEAT_ACK]'):
                    # Silent acknowledgment
                    pass
                elif message.startswith('[ACK]'):
                    ack_text = message.replace('[ACK] ', '')
                    print(f"[F.R.E.D.] {ack_text}")
                elif message.startswith('[AUDIO_BASE64:'):
                    # Handle incoming audio from F.R.E.D.
                    try:
                        header_end = message.find(']')
                        format_info = message[14:header_end]
                        audio_b64 = message[header_end + 1:]
                        
                        print(f"[TRANSMISSION] Incoming voice data from F.R.E.D. ({format_info})")
                        self._play_audio_from_base64(audio_b64, format_info)
                        
                    except Exception as e:
                        print(f"[ERROR] Audio processing failure: {e}")
                else:
                    print(f'\n[F.R.E.D.] {message}')
                
                if not message.startswith('[HEARTBEAT_ACK]'):
                    print('[PIP-BOY] Standing by for commands...')
            
            @self.data_channel.on('close')
            def on_close():
                print('[CRITICAL] Connection to F.R.E.D. mainframe terminated')
                raise Exception("Data channel closed")
            
            # Create offer and connect
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            print(f"🔗 Connecting to {self.server_url}/offer...")
            
            headers = {
                'Authorization': 'Bearer fred_pi_glasses_2024',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(f'{self.server_url}/offer?client_type=local_stt', json={
                'sdp': self.pc.localDescription.sdp,
                'type': self.pc.localDescription.type
            }, headers=headers, timeout=15)
            
            if response.status_code == 200:
                answer = RTCSessionDescription(**response.json())
                await self.pc.setRemoteDescription(answer)
                print("🚀 F.R.E.D. Pi Glasses connected and ready!")
            else:
                print(f"❌ Server error: {response.status_code}")
                raise Exception(f"Server returned {response.status_code}")
                
        except Exception as e:
            print(f"❌ [WEBRTC] Connection failed: {e}")
            raise
    
    async def _run_main_loop(self):
        """Main operational loop"""
        start_time = time.time()
        heartbeat_interval = 30
        last_heartbeat = start_time
        
        while self.is_running:
            try:
                await asyncio.sleep(1)
                
                # Send heartbeat periodically
                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    if self.data_channel and self.data_channel.readyState == 'open':
                        self.data_channel.send('[HEARTBEAT]')
                        last_heartbeat = current_time
                        if int(current_time) % 120 == 0:  # Every 2 minutes
                            print("[VITAL-MONITOR] Pip-Boy status confirmed")
                    else:
                        raise Exception("Data channel not open")
                        
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Field operative terminating connection")
                break
            except Exception as e:
                print(f"[CRITICAL] Connection to mainframe lost: {e}")
                break
        
        await self._cleanup()
    
    def _handle_transcription(self, text: str):
        """Handle transcribed speech from STT service"""
        print(f"🎤 [COMMAND] '{text}'")
        
        # Send transcription via WebRTC data channel
        if self.data_channel and self.data_channel.readyState == 'open':
            try:
                # Send as JSON for local STT client
                message = json.dumps({
                    'type': 'transcription',
                    'text': text
                })
                self.data_channel.send(message)
                print(f"📡 [TRANSMITTED] '{text}' sent to F.R.E.D. mainframe")
            except Exception as e:
                print(f"❌ [COMM] Transmission failed: {e}")
        else:
            print("❌ [COMM] No data channel available")
    
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
        
        if self.pc:
            await self.pc.close()
        
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
        'http://localhost:8080',  # WebRTC server port
        'http://127.0.0.1:8080',
        'http://localhost:5000',
        'http://127.0.0.1:5000'
    ]
    
    for url in local_addresses:
        try:
            response = requests.get(f'{url}/', timeout=2)
            if response.status_code == 200:
                print(f"✅ Found F.R.E.D. mainframe at: {url}")
                return url
        except:
            continue
    
    # Check tunnel info
    try:
        with open('../tunnel_info.json', 'r') as f:
            tunnel_info = json.load(f)
            tunnel_url = tunnel_info.get('webrtc_server')
            if tunnel_url:
                print(f"🌐 Found tunnel URL: {tunnel_url}")
                return tunnel_url
    except:
        pass
    
    print("❌ [CRITICAL] No F.R.E.D. server found!")
    print("\n[VAULT-TEC] Troubleshooting protocols:")
    print("1. Start F.R.E.D. WebRTC server: python start_fred_with_webrtc.py")
    print("2. Use local URL: --server http://localhost:8080")
    print("3. Use ngrok tunnel URL")
    
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