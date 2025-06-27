#!/usr/bin/env python3
"""
F.R.E.D. Pi Glasses Client - Consolidated Local STT & Communication
ShelterNet Approved™ | OLLIE-TEC Advanced Wasteland Computing Division

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
from typing import Optional, Callable
import vosk
import librosa
import scipy

# Import Ollie-Tec theming
from ollietec_theme import apply_theme, banner
from ollie_print import olliePrint, olliePrint_simple

# WebRTC imports
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

# Apply theming to all prints
apply_theme()


class LightweightSpeakerVerification:
    """Ultra-lightweight speaker verification for Pi Zero 2W"""
    
    def __init__(self):
        self.user_profile = None  # Single user profile to save memory
        self.sample_rate = 16000
        self.mfcc_features = 13  # Minimal MFCC features
        self.is_enrolled = False
        
    def extract_mfcc_features(self, audio_data):
        """Extract minimal MFCC features"""
        try:
            # Use minimal settings for Pi Zero 2W
            mfccs = librosa.feature.mfcc(
                y=audio_data.astype(np.float32), 
                sr=self.sample_rate,
                n_mfcc=self.mfcc_features,
                n_fft=512,  # Smaller FFT for speed
                hop_length=160  # Larger hop for speed
            )
            return np.mean(mfccs, axis=1)  # Average across time
        except (ImportError, Exception):
            # Fallback: basic spectral features without librosa
            return self._basic_spectral_features(audio_data)
    
    def _basic_spectral_features(self, audio_data):
        """Fallback spectral features without librosa"""
        # Convert to frequency domain
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Extract basic features
        features = []
        features.append(np.mean(magnitude))  # Spectral centroid approximation
        features.append(np.std(magnitude))   # Spectral spread
        features.append(np.max(magnitude))   # Peak magnitude
        
        return np.array(features)
    
    def enroll_user(self, audio_chunks):
        """Create user voice profile from audio samples"""
        if not audio_chunks:
            return False
            
        features_list = []
        for chunk in audio_chunks:
            if len(chunk) > 0:
                features = self.extract_mfcc_features(chunk)
                if features is not None and len(features) > 0:
                    features_list.append(features)
        
        if features_list:
            # Create average profile
            self.user_profile = np.mean(features_list, axis=0)
            self.is_enrolled = True
            olliePrint_simple("✅ [VOICE-ID] User voice profile created", 'success')
            return True
        else:
            olliePrint_simple("❌ [VOICE-ID] Enrollment failed", 'error')
            return False
    
    def verify_speaker(self, audio_chunk):
        """Verify if audio matches enrolled user"""
        if not self.is_enrolled or self.user_profile is None or len(audio_chunk) == 0:
            return True, 1.0  # Allow all speech if not enrolled
        
        try:
            features = self.extract_mfcc_features(audio_chunk)
            if features is None or len(features) == 0:
                return True, 0.5  # Benefit of doubt
            
            # Cosine similarity (memory efficient)
            similarity = np.dot(features, self.user_profile) / (
                np.linalg.norm(features) * np.linalg.norm(self.user_profile) + 1e-8
            )
            
            # Convert to confidence score
            confidence = max(0.0, min(1.0, similarity))
            is_user = confidence > 0.6  # Tunable threshold
            
            return is_user, confidence
        except Exception as e:
            olliePrint_simple(f"Speaker verification error: {e}", 'warning')
            return True, 0.5  # Default to allowing speech


class FREDPiSTTService:
    """
    ShelterNet Speech Recognition Module
    Optimized for Pi with Vosk small English model + Enhanced Text Processing
    """
    
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.is_initialized = False
        
        # Audio processing configuration (optimized for Pi)
        self.sample_rate = 16000
        self.channels = 1
        
        # Enhanced speech detection settings
        self.speech_buffer = []
        self.partial_buffer = ""  # Track partial results separately
        self.last_speech_time = 0
        self.silence_duration = 0.8  # Reduced from 1.2 for faster processing
        self.silence_threshold = 0.002
        
        # Processing control
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_processing = False
        self.processing_thread = None
        self.transcription_callback: Optional[Callable] = None
        
        # Enhanced wake word detection
        self.wake_words = [
            "fred", "hey fred", "okay fred", 
            "hi fred", "excuse me fred", "fred are you there"
        ]
        self.stop_words = [
            "goodbye", "bye fred", "stop listening", 
            "that's all", "thank you fred", "sleep now"
        ]
        
        # Speaker verification
        self.speaker_verifier = LightweightSpeakerVerification()
        
        # Enhanced performance monitoring
        self._transcription_count = 0
        self._confidence_sum = 0.0
        self._start_time = time.time()
        
    def initialize(self):
        """Initialize the Vosk model optimized for Pi"""
        if self.is_initialized:
            return True
            
        try:
            olliePrint_simple("Voice recognition initializing...")
            
            # Set Vosk log level to reduce noise
            vosk.SetLogLevel(-1)
            
            # Look for Vosk model in common locations (Pi client specific paths)
            model_paths = [
                "models/vosk-model-small-en-us-0.15",         # Optimized for Pi resources (preferred)
                "../models/vosk-model-small-en-us-0.15",     # Pi-optimized fallback
                "./vosk-model-small-en-us-0.15",             # Local Pi-optimized
                "/home/raspberry/FRED-V2/pi_client/models/vosk-model-small-en-us-0.15",  # Pi installation path
                "/opt/vosk/models/vosk-model-small-en-us-0.15",  # System Pi installation
                "models/vosk-model-en-us-0.22",               # Larger model only if Pi has resources
                "../models/vosk-model-en-us-0.22",
                "./vosk-model-en-us-0.22"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                olliePrint_simple("Vosk model not found. Please run install_vosk_model.sh", level='error')
                return False
            
            olliePrint_simple(f"Loading model: {os.path.basename(model_path)}")
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            
            # Configure recognizer for better accuracy
            self.recognizer.SetWords(True)  # Enable word-level results
            self.recognizer.SetPartialWords(True)  # Enable partial word results
            
            self.is_initialized = True
            olliePrint_simple("Voice recognition ready", 'success')
            return True
            
        except Exception as e:
            olliePrint_simple(f"Voice recognition failed: {e}", level='error')
            return False

    def start_processing(self, callback: Callable):
        """Start the audio processing with wake word detection"""
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        olliePrint_simple("🎤 [ARMLINK STT] Starting audio processing...")
        self.transcription_callback = callback
        self.is_processing = True
        
        # Start processing thread for audio capture
        self.processing_thread = threading.Thread(
            target=self._audio_capture_loop, 
            daemon=True
        )
        self.processing_thread.start()
        
        olliePrint_simple("👂 [ARMLINK STT] Listening for wake word...")
        return True
    
    def stop_processing(self):
        """Stop audio processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        olliePrint_simple("🔇 [ARMLINK STT] Voice recognition offline", 'warning')
    
    def _audio_capture_loop(self):
        """Audio capture and processing loop using sounddevice"""
        try:
            import sounddevice as sd
            olliePrint_simple("🎤 [AUDIO] Starting audio capture...")
            
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
                olliePrint_simple("✅ [AUDIO] Audio capture ONLINE", 'success')
                
                last_buffer_check = time.time()
                
                while self.is_processing:
                    try:
                        if not self.audio_queue.empty():
                            audio_chunk = self.audio_queue.get()
                            self._process_audio_chunk(audio_chunk)
                        else:
                            time.sleep(0.01)
                        
                        # Periodic buffer check - ensure speech doesn't get stuck
                        current_time = time.time()
                        if current_time - last_buffer_check > 2.0:  # Check every 2 seconds
                            if self.is_listening and self.speech_buffer:
                                time_since_speech = current_time - self.last_speech_time
                                if time_since_speech > 2.0:  # Force processing if > 2 seconds old
                                    olliePrint_simple(f"⏰ [FORCED] Processing stuck buffer after {time_since_speech:.1f}s")
                                    self._process_complete_utterance()
                            last_buffer_check = current_time
                            
                    except Exception as e:
                        olliePrint_simple(f"Audio processing error: {e}", 'error')
                        time.sleep(0.1)
                        
        except ImportError:
            olliePrint_simple("❌ [AUDIO] sounddevice not available", 'error')
        except Exception as e:
            olliePrint_simple(f"❌ [AUDIO] Audio capture failed: {e}", 'error')
    
    def _transcribe_audio(self, audio_chunk: np.ndarray) -> tuple:
        """Enhanced transcribe audio using Vosk with confidence tracking"""
        try:
            if len(audio_chunk) < 1600:  # Less than 0.1 seconds
                return "", 0.0, False, []
            
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
                # Final result - this is what we want for speech buffer
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '').strip()
                confidence = result.get('conf', 0.0)
                words = result.get('result', [])
                
                # Show word-level confidence if available
                if words and text:
                    self._display_word_confidence(words)
                
                self._transcription_count += 1
                if confidence > 0:
                    self._confidence_sum += confidence
                
                return text, confidence, True, words  # is_final=True
            else:
                # Partial result - only for live feedback
                result = json.loads(self.recognizer.PartialResult())
                partial_text = result.get('partial', '').strip()
                
                # Show live feedback for partial results
                if partial_text and partial_text != self.partial_buffer:
                    olliePrint_simple(f"🎤 [LISTENING] {partial_text}...", 'muted')
                    self.partial_buffer = partial_text
                
                return partial_text, 0.0, False, []  # is_final=False
            
        except Exception as e:
            olliePrint_simple(f"Transcription error: {e}", 'error')
            return "", 0.0, False, []
    
    def _display_word_confidence(self, words):
        """Display word-level confidence with color coding"""
        confidence_display = []
        for word in words:
            word_text = word.get('word', '')
            word_conf = word.get('conf', 0.0)
            
            # Color code by confidence
            if word_conf > 0.8:
                confidence_display.append(f"'{word_text}'({word_conf:.2f})")
            elif word_conf > 0.6:
                confidence_display.append(f"'{word_text}'({word_conf:.2f})")
            else:
                confidence_display.append(f"'{word_text}'({word_conf:.2f})")
        
        if confidence_display:
            olliePrint_simple(f"📝 [CONFIDENCE] {' '.join(confidence_display)}")
    
    def _handle_transcribed_text(self, text: str, confidence: float, is_final: bool, words: list, audio_chunk: np.ndarray):
        """Process transcribed text with wake word detection and speaker verification"""
        if not text or len(text.strip()) < 2:
            return
            
        text = text.strip().lower()
        
        # Skip common Vosk artifacts
        if text in ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]:
            return
        
        # Speaker verification (simplified logging)
        is_authorized_user, speaker_confidence = self.speaker_verifier.verify_speaker(audio_chunk)
        
        if not is_authorized_user:
            if speaker_confidence < 0.3:  # Only log very low confidence
                olliePrint_simple(f"Unverified speaker (confidence: {speaker_confidence:.2f})", 'warning')
            return
        
        # Manual confidence calculation - use manual average instead of Vosk confidence
        manual_confidence = min(0.8, len(text) / 10.0)  # Longer text = higher confidence, capped at 0.8
        
        # Enhanced logging with manual confidence
        confidence_emoji = "🟢" if manual_confidence > 0.6 else "🟡" if manual_confidence > 0.3 else "🔴"
        if is_final:
            olliePrint_simple(f"{confidence_emoji} '{text}' (manual conf: {manual_confidence:.2f})")
        
        # Wake word detection
        wake_detected = any(wake in text for wake in self.wake_words)
        stop_detected = any(stop in text for stop in self.stop_words)
        
        if not self.is_listening and wake_detected:
            self.is_listening = True
            olliePrint_simple("Wake word detected - listening", 'success')
            # Send acknowledgment
            if is_final and self.transcription_callback:
                self.transcription_callback("_acknowledge_Wake word confirmed")
            return
        
        if self.is_listening and stop_detected:
            self.is_listening = False
            olliePrint_simple("Stop word detected - standby", 'warning')
            self._process_complete_utterance() # Process anything in buffer before stopping
            return
        
        # Process speech when listening
        if self.is_listening and is_final and len(text) > 2:
            # Clean up text by removing wake words
            for wake in self.wake_words:
                text = text.replace(wake, "").strip()
            
            if len(text) > 2:
                # Buffer the final text instead of sending immediately
                self.speech_buffer.append(text)

                # Update performance stats with manual confidence
                self._transcription_count += 1
                self._confidence_sum += manual_confidence
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """Enhanced audio chunk processing"""
        try:
            # Calculate audio level for voice activity detection
            audio_level = np.abs(audio_chunk).mean()
            
            # If too quiet, check if we should process a completed utterance
            if audio_level < self.silence_threshold:
                if self.is_listening and self.speech_buffer and self.last_speech_time > 0:
                    silence_duration = time.time() - self.last_speech_time
                    if silence_duration > self.silence_duration:
                        olliePrint_simple(f"🔇 [SILENCE] {silence_duration:.1f}s detected - processing utterance")
                        self._process_complete_utterance()
                return

            # If we are here, there is audible sound. Update the last speech time.
            self.last_speech_time = time.time()

            # Transcribe the audio chunk
            text, confidence, is_final, words = self._transcribe_audio(audio_chunk)
            
            # Handle the result of the transcription
            if text and len(text.strip()) > 0:
                self._handle_transcribed_text(text.strip(), confidence, is_final, words, audio_chunk)
                
        except Exception as e:
            olliePrint_simple(f"Audio chunk processing error: {e}", 'error')
    
    def _process_complete_utterance(self):
        """Process complete buffered utterance with enhanced filtering"""
        if not self.speech_buffer:
            return
            
        # Join with better sentence reconstruction
        complete_text = " ".join(self.speech_buffer).strip()
        
        # Filter out very short or low-quality utterances
        if len(complete_text.split()) < 2:  # Less than 2 words
            olliePrint_simple(f"⚠️ [FILTER] Utterance too short: '{complete_text}'", 'warning')
            self.speech_buffer = []
            return
        
        olliePrint_simple(f"🗣️ [COMPLETE] Processing: '{complete_text}'", 'success')
        
        # Clear buffer
        self.speech_buffer = []
        self.partial_buffer = ""  # Clear partial buffer too
        
        # Send to callback (everything after wake word - Option A)
        if self.transcription_callback:
            self.transcription_callback(complete_text)
        
        # Performance stats
        if self._transcription_count > 0:
            avg_confidence = self._confidence_sum / self._transcription_count
            olliePrint_simple(f"📊 [STATS] Avg confidence: {avg_confidence:.2f}")
        
        # Resume listening
        olliePrint_simple("👂 [READY] Listening for next command...")

    def enroll_user_voice(self):
        """Simple user voice enrollment process"""
        olliePrint_simple("🎤 [ENROLLMENT] Voice enrollment starting...")
        olliePrint_simple("📋 [INSTRUCTIONS] Say 'Hello Fred' 5 times when prompted")
        
        enrollment_samples = []
        
        for i in range(5):
            olliePrint_simple(f"🎙️ [{i+1}/5] Say 'Hello Fred' now...")
            
            # Collect audio for 3 seconds
            start_time = time.time()
            audio_buffer = []
            
            while time.time() - start_time < 3.0:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_buffer.append(chunk)
                time.sleep(0.01)
            
            if audio_buffer:
                # Concatenate audio chunks
                audio_sample = np.concatenate(audio_buffer)
                enrollment_samples.append(audio_sample)
                olliePrint_simple(f"✅ [{i+1}/5] Sample recorded", 'success')
            else:
                olliePrint_simple(f"❌ [{i+1}/5] No audio detected", 'error')
            
            if i < 4:  # Don't wait after last sample
                time.sleep(1)
        
        # Create voice profile
        success = self.speaker_verifier.enroll_user(enrollment_samples)
        if success:
            olliePrint_simple("🎉 [ENROLLMENT] Voice profile created successfully!", 'success')
            olliePrint_simple("🔐 [SECURITY] F.R.E.D. will now only respond to your voice")
        else:
            olliePrint_simple("❌ [ENROLLMENT] Voice enrollment failed", 'error')
        
        return success


class FREDPiClient:
    """F.R.E.D. Pi Glasses Client - WebRTC interface with local STT"""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.stt_service = FREDPiSTTService()
        self.is_running = False
        
        # WebRTC setup
        self.pc = None
        self.data_channel = None
        self.loop = None  # Store the event loop for thread-safe operations
        
        # Camera setup  
        self.camera = None
        
    async def start(self):
        """Start the F.R.E.D. Pi client"""
        olliePrint(banner("F.R.E.D. Pi Glasses v2.0"))
        olliePrint_simple("[SHELTER-CORE] Booting field AI interface...")
        
        # Store the event loop for thread-safe operations
        self.loop = asyncio.get_running_loop()
        
        # Initialize camera
        await self._init_camera()
        
        # Start STT service
        if not self.stt_service.start_processing(self._handle_transcription):
            olliePrint_simple("❌ [CRITICAL] STT initialization failed", 'error')
            return False
        
        # Setup WebRTC connection
        await self._setup_webrtc()
        
        self.is_running = True
        olliePrint_simple("[SHELTER-NET] F.R.E.D. Pi Glasses ONLINE!", 'success')
        olliePrint_simple("[ARMLINK] Ready for wasteland operations...", 'success')
        
        # Main loop
        await self._run_main_loop()
        
    async def _init_camera(self):
        """Initialize Picamera2 for vision"""
        try:
            olliePrint_simple("📸 [VISION] Initializing ArmLink camera systems...")
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Configure for optimal quality
            sensor_modes = self.camera.sensor_modes
            max_mode = max(sensor_modes, key=lambda x: x['size'][0] * x['size'][1])
            max_res = max_mode['size']
            olliePrint_simple(f"🎯 [VISION] Using native resolution {max_res}")
            
            config = self.camera.create_video_configuration(
                main={"size": max_res, "format": "RGB888"},
                controls={
                    "FrameRate": 5,  # Lower FPS for processing
                    "Brightness": 0.1,
                    "Contrast": 1.1,
                    "Saturation": 1.0,
                },
                buffer_count=2  # Minimize buffer for low latency
            )
            
            self.camera.configure(config)
            self.camera.start()
            olliePrint_simple("✅ [VISION] Camera systems ONLINE", 'success')
            
        except Exception as e:
            olliePrint_simple(f"❌ [VISION] Camera initialization failed: {e}", 'error')
            self.camera = None

    def create_local_tracks(self, video=True):
        """Create a video track for WebRTC"""
        tracks = []
        
        if video and self.camera:
            olliePrint_simple("🎥 Setting up video with Picamera2...")
            try:
                from aiortc import VideoStreamTrack
                import av

                class PiCamera2Track(VideoStreamTrack):
                    """Video track that streams video from Picamera2 camera"""
                    def __init__(self, camera):
                        super().__init__()
                        self.camera = camera
                        self.frame_count = 0
                        self.start_time = time.time()
                        olliePrint_simple("✅ Picamera2 video track initialized", 'success')

                    async def recv(self):
                        """Minimal video track to maintain WebRTC connection - no actual camera streaming"""
                        pts, time_base = await self.next_timestamp()
                        
                        # Send minimal placeholder frames to maintain WebRTC connection
                        # Actual image capture is now handled via data channel on-demand
                        array = np.zeros((240, 320, 3), dtype=np.uint8)  # Minimal placeholder
                        
                        # Convert to video frame for aiortc
                        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                        frame.pts = pts
                        frame.time_base = time_base

                        self.frame_count += 1
                        if self.frame_count == 1:
                            olliePrint_simple("🎥 [VIDEO] Minimal WebRTC video track established (on-demand capture mode)", 'success')
                        elif self.frame_count % 1000 == 0:  # Very infrequent logging
                            olliePrint_simple(f"📡 [VIDEO] Connection maintained ({self.frame_count} placeholder frames)")

                        return frame

                tracks.append(PiCamera2Track(self.camera))
                olliePrint_simple("✅ Picamera2 video track created successfully!", 'success')
                
            except Exception as e:
                olliePrint_simple(f"❌ Video track creation failed: {e}", 'error')
        
        return tracks

    async def _setup_webrtc(self):
        """Setup WebRTC connection to F.R.E.D. server"""
        try:
            olliePrint_simple("🔗 [WEBRTC] Establishing secure link to F.R.E.D. mainframe...")
            
            # Configure ICE servers
            try:
                ice_servers = [
                    RTCIceServer("stun:stun.l.google.com:19302"),
                    RTCIceServer("stun:stun1.l.google.com:19302")
                ]
                config = RTCConfiguration(iceServers=ice_servers)
                self.pc = RTCPeerConnection(configuration=config)
            except Exception:
                olliePrint_simple("🔧 Using legacy WebRTC configuration", 'warning')
                self.pc = RTCPeerConnection()
            
            # Suppress noisy ICE transaction timeout errors
            import logging
            logging.getLogger('aioice.stun').setLevel(logging.ERROR)
            
            # Set up data channel for communication
            self.data_channel = self.pc.createDataChannel('chat')
            
            @self.data_channel.on('open')
            def on_open():
                olliePrint_simple('[SHELTER-NET] Secure connection established with F.R.E.D. mainframe!', 'success')
                olliePrint_simple('[ARMLINK] Audio/visual sensors ONLINE - ready for wasteland operations...', 'success')
            
            @self.data_channel.on('message')
            def on_message(message):
                # Enhanced debug logging for audio troubleshooting
                if message.startswith('[HEARTBEAT_ACK]'):
                    # Silent acknowledgment
                    pass
                elif message == '[CAPTURE_REQUEST]':
                    # Handle fresh image capture request from F.R.E.D. vision service
                    olliePrint_simple("📸 [CAPTURE] Fresh image capture requested by F.R.E.D.")
                    self._capture_and_send_image()
                elif message.startswith('[ACK]'):
                    ack_text = message.replace('[ACK] ', '')
                    olliePrint_simple(f"[F.R.E.D.] {ack_text}")
                elif message.startswith('[AUDIO_BASE64:'):
                    # Handle incoming audio from F.R.E.D.
                    try:
                        header_end = message.find(']')
                        format_info = message[14:header_end]
                        audio_b64 = message[header_end + 1:]
                        
                        olliePrint_simple(f"[TRANSMISSION] Incoming voice data from F.R.E.D. ({format_info})")
                        self._play_audio_from_base64(audio_b64, format_info)
                        
                    except Exception as e:
                        olliePrint_simple(f"[ERROR] Audio processing failure: {e}", 'error')
                        import traceback
                        traceback.print_exc()
                else:
                    # Handle F.R.E.D.'s text responses - display them prominently on Pi terminal
                    if len(message.strip()) > 0:
                        olliePrint_simple(f'\n[F.R.E.D.] {message}', 'success')
                
                if not message.startswith('[HEARTBEAT_ACK]'):
                    olliePrint_simple('[ARMLINK] Standing by for commands...')
            
            @self.data_channel.on('close')
            def on_close():
                olliePrint_simple('[CRITICAL] Connection to F.R.E.D. mainframe terminated', 'error')
                raise Exception("Data channel closed")
            
            # Add video track. Audio is processed locally for STT and not streamed.
            tracks = self.create_local_tracks(video=True)
            
            if not tracks:
                olliePrint_simple("⚠️  No media tracks available - connecting with data channel only", 'warning')
            
            for track in tracks:
                self.pc.addTrack(track)
                track_kind = getattr(track, 'kind', 'unknown')
                track_type = type(track).__name__
                olliePrint_simple(f"📡 Added {track_kind} track ({track_type})")
            
            # Create offer and connect
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            olliePrint_simple(f"🔗 Connecting to {self.server_url}/offer...")
            
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
                olliePrint_simple("🚀 F.R.E.D. Pi Glasses connected and ready!", 'success')
            else:
                olliePrint_simple(f"❌ Server error: {response.status_code}", 'error')
                raise Exception(f"Server returned {response.status_code}")
                
        except Exception as e:
            olliePrint_simple(f"❌ [WEBRTC] Connection failed: {e}", 'error')
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
                            olliePrint_simple("[VITAL-MONITOR] ArmLink status confirmed")
                    else:
                        raise Exception("Data channel not open")
                        
            except KeyboardInterrupt:
                olliePrint_simple("\n[SHUTDOWN] Field operative terminating connection", 'warning')
                break
            except Exception as e:
                olliePrint_simple(f"[CRITICAL] Connection to mainframe lost: {e}", 'error')
                break
        
        await self._cleanup()
    
    def _handle_transcription(self, text: str):
        """Handle transcribed speech from STT service"""
        olliePrint_simple(f"🎤 [COMMAND] '{text}'", 'success')
        
        # Send transcription via WebRTC data channel using thread-safe approach
        if self.data_channel and self.data_channel.readyState == 'open' and self.loop:
            try:
                # Schedule the async operation in the main event loop
                future = asyncio.run_coroutine_threadsafe(
                    self._send_transcription_async(text), 
                    self.loop
                )
                # Wait for completion with timeout
                future.result(timeout=5.0)
                olliePrint_simple(f"📡 [TRANSMITTED] '{text}' sent to F.R.E.D. mainframe", 'success')
            except Exception as e:
                olliePrint_simple(f"❌ [COMM] Transmission failed: {e}", 'error')
        else:
            olliePrint_simple("❌ [COMM] No data channel available", 'error')
    
    async def _send_transcription_async(self, text: str):
        """Async helper to send transcription via data channel"""
        # Send as plain text - revert to original working format
        self.data_channel.send(text)
    
    def _capture_and_send_image(self):
        """Capture fresh high-resolution image and send to F.R.E.D. mainframe"""
        try:
            if not self.camera:
                olliePrint_simple("❌ [CAPTURE] No camera available", 'error')
                return
            
            # Capture high-resolution image
            image_array = self.camera.capture_array("main")
            height, width = image_array.shape[:2]
            
            olliePrint_simple(f"📸 [CAPTURE] Captured fresh {width}x{height} image")
            
            # Convert to PIL Image and compress for transmission
            from PIL import Image
            import io
            
            image = Image.fromarray(image_array)
            
            # Compress to JPEG for efficient transmission
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)  # High quality but efficient
            image_bytes = buffer.getvalue()
            
            # Encode to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Send via data channel with format header
            message = f"[IMAGE_DATA:jpeg]{image_b64}"
            
            if self.data_channel and self.data_channel.readyState == 'open':
                self.data_channel.send(message)
                olliePrint_simple(f"📡 [SENT] Fresh image sent to F.R.E.D. ({len(image_bytes)} bytes)", 'success')
            else:
                olliePrint_simple("❌ [CAPTURE] No data channel available", 'error')
                
        except Exception as e:
            olliePrint_simple(f"[CAPTURE] Image capture failed: {e}", 'error')
            import traceback
            traceback.print_exc()
    
    def _play_audio_from_base64(self, audio_b64, format_type='wav'):
        """Play audio from base64 data"""
        try:
            audio_data = base64.b64decode(audio_b64)
            olliePrint_simple(f"[AUDIO] F.R.E.D. transmission received ({len(audio_data)} bytes)")
            
            with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Try aplay first
            try:
                result = subprocess.run(['aplay', temp_file_path], check=True, capture_output=True, text=True)
                olliePrint_simple("[SUCCESS] F.R.E.D. voice transmission complete (aplay)", 'success')
            except subprocess.CalledProcessError as e:
                olliePrint_simple(f"[AUDIO] aplay failed: {e.stderr}", 'warning')
                # Try paplay as fallback
                try:
                    result = subprocess.run(['paplay', temp_file_path], check=True, capture_output=True, text=True)
                    olliePrint_simple("[SUCCESS] F.R.E.D. voice transmission complete (paplay)", 'success')
                except subprocess.CalledProcessError as e2:
                    olliePrint_simple(f"[AUDIO] paplay also failed: {e2.stderr}", 'error')
                    # Try mpg123 as last resort
                    try:
                        result = subprocess.run(['mpg123', temp_file_path], check=True, capture_output=True, text=True)
                        olliePrint_simple("[SUCCESS] F.R.E.D. voice transmission complete (mpg123)", 'success')
                    except subprocess.CalledProcessError as e3:
                        olliePrint_simple(f"[CRITICAL] All audio players failed - last error: {e3.stderr}", 'error')
            except FileNotFoundError:
                olliePrint_simple("[AUDIO] aplay not found, trying paplay...", 'warning')
                try:
                    result = subprocess.run(['paplay', temp_file_path], check=True, capture_output=True, text=True)
                    olliePrint_simple("[SUCCESS] F.R.E.D. voice transmission complete (paplay)", 'success')
                except Exception as e:
                    olliePrint_simple(f"[CRITICAL] Audio playback failed: {e}", 'error')
            
            # Cleanup temp file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass  # Silent cleanup
                
        except Exception as e:
            olliePrint_simple(f"[CRITICAL] Audio playback system failure: {e}", 'error')
            import traceback
            traceback.print_exc()
    
    async def _cleanup(self):
        """Cleanup resources"""
        olliePrint_simple("[CLEANUP] Shutting down ArmLink systems...", 'warning')
        
        self.stt_service.stop_processing()
        
        if self.camera:
            try:
                self.camera.stop()
                olliePrint_simple("📸 [VISION] Camera systems offline", 'warning')
            except Exception as e:
                olliePrint_simple(f"⚠️ [VISION] Camera cleanup error: {e}", 'warning')
        
        if self.pc:
            await self.pc.close()
        
        self.is_running = False
        olliePrint_simple("[SHELTER-CORE] All systems offline. Stay safe out there!", 'warning')


def get_server_url(provided_url=None):
    """Auto-discover F.R.E.D. server or use provided URL"""
    if provided_url:
        if not provided_url.startswith(('http://', 'https://')):
            provided_url = f'http://{provided_url}'
        olliePrint_simple(f"🔗 Using provided F.R.E.D. server: {provided_url}")
        return provided_url
    
    olliePrint_simple("🔍 Auto-discovering F.R.E.D. mainframe...")
    
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
                olliePrint_simple(f"✅ Found F.R.E.D. mainframe at: {url}", 'success')
                return url
        except:
            continue
    
    # Check tunnel info
    try:
        with open('../tunnel_info.json', 'r') as f:
            tunnel_info = json.load(f)
            tunnel_url = tunnel_info.get('webrtc_server')
            if tunnel_url:
                olliePrint_simple(f"🌐 Found tunnel URL: {tunnel_url}", 'success')
                return tunnel_url
    except:
        pass
    
    olliePrint_simple("❌ [CRITICAL] No F.R.E.D. server found!", 'error')
    olliePrint_simple("\n[SHELTER-CORE] Troubleshooting protocols:", 'warning')
    olliePrint_simple("1. Start F.R.E.D. WebRTC server: python start_fred_with_webrtc.py")
    olliePrint_simple("2. Use local URL: --server http://localhost:8080")
    olliePrint_simple("3. Use ngrok tunnel URL")
    
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
        olliePrint_simple("\n[SHUTDOWN] Field operative terminating connection", 'warning')
    except Exception as e:
        olliePrint_simple(f"\n[CRITICAL] System failure: {e}", 'error')
        olliePrint_simple("\n[SHELTER-CORE] Troubleshooting protocols:", 'warning')
        olliePrint_simple("1. Verify F.R.E.D. mainframe is operational")
        olliePrint_simple("2. Check wasteland communication network")
        olliePrint_simple("3. Try manual server specification with --server")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main()) 