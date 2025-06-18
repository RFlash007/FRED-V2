#!/usr/bin/env python3
"""
Pi-Based Speech-to-Text Service for F.R.E.D. Glasses
Optimized for Raspberry Pi with Vosk small English model
"""

import time
import threading
import queue
import numpy as np
import logging
import json
import os
from typing import Optional, Callable
import vosk
from ollietec_theme import apply_theme

apply_theme()

logger = logging.getLogger(__name__)

class PiSTTService:
    """Optimized STT service for Raspberry Pi using Vosk small English"""
    
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.is_initialized = False
        
        # Audio processing configuration (optimized for Pi)
        self.sample_rate = 16000
        self.channels = 1
        self.block_duration = 0.5  # Shorter blocks for streaming
        self.blocksize = int(self.block_duration * self.sample_rate)
        
        # Speech detection settings
        self.speech_buffer = []
        self.last_speech_time = 0
        self.silence_duration = 1.0  # Time before processing complete utterance
        self.silence_threshold = 0.002  # Audio level threshold
        
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
        """Initialize the Vosk model"""
        if self.is_initialized:
            return True
            
        try:
            print("[PIP-BOY STT] Initializing voice recognition systems...")
            print("🔧 Loading Vosk small English model...")
            
            # Set log level to reduce Vosk verbosity
            vosk.SetLogLevel(-1)
            
            # Check if model exists, if not download it
            model_path = self._ensure_model_downloaded()
            if not model_path:
                return False
            
            # Initialize Vosk model and recognizer
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            
            # Configure recognizer for better performance
            self.recognizer.SetWords(True)  # Enable word-level timestamps
            self.recognizer.SetPartialWords(False)  # Disable for better performance
            
            print("✅ [PIP-BOY STT] Voice recognition ONLINE")
            print(f"📊 Model: Vosk small English")
            print(f"🔧 Sample rate: {self.sample_rate}Hz, optimized for Pi")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"[CRITICAL] STT initialization failed: {e}")
            print(f"❌ [PIP-BOY STT] Voice recognition FAILED: {e}")
            return False
    
    def _ensure_model_downloaded(self):
        """Ensure Vosk small English model is available"""
        model_dir = "vosk-model-small-en-us-0.15"
        model_path = os.path.join(os.path.expanduser("~"), model_dir)
        
        if os.path.exists(model_path) and os.path.isdir(model_path):
            print(f"📁 [MODEL] Found existing model at {model_path}")
            return model_path
        
        print("📥 [MODEL] Vosk small English model not found")
        print("🔗 [INFO] Please download the model manually:")
        print("   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        print("   unzip vosk-model-small-en-us-0.15.zip -d ~/")
        print(f"   Expected location: {model_path}")
        
        # Try alternative locations
        alt_paths = [
            "/opt/vosk-model-small-en-us-0.15",
            "/usr/local/share/vosk-model-small-en-us-0.15",
            "./vosk-model-small-en-us-0.15"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path) and os.path.isdir(alt_path):
                print(f"📁 [MODEL] Found model at alternative location: {alt_path}")
                return alt_path
        
        return None
    
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
        accumulated_audio = b''
        
        while self.is_processing:
            try:
                if self.audio_queue.empty():
                    time.sleep(0.05)
                    continue
                
                audio_chunk = self.audio_queue.get()
                
                # Calculate audio level for voice activity detection
                audio_level = np.abs(audio_chunk).mean()
                
                # Skip if too quiet
                if audio_level < self.silence_threshold:
                    # Check if we should process buffered speech
                    if self.is_listening and accumulated_audio:
                        if time.time() - self.last_speech_time > self.silence_duration:
                            self._finalize_recognition()
                            accumulated_audio = b''
                    continue
                
                # Convert to bytes for Vosk
                audio_bytes = audio_chunk.tobytes()
                accumulated_audio += audio_bytes
                
                # Process with Vosk
                if self.recognizer.AcceptWaveform(audio_bytes):
                    # Final result
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        self._handle_transcribed_text(text.lower())
                        self.last_speech_time = time.time()
                else:
                    # Partial result - update last speech time if we got something
                    partial = json.loads(self.recognizer.PartialResult())
                    if partial.get('partial', '').strip():
                        self.last_speech_time = time.time()
                        
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                time.sleep(0.1)
    
    def _finalize_recognition(self):
        """Get final recognition result"""
        try:
            final_result = json.loads(self.recognizer.FinalResult())
            text = final_result.get('text', '').strip()
            if text:
                self._handle_transcribed_text(text.lower())
                
            # Reset recognizer for next utterance
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            self.recognizer.SetPartialWords(False)
            
        except Exception as e:
            logger.error(f"Error finalizing recognition: {e}")
    
    def _handle_transcribed_text(self, text: str):
        """Handle transcribed text with wake word detection"""
        if not text:
            return
            
        print(f"🎙️ [DETECTED] '{text}'")
        
        # Performance monitoring
        self._transcription_count += 1
        if self._transcription_count % 25 == 0:
            elapsed = time.time() - self._start_time
            avg_time = elapsed / self._transcription_count if self._transcription_count > 0 else 0
            print(f"📊 [PERFORMANCE] {self._transcription_count} transcriptions, avg: {avg_time:.2f}s")
        
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
            if len(text.split()) > 0:  # Has words
                print(f"📝 [BUFFER] Adding: '{text}'")
                self.last_speech_time = time.time()
                self.speech_buffer.append(text)
                
                # Process immediately for more responsive interaction
                if len(self.speech_buffer) >= 1:  # Process after getting some speech
                    self._process_complete_utterance()
    
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

# Global instance
pi_stt_service = PiSTTService() 