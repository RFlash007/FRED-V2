#!/usr/bin/env python3
"""
Pi-Based Speech-to-Text Service for F.R.E.D. Glasses
Optimized for Raspberry Pi with tiny.en model and int8 quantization
"""

import time
import threading
import queue
import numpy as np
import logging
from typing import Optional, Callable
from faster_whisper import WhisperModel
from ollietec_theme import apply_theme

apply_theme()

logger = logging.getLogger(__name__)

class PiSTTService:
    """Optimized STT service for Raspberry Pi with minimal latency"""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        
        # Audio processing configuration (optimized for Pi)
        self.sample_rate = 16000
        self.channels = 1
        self.block_duration = 3  # Shorter blocks for Pi responsiveness
        self.blocksize = int(self.block_duration * self.sample_rate)
        
        # Speech detection settings
        self.speech_buffer = []
        self.last_speech_time = 0
        self.silence_duration = 0.8  # Slightly longer for Pi processing
        self.silence_threshold = 0.002  # Adjusted for Pi microphones
        
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
        """Initialize the Whisper model optimized for Pi"""
        if self.is_initialized:
            return True
            
        try:
            print("[PIP-BOY STT] Initializing voice recognition systems...")
            print("🔧 Loading tiny.en model with int8 quantization...")
            
            # Optimal settings for Raspberry Pi
            self.model = WhisperModel(
                "tiny.en",              # Fastest English model
                device="cpu",           # Pi doesn't have CUDA
                compute_type="int8",    # Optimal quantization for Pi
                cpu_threads=4,          # Use all Pi cores efficiently
                num_workers=1           # Single worker to avoid memory issues
            )
            
            print("✅ [PIP-BOY STT] Voice recognition ONLINE")
            print(f"📊 Model: tiny.en (int8 quantized)")
            print(f"🔧 CPU threads: 4, Memory optimized for Pi")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"[CRITICAL] STT initialization failed: {e}")
            print(f"❌ [PIP-BOY STT] Voice recognition FAILED: {e}")
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

# Global instance
pi_stt_service = PiSTTService() 