#!/usr/bin/env python3
"""
Pi-Based Speech-to-Text Service for F.R.E.D. Glasses
Upgraded with dynamic calibration and robust utterance detection.
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
    """
    Optimized STT service for Raspberry Pi, featuring dynamic noise calibration
    and robust end-of-speech detection.
    """
    
    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.is_initialized = False
        
        # Audio processing configuration
        self.sample_rate = 16000
        self.channels = 1
        self.block_duration_s = 2  # Process audio in 2-second chunks
        
        # VAD & Speech detection settings
        self.speech_buffer = []
        self.last_speech_time = 0
        self.silence_duration_s = 0.8  # Seconds of silence to trigger end-of-speech
        self.silence_threshold = 0.01  # Initial threshold, will be calibrated
        self.calibration_duration_s = 2 # How long to listen for ambient noise
        
        # Processing control
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_listening = False # True when wake word is detected
        self.is_processing = False # True when the service is running
        self.processing_thread: Optional[threading.Thread] = None
        self.transcription_callback: Optional[Callable[[str], None]] = None
        
        # Wake/Stop words
        self.wake_words = [
            "fred", "hey fred", "okay fred", "hi fred", "excuse me fred"
        ]
        self.stop_words = [
            "goodbye", "bye fred", "stop listening", "that's all", "sleep now"
        ]
        
    def initialize(self):
        """Initialize the Whisper model optimized for Pi."""
        if self.is_initialized:
            return True
            
        try:
            print("[PIP-BOY STT] Initializing voice recognition systems...")
            print("🔧 Loading tiny.en model with int8 quantization...")
            
            # Optimal settings for Raspberry Pi
            self.model = WhisperModel(
                "tiny.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1
            )
            
            print("✅ [PIP-BOY STT] Voice recognition ONLINE")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"[CRITICAL] STT initialization failed: {e}")
            print(f"❌ [PIP-BOY STT] Voice recognition FAILED: {e}")
            return False
    
    def calibrate(self):
        """Calibrate the silence threshold based on ambient noise."""
        print("🎤 Calibrating microphone... Please be quiet for a moment.")
        
        calibration_samples = []
        calibration_end_time = time.time() + self.calibration_duration_s
        
        while time.time() < calibration_end_time:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
                audio_level = np.abs(audio_chunk).mean()
                calibration_samples.append(audio_level)
            except queue.Empty:
                continue
        
        if calibration_samples:
            # Set threshold to 150% of the average ambient noise
            ambient_noise = np.mean(calibration_samples)
            self.silence_threshold = ambient_noise * 1.5
            print(f"✅ Calibration complete. Noise threshold set to: {self.silence_threshold:.6f}")
        else:
            print("⚠️ Calibration failed. Could not get audio. Using default threshold.")
        
        # Clear the queue of any calibration audio
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
            
    def start_processing(self, callback: Callable[[str], None]):
        """Start the audio processing loop and calibrate."""
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        self.transcription_callback = callback
        self.is_processing = True
        
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        
        print("🎤 [PIP-BOY STT] Voice service activated.")
        return True
    
    def stop_processing(self):
        """Stop audio processing."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("🔇 [PIP-BOY STT] Voice service offline.")
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk for processing."""
        if self.is_processing and not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def _process_audio_loop(self):
        """Main audio processing loop."""
        # Run calibration at the start of the loop
        self.calibrate()
        print("👂 Listening for wake word...")
        
        audio_buffer = np.array([], dtype=np.float32)

        while self.is_processing:
            try:
                # Wait for a chunk of audio
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Append new data to the buffer
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                # Process buffer only if it's large enough
                if len(audio_buffer) < self.sample_rate * self.block_duration_s:
                    continue

                # --- Voice Activity Detection ---
                audio_level = np.abs(audio_buffer).mean()
                
                # If silent, check if we need to process a completed utterance
                if audio_level < self.silence_threshold:
                    if self.is_listening and self.speech_buffer and \
                       time.time() - self.last_speech_time > self.silence_duration_s:
                        self._process_complete_utterance()
                    # Clear buffer during silence
                    audio_buffer = np.array([], dtype=np.float32)
                    continue

                # --- Transcription ---
                # We have sound, so transcribe it
                text = self._transcribe_audio(audio_buffer)
                audio_buffer = np.array([], dtype=np.float32) # Clear buffer after processing
                
                if text:
                    self._handle_transcribed_text(text)

            except queue.Empty:
                # This is normal, just means no audio is coming in
                if self.is_listening and self.speech_buffer and \
                   time.time() - self.last_speech_time > self.silence_duration_s:
                    self._process_complete_utterance()
                continue
            except Exception as e:
                logger.error(f"Audio processing loop error: {e}")
                time.sleep(0.1)
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using optimized Whisper settings."""
        if self.model is None: return ""
        
        try:
            # Normalize audio to float32 [-1, 1] for Whisper
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            segments, _ = self.model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            
            return " ".join(segment.text for segment in segments).strip()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def _handle_transcribed_text(self, text: str):
        """Handle transcribed text with wake/stop word logic."""
        text = text.lower()
        print(f"🎙️  [DETECTED] '{text}'")
        
        if not self.is_listening:
            if any(wake_word in text for wake_word in self.wake_words):
                print("👋 [WAKE] Wake word detected! Listening for command...")
                self.is_listening = True
                self.speech_buffer = [] # Clear buffer for new command
                self.last_speech_time = time.time()
            return

        if self.is_listening:
            # Update last speech time
            self.last_speech_time = time.time()
            
            # Check for stop words
            if any(stop_word in text for stop_word in self.stop_words):
                print("💤 [SLEEP] Stop word detected. Returning to standby.")
                self.is_listening = False
                if self.transcription_callback:
                    self.transcription_callback("goodbye") # Signal end
                return
            
            # Add meaningful text to buffer
            if text:
                print(f"📝 [BUFFER] Adding: '{text}'")
                self.speech_buffer.append(text)
    
    def _process_complete_utterance(self):
        """Process the complete buffered utterance."""
        if not self.speech_buffer:
            return
            
        complete_text = " ".join(self.speech_buffer).strip()
        print(f"🗣️  [SENDING] Processing complete command: '{complete_text}'")
        
        self.speech_buffer = []
        
        # We've processed the command, go back to sleep until next wake word
        self.is_listening = False 
        print("👂 Listening for wake word...")
        
        if self.transcription_callback and complete_text:
            self.transcription_callback(complete_text)

# Singleton instance
pi_stt_service = PiSTTService()

if __name__ == "__main__":
    # Example usage for direct testing
    logging.basicConfig(level=logging.INFO)
    
    def test_callback(text: str):
        print(f"\n[CALLBACK] Received final text: '{text}'\n")

    pi_stt_service.start_processing(test_callback)
    
    try:
        # Simulate audio coming from another source
        # In the real app, client.py's audio capture loop calls add_audio_chunk
        print("\n--- Mocking audio input for testing ---")
        print("This test requires you to speak into the default microphone.")
        import sounddevice as sd

        def audio_callback(indata, frames, time, status):
            pi_stt_service.add_audio_chunk(indata.copy())

        with sd.InputStream(callback=audio_callback,
                            samplerate=pi_stt_service.sample_rate,
                            channels=pi_stt_service.channels,
                            dtype='float32'):
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n--- Shutting down test ---")
        pi_stt_service.stop_processing() 