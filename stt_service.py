import io
import threading
import time
import logging
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import torch
from collections import deque
import re
from datetime import datetime
import os
import psutil  # Add for system optimization
import sounddevice as sd  # ADD: Direct audio capture like old system
import queue  # ADD: Proper queue for audio processing
import sys  # ADD: For printing to stderr
from config import config

# === TERMINAL TRANSCRIPTION LOGGING ===
def print_transcription_to_terminal(text, source="TRANSCRIPTION"):
    """Print transcription results to terminal with clear formatting"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    separator = "=" * 50
    print(f"\n{separator}")
    print(f"🎤 {source} [{timestamp}]")
    print(f"📝 Text: '{text}'")
    print(f"{separator}\n")

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self):
        self.model = None
        self.is_initialized = False
        
        # FIX 1: Use proper queue instead of deque like old system
        self.audio_queue = queue.Queue()  # Changed from deque to queue.Queue
        self.speech_buffer = []  # Text-based buffering like v1
        self.processing_thread = None
        self.is_processing = False
        self.transcription_callback = None
        self.is_listening = False  # Binary listening state like v1
        self.last_speech_time = 0  # Track when we last detected speech
        self.processing_audio = False  # Prevent double processing
        
        # FIX 2: Match old system audio settings exactly
        self.sample_rate = config.STT_SAMPLE_RATE
        self.channels = config.STT_CHANNELS
        self.block_duration = config.STT_BLOCK_DURATION  # 5 second blocks like v1
        self.blocksize = config.get_stt_blocksize()
        
        # FIX 3: Match old system VAD settings exactly
        self.silence_threshold = config.STT_SILENCE_THRESHOLD  # Exact same as old system
        self.calibration_samples = []
        self.calibration_duration = config.STT_CALIBRATION_DURATION  # seconds
        self.silence_duration = config.STT_SILENCE_DURATION  # Exact same as old system (not 0.8)
        
        # ADD: Audio stream components like old system
        self.stream = None
        self.terminate_event = threading.Event()
        self.is_running = False
        
        # Wake words and responses like v1
        self.wake_words = config.WAKE_WORDS
        self.stop_words = config.STOP_WORDS
        self.acknowledgments = config.ACKNOWLEDGMENTS
        
    def initialize(self):
        """Initialize the Whisper model with maximum performance optimization"""
        if self.is_initialized:
            return True
            
        try:
            # Detect system capabilities
            cpu_cores = os.cpu_count()
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            # Try GPU first, fallback to CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "float32"
            
            logger.info(f"System specs: {cpu_cores} CPU cores, {available_memory:.1f}GB available RAM")
            logger.info(f"Initializing Whisper model on {device} with {compute_type}")
            
            # Optimize CPU threads - use all available cores
            cpu_threads = cpu_cores
            
            # FIX 4: Use medium model like old system for better accuracy
            model_size = config.STT_MODEL_SIZE  # Changed from "base" to "medium"
            logger.info(f"Using {model_size} model for better accuracy like old system")
            
            logger.info(f"Configuring Whisper with {cpu_threads} CPU threads")
            
            # Initialize with maximum performance settings
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,  # Use all available cores
                num_workers=1,  # Single worker for real-time processing
            )
            
            # Set process priority for better real-time performance
            try:
                current_process = psutil.Process()
                if os.name == 'nt':  # Windows
                    current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                else:  # Linux/Mac
                    current_process.nice(-10)  # Higher priority
                logger.info("Set high process priority for better real-time performance")
            except Exception as e:
                logger.warning(f"Could not set process priority: {e}")
            
            self.is_initialized = True
            logger.info(f"STT Service initialized successfully with {model_size} model using {cpu_threads} threads")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            return False
    
    # ADD: Direct audio callback like old system
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream - EXACTLY like old system"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())
    
    def start_processing(self, callback):
        """Start the audio processing thread with direct audio capture"""
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        print("🎤 Enhanced debugging enabled - Terminal transcription logging active")
        self.transcription_callback = callback
        self.is_processing = True
        self.is_running = True
        self.terminate_event.clear()
        
        # FIX 5: Start direct audio stream like old system
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.blocksize
            )
            self.stream.start()
            print("[DEBUG] Direct audio stream started successfully")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            return False
        
        # Calibrate silence threshold like v1
        self.calibrate_silence_threshold()
        
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        
        print("[DEBUG] Processing thread started successfully")
        print("Voice system initialized. Waiting for wake word...")  # Like Transcribe.py
        logger.info("STT processing started - waiting for wake word...")
        return True
    
    def stop_processing(self):
        """Stop the audio processing"""
        self.is_processing = False
        self.is_running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        self.terminate_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logger.info("STT processing stopped")
    
    def calibrate_silence_threshold(self):
        """Calibrate the silence threshold based on ambient noise - EXACTLY like old system"""
        print("Calibrating microphone... Please remain quiet.")
        start_time = time.time()
        
        while time.time() - start_time < self.calibration_duration:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_level = np.abs(audio_data).mean()  # Simple amplitude like v1
                self.calibration_samples.append(audio_level)
                print(f"[DEBUG] Calibration sample: {audio_level:.6f}")
            time.sleep(0.1)
        
        if self.calibration_samples:
            # Set threshold slightly above the average ambient noise like v1
            avg_noise = np.mean(self.calibration_samples)
            self.silence_threshold = avg_noise * 1.1  # Exact same formula as old system
            print(f"Silence threshold calibrated to: {self.silence_threshold:.6f}")
        else:
            print("[DEBUG] No calibration samples collected, using default threshold")
            print("Using default silence threshold")
        print()
    
    def _process_audio_loop(self):
        """Main audio processing loop - EXACTLY like old system"""
        while self.is_processing and not self.terminate_event.is_set():
            try:
                if self.audio_queue.empty():
                    time.sleep(0.1)
                    continue
                
                # Get audio chunk - flatten and convert like old system
                audio_chunk = self.audio_queue.get()
                audio_data = audio_chunk.flatten().astype(np.float32)  # EXACT same as old system
                
                # Calculate audio level - EXACT same as old system
                audio_level = np.abs(audio_data).mean()
                
                # Debug audio levels periodically - like old system
                if self.is_listening:
                    print(f"\rAudio level: {audio_level:.6f} (Threshold: {self.silence_threshold:.6f})", end="")
                
                # Only process audio if level is above threshold
                if audio_level > self.silence_threshold:
                    print(f"\nDetected text processing... Level: {audio_level:.6f} > {self.silence_threshold:.6f}")
                    try:
                        # FIX 6: Use EXACT transcription settings as old system
                        segments, _ = self.model.transcribe(
                            audio_data,  # Use flattened audio like old system
                            language="en",
                            beam_size=5,  # Same as old system
                            word_timestamps=True  # Same as old system
                            # REMOVED all the extra parameters that old system doesn't use
                        )

                        for segment in segments:
                            text = segment.text.strip().lower()
                            
                            if text and text != "thanks for watching!":
                                print(f"\nDetected text: {text}")  # EXACT same debug as old system
                                
                                # === TERMINAL LOGGING FOR TRANSCRIPTION ===
                                print_transcription_to_terminal(text, "SPEECH-TO-TEXT")
                                
                                # Check for wake words when not listening - EXACT same logic
                                if not self.is_listening:
                                    wake_word_found = any(wake_word in text for wake_word in self.wake_words)
                                    if wake_word_found:
                                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Wake word detected! Listening...")
                                        # REMOVED: Acknowledgment response - F.R.E.D. now listens silently
                                        # response = np.random.choice(self.acknowledgments)
                                        # print(f"[DEBUG] Sending acknowledgment: '{response}'")
                                        
                                        # === TERMINAL LOGGING FOR WAKE WORD ===
                                        print_transcription_to_terminal("WAKE WORD DETECTED -> Activating F.R.E.D. (Silent Mode)", "WAKE WORD")
                                        
                                        # REMOVED: Acknowledgment callback
                                        # if self.transcription_callback:
                                        #     # Send acknowledgment
                                        #     self.transcription_callback(f"_acknowledge_{response}")
                                        
                                        self.is_listening = True
                                        self.speech_buffer = []
                                        self.last_speech_time = time.time()
                                        print("[DEBUG] Now listening for commands silently. Buffer cleared.")
                                        continue

                                # Process speech while listening - EXACT same logic as old system
                                if self.is_listening:
                                    # Check for stop words - EXACT same logic
                                    stop_word_found = any(stop_word in text for stop_word in self.stop_words)
                                    if stop_word_found:
                                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stop word detected. Going to sleep.")
                                        
                                        # === TERMINAL LOGGING FOR STOP WORD ===
                                        print_transcription_to_terminal("STOP WORD DETECTED -> Deactivating F.R.E.D.", "STOP WORD")
                                        
                                        if self.transcription_callback:
                                            self.transcription_callback("goodbye")
                                        self.is_listening = False
                                        self.speech_buffer = []
                                        print("[DEBUG] Stopped listening. Buffer cleared.")
                                        continue
                                    
                                    # Add speech to buffer if it's not too short - EXACT same logic
                                    if len(text.split()) > 1:  # Only add if more than one word
                                        print(f"\nAdding to speech buffer: {text}")  # EXACT same debug as old system
                                        self.last_speech_time = time.time()
                                        self.speech_buffer.append(text)
                                        print(f"[DEBUG] Added to speech buffer ({len(text.split())} words): '{text}'")
                                        print(f"[DEBUG] Buffer now contains {len(self.speech_buffer)} segments: {self.speech_buffer}")
                                        
                                        # === TERMINAL LOGGING FOR BUFFERED SPEECH ===
                                        print_transcription_to_terminal(f"[BUFFERING] {text} (Total segments: {len(self.speech_buffer)})", "VOICE COMMAND")
                                    else:
                                        print(f"[DEBUG] Ignoring short text ({len(text.split())} word): '{text}'")

                    except Exception as e:
                        print(f"\nError during transcription: {str(e)}")  # EXACT same error message
                        logger.error(f"Error during transcription: {str(e)}")
                else:
                    # Check for complete utterance - EXACT same logic as old system
                    if (self.is_listening and self.speech_buffer and 
                        time.time() - self.last_speech_time > self.silence_duration):
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        complete_utterance = " ".join(self.speech_buffer)
                        print(f"\n[{timestamp}] Processing complete utterance: {complete_utterance}")  # EXACT same as old system
                        
                        # === TERMINAL LOGGING FOR COMPLETE UTTERANCE ===
                        print_transcription_to_terminal(f"COMPLETE COMMAND: '{complete_utterance}'", "FINAL TRANSCRIPTION")
                        
                        self.speech_buffer = []
                        
                        # Temporarily stop listening while processing - EXACT same as old system
                        self.is_listening = False
                        print("[DEBUG] Temporarily stopped listening for processing")
                        
                        try:
                            if self.transcription_callback:
                                print("\nProcessing message through callback...")  # EXACT same as old system
                                print(f"[DEBUG] Sending complete utterance to callback: '{complete_utterance}'")
                                self.transcription_callback(complete_utterance)
                            
                            # Resume listening after response - EXACT same as old system
                            self.is_listening = True
                            print("\nListening for next input...")  # EXACT same as old system
                            print("[DEBUG] Resumed listening after processing")
                        except Exception as e:
                            print(f"\nError in callback processing: {str(e)}")  # EXACT same error message
                            logger.error(f"Error in callback processing: {str(e)}")
                            self.is_listening = True  # Ensure we resume listening

            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                print(f"[DEBUG] Audio processing loop error: {e}")
                time.sleep(0.5)
            
            # ADD: Same timing as old system
            time.sleep(0.1)
    
    def _transcribe_audio(self, audio_chunk):
        """Transcribe audio using Whisper with optimized settings"""
        try:
            if len(audio_chunk) < 1600:  # Less than 0.1 seconds
                return ""
            
            # Optimized transcription settings for maximum performance
            segments, info = self.model.transcribe(
                audio_chunk,
                language="en",
                beam_size=5,  # Higher beam size for better accuracy
                word_timestamps=True,  # Better phrase detection
                condition_on_previous_text=False,  # Faster processing
                vad_filter=False,  # We handle our own VAD
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,  # Default
                log_prob_threshold=-1.0,  # Default
                no_speech_threshold=0.6,  # Slightly more permissive
                initial_prompt=None,  # No prompt bias
            )
            
            # Combine all segments efficiently
            text = ""
            for segment in segments:
                text += segment.text
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def transcribe_file(self, audio_file_path):
        """Transcribe an audio file (for testing)"""
        if not self.is_initialized:
            if not self.initialize():
                return ""
        
        try:
            segments, info = self.model.transcribe(
                audio_file_path,
                language="en",
                beam_size=5
            )
            
            text = ""
            for segment in segments:
                text += segment.text
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return ""

# Global STT service instance
stt_service = STTService() 