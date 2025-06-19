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
from ollietec_theme import apply_theme

apply_theme()

# === TERMINAL TRANSCRIPTION LOGGING ===
def print_transcription_to_terminal(text, source="TRANSCRIPTION"):
    """Print transcription results to terminal with clear formatting"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    separator = "=" * 50
    message = (
        f"\n{separator}\n"
        f"🎤 {source} [{timestamp}]\n"
        f"📝 Text: '{text}'\n"
        f"{separator}\n"
    )
    print("".join(message))

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self):
        self.model = None
        self.is_initialized = False
        
        # FIX 1: Use proper queue instead of deque like old system
        self.audio_queue = queue.Queue()  # Now stores (audio_data, from_pi) tuples
        self.speech_buffer = []  # Text-based buffering like v1
        self.processing_thread = None
        self.is_processing = False
        self.transcription_callback = None
        self.is_listening = False  # Binary listening state like v1
        self.last_speech_time = 0  # Track when we last detected speech
        self.processing_audio = False  # Prevent double processing
        self.is_speaking = False  # Indicates F.R.E.D. TTS is playing to avoid feedback loop
        
        # FIX 2: Match old system audio settings exactly
        self.sample_rate = config.STT_SAMPLE_RATE
        self.channels = config.STT_CHANNELS
        self.block_duration = config.STT_BLOCK_DURATION  # 5 second blocks like v1
        self.blocksize = config.get_stt_blocksize()
        
        # FIX 3: Match old system VAD settings exactly
        self.silence_threshold = config.STT_SILENCE_THRESHOLD  # Default for local mic (will be recalibrated)
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
        
        # Counter to throttle debug logs for Pi audio chunks
        self._pi_chunk_counter = 0
        
        # Phrases to ignore due to Whisper hallucinations on silence
        self._ignore_phrases = {
            "thanks for watching!", "thanks for watching", " thanks for watching!", " thanks for watching",
            "thank you for watching", "thank you for watching!", " thank you for watching",
            "please subscribe", "like and subscribe", "don't forget to subscribe",
            "see you next time", "see you later", "goodbye", " goodbye",
            "music", " music", "♪", "[music]", "[Music]", "(music)", "(Music)",
            "f.r.e.d.", "f.r.e.d", "f.r.e.d", "f r e d",
        }
        
        # Separate threshold for Pi audio
        self.pi_silence_threshold = config.STT_PI_SILENCE_THRESHOLD
        self._pi_threshold_set = False
        
        # Throttle frequency of continuous audio-level debug prints
        self._last_level_log = 0.0
        self._debug_counter = 0  # For concise debug output
        
    def initialize(self):
        """Initialize the Whisper model with maximum performance optimization"""
        if self.is_initialized:
            return True
            
        try:
            # Detect system capabilities
            cpu_cores = os.cpu_count()
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            # Try GPU first, fallback to CPU with quantization
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use quantized model for speed while maintaining accuracy
            if device == "cuda":
                compute_type = config.STT_COMPUTE_TYPE if hasattr(config, 'STT_COMPUTE_TYPE') else "int8"
            else:
                compute_type = "int8"  # Always use quantized on CPU
            
            logger.info(f"[SHELTER-NET] Speech recognition matrix: {cpu_cores} cores, {available_memory:.1f}GB RAM")
            logger.info(f"[NEURAL-NET] Initializing Whisper large-v3 (quantized) on {device.upper()}")
            
            # Optimize CPU threads - use most but not all cores to avoid blocking
            cpu_threads = max(1, cpu_cores - 1)
            
            # Use large-v3 model with quantization for best accuracy/speed balance
            model_size = config.STT_MODEL_SIZE
            logger.info(f"[ARC-MODE] Loading {model_size} model with {compute_type} quantization")
            
            # Initialize with optimized settings for real-time accuracy
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=1,  # Single worker for real-time processing
                download_root=None,  # Use default cache
                local_files_only=False  # Allow downloads if needed
            )
            
            # Set process priority for better real-time performance
            try:
                current_process = psutil.Process()
                if os.name == 'nt':  # Windows
                    current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                else:  # Linux/Mac
                    current_process.nice(-10)  # Higher priority
                logger.info("[SYSTEM] High priority mode enabled for real-time processing")
            except Exception as e:
                logger.warning(f"[WARNING] Could not set process priority: {e}")
            
            self.is_initialized = True
            logger.info(f"[SUCCESS] Speech recognition online - {model_size} quantized model ready")
            return True
            
        except Exception as e:
            logger.error(f"[CRITICAL] Speech recognition initialization failed: {e}")
            return False
    
    # ADD: Direct audio callback like old system
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream - EXACTLY like old system"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        self.audio_queue.put((indata.copy(), False))  # (audio_data, from_pi)
    
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
        
        # FIX 5: Start direct audio stream like old system (ONLY if no Pi glasses connected)
        # Skip local audio capture if Pi glasses are providing audio via WebRTC
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.blocksize
            )
            self.stream.start()
            print("[ARMLINK AUDIO] Direct audio stream active")
        except Exception as e:
            logger.warning(f"Failed to start local audio stream (this is normal if using Pi glasses): {e}")
            # Don't return False - Pi glasses audio will be handled via WebRTC
            self.stream = None
            print("🍇 [ARMLINK AUDIO] Local capture off - using Pi glasses via WebRTC")
        
        # Calibrate silence threshold like v1
        self.calibrate_silence_threshold()
        
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        
        print("[ARMLINK STT] Processing thread active - awaiting wake word...")
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
        if self.stream is None:
            self.silence_threshold = self.pi_silence_threshold  # Use Pi default threshold
            print(f"🍇 [ARMLINK AUDIO] Pi glasses mode - default threshold {self.silence_threshold:.6f}")
            return
            
        print("[AUDIO] Calibrating microphone... Please remain quiet.")
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
            print(f"[AUDIO] Silence threshold calibrated to: {self.silence_threshold:.6f}")
        else:
            print("[WARNING] No calibration samples collected - using default silence threshold")
    
    def _process_audio_loop(self):
        """Main audio processing loop - EXACTLY like old system"""
        while self.is_processing and not self.terminate_event.is_set():
            try:
                if self.audio_queue.empty():
                    time.sleep(0.1)
                    continue
                
                # Get audio chunk - handle both old and new queue formats
                queue_item = self.audio_queue.get()
                
                # Handle both old and new queue formats
                if isinstance(queue_item, tuple):
                    audio_chunk, from_pi = queue_item

                    # Switch threshold on first Pi audio
                    if from_pi and not self._pi_threshold_set:
                        self.silence_threshold = self.pi_silence_threshold
                        self._pi_threshold_set = True
                        print(f"[DEBUG] Pi silence threshold set to {self.silence_threshold:.6f}")

                    if from_pi:
                        # Throttle verbose logging to avoid console spam – print every 20th chunk (~10 s)
                        self._pi_chunk_counter += 1
                        if self._pi_chunk_counter % 20 == 0:
                            print(f"[DEBUG] Processing Pi audio chunk #{self._pi_chunk_counter} ({len(audio_chunk)} samples)")
                else:
                    audio_chunk, from_pi = queue_item, False
                    print(f"[DEBUG] Processing local audio chunk ({len(audio_chunk)} samples)")
                
                # ENHANCED: Proper audio normalization (preserve quality from WebRTC)
                if isinstance(audio_chunk, np.ndarray):
                    if audio_chunk.dtype == np.float32:
                        # Already normalized from WebRTC pipeline
                        audio_data = audio_chunk.flatten() if audio_chunk.ndim > 1 else audio_chunk
                    elif np.issubdtype(audio_chunk.dtype, np.integer):
                        # Legacy int16 conversion (local mic)
                        audio_data = audio_chunk.astype(np.float32) / 32768.0
                    else:
                        audio_data = audio_chunk.flatten().astype(np.float32)
                else:
                    # Fallback – ensure numpy
                    audio_data = np.asarray(audio_chunk, dtype=np.float32)
                    if np.max(np.abs(audio_data)) > 1.1:  # Not normalized
                        audio_data = audio_data / 32768.0
                
                # Calculate audio level - EXACT same as old system
                audio_level = np.abs(audio_data).mean()
                
                # Debug audio levels periodically - like old system
                if self.is_listening:
                    now = time.time()
                    if now - self._last_level_log > 0.5:  # print at most twice per second
                        print(f"\rAudio level: {audio_level:.6f} (Threshold: {self.silence_threshold:.6f})", end="")
                        self._last_level_log = now
                
                # Skip processing if F.R.E.D. is speaking to avoid feedback
                if self.is_speaking:
                    continue
                
                # Only process audio if level is above threshold
                if audio_level > self.silence_threshold:
                    # Concise processing indicator
                    self._debug_counter += 1
                    if self._debug_counter % 5 == 0:  # Every 5th detection
                        print(f"[PROCESSING] Audio level: {audio_level:.6f}")
                        
                    try:
                        # Optimized transcription settings for accuracy
                        segments_gen, info = self.model.transcribe(
                            audio_data,
                            language="en",
                            beam_size=config.STT_BEAM_SIZE if hasattr(config, 'STT_BEAM_SIZE') else 5,
                            temperature=config.STT_TEMPERATURE if hasattr(config, 'STT_TEMPERATURE') else 0.0,
                            condition_on_previous_text=False,  # Prevent context contamination
                            word_timestamps=True,
                            vad_filter=True,  # Use Whisper's built-in VAD
                            vad_parameters={"min_silence_duration_ms": 500},  # 0.5s minimum silence
                            initial_prompt="This is a voice command to F.R.E.D., an AI assistant.",  # Context prompt
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                            no_speech_threshold=0.5  # Lower threshold for better detection
                        )

                        # Convert generator to list for safe multiple passes
                        segments = list(segments_gen)
                        if from_pi and segments and self._debug_counter % 10 == 0:  # Concise Pi logging
                            print(f"[ARMLINK] Processed {len(segments)} speech segments")

                        for segment in segments:
                            text = segment.text.strip().lower()
                            
                            # Enhanced hallucination filtering
                            if any(phrase in text for phrase in self._ignore_phrases):
                                if self._debug_counter % 20 == 0:  # Occasional hallucination logging
                                    print(f"[FILTER] Blocked hallucination: '{text[:30]}...'")
                                continue
                            
                            # Filter very short or repetitive text
                            if len(text) < 2 or text.count(text[0]) > len(text) * 0.8:
                                continue
                            
                            if text and len(text.split()) > 0:
                                source_type = "Pi Glasses" if from_pi else "Local Computer"
                                print(f"[RECOGNITION] {source_type}: {text}")
                                
                                # === TERMINAL LOGGING FOR TRANSCRIPTION ===
                                print_transcription_to_terminal(f"[{source_type}] {text}", "SPEECH-TO-TEXT")
                                
                                # Check for wake words when not listening
                                if not self.is_listening:
                                    wake_word_found = any(wake_word in text for wake_word in self.wake_words)
                                    if wake_word_found:
                                        print(f"[WAKE] F.R.E.D. activated - listening mode engaged")
                                        print_transcription_to_terminal("F.R.E.D. ACTIVATED - Ready for commands", "WAKE WORD")
                                        
                                        self.is_listening = True
                                        self.speech_buffer = []
                                        self.last_speech_time = time.time()
                                        continue

                                # Process speech while listening
                                if self.is_listening:
                                    # Check for stop words
                                    stop_word_found = any(stop_word in text for stop_word in self.stop_words)
                                    if stop_word_found:
                                        print(f"[SLEEP] F.R.E.D. deactivated")
                                        print_transcription_to_terminal("F.R.E.D. DEACTIVATED", "STOP WORD")
                                        
                                        if self.transcription_callback:
                                            self.transcription_callback("goodbye", from_pi)
                                        self.is_listening = False
                                        self.speech_buffer = []
                                        continue
                                    
                                    # Add speech to buffer if it's substantial
                                    words = text.split()
                                    if len(words) > 1:  # More than one word
                                        self.last_speech_time = time.time()
                                        self.speech_buffer.append(text)
                                        print(f"[BUFFER] Added: '{text}' ({len(self.speech_buffer)} segments)")
                                        print_transcription_to_terminal(f"[COMMAND] {text} (Buffer: {len(self.speech_buffer)})", "VOICE COMMAND")

                    except Exception as e:
                        print(f"[ERROR] Transcription failed: {str(e)}")
                        logger.error(f"Error during transcription: {str(e)}")
                else:
                    # Check for complete utterance with improved logic
                    if (self.is_listening and self.speech_buffer and 
                        time.time() - self.last_speech_time > self.silence_duration):
                        
                        complete_utterance = " ".join(self.speech_buffer)
                        print(f"[COMMAND] Processing: '{complete_utterance}'")
                        print_transcription_to_terminal(f"FINAL COMMAND: '{complete_utterance}'", "FINAL TRANSCRIPTION")
                        
                        self.speech_buffer = []
                        self.is_listening = False
                        
                        try:
                            if self.transcription_callback:
                                source_type = "Pi Glasses" if from_pi else "Local Computer"
                                print(f"[RELAY] Sending to F.R.E.D. from {source_type}")
                                self.transcription_callback(complete_utterance, from_pi)
                            
                            # Resume listening
                            self.is_listening = True
                            print("[READY] Listening for next command")
                        except Exception as e:
                            print(f"[ERROR] Command processing failed: {str(e)}")
                            logger.error(f"Error in callback processing: {str(e)}")
                            self.is_listening = True

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

    def set_speaking_state(self, speaking: bool):
        """Externally notify STT service that F.R.E.D. is currently speaking (True) or silent (False)."""
        self.is_speaking = speaking

# Global STT service instance
stt_service = STTService() 