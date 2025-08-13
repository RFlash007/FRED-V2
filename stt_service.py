import io
import threading
import time
import numpy as np
import json
import vosk
import queue
import sys
import os
import sounddevice as sd
from collections import deque
import re
from datetime import datetime
from config import config
 
def olliePrint_simple(*args, **kwargs):
    """No-op printer to preserve runtime without output."""
    return None

def log_model_io(*args, **kwargs):
    """No-op model IO logger."""
    return None

 

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
    olliePrint_simple("".join(message))


class STTService:
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.is_initialized = False
        
        # Audio processing configuration (matching Pi client)
        self.sample_rate = config.STT_SAMPLE_RATE
        self.channels = config.STT_CHANNELS
        self.block_duration = config.STT_BLOCK_DURATION
        self.blocksize = config.get_stt_blocksize()
        
        # Processing control
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.is_listening = False
        self.is_processing = False
        self.processing_thread = None
        self.transcription_callback = None
        self.last_speech_time = 0
        self.is_speaking = False  # Indicates F.R.E.D. TTS is playing; used for barge-in mode
        self._barge_in_mode = False  # When True, only listen for interrupt grammar
        
        # Enhanced speech detection settings
        self.silence_duration = config.STT_SILENCE_DURATION
        self.silence_threshold = config.STT_SILENCE_THRESHOLD
        self.pi_silence_threshold = config.STT_PI_SILENCE_THRESHOLD
        self.calibration_duration = config.STT_CALIBRATION_DURATION
        self.calibration_samples = []
        
        # Audio stream components
        self.stream = None
        self.terminate_event = threading.Event()
        self.is_running = False
        
        # Wake words and responses
        self.wake_words = config.WAKE_WORDS
        self.stop_words = config.STOP_WORDS
        self.acknowledgments = config.ACKNOWLEDGMENTS
        
        # Counter to throttle debug logs for Pi audio chunks
        self._pi_chunk_counter = 0
        
        # Phrases to ignore due to Vosk artifacts on silence
        self._ignore_phrases = {
            "uh", "um", "ah", "eh", "mm", "hmm", "yeah", "yes", "no", "oh", "okay", "ok",
            "the", "a", "an", "and", "or", "but", "so", "well", "like", "you know",
            "i", "me", "my", "mine", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "will", "would", "could", "should",
            "may", "might", "must", "can", "cannot", "can't", "won't", "wouldn't",
            "couldn't", "shouldn't", "mustn't", "needn't", "daren't", "mayn't",
            "oughtn't", "mightn't"
        }
        
        # Throttle frequency of continuous audio-level debug prints
        self._last_level_log = 0.0
        self._debug_counter = 0
        
    def initialize(self):
        """Initialize the Vosk model with optimal settings"""
        if self.is_initialized:
            return True
            
        try:
            olliePrint_simple("Speech recognition initializing with Vosk...")
            
            # Set Vosk log level from configuration
            vosk.SetLogLevel(config.VOSK_LOG_LEVEL)
            
            # Look for Vosk model in configured locations
            model_paths = config.VOSK_MODEL_PATHS
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                olliePrint_simple("⚠️  Vosk model missing (speech-to-text unavailable)", level='warning')
                return False
            
            olliePrint_simple(f"Loading Vosk model from: {model_path}")
            self.model = vosk.Model(model_path)
            
            # Enhanced recognizer with better configuration
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            
            # Configure recognizer from configuration
            self.recognizer.SetWords(config.VOSK_ENABLE_WORDS)
            self.recognizer.SetPartialWords(config.VOSK_ENABLE_PARTIAL_WORDS)
            
            self.is_initialized = True
            olliePrint_simple("Speech recognition online - Vosk model ready")
            return True
            
        except Exception as e:
            olliePrint_simple(f"Speech recognition initialization failed: {e}", level='error')
            return False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            olliePrint_simple(f"Audio status: {status}")
        self.audio_queue.put((indata.copy(), False))  # (audio_data, from_pi)
    
    def start_processing(self, callback):
        """Start the audio processing thread with direct audio capture"""
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        olliePrint_simple("🎤 Enhanced debugging enabled - Terminal transcription logging active")
        self.transcription_callback = callback
        self.is_processing = True
        self.is_running = True
        self.terminate_event.clear()
        
        # Start direct audio stream (ONLY if no Pi glasses connected)
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.blocksize
            )
            self.stream.start()
            olliePrint_simple("[ARMLINK AUDIO] Direct audio stream active")
        except Exception as e:
            olliePrint_simple(f"Failed to start local audio stream (this is normal if using Pi glasses): {e}")
            self.stream = None
            olliePrint_simple("🍇 [ARMLINK AUDIO] Local capture off - using Pi glasses via WebRTC")
        
        # Calibrate silence threshold
        self.calibrate_silence_threshold()
        
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        
        olliePrint_simple("[ARMLINK STT] Processing thread active - awaiting wake word...")
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
        olliePrint_simple("STT processing stopped")
    
    def calibrate_silence_threshold(self):
        """Calibrate the silence threshold based on ambient noise"""
        if self.stream is None:
            self.silence_threshold = self.pi_silence_threshold
            olliePrint_simple(f"🍇 [ARMLINK AUDIO] Pi glasses mode - default threshold {self.silence_threshold:.6f}")
            return
            
        olliePrint_simple("[AUDIO] Calibrating microphone... Please remain quiet.")
        start_time = time.time()
        
        while time.time() - start_time < self.calibration_duration:
            if not self.audio_queue.empty():
                audio_data, _ = self.audio_queue.get()
                audio_level = np.abs(audio_data).mean()
                self.calibration_samples.append(audio_level)
                olliePrint_simple(f"[DEBUG] Calibration sample: {audio_level:.6f}")
            time.sleep(0.1)
        
        if self.calibration_samples:
            avg_noise = np.mean(self.calibration_samples)
            self.silence_threshold = avg_noise * 1.5  # More aggressive threshold for Vosk
            olliePrint_simple(f"[AUDIO] Silence threshold calibrated to: {self.silence_threshold:.6f}")
        else:
            olliePrint_simple("[WARNING] No calibration samples collected - using default silence threshold")
    
    def _process_audio_loop(self):
        """Main audio processing loop"""
        olliePrint_simple("[STT] Audio processing loop started")
        
        while self.is_processing and self.is_running:
            try:
                # If F.R.E.D. is speaking, enable barge-in limited grammar instead of pausing
                if self.is_speaking:
                    # Process audio but only detect interrupt keywords quickly
                    if not self.audio_queue.empty():
                        audio_data, from_pi = self.audio_queue.get()
                        if len(audio_data.shape) == 2 and audio_data.shape[1] == 2:
                            audio_data = np.mean(audio_data, axis=1)
                        audio_level = np.abs(audio_data).mean()
                        threshold = self.pi_silence_threshold if from_pi else self.silence_threshold
                        if audio_level > max(threshold * 1.3, threshold + 0.0005):  # slightly stricter during speaking
                            audio_int16 = (audio_data * 32768).astype(np.int16)
                            if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
                                result = json.loads(self.recognizer.Result())
                                text = result.get('text', '').strip().lower()
                                if text:
                                    # Hard stop if user says 'stop'
                                    if 'stop' in text:
                                        try:
                                            # Defer import to avoid circulars
                                            from conversation_orchestrator import InteractionOrchestrator
                                        except Exception:
                                            InteractionOrchestrator = None
                                        try:
                                            # Global orchestrator is created in app, but importing here is avoided; emit via socket instead
                                            import socketio
                                        except Exception:
                                            pass
                                        # Use callback pathway to main app via transcription callback
                                        if self.transcription_callback:
                                            self.transcription_callback('stop', from_pi)
                                        continue
                                    # Wake-word barge-in: require 'fred'
                                    if 'fred' in text:
                                        # Extract remainder after 'fred' and forward with wake word retained
                                        remainder = text.split('fred', 1)[1].strip()
                                        if remainder:
                                            if self.transcription_callback:
                                                self.transcription_callback(f"fred {remainder}", from_pi)
                                        else:
                                            if self.transcription_callback:
                                                self.transcription_callback('_acknowledge_', from_pi)
                                        continue
                    time.sleep(0.05)
                    continue
                
                # Process audio from queue
                if not self.audio_queue.empty():
                    audio_data, from_pi = self.audio_queue.get()
                    
                    # Calculate audio level
                    if len(audio_data.shape) == 2 and audio_data.shape[1] == 2:
                        audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono
                    
                    audio_level = np.abs(audio_data).mean()
                    
                    # Use appropriate threshold
                    threshold = self.pi_silence_threshold if from_pi else self.silence_threshold
                    
                    # Process speech detection
                    if audio_level > threshold:
                        self.last_speech_time = time.time()
                        
                        # Convert to format Vosk expects
                        audio_int16 = (audio_data * 32768).astype(np.int16)
                        
                        # Process with Vosk
                        if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
                            result = json.loads(self.recognizer.Result())
                            text = result.get('text', '').strip()
                            
                            if text and self._is_valid_text(text):
                                self._handle_transcribed_text(text, True, from_pi)
                        else:
                            # Partial result
                            partial_result = json.loads(self.recognizer.PartialResult())
                            partial_text = partial_result.get('partial', '').strip()
                            
                            if partial_text and self._is_valid_partial_text(partial_text):
                                self._handle_partial_text(partial_text, from_pi)
                    else:
                        # Check for complete utterance
                        if (self.is_listening and self.speech_buffer and 
                            time.time() - self.last_speech_time > self.silence_duration):
                            
                            complete_utterance = " ".join(self.speech_buffer)
                            olliePrint_simple(f"[COMMAND] Processing: '{complete_utterance}'")
                            print_transcription_to_terminal(f"FINAL COMMAND: '{complete_utterance}'", "FINAL TRANSCRIPTION")
                            
                            self.speech_buffer = []
                            self.is_listening = False
                            
                            try:
                                if self.transcription_callback:
                                    source_type = "Pi Glasses" if from_pi else "Local Computer"
                                    olliePrint_simple(f"[RELAY] Sending to F.R.E.D. from {source_type}")
                                    self.transcription_callback(complete_utterance, from_pi)
                                
                                # Resume listening
                                self.is_listening = True
                                olliePrint_simple("[READY] Listening for next command")
                            except Exception as e:
                                olliePrint_simple(f"[ERROR] Command processing failed: {str(e)}")
                                self.is_listening = True

            except Exception as e:
                olliePrint_simple(f"Error in audio processing loop: {e}")
                time.sleep(0.5)
            
            time.sleep(0.1)
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text is valid for processing"""
        text_clean = text.strip().lower()
        
        # Filter out empty or very short text
        if not text_clean or len(text_clean) < config.STT_MIN_WORD_LENGTH:
            return False
        
        # Filter out single character repetitions (e.g., "aaaa", "hhhh")
        if len(set(text_clean)) == 1 and len(text_clean) > 2:
            return False
        
        # Filter out common artifacts and filler words
        if text_clean in self._ignore_phrases:
            return False
        
        # Filter out very repetitive text
        words = text_clean.split()
        if len(words) > 1 and len(set(words)) == 1:  # All words are the same
            return False
        
        return True
    
    def _is_valid_partial_text(self, text: str) -> bool:
        """Check if partial text is valid for display"""
        text_clean = text.strip().lower()
        
        if not text_clean or len(text_clean) < config.STT_MIN_PHRASE_LENGTH:
            return False
        
        if text_clean in self._ignore_phrases:
            return False
        
        return True
    
    def _handle_transcribed_text(self, text: str, is_final: bool, from_pi: bool):
        """Handle transcribed text with wake word detection"""
        original_text = text.strip()
        text_lower = original_text.lower()
        
        if not text_lower:
            return
        
        # Check for wake words
        for wake_word in self.wake_words:
            if wake_word.lower() in text_lower:
                self.is_listening = True
                olliePrint_simple(f"🎯 Wake word detected: '{wake_word}' in '{original_text}'")
                print_transcription_to_terminal(f"WAKE WORD: '{wake_word}' detected", "WAKE DETECTION")
                
                # Remove wake word from text and add remainder if substantial
                text_without_wake = text_lower.replace(wake_word.lower(), "").strip()
                if text_without_wake and len(text_without_wake.split()) >= config.STT_MIN_PHRASE_LENGTH:
                    self.speech_buffer.append(text_without_wake)
                    self.last_speech_time = time.time()
                return
        
        # Check for stop words
        for stop_word in self.stop_words:
            if stop_word.lower() in text_lower:
                self.is_listening = False
                olliePrint_simple(f"🛑 Stop word detected: '{stop_word}'")
                print_transcription_to_terminal(f"STOP WORD: '{stop_word}' detected", "STOP DETECTION")
                self.speech_buffer = []
                return
        
        # Add to speech buffer if listening and text is substantial
        if self.is_listening and is_final:
            words = text_lower.split()
            if len(words) >= config.STT_MIN_PHRASE_LENGTH:
                self.speech_buffer.append(original_text)  # Keep original case
                self.last_speech_time = time.time()
                olliePrint_simple(f"📝 Speech: '{original_text}'")
                print_transcription_to_terminal(f"SPEECH: '{original_text}'", "SPEECH BUFFER")
                log_model_io("VOSK_STT", "[audio]", original_text)
    
    def _handle_partial_text(self, text: str, from_pi: bool):
        """Handle partial transcription results"""
        text = text.strip().lower()
        
        if not text or text in self._ignore_phrases:
            return
        
        # Show partial results for user feedback
        if len(text) > 3:  # Only show meaningful partial results
            source = "Pi" if from_pi else "Local"
            olliePrint_simple(f"🎤 [{source}] '{text}...'")
    
    def process_audio_from_webrtc(self, audio_data, from_pi=True):
        """Process audio data from WebRTC connection"""
        if not self.is_processing:
            return
        
        # Add to processing queue
        self.audio_queue.put((audio_data, from_pi))
        
        self._pi_chunk_counter += 1
        if self._pi_chunk_counter % 50 == 0:  # Reduced logging frequency
            olliePrint_simple(f"WebRTC audio chunks: {self._pi_chunk_counter}")
    
    def transcribe_file(self, audio_file_path):
        """Transcribe an audio file using Vosk"""
        if not self.is_initialized:
            if not self.initialize():
                return "STT service not initialized"
        
        try:
            import wave
            
            # Read the audio file
            with wave.open(audio_file_path, 'rb') as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != self.sample_rate:
                    olliePrint_simple("Audio file must be WAV format mono PCM.")
                    return "Audio format not supported"
                
                # Create a new recognizer for file processing
                file_recognizer = vosk.KaldiRecognizer(self.model, wf.getframerate())
                
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if file_recognizer.AcceptWaveform(data):
                        result = json.loads(file_recognizer.Result())
                        if result.get('text'):
                            results.append(result['text'])
                
                # Get final result
                final_result = json.loads(file_recognizer.FinalResult())
                if final_result.get('text'):
                    results.append(final_result['text'])

                transcript = ' '.join(results)
                log_model_io("VOSK_STT", f"[file:{audio_file_path}]", transcript)
                return transcript
                
        except Exception as e:
            olliePrint_simple(f"File transcription error: {e}")
            return f"Transcription failed: {e}"
    
    def set_speaking_state(self, speaking: bool):
        """Set whether F.R.E.D. is currently speaking (to avoid feedback)"""
        self.is_speaking = speaking
        if speaking:
            olliePrint_simple("[AUDIO] F.R.E.D. speaking - STT paused")
        else:
            olliePrint_simple("[AUDIO] F.R.E.D. finished - STT resumed")


# Global STT service instance
stt_service = STTService() 
