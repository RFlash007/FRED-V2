#!/usr/bin/env python3
"""
Stewie Voice Cloning Service for F.R.E.D.
Uses Coqui XTTS-v2 for high-quality voice cloning
OLLIE-TEC Advanced Voice Synthesis Division
"""

import os
import torch
import torchaudio
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import librosa
import soundfile as sf
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from config import config
from ollie_print import olliePrint_simple


class StewieVoiceClone:
    """
    Advanced voice cloning system for Stewie Griffin using Coqui XTTS-v2
    Post-apocalyptic voice synthesis at its finest!
    """
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 22050  # XTTS v2 standard sample rate
        self.stewie_voice_samples = []
        self.voice_embedding = None
        
        # Create necessary directories
        self._create_directories()
        
        olliePrint_simple(f"[STEWIE-CLONE] Voice cloning system initializing on {self.device.upper()}", 'audio')
    
    def _create_directories(self):
        """Create necessary directories for voice cloning"""
        directories = [
            config.STEWIE_VOICE_SAMPLES_DIR,
            config.STEWIE_VOICE_CLONE_TEMP_DIR,
            config.STEWIE_VOICE_CLONE_MODEL_CACHE
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            olliePrint_simple(f"[STEWIE-CLONE] Directory ready: {directory}")
    
    def initialize(self):
        """Initialize the XTTS-v2 model for voice cloning"""
        try:
            if self.is_initialized:
                return True
            
            olliePrint_simple("[STEWIE-CLONE] Loading Coqui XTTS-v2 model...", 'audio')
            
            # Initialize TTS with XTTS-v2
            self.model = TTS(config.XTTS_MODEL_NAME).to(self.device)
            
            # Load and process Stewie voice samples
            self._load_stewie_samples()
            
            self.is_initialized = True
            olliePrint_simple("[STEWIE-CLONE] Voice cloning system ONLINE! Ready to replicate Stewie's voice", 'success')
            return True
            
        except Exception as e:
            olliePrint_simple(f"[STEWIE-CLONE] Initialization failed: {e}", 'error')
            return False
    
    def _load_stewie_samples(self):
        """Load and validate Stewie voice samples"""
        self.stewie_voice_samples = []
        samples_dir = Path(config.STEWIE_VOICE_SAMPLES_DIR)
        
        if not samples_dir.exists():
            olliePrint_simple(f"[STEWIE-CLONE] WARNING: Samples directory not found: {samples_dir}", 'warning')
            olliePrint_simple("[STEWIE-CLONE] Please add Stewie voice samples to the directory", 'warning')
            return
        
        # Look for Stewie voice samples
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        found_samples = []
        
        for ext in audio_extensions:
            found_samples.extend(samples_dir.glob(f'*{ext}'))
        
        if not found_samples:
            olliePrint_simple("[STEWIE-CLONE] No voice samples found! Please add Stewie audio clips", 'warning')
            return
        
        # Process each sample
        for sample_path in found_samples:
            try:
                processed_sample = self._process_voice_sample(sample_path)
                if processed_sample:
                    self.stewie_voice_samples.append(processed_sample)
                    olliePrint_simple(f"[STEWIE-CLONE] ✅ Loaded sample: {sample_path.name}")
            except Exception as e:
                olliePrint_simple(f"[STEWIE-CLONE] ❌ Failed to load {sample_path.name}: {e}", 'error')
        
        olliePrint_simple(f"[STEWIE-CLONE] Successfully loaded {len(self.stewie_voice_samples)} voice samples", 'success')
    
    def _process_voice_sample(self, sample_path: Path) -> Optional[str]:
        """Process a voice sample for optimal cloning"""
        try:
            # Load audio
            audio, sr = librosa.load(str(sample_path), sr=self.sample_rate)
            
            # Basic audio processing
            audio = self._enhance_audio(audio)
            
            # Save processed sample
            processed_path = Path(config.STEWIE_VOICE_CLONE_TEMP_DIR) / f"processed_{sample_path.stem}.wav"
            sf.write(str(processed_path), audio, self.sample_rate)
            
            return str(processed_path)
            
        except Exception as e:
            olliePrint_simple(f"[STEWIE-CLONE] Audio processing error for {sample_path}: {e}", 'error')
            return None
    
    def _enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Minimal audio processing to preserve original quality"""
        # Only basic normalization - no other processing
        audio = librosa.util.normalize(audio) * 0.8  # Very gentle normalization
        
        # Skip all other processing that might cause artifacts
        # No trimming, no preemphasis, no effects - keep it pure
        
        return audio
    
    def get_best_voice_sample(self) -> Optional[str]:
        """Get the best voice sample for cloning (longest clean sample)"""
        if not self.stewie_voice_samples:
            return None
        
        best_sample = None
        best_duration = 0
        
        for sample_path in self.stewie_voice_samples:
            try:
                audio, sr = librosa.load(sample_path, sr=None)
                duration = len(audio) / sr
                
                if duration > best_duration and duration < 30:  # Prefer samples under 30 seconds
                    best_duration = duration
                    best_sample = sample_path
            except:
                continue
        
        return best_sample or self.stewie_voice_samples[0]
    
    def clone_voice(self, text: str, output_path: str) -> bool:
        """
        Clone Stewie's voice to speak the given text
        
        Args:
            text: Text for Stewie to speak
            output_path: Where to save the generated audio
            
        Returns:
            bool: Success status
        """
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return False
            
            if not self.stewie_voice_samples:
                olliePrint_simple("[STEWIE-CLONE] No voice samples available for cloning", 'error')
                return False
            
            # Get the best voice sample
            speaker_wav = self.get_best_voice_sample()
            if not speaker_wav:
                olliePrint_simple("[STEWIE-CLONE] No suitable voice sample found", 'error')
                return False
            
            olliePrint_simple(f"[STEWIE-CLONE] Generating Stewie's voice: '{text[:50]}...'", 'audio')
            olliePrint_simple(f"[STEWIE-CLONE] Using voice sample: {Path(speaker_wav).name}")
            
            # Generate speech using minimal, stable settings
            self.model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=config.FRED_LANGUAGE,
                file_path=output_path
                # Using only basic parameters to avoid distortion
            )
            
            # Verify output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                olliePrint_simple(f"[STEWIE-CLONE] ✅ Voice cloning successful! Output: {file_size} bytes", 'success')
                return True
            else:
                olliePrint_simple("[STEWIE-CLONE] ❌ Output file not generated", 'error')
                return False
                
        except Exception as e:
            olliePrint_simple(f"[STEWIE-CLONE] Voice cloning failed: {e}", 'error')
            import traceback
            traceback.print_exc()
            return False
    
    def validate_samples(self) -> Dict[str, Any]:
        """Validate available voice samples and return statistics"""
        stats = {
            'total_samples': len(self.stewie_voice_samples),
            'samples': [],
            'total_duration': 0,
            'average_duration': 0,
            'quality_score': 0
        }
        
        for sample_path in self.stewie_voice_samples:
            try:
                audio, sr = librosa.load(sample_path, sr=None)
                duration = len(audio) / sr
                
                # Basic quality metrics
                rms = librosa.feature.rms(y=audio)[0]
                quality = np.mean(rms) * 100  # Simple quality metric
                
                sample_info = {
                    'path': sample_path,
                    'name': Path(sample_path).name,
                    'duration': duration,
                    'quality': quality,
                    'sample_rate': sr
                }
                
                stats['samples'].append(sample_info)
                stats['total_duration'] += duration
                
            except Exception as e:
                olliePrint_simple(f"[STEWIE-CLONE] Sample validation error: {e}", 'error')
        
        if stats['total_samples'] > 0:
            stats['average_duration'] = stats['total_duration'] / stats['total_samples']
            stats['quality_score'] = np.mean([s['quality'] for s in stats['samples']])
        
        return stats
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            temp_dir = Path(config.STEWIE_VOICE_CLONE_TEMP_DIR)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                olliePrint_simple("[STEWIE-CLONE] Temporary files cleaned up", 'success')
        except Exception as e:
            olliePrint_simple(f"[STEWIE-CLONE] Cleanup error: {e}", 'warning')


# Global instance
stewie_voice_clone = StewieVoiceClone()


def initialize_stewie_voice():
    """Initialize Stewie voice cloning system"""
    if config.STEWIE_VOICE_ENABLED:
        return stewie_voice_clone.initialize()
    else:
        olliePrint_simple("[STEWIE-CLONE] Voice cloning disabled in config", 'warning')
        return False


def generate_stewie_speech(text: str, output_path: str) -> bool:
    """Generate speech using Stewie's cloned voice"""
    if not config.STEWIE_VOICE_ENABLED:
        return False
    
    return stewie_voice_clone.clone_voice(text, output_path)


def validate_stewie_samples() -> Dict[str, Any]:
    """Validate Stewie voice samples"""
    return stewie_voice_clone.validate_samples()


def cleanup_stewie_voice():
    """Cleanup Stewie voice cloning resources"""
    stewie_voice_clone.cleanup() 