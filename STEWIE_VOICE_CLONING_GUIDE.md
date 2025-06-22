# Stewie Voice Cloning for F.R.E.D.
## Advanced Voice Synthesis with Coqui XTTS-v2

### Overview

This implementation allows F.R.E.D. to speak with Stewie Griffin's voice using advanced voice cloning technology powered by Coqui XTTS-v2. The system processes Stewie voice samples and generates high-quality speech that plays on the Raspberry Pi glasses.

### Architecture

```
F.R.E.D. Server (Main Computer)
├── Stewie Voice Cloning Service (stewie_voice_clone.py)
│   ├── Coqui XTTS-v2 Model Loading
│   ├── Voice Sample Processing
│   ├── Speech Generation
│   └── Audio Enhancement
├── TTS Integration (app.py)
│   ├── Priority: Stewie Voice Cloning
│   ├── Fallback: Standard XTTS
│   └── Audio Transmission to Pi
└── Configuration (config.py)
    ├── Voice Cloning Settings
    ├── Quality Parameters
    └── Directory Structure

Raspberry Pi Glasses
├── Audio Reception (fred_pi_client.py)
├── Base64 Decoding
└── Audio Playback (aplay/paplay)
```

### Features

- **High-Quality Voice Cloning**: Uses Coqui XTTS-v2 for realistic voice synthesis
- **Multiple Voice Samples**: Supports multiple Stewie audio clips for better cloning
- **Automatic Sample Processing**: Enhances audio quality and removes silence
- **Fallback System**: Falls back to standard TTS if voice cloning fails
- **Real-time Generation**: Generates speech in real-time for Pi glasses
- **Quality Controls**: Configurable temperature, speed, and quality settings

### Installation

#### 1. Install Required Packages

```bash
# Install voice cloning dependencies
pip install -r requirements_stewie_voice.txt

# Or install manually:
pip install TTS>=0.22.0 librosa>=0.10.0 soundfile>=0.12.1 torch>=2.0.0 torchaudio>=2.0.0
```

#### 2. Run Setup Script

```bash
# Basic setup (creates directories, validates installation)
python setup_stewie_voice.py

# Setup with voice samples from a directory
python setup_stewie_voice.py --voice-samples /path/to/stewie/audio/clips

# Skip validation if packages are already installed
python setup_stewie_voice.py --skip-validation
```

#### 3. Add Voice Samples

Place Stewie voice samples in the `voice_samples/stewie/` directory:

**Supported Formats:**
- WAV (recommended)
- MP3
- M4A
- FLAC
- OGG
- AAC

**Sample Quality Guidelines:**
- Duration: 3-30 seconds per sample
- Quality: Clear speech with minimal background noise
- Content: Varied speech patterns and emotions
- Quantity: 3-10 samples for best results

### Configuration

Edit `config.py` to customize voice cloning:

```python
# Enable Stewie voice cloning
STEWIE_VOICE_ENABLED = True

# Voice sample settings
STEWIE_VOICE_SAMPLES_DIR = "voice_samples/stewie"
STEWIE_VOICE_CLONE_QUALITY = "high"  # "fast", "standard", "high"

# Speech parameters
STEWIE_VOICE_CLONE_SPEED = 1.0        # Speech speed (0.5-2.0)
STEWIE_VOICE_CLONE_TEMPERATURE = 0.7   # Voice variation (0.1-1.0)
STEWIE_VOICE_CLONE_REPETITION_PENALTY = 1.1  # Prevent repetition

# Advanced settings
STEWIE_VOICE_CLONE_TOP_K = 50         # Top-k sampling
STEWIE_VOICE_CLONE_TOP_P = 0.8        # Top-p sampling
```

### Usage

#### 1. Start F.R.E.D. with Voice Cloning

```bash
python start_fred_with_webrtc.py
```

You should see:
```
[STEWIE-CLONE] Voice cloning system initializing on CUDA
[STEWIE-CLONE] Found 5 voice samples
[STEWIE-CLONE] Total duration: 45.2s
[STEWIE-CLONE] Stewie voice cloning ACTIVE!
Voice synthesis ready on CUDA with STEWIE VOICE CLONING
```

#### 2. Connect Pi Glasses

```bash
# On Raspberry Pi
cd pi_client
python fred_pi_client.py
```

#### 3. Test Voice Cloning

Speak to F.R.E.D. through the Pi glasses or web interface. All responses will use Stewie's cloned voice.

### Voice Sample Optimization

#### Best Practices for Voice Samples

1. **Clean Audio**: Remove background noise and echo
2. **Consistent Quality**: Use similar recording conditions for all samples
3. **Varied Content**: Include different emotions and speech patterns
4. **Optimal Length**: 5-20 seconds per sample is ideal
5. **Clear Speech**: Avoid mumbling or very fast speech

#### Sample Processing Pipeline

The system automatically:
1. **Normalizes** audio levels
2. **Trims** silence from beginning and end
3. **Applies** gentle noise reduction
4. **Resamples** to 22050 Hz (XTTS-v2 standard)
5. **Selects** best sample for each generation

### Troubleshooting

#### Common Issues

**1. "No voice samples found"**
```bash
# Check if samples directory exists and contains audio files
ls -la voice_samples/stewie/
```

**2. "Voice cloning initialization failed"**
```bash
# Verify all packages are installed
python setup_stewie_voice.py --skip-test
```

**3. "CUDA out of memory"**
```bash
# Switch to CPU mode in config.py or reduce quality settings
STEWIE_VOICE_CLONE_QUALITY = "fast"
```

**4. "Poor voice quality"**
```bash
# Try different voice samples or adjust parameters
STEWIE_VOICE_CLONE_TEMPERATURE = 0.5  # Less variation
STEWIE_VOICE_CLONE_SPEED = 0.9        # Slower speech
```

#### Debug Commands

```bash
# Test voice cloning system
python -c "from stewie_voice_clone import *; test_voice_cloning()"

# Validate voice samples
python -c "from stewie_voice_clone import validate_stewie_samples; print(validate_stewie_samples())"

# Generate test speech
python -c "from stewie_voice_clone import generate_stewie_speech; generate_stewie_speech('Test message', 'test.wav')"
```

### Performance

#### System Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 8GB
- Storage: 2GB free space
- GPU: Optional (CPU works but slower)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB
- Storage: 5GB free space
- GPU: NVIDIA GPU with 4GB+ VRAM

#### Performance Benchmarks

| Quality Setting | Generation Time | GPU Memory | Audio Quality |
|----------------|-----------------|------------|---------------|
| Fast           | 2-5 seconds     | 2GB        | Good          |
| Standard       | 5-10 seconds    | 3GB        | Very Good     |
| High           | 10-20 seconds   | 4GB        | Excellent     |

### API Reference

#### StewieVoiceClone Class

```python
from stewie_voice_clone import StewieVoiceClone

# Initialize voice cloning
stewie = StewieVoiceClone()
stewie.initialize()

# Generate speech
success = stewie.clone_voice("What the devil?", "output.wav")

# Validate samples
stats = stewie.validate_samples()
print(f"Found {stats['total_samples']} samples")
```

#### Utility Functions

```python
from stewie_voice_clone import *

# Initialize system
initialize_stewie_voice()

# Generate speech
generate_stewie_speech("Hello there", "hello.wav")

# Validate samples
stats = validate_stewie_samples()

# Cleanup
cleanup_stewie_voice()
```

### Integration with F.R.E.D.

The voice cloning is automatically integrated into F.R.E.D.'s speech system:

1. **Priority System**: Stewie voice has priority over standard TTS
2. **Fallback**: Falls back to standard TTS if voice cloning fails
3. **Pi Integration**: Audio is automatically sent to Pi glasses
4. **Real-time**: Speech is generated in real-time during conversations

### Advanced Configuration

#### Custom Voice Processing

```python
# Custom audio enhancement
def custom_enhance_audio(audio):
    # Your custom processing here
    return enhanced_audio

# Apply in stewie_voice_clone.py
stewie._enhance_audio = custom_enhance_audio
```

#### Multiple Voice Profiles

```python
# Support multiple characters
VOICE_PROFILES = {
    'stewie': {
        'samples_dir': 'voice_samples/stewie',
        'temperature': 0.7,
        'speed': 1.0
    },
    'brian': {
        'samples_dir': 'voice_samples/brian',
        'temperature': 0.8,
        'speed': 0.9
    }
}
```

### License and Credits

- **Coqui XTTS-v2**: [Mozilla Public License 2.0]
- **F.R.E.D. Integration**: [OLLIE-TEC Advanced Computing Division]
- **Voice Samples**: [Provided by user - ensure proper licensing]

### Support

For issues and questions:
1. Check troubleshooting section above
2. Validate installation with setup script
3. Test with different voice samples
4. Check F.R.E.D. logs for error messages

---

**OLLIE-TEC Advanced Voice Synthesis Division**
*Post-apocalyptic voice cloning at its finest!* 