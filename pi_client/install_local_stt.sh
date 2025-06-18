#!/bin/bash

echo "🤖 F.R.E.D. Pi Client Local STT Installation"
echo "============================================="
echo "Setting up on-device speech recognition with tiny.en model"
echo

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for faster-whisper
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    portaudio19-dev \
    alsa-utils \
    pulseaudio \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    cmake \
    pkg-config

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Download and cache the tiny.en model
echo "🧠 Downloading tiny.en Whisper model..."
python3 -c "
import os
from faster_whisper import WhisperModel
print('Downloading tiny.en model with int8 quantization...')
model = WhisperModel('tiny.en', device='cpu', compute_type='int8')
print('✅ Model downloaded successfully!')
print('Model cached for offline use')
"

# Test audio setup
echo "🎤 Testing audio setup..."
python3 -c "
import sounddevice as sd
import numpy as np
print('Available audio devices:')
print(sd.query_devices())
print()
print('Testing audio capture...')
try:
    # Test 1-second recording
    audio = sd.rec(int(1 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    level = np.abs(audio).mean()
    print(f'✅ Audio capture working! Level: {level:.4f}')
except Exception as e:
    print(f'❌ Audio test failed: {e}')
    print('Check microphone permissions and connections')
"

# Test the local STT service
echo "🧪 Testing local STT service..."
python3 -c "
try:
    from pi_stt_service import pi_stt_service
    print('✅ Pi STT service import successful!')
    
    if pi_stt_service.initialize():
        print('✅ Whisper model initialization successful!')
        print('🎯 Ready for local speech recognition')
        pi_stt_service.stop_processing()
    else:
        print('❌ STT initialization failed')
except Exception as e:
    print(f'❌ STT test failed: {e}')
"

echo
echo "🎉 F.R.E.D. Pi Client Local STT Installation Complete!"
echo
echo "📝 Next Steps:"
echo "1. Run the new client: python3 client_with_local_stt.py --server https://your-ngrok-url"
echo "2. Say a wake word like 'Hey Fred' to activate"
echo "3. Speech will be processed locally on the Pi!"
echo
echo "💡 Performance Tip: The tiny.en model is optimized for Pi Zero hardware"
echo "🔋 Battery Note: Local processing uses more CPU but reduces network usage"
echo 