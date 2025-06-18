#!/bin/bash

echo "🤖 F.R.E.D. Pi Client Local STT Installation"
echo "============================================="
echo "Setting up on-device speech recognition with Vosk small English model"
echo

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for Vosk
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
    wget \
    unzip \
    curl

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Download Vosk small English model
echo "🧠 Downloading Vosk small English model..."
VOSK_MODEL="vosk-model-small-en-us-0.15"
VOSK_URL="https://alphacephei.com/vosk/models/${VOSK_MODEL}.zip"
VOSK_HOME="${HOME}/${VOSK_MODEL}"

if [ ! -d "$VOSK_HOME" ]; then
    echo "📥 Downloading ${VOSK_MODEL}..."
    cd "$HOME"
    wget "$VOSK_URL"
    
    if [ $? -eq 0 ]; then
        echo "📂 Extracting model..."
        unzip "${VOSK_MODEL}.zip"
        rm "${VOSK_MODEL}.zip"
        echo "✅ Vosk model installed at: $VOSK_HOME"
    else
        echo "❌ Model download failed. You can manually download:"
        echo "   wget $VOSK_URL"
        echo "   unzip ${VOSK_MODEL}.zip -d ~/"
    fi
else
    echo "✅ Vosk model already exists at: $VOSK_HOME"
fi

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
        print('✅ Vosk model initialization successful!')
        print('🎯 Ready for local speech recognition')
        pi_stt_service.stop_processing()
    else:
        print('❌ STT initialization failed')
        print('Make sure the Vosk model is downloaded correctly')
except Exception as e:
    print(f'❌ STT test failed: {e}')
"

echo
echo "🎉 F.R.E.D. Pi Client Local STT Installation Complete!"
echo
echo "📝 Next Steps:"
echo "1. Run the client: python3 client.py --server https://your-ngrok-url"
echo "2. Say a wake word like 'Hey Fred' to activate"
echo "3. Speech will be processed locally on the Pi using Vosk!"
echo
echo "💡 Performance Tip: Vosk small English model is optimized for Pi hardware"
echo "🔋 Battery Note: Local processing uses more CPU but reduces network usage"
echo "🎯 Model: ${VOSK_MODEL} (~40MB)"
echo 