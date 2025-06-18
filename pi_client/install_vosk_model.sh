#!/bin/bash
# F.R.E.D. Vosk Model Installation Script
# Download and install the small English Vosk model for local STT

echo "🤖 [F.R.E.D. SETUP] Installing Vosk Speech Recognition Model"
echo "════════════════════════════════════════════════════════════"

# Create models directory
echo "📁 [SETUP] Creating models directory..."
mkdir -p models
cd models

# Check if model already exists
if [ -d "vosk-model-small-en-us-0.15" ]; then
    echo "✅ [MODEL] Vosk model already installed!"
    echo "📍 [LOCATION] $(pwd)/vosk-model-small-en-us-0.15"
    exit 0
fi

# Download model
echo "⬇️  [DOWNLOAD] Fetching Vosk small English model (~50MB)..."
echo "🌐 [SOURCE] https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

if command -v wget &> /dev/null; then
    wget -O vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
elif command -v curl &> /dev/null; then
    curl -L -o vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
else
    echo "❌ [ERROR] Neither wget nor curl available. Please install one of them."
    exit 1
fi

# Check download success
if [ ! -f "vosk-model-small-en-us-0.15.zip" ]; then
    echo "❌ [ERROR] Download failed!"
    exit 1
fi

echo "✅ [DOWNLOAD] Model downloaded successfully"

# Extract model
echo "📦 [EXTRACT] Unpacking model..."
if command -v unzip &> /dev/null; then
    unzip -q vosk-model-small-en-us-0.15.zip
elif command -v python3 &> /dev/null; then
    python3 -c "
import zipfile
with zipfile.ZipFile('vosk-model-small-en-us-0.15.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
"
else
    echo "❌ [ERROR] Neither unzip nor python3 available for extraction"
    exit 1
fi

# Cleanup
echo "🧹 [CLEANUP] Removing zip file..."
rm vosk-model-small-en-us-0.15.zip

# Verify installation
if [ -d "vosk-model-small-en-us-0.15" ]; then
    echo "✅ [SUCCESS] Vosk model installed successfully!"
    echo "📍 [LOCATION] $(pwd)/vosk-model-small-en-us-0.15"
    echo "📊 [SIZE] $(du -sh vosk-model-small-en-us-0.15 | cut -f1)"
    echo ""
    echo "🎤 [READY] Local speech recognition is now available!"
    echo "🚀 [USAGE] Run: python3 client.py --server YOUR_SERVER_URL"
else
    echo "❌ [ERROR] Model extraction failed!"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "🎯 [F.R.E.D.] Vault-Tec speech recognition protocols ONLINE" 