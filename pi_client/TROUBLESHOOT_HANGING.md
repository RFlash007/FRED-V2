# F.R.E.D. Pi Client Hanging - Troubleshooting Guide

## 🔍 Problem Description

Your Pi client connects to the WebRTC server successfully but then hangs without any further output:

```
✅ [SUCCESS] WebRTC connection established with local STT
🔗 [CONNECTION] State: connected
[HANGS HERE]
```

## 🎯 Root Cause Analysis

The hanging occurs because:

1. **WebRTC Connection** ✅ - Establishes successfully 
2. **Data Channel** ❌ - Fails to open or times out
3. **STT Service** ❌ - Cannot initialize due to missing Vosk model
4. **Audio Processor** ❌ - Fails silently, causing the main loop to hang

## 🔧 Solutions

### Solution 1: Install Missing Vosk Model (Recommended)

The client expects a Vosk model for local speech recognition:

```bash
# Quick fix - run the installation script
cd ~/FRED-V2/pi_client
chmod +x install_vosk_model.sh
./install_vosk_model.sh
```

**Manual Installation:**
```bash
# Create models directory
mkdir -p models
cd models

# Download Vosk small English model (~50MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

# Extract the model
unzip vosk-model-small-en-us-0.15.zip

# Verify installation
ls -la vosk-model-small-en-us-0.15/
```

### Solution 2: Use Original Server-Side STT

If you don't want local STT, check if you have an older client:

```bash
# Look for the original client file
ls -la client*.py

# If you have client_original.py or similar, use that instead
python3 client_original.py --server YOUR_SERVER_URL
```

### Solution 3: Debug Mode (Updated Client)

The updated client now includes better debugging. Run it to see exactly where it fails:

```bash
python3 client.py --server YOUR_SERVER_URL
```

You should now see more detailed output like:
```
🎤 [INITIALIZING] Local voice recognition...
❌ [MODEL ERROR] Vosk model not found at 'models/vosk-model-small-en-us-0.15'
📋 [INSTALLATION HELP] To fix this:
   1. Create models directory: mkdir -p models
   2. Download model: wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   3. Extract: unzip vosk-model-small-en-us-0.15.zip -d models/
   4. Or set VOSK_MODEL_PATH environment variable
💡 [ALTERNATIVE] Use server-side STT by running original client.py
❌ [CRITICAL] Failed to start voice recognition
💡 [FALLBACK] Data channel is open but STT failed
```

## 🚀 Expected Working Flow

When everything works correctly, you should see:

```
✅ [SUCCESS] WebRTC connection established with local STT
🔗 [CONNECTION] State: connected
✅ [DATA CHANNEL] Successfully opened!
🎤 [INITIALIZING] Local voice recognition...
📁 [MODEL] Loading from: models/vosk-model-small-en-us-0.15
✅ [PIP-BOY STT] Voice recognition ONLINE
🎤 [SUCCESS] Local voice recognition active
👂 Listening for wake word...
💓 [PING] Keep-alive sent
```

## 🔬 Advanced Debugging

### Check Model Installation:
```bash
ls -la models/vosk-model-small-en-us-0.15/
# Should show files like: am/, conf/, graph/, words.txt
```

### Test Audio Devices:
```bash
# List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
arecord -D hw:1,0 -d 3 -r 16000 -c 1 -f S16_LE test.wav
aplay test.wav
```

### Check Dependencies:
```bash
pip3 install vosk sounddevice numpy
```

### Manual STT Test:
```bash
cd pi_client
python3 pi_stt_service.py
# This will test the STT service directly
```

## 🎯 Environment Variables

You can customize the model path:

```bash
export VOSK_MODEL_PATH="/path/to/your/vosk/model"
python3 client.py --server YOUR_SERVER_URL
```

## 📊 Performance Notes

- **Vosk Model Size**: ~50MB download, ~120MB extracted
- **Memory Usage**: ~150-200MB during operation
- **CPU Usage**: 60-80% on Pi Zero 2W, 30-40% on Pi 4
- **First Load**: Takes 5-10 seconds to initialize

## 🔄 Alternative: Server-Side STT

If local STT is causing issues, you can fall back to server-side processing by modifying the client type in `client.py`:

```python
# Change this line:
'client_type': 'pi_glasses_with_local_stt'
# To:
'client_type': 'pi_glasses_server_stt'
```

This will send raw audio to the server instead of processing locally.

---

🎮 **Ready to test!** After installing the Vosk model, your F.R.E.D. Pi client should connect and work properly. 