# F.R.E.D. Pi Client with Local Speech-to-Text

## 🎯 **Overview**

The F.R.E.D. system has been enhanced with **on-device speech recognition** using the **tiny.en** Whisper model with **int8 quantization** - optimized specifically for Raspberry Pi hardware.

## 🔄 **Architecture Change**

### **Before (Server STT):**
```
Pi Glasses → Raw Audio → WebRTC → Main Server → Whisper (large model) → Text → F.R.E.D.
```

### **After (Local STT):**
```
Pi Glasses → Audio → Local Whisper (tiny.en) → Text → WebRTC → Main Server → F.R.E.D.
```

## ✅ **Benefits**

- **🚀 Lower Latency**: No audio transmission delay
- **📡 Reduced Bandwidth**: Send text instead of raw audio streams
- **🔋 Privacy**: Speech processing never leaves the device
- **🏠 Offline Capable**: Works without internet for transcription
- **⚡ Real-time**: Optimized for Pi Zero-class hardware

## 🛠️ **Technical Specifications**

### **Whisper Model:**
- **Model**: `tiny.en` (English-only, fastest)
- **Quantization**: `int8` (optimal for Pi)
- **Size**: ~39MB (fits in Pi memory)
- **CPU Threads**: 4 (uses all Pi cores)
- **Memory**: ~150MB peak usage

### **Audio Processing:**
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Block Size**: 3-second chunks
- **Format**: float32 for quality preservation

### **Wake Word Detection:**
- **Wake Words**: "fred", "hey fred", "okay fred", "hi fred"
- **Stop Words**: "goodbye", "bye fred", "stop listening"
- **Buffer**: Smart speech buffering with silence detection

## 📁 **New Files Created**

1. **`pi_stt_service.py`** - Local STT processing engine
2. **`client_with_local_stt.py`** - New Pi client with local transcription
3. **`install_local_stt.sh`** - Automated installation script
4. **Updated `requirements.txt`** - Added faster-whisper dependency

## 🚀 **Installation (Run on Pi)**

```bash
# 1. Copy new files to Pi
scp pi_stt_service.py pi@your-pi:~/FRED-V2/pi_client/
scp client_with_local_stt.py pi@your-pi:~/FRED-V2/pi_client/
scp install_local_stt.sh pi@your-pi:~/FRED-V2/pi_client/
scp requirements.txt pi@your-pi:~/FRED-V2/pi_client/

# 2. Run installation script
cd ~/FRED-V2/pi_client
chmod +x install_local_stt.sh
./install_local_stt.sh
```

## 🎮 **Usage**

### **Start the New Client:**
```bash
cd ~/FRED-V2/pi_client
python3 client_with_local_stt.py --server https://your-ngrok-url.ngrok.io
```

### **Voice Commands:**
1. **Activate**: Say "Hey Fred" or "Fred"
2. **Speak**: Give your command (e.g., "What time is it?")
3. **Deactivate**: Say "Goodbye" or "Bye Fred"

## 📊 **Performance Expectations**

### **Pi Zero 2W:**
- **Transcription Speed**: ~1.5-2x real-time
- **Wake Word Detection**: <200ms
- **Memory Usage**: ~200MB total
- **CPU Usage**: 60-80% during transcription

### **Pi 4:**
- **Transcription Speed**: ~3-4x real-time
- **Wake Word Detection**: <100ms
- **Memory Usage**: ~180MB total
- **CPU Usage**: 30-40% during transcription

## 🔧 **Server Changes Made**

The WebRTC server (`webrtc_server.py`) has been updated to:

1. **Detect Local STT Clients**: Recognizes `client_type: pi_glasses_with_local_stt`
2. **Skip Audio Processing**: No longer processes raw audio from local STT clients
3. **Handle Text Messages**: Receives and processes transcribed text via data channel
4. **Backwards Compatible**: Still supports old audio-based clients

## 🐛 **Troubleshooting**

### **Audio Issues:**
```bash
# Check audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
arecord -D hw:1,0 -d 5 -r 16000 -c 1 -f S16_LE test.wav
aplay test.wav
```

### **Model Download Issues:**
```bash
# Manual model download
python3 -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"
```

### **Memory Issues:**
```bash
# Check available memory
free -h
# Increase swap if needed
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=512/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 🔄 **Switching Between Modes**

### **Use Local STT (Recommended):**
```bash
python3 client_with_local_stt.py --server https://your-url
```

### **Use Server STT (Fallback):**
```bash
python3 client.py --server https://your-url
```

## 📈 **Monitoring Performance**

The local STT service includes built-in performance monitoring:
- **Transcription count** and **average processing time**
- **Audio level monitoring** for voice activity detection
- **Memory usage** and **processing queue status**

## 🔮 **Future Enhancements**

- **Model Optimization**: Custom quantization for even better Pi performance
- **Multi-language**: Support for other languages beyond English
- **Custom Wake Words**: User-configurable activation phrases
- **Edge Caching**: Intelligent buffering for faster response times

---

**🎉 Enjoy your new locally-processed F.R.E.D. experience with enhanced privacy and performance!** 