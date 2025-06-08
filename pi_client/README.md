# F.R.E.D. Pi Client - EDITH Glasses

Scripts for the Raspberry Pi Zero 2 W glasses that provide EDITH-style ambient intelligence. The Pi captures video and audio, streams them to the F.R.E.D. server, and receives responses back for display.

## Hardware Setup

### Required Components:
- Raspberry Pi Zero 2 W (mounted in glasses frame)
- USB microphone or I2S microphone
- Pi Camera or USB webcam
- Optional: Small OLED/LCD display for AR overlays

### Camera Setup:
```bash
# Enable camera interface
sudo raspi-config
# Navigate to Interface Options -> Camera -> Enable

# Test camera
raspistill -o test.jpg
```

### Audio Setup:
```bash
# Test microphone
arecord -l  # List audio devices
arecord -D hw:1,0 -d 5 test.wav  # Record 5 seconds
aplay test.wav  # Play back to verify
```

## Software Installation

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Audio (if needed):
```bash
# Create/edit ~/.asoundrc for USB microphone
cat > ~/.asoundrc << EOF
pcm.!default {
    type asym
    playback.pcm "hw:0,0"
    capture.pcm "hw:1,0"
}
EOF
```

## Usage

### 1. Start F.R.E.D. Server (on main computer):
```bash
# On your main computer, start both servers
python start_fred_with_webrtc.py
```

### 2. Start Pi Client (on Raspberry Pi):
```bash
# Replace YOUR_FRED_SERVER_IP with actual IP address
python client.py --server http://YOUR_FRED_SERVER_IP:8080
```

### 3. Talk to F.R.E.D.:
- Say "Hey F.R.E.D." or "Fred" to wake up
- Speak your command
- F.R.E.D.'s response will appear in the Pi terminal
- Say "goodbye" or "bye fred" to stop listening

## Expected Behavior

### ✅ What Works Now:
1. **Audio Capture**: Pi captures your voice
2. **WebRTC Streaming**: Audio streams to F.R.E.D. server
3. **Wake Word Detection**: F.R.E.D. listens for "fred", "hey fred", etc.
4. **Voice Processing**: F.R.E.D. processes your commands
5. **Text Responses**: F.R.E.D.'s responses appear in Pi terminal

### 🔄 Communication Flow:
```
Pi Glasses → WebRTC Server → F.R.E.D. Main Server → AI Processing
                                                              ↓
Pi Terminal ← WebRTC Server ← F.R.E.D. Main Server ← AI Response
```

### 📋 Example Session:
```
$ python client.py --server http://192.168.1.100:8080
Data channel opened
[Pi detects wake word "Hey Fred"]
[You say: "What's the weather like?"]
FRED: I'd be happy to help you check the weather! However, I'll need to know your location first. Could you tell me what city or area you'd like the weather for?
[You say: "Seattle"]
FRED: Let me check the current weather for Seattle...
[F.R.E.D. provides weather information]
```

## Troubleshooting

### Connection Issues:
```bash
# Test network connectivity
ping YOUR_FRED_SERVER_IP

# Check if WebRTC server is running
curl http://YOUR_FRED_SERVER_IP:8080
```

### Audio Issues:
```bash
# List audio devices
arecord -l
alsamixer  # Adjust microphone levels

# Test audio capture
arecord -D hw:1,0 -d 5 -f cd test.wav
```

### Camera Issues:
```bash
# Check camera detection
ls /dev/video*
v4l2-ctl --list-devices
```

## Future Enhancements

### Coming Soon:
- **Computer Vision**: Object detection and scene understanding
- **AR Display**: Visual overlays on glasses display
- **Edge Processing**: Local wake word detection
- **Gesture Control**: Hand gesture recognition
- **Environmental Awareness**: Ambient intelligence features

## Network Requirements

- F.R.E.D. server and Pi must be on same network
- WebRTC server runs on port 8080
- Main F.R.E.D. server runs on port 5000
- Ensure firewall allows these ports
