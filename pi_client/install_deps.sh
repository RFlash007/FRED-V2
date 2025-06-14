#!/bin/bash
# F.R.E.D. Pi Client Dependencies Installation

echo "🍇 Installing F.R.E.D. Pi Client Dependencies..."
echo "This will update your system and install necessary packages."

# Stop on first error
set -e

# Update system package list
echo "🔄 Updating package lists..."
sudo apt-get update

# Install system dependencies
# This includes pip, Python headers, libcamera apps, audio dependencies, and the official Picamera2 package
echo "📦 Installing system packages (this may take a while)..."
sudo apt-get install -y python3-pip python3-dev libcamera-apps portaudio19-dev python3-picamera2

# Remove picamera2 from requirements.txt to avoid conflicts with the system package
# It's better to manage it via apt on the Pi
if grep -q "picamera2" "requirements.txt"; then
    echo "Removing 'picamera2' from requirements.txt to use system version..."
    sed -i '/^picamera2/d' requirements.txt
fi

# Install remaining Python dependencies from requirements.txt
echo "🐍 Installing Python packages..."
pip3 install -r requirements.txt

echo "✅ Installation complete!"
echo ""
echo "💡 To enable the camera, run 'sudo raspi-config', navigate to 'Interface Options', and enable 'Legacy Camera' or 'Camera'."
echo ""
echo "📷 You can test the camera directly with the command:"
echo "   libcamera-hello -t 5000"
echo ""
echo "🚀 Then, test your Pi client with:"
echo "   python3 client.py" 