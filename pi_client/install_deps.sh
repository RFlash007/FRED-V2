#!/bin/bash
# F.R.E.D. Pi Client Dependencies Installation

# Move to the project root directory (one level up from pi_client)
cd "$(dirname "$0")/.."

echo "🍇 Installing F.R.E.D. Pi Client Dependencies..."
echo "This will update your system and install necessary packages."

# Stop on first error
set -e

# Update system package list
echo "🔄 Updating package lists..."
sudo apt-get update

# Install system dependencies
# Includes pip, Python headers, venv, libcamera, audio, and official Picamera2
echo "📦 Installing system packages (this may take a while)..."
sudo apt-get install -y python3-pip python3-dev python3-venv libcamera-apps portaudio19-dev python3-picamera2

# --- Virtual Environment Setup ---
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment '$VENV_DIR' already exists. Skipping re-creation."
else
    echo "🐍 Creating Python virtual environment at project root..."
    python3 -m venv $VENV_DIR
fi

echo " activating virtual environment..."
source $VENV_DIR/bin/activate

# Remove picamera2 from requirements.txt to avoid conflicts with the system package
REQUIREMENTS_FILE="pi_client/requirements.txt"
if grep -q "picamera2" "$REQUIREMENTS_FILE"; then
    echo "Removing 'picamera2' from $REQUIREMENTS_FILE to use system version..."
    # Use a temporary file and move to be compatible with sed on different systems
    sed '/^picamera2/d' "$REQUIREMENTS_FILE" > "$REQUIREMENTS_FILE.tmp" && mv "$REQUIREMENTS_FILE.tmp" "$REQUIREMENTS_FILE"
fi

# Install/upgrade pip in the virtual environment
echo "🐍 Upgrading pip in the virtual environment..."
pip install --upgrade pip

# Install Python dependencies from requirements.txt into the virtual environment
echo "🐍 Installing Python packages into the virtual environment..."
pip install -r "$REQUIREMENTS_FILE"

# Deactivate the environment for a clean exit, user will activate it to run the script
deactivate

echo "✅ Installation complete!"
echo ""
echo "💡 To enable the camera (if you haven't already), run 'sudo raspi-config', navigate to 'Interface Options', and enable 'Camera', then reboot."
echo ""
echo "📷 You can test the camera directly with the command:"
echo "   libcamera-hello -t 5000"
echo ""
echo "🚀 To run the client, from the FRED-V2 directory, you must first activate the environment:"
echo "   source .venv/bin/activate"
echo "   python3 pi_client/client.py" 