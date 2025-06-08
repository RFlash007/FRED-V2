#!/usr/bin/env python3
"""
F.R.E.D. with WebRTC Startup Script
Runs both the main F.R.E.D. server and WebRTC server for Pi glasses integration
"""

import subprocess
import threading
import time
import sys
import os
from config import config

def run_fred_server():
    """Run the main F.R.E.D. Flask server"""
    print("🤖 Starting F.R.E.D. main server...")
    subprocess.run([sys.executable, "app.py"], cwd=os.getcwd())

def run_webrtc_server():
    """Run the WebRTC server for Pi glasses"""
    print("📡 Starting WebRTC server for Pi glasses...")
    # Give main server time to start first
    time.sleep(3)
    subprocess.run([sys.executable, "webrtc_server.py"], cwd=os.getcwd())

def main():
    print("🚀 Starting F.R.E.D. with Pi Glasses Support")
    print("=" * 50)
    print(f"Main F.R.E.D. server: http://localhost:{config.PORT}")
    print("WebRTC server: http://localhost:8080")
    print(f"Vision processing: {config.VISION_MODEL} (Pi glasses only)")
    print("Pi client should connect to WebRTC server")
    print("=" * 50)
    
    # Start both servers in separate threads
    fred_thread = threading.Thread(target=run_fred_server, daemon=True)
    webrtc_thread = threading.Thread(target=run_webrtc_server, daemon=True)
    
    try:
        fred_thread.start()
        webrtc_thread.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down F.R.E.D. servers...")
        sys.exit(0)

if __name__ == "__main__":
    main() 