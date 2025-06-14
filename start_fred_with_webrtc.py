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
import json
import requests
from config import config

# Global variable to store tunnel info
tunnel_info = {"webrtc_url": None, "main_url": None}

def start_ngrok_tunnel():
    """Start ngrok tunnel for WebRTC server and return public URL"""
    try:
        print("🌐 Starting ngrok tunnel for remote access...")
        
        # Start ngrok for WebRTC server (using config port)
        ngrok_process = subprocess.Popen([
            "ngrok", "http", str(config.WEBRTC_PORT), 
            "--log=stdout",
            "--log-level=info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give ngrok time to start
        time.sleep(3)
        
        # Get tunnel URL from ngrok API
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
            tunnels = response.json()
            
            for tunnel in tunnels.get('tunnels', []):
                if tunnel['config']['addr'] == f'http://localhost:{config.WEBRTC_PORT}':
                    public_url = tunnel['public_url']
                    tunnel_info["webrtc_url"] = public_url
                    print(f"✅ ngrok tunnel active: {public_url}")
                    
                    # Save tunnel info for Pi client
                    with open('tunnel_info.json', 'w') as f:
                        json.dump({"webrtc_server": public_url}, f)
                    
                    return ngrok_process
        except Exception as e:
            print(f"⚠️  Could not get ngrok tunnel URL: {e}")
            print("   Check if ngrok is properly configured with auth token")
            
        return ngrok_process
        
    except Exception as e:
        print(f"❌ Failed to start ngrok tunnel: {e}")
        return None

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
    
    # Start ngrok tunnel first
    ngrok_process = start_ngrok_tunnel()
    
    print(f"Main F.R.E.D. server: http://localhost:{config.PORT}")
    print(f"WebRTC server: http://localhost:{config.WEBRTC_PORT}")
    
    if tunnel_info["webrtc_url"]:
        print(f"Remote WebRTC access: {tunnel_info['webrtc_url']}")
        print("🌍 Pi glasses can connect from anywhere!")
    else:
        print("⚠️  No remote tunnel - Pi glasses limited to local network")
    
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
        
        # Cleanup ngrok process
        if ngrok_process:
            ngrok_process.terminate()
            print("🌐 ngrok tunnel stopped")
        
        # Cleanup tunnel info file
        if os.path.exists('tunnel_info.json'):
            os.remove('tunnel_info.json')
            
        sys.exit(0)

if __name__ == "__main__":
    main() 