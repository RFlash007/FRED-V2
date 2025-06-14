#!/usr/bin/env python3
"""
F.R.E.D. with WebRTC Startup Script
Runs both the main F.R.E.D. server and WebRTC server for Pi glasses integration
"""

import asyncio
import threading
import sys
import os
import json
import requests
import time
import aiohttp
from config import config

# Global variable to store tunnel info
tunnel_info = {"webrtc_server": None, "main_url": None}

# It's better to import the functions directly
from app import run_app as run_fred_server
from webrtc_server import main as run_webrtc_server_async

async def start_ngrok_tunnel():
    """Start ngrok tunnel asynchronously."""
    try:
        print("🌐 Starting ngrok tunnel for remote access...")
        ngrok_process = await asyncio.create_subprocess_exec(
            "ngrok", "http", str(config.WEBRTC_PORT),
            "--log", "stdout",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # It's tricky to get the URL when ngrok is a subprocess in asyncio
        # We will use the API method which is more reliable.
        await asyncio.sleep(3) # Give ngrok time to start up the API

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://localhost:4040/api/tunnels", timeout=5) as response:
                    tunnels = await response.json()
                    for tunnel in tunnels.get('tunnels', []):
                        if tunnel['config']['addr'].endswith(str(config.WEBRTC_PORT)):
                            public_url = tunnel['public_url']
                            tunnel_info["webrtc_server"] = public_url
                            print(f"✅ ngrok tunnel active: {public_url}")
                            # Save tunnel info for Pi client
                            with open('tunnel_info.json', 'w') as f:
                                json.dump({"webrtc_server": public_url}, f)
                            return ngrok_process
            except Exception as e:
                print(f"⚠️  Could not get ngrok tunnel URL via API: {e}")
        
        return ngrok_process
    except Exception as e:
        print(f"❌ Failed to start ngrok tunnel: {e}")
        return None

async def run_server(name, command):
    """Run a server command as a subprocess."""
    print(f"🚀 Starting {name}...")
    process = await asyncio.create_subprocess_exec(
        sys.executable, command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=os.getcwd()
    )
    return process

async def run_with_ngrok():
    """Run the WebRTC server with optional ngrok tunnel."""
    ngrok_process = None
    
    if config.NGROK_ENABLED:
        print("🌐 ngrok is enabled - starting tunnel...")
        ngrok_process = await start_ngrok_tunnel()
        if ngrok_process:
            print("✅ ngrok tunnel started successfully")
        else:
            print("⚠️ ngrok tunnel failed to start, continuing with local access only")
    else:
        print("🏠 ngrok disabled - local network access only")
    
    try:
        await run_webrtc_server_async()
    finally:
        if ngrok_process:
            print("🌐 Shutting down ngrok tunnel...")
            ngrok_process.terminate()
            await ngrok_process.wait()

def main():
    """
    Main function to start and manage F.R.E.D. servers.
    - F.R.E.D. Flask server runs in a separate thread.
    - WebRTC aiohttp server runs in the main thread's asyncio loop.
    """
    print("🚀 Starting F.R.E.D. with Pi Glasses Support")
    print("=" * 50)

    # 1. Run the Flask/SocketIO server in its own thread
    # This is necessary because it uses its own blocking eventlet server.
    fred_thread = threading.Thread(target=run_fred_server, daemon=True)
    fred_thread.start()
    print("🤖 F.R.E.D. main server thread started.")
    
    # Give the main server a moment to start before the WebRTC server,
    # as the WebRTC server may try to connect to it.
    time.sleep(5)

    # 2. Run the aiohttp WebRTC server with ngrok tunnel in the main thread using asyncio
    try:
        print("📡 Starting WebRTC server with tunnel support...")
        asyncio.run(run_with_ngrok())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down F.R.E.D. servers...")
    finally:
        # The servers are daemonized or will shut down on loop completion
        print("✅ Shutdown sequence initiated.")
        # Note: Proper cleanup of the WebRTC runner is handled within webrtc_server.py
        # on loop cancellation. The fred_thread is a daemon and will exit with the main thread.

if __name__ == "__main__":
    main() 