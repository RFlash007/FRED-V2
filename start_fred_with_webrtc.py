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
from ollietec_theme import apply_theme, banner
from ollie_print import olliePrint, olliePrint_simple

# Global variable to store tunnel info
tunnel_info = {"webrtc_server": None, "main_url": None}

# It's better to import the functions directly
from app import run_app as run_fred_server
from webrtc_server import main as run_webrtc_server_async

# Apply OLLIE-TEC theming to all prints
apply_theme()

async def start_ngrok_tunnel():
    """Start ngrok tunnel asynchronously."""
    try:
        olliePrint_simple("[SHELTER-NET] Establishing external communication tunnel...")
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
                            olliePrint_simple(f"[SUCCESS] External tunnel established: {public_url}", 'success')
                            olliePrint_simple(f"📱 Pi Command: python client.py --server {public_url}")
                            # Save tunnel info for Pi client
                            with open('tunnel_info.json', 'w') as f:
                                json.dump({"webrtc_server": public_url}, f)
                            return ngrok_process
            except Exception as e:
                olliePrint_simple(f"[WARNING] Tunnel status unavailable via API: {e}", 'warning')
        
        return ngrok_process
    except Exception as e:
        olliePrint_simple(f"[ERROR] External tunnel establishment failed: {e}", 'error')
        return None

async def run_server(name, command):
    """Run a server command as a subprocess."""
    olliePrint_simple(f"🚀 Starting {name}...")
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
        olliePrint_simple("[SHELTER-NET] External tunnel protocols enabled...")
        ngrok_process = await start_ngrok_tunnel()
        if ngrok_process:
            olliePrint_simple("[SUCCESS] External communications link established", 'success')
        else:
            olliePrint_simple("[WARNING] External tunnel failed - operating in local mode only", 'warning')
    else:
        olliePrint_simple("[LOCAL] External tunnel disabled - shelter network access only")
    
    try:
        await run_webrtc_server_async()
    finally:
        if ngrok_process:
            olliePrint_simple("[SHELTER-NET] Terminating external communication tunnel...")
            ngrok_process.terminate()
            await ngrok_process.wait()

def main():
    """
    Main function to start and manage F.R.E.D. servers.
    - F.R.E.D. Flask server runs in a separate thread.
    - WebRTC aiohttp server runs in the main thread's asyncio loop.
    """
    # Single clean startup banner
    olliePrint(banner("F.R.E.D. Mainframe v2.0"))
    olliePrint_simple("[MAINFRAME] F.R.E.D. core intelligence systems initializing...")

    # 1. Run the Flask/SocketIO server in its own thread
    # This is necessary because it uses its own blocking eventlet server.
    fred_thread = threading.Thread(target=run_fred_server, daemon=True)
    fred_thread.start()
    
    # Give the main server a moment to start before the WebRTC server,
    # as the WebRTC server may try to connect to it.
    time.sleep(5)

    # 2. Run the aiohttp WebRTC server with ngrok tunnel in the main thread using asyncio
    try:
        olliePrint_simple("[NETWORK] Initializing wasteland communication protocols...")
        asyncio.run(run_with_ngrok())
    except KeyboardInterrupt:
        olliePrint_simple("\n[SHUTDOWN] F.R.E.D. mainframe shutting down...", 'warning')
    finally:
        # The servers are daemonized or will shut down on loop completion
        olliePrint_simple("[SHELTER-CORE] All systems offline. Stay safe out there.", 'warning')
        # Note: Proper cleanup of the WebRTC runner is handled within webrtc_server.py
        # on loop cancellation. The fred_thread is a daemon and will exit with the main thread.

if __name__ == "__main__":
    main() 