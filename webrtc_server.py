from aiohttp import web
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from stt_service import stt_service
from vision_service import vision_service
import socketio
import json
import hashlib
import time
import logging
import argparse
import ssl
import numpy as np
from scipy.signal import resample_poly
from ollietec_theme import apply_theme, banner

# Import configuration
from config import config

# Get security configuration from config
FRED_AUTH_TOKEN = config.FRED_AUTH_TOKEN
MAX_CONNECTIONS = config.MAX_PI_CONNECTIONS

apply_theme()

pcs = set()
data_channels = set()  # Store data channels for sending responses back to Pi
pi_clients = set()  # Track Pi clients for vision processing
client_video_tracks = {}  # Store video tracks for on-demand frame requests
connection_timestamps = {}  # Track connection times for rate limiting

async def request_frame_from_client(client_ip):
    """Request a fresh frame from a specific client"""
    if client_ip not in client_video_tracks:
        print(f"⚠️ No video track available for {client_ip}")
        return None
    
    try:
        track = client_video_tracks[client_ip]
        print(f"📸 Requesting fresh frame from {client_ip}")
        frame = await track.recv()
        print(f"✅ Fresh frame received from {client_ip} (size: {frame.width}x{frame.height})")
        return frame
    except Exception as e:
        print(f"❌ Failed to request frame from {client_ip}: {e}")
        return None

# SocketIO client to connect to main F.R.E.D. server
sio_client = socketio.AsyncClient()

def authenticate_request(request):
    """Basic authentication for Pi glasses connections"""
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Bearer '):
        return False
    
    token = auth_header[7:]  # Remove 'Bearer ' prefix
    return token == FRED_AUTH_TOKEN

def check_rate_limit(client_ip):
    """Check if client is within rate limits"""
    current_time = time.time()
    
    # Clean old connections (older than 1 hour)
    old_connections = [ip for ip, timestamp in connection_timestamps.items() 
                      if current_time - timestamp > 3600]
    for ip in old_connections:
        del connection_timestamps[ip]
    
    # Check if we're at max connections
    if len(pcs) >= MAX_CONNECTIONS:
        return False
    
    # Allow connection
    connection_timestamps[client_ip] = current_time
    return True

async def index(request):
    return web.Response(text="FRED WebRTC Server Running - Pi Glasses Gateway")

async def offer(request):
    # Get client IP for rate limiting
    client_ip = request.remote
    
    # Check rate limits
    if not check_rate_limit(client_ip):
        logging.warning(f"Rate limit exceeded for {client_ip}")
        return web.json_response({'error': 'Too many connections'}, status=429)
    
    # Authenticate request
    if not authenticate_request(request):
        logging.warning(f"Unauthorized connection attempt from {client_ip}")
        return web.json_response({'error': 'Unauthorized'}, status=401)
    
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        logging.info(f"[VAULT-NET] New Pip-Boy connection from {client_ip}")

        # Initialize media recorder (blackhole – we don't save media)
        recorder = MediaBlackhole()
        # Start the recorder (MediaBlackhole.start is *not* a coroutine)
        try:
            await recorder.start()
        except Exception as e:
            logging.warning(f"Recorder failed to start: {e}")
        
        # -------------------------------------------------------------
        #  STT SET-UP (run only once – on first connection)
        # -------------------------------------------------------------
        if not getattr(stt_service, "is_processing", False):
            # Define a synchronous callback that will forward the final
            # transcription to the main F.R.E.D. server and relay the
            # response back to all connected Pi clients.
            def process_pi_transcription(text, from_pi=False):
                """Handle text recognised from Pi audio."""
                if not text or not text.strip():
                    return

                logging.info(f"[PIP-BOY COMM] Field operative transmission → '{text}'")

                # Prepare chat request for main server
                payload = {
                    "message": text,
                    "model": config.DEFAULT_MODEL,
                    "mute_fred": False,  # Enable TTS for Pi glasses - F.R.E.D. speaks through the glasses
                    "from_pi_glasses": True,
                }

                def _call_fred():
                    try:
                        import requests, json as _json
                        resp = requests.post("http://localhost:5000/chat", json=payload, stream=True, timeout=60)
                        if resp.status_code == 200:
                            buffer = ""
                            for line in resp.iter_lines():
                                if not line:
                                    continue
                                try:
                                    data = _json.loads(line.decode("utf-8"))
                                except Exception:
                                    continue

                                if "response" in data:
                                    # incremental content from server – buffer and forward
                                    fragment = data["response"]
                                    buffer += fragment
                                    for ch in data_channels.copy():
                                        try:
                                            ch.send(fragment)
                                        except Exception:
                                            data_channels.discard(ch)
                        else:
                            logging.error(f"Chat request failed: {resp.status_code}")
                    except Exception as exc:
                        logging.error(f"Error relaying message to F.R.E.D.: {exc}")

                import threading as _threading
                _threading.Thread(target=_call_fred, daemon=True).start()

            # Start STT processing – this will spin up the processing thread
            try:
                stt_service.start_processing(process_pi_transcription)
                logging.info("STT processing thread started for Pi audio")
            except Exception as stt_err:
                logging.error(f"Unable to start STT processing: {stt_err}")
        
        # Store data channel when created
        @pc.on('datachannel')
        def on_datachannel(channel):
            data_channels.add(channel)
            pi_clients.add(channel)  # Track Pi clients
            print(f"[PIP-BOY] Data channel '{channel.label}' established with field operative at {client_ip}")
            
            # Notify vision service that Pi is connected
            vision_service.set_pi_connection_status(True)
            print(f"[OPTICS] Pip-Boy visual sensors ONLINE - initiating reconnaissance protocols")
            
            # Check if this is a local STT client
            client_type = params.get('client_type', 'unknown')
            is_local_stt = 'local_stt' in client_type
            
            if is_local_stt:
                print(f"🧠 [LOCAL STT] Client using on-device transcription - text-only mode")
            
            @channel.on('message')
            def on_message(message):
                if message == '[HEARTBEAT]':
                    print(f"[VITAL-MONITOR] Pip-Boy heartbeat confirmed from {client_ip}")
                    channel.send('[HEARTBEAT_ACK]')
                    return

                try:
                    data = json.loads(message)
                    if data.get('type') == 'transcription':
                        text = data.get('text', '').strip()
                        if not text:
                            return
                        
                        print(f"🗣️  [PI TRANSCRIPTION] '{text}' from {client_ip}")
                        
                        # --- Simplified and Robust Response Handling ---
                        try:
                            # 1. Prepare payload for F.R.E.D.
                            payload = {
                                "message": text,
                                "model": config.DEFAULT_MODEL,
                                "mute_fred": False, # Ensure TTS for Pi glasses
                                "from_pi_glasses": True,
                            }
                            
                            # 2. Call F.R.E.D. server directly
                            import requests
                            resp = requests.post("http://localhost:5000/chat", json=payload, timeout=60)
                            resp.raise_for_status() # Raise an exception for bad status codes
                            
                            response_data = resp.json()
                            
                            # 3. Relay complete response back to Pi
                            fred_text = response_data.get("response", "No text response.")
                            fred_audio_b64 = response_data.get("audio")

                            if fred_audio_b64:
                                print(f"🔊 Relaying audio response to {client_ip}")
                                channel.send(json.dumps({
                                    "type": "audio",
                                    "audio": fred_audio_b64
                                }))
                            else:
                                print(f"💬 Relaying text-only response to {client_ip}")
                                channel.send(json.dumps({
                                    "type": "text",
                                    "text": fred_text
                                }))

                        except requests.exceptions.RequestException as e:
                            logging.error(f"Failed to get response from F.R.E.D.: {e}")
                            channel.send(json.dumps({"type": "status", "status": f"Error: {e}"}))
                        except Exception as e:
                            logging.error(f"Error processing transcription on server: {e}")
                            channel.send(json.dumps({"type": "status", "status": f"Server Error: {e}"}))

                except json.JSONDecodeError:
                    logging.warning(f"Received non-JSON message from {client_ip}: {message}")
            
            @channel.on('close')
            def on_close():
                data_channels.discard(channel)
                pi_clients.discard(channel)
                print(f"[PIP-BOY] Data channel '{channel.label}' closed from {client_ip}")
                
                # If no more Pi clients, stop vision processing
                if not pi_clients:
                    vision_service.set_pi_connection_status(False)
                    print(f"[OPTICS] All Pip-Boys disconnected - reconnaissance protocols OFFLINE")

        @pc.on('track')
        async def on_track(track):
            print(f"[SIGNAL] {track.kind.upper()} link established with field operative {client_ip}")
            
            if track.kind == 'audio':
                # Local STT clients don't send processable audio
                print("[AUDIO] Client uses local STT - audio stream will be ignored.")
                # We still need to consume the track to prevent buffer issues
                recorder.addTrack(track)
            
            if track.kind == 'video':
                print(f"[OPTICS] Video feed from {client_ip} is now available for on-demand analysis.")
                client_video_tracks[client_ip] = track
                # Don't record video unless explicitly requested
                # recorder.addTrack(track)

        @pc.on('connectionstatechange')
        async def on_connectionstatechange():
            
            state = pc.connectionState
            logging.info(f"Connection state is {state}")
            
            if state == "failed" or state == "closed":
                await pc.close()
                pcs.discard(pc)
                if client_ip in client_video_tracks:
                    del client_video_tracks[client_ip]
                logging.info(f"Connection from {client_ip} closed.")

        await pc.setRemoteDescription(offer)
        await recorder.start() # Start recording (or blackholing) media
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        })
    
    except Exception as e:
        logging.error(f"Failed to establish WebRTC connection with {client_ip}: {e}")
        await pc.close()
        pcs.discard(pc)
        return web.json_response({'error': str(e)}, status=500)

# SocketIO event handlers for receiving responses from main F.R.E.D. server
@sio_client.event
async def connect():
    logging.info("[VAULT-NET] ✅ Connection to F.R.E.D. main server established.")
    # Emit connection confirmation
    await sio_client.emit('webrtc_server_connected')
    print("[BRIDGE] Wasteland communication network ONLINE - standing by for field operations")

@sio_client.event
async def voice_response(data):
    """Forward F.R.E.D.'s responses back to Pi clients"""
    if 'response' in data:
        response_text = data['response']
        # Send response to all connected Pi clients
        for channel in data_channels.copy():
            try:
                channel.send(response_text)
                print(f"📱 Text sent to Pi: {response_text[:50]}...")
            except Exception as e:
                print(f"Failed to send to Pi: {e}")
                data_channels.discard(channel)

@sio_client.event
async def fred_acknowledgment(data):
    """Forward acknowledgments back to Pi clients"""
    ack_text = data.get('text', '')
    for channel in data_channels.copy():
        try:
            channel.send(f"[ACK] {ack_text}")
            print(f"Sent acknowledgment to Pi: {ack_text}")
        except Exception as e:
            print(f"Failed to send acknowledgment to Pi: {e}")
            data_channels.discard(channel)

@sio_client.event
async def fred_audio(data):
    """Forward F.R.E.D.'s audio responses to Pi clients"""
    audio_b64 = data.get('audio_data', '')
    text = data.get('text', '')
    audio_format = data.get('format', 'wav')
    
    if audio_b64:
        print(f"[TRANSMISSION] Audio matrix received from F.R.E.D. ({len(audio_b64)} chars) → '{text[:50]}...'")
        print(f"[NETWORK] {len(data_channels)} Pip-Boy device(s) in communication range")
        
        # Send audio to all connected Pi clients
        sent_count = 0
        for channel in data_channels.copy():
            try:
                message = f"[AUDIO_BASE64:{audio_format}]{audio_b64}"
                channel.send(message)
                sent_count += 1
                print(f"[RELAY] Voice data transmitted to Pip-Boy #{sent_count} → '{text[:50]}...'")
            except Exception as e:
                print(f"[ERROR] Pip-Boy #{sent_count+1} transmission failure: {e}")
                data_channels.discard(channel)
        
        if sent_count == 0:
            print("[WARNING] No Pip-Boy devices available - audio transmission failed")
        else:
            print(f"[SUCCESS] Voice transmission complete → {sent_count} field operative(s) reached")
        
        # Inform STT to pause listening during playback
        estimated_bytes = int(len(audio_b64) * 3 / 4)  # rough base64 decode
        playback_seconds = max(1, estimated_bytes // 32000 + 1)  # 32kB ≈ 1s at 16-kHz s16 mono
        stt_service.set_speaking_state(True)
        # Schedule automatic resume after playback finishes
        loop = asyncio.get_event_loop()
        loop.call_later(playback_seconds, lambda: stt_service.set_speaking_state(False))
    else:
        print("[ERROR] No audio data in transmission from F.R.E.D. mainframe")

async def cleanup(app):
    # This is still valuable for graceful shutdown
    print("🧹 Cleaning up server resources...")
    for pc in pcs:
        if pc.connectionState != "closed":
            await pc.close()
    pcs.clear()
    if 'fred_client_task' in app:
        app['fred_client_task'].cancel()
        await app['fred_client_task']

async def init_app(app):
    """Initialize the WebRTC server and connect to main F.R.E.D. server"""
    try:
        # Connect to main F.R.E.D. server
        await sio_client.connect('http://localhost:5000')
        print("Connected to F.R.E.D. main server for response forwarding")
    except Exception as e:
        print(f"Failed to connect to F.R.E.D. main server: {e}")

async def main():
    parser = argparse.ArgumentParser(description="F.R.E.D. WebRTC server")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_startup.append(init_app)
    app.on_shutdown.append(cleanup)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=args.host, port=args.port, ssl_context=ssl_context)
    await site.start()
    
    print(banner("WebRTC Server"))
    print(f"🔐 Authentication: {'Enabled' if FRED_AUTH_TOKEN else 'Disabled'}")
    print(f"🔢 Max connections: {MAX_CONNECTIONS}")
    print(f"🚀 Listening on http://{args.host}:{args.port}")

    # Keep server running until interrupted
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⌨️ Server shutting down manually...")
    finally:
        # This cleanup is for when running the file directly
        if runner:
            asyncio.run(runner.cleanup())
        print("✅ Server shutdown complete.")
