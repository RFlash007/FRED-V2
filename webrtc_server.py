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

# Import configuration
from config import config

# Get security configuration from config
FRED_AUTH_TOKEN = config.FRED_AUTH_TOKEN
MAX_CONNECTIONS = config.MAX_PI_CONNECTIONS

pcs = set()
data_channels = set()  # Store data channels for sending responses back to Pi
pi_clients = set()  # Track Pi clients for vision processing
connection_timestamps = {}  # Track connection times for rate limiting

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
        
        logging.info(f"New Pi glasses connection from {client_ip}")

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

                logging.info(f"[PI TRANSCRIPTION] -> '{text}'")

                # Prepare chat request for main server
                payload = {
                    "message": text,
                    "model": config.DEFAULT_MODEL,
                    "mute_fred": True,  # Disable TTS on main server for Pi-originated queries
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
            print(f"Data channel '{channel.label}' opened from {client_ip}")
            
            # Notify vision service that Pi is connected
            vision_service.set_pi_connection_status(True)
            print(f"Pi glasses connected - starting vision processing")
            
            @channel.on('message')
            def on_message(message):
                # Handle heartbeat messages
                if message == '[HEARTBEAT]':
                    print(f"💓 Heartbeat received from {client_ip}")
                    channel.send('[HEARTBEAT_ACK]')
                else:
                    print(f"Message from Pi glasses: {message}")
            
            @channel.on('close')
            def on_close():
                data_channels.discard(channel)
                pi_clients.discard(channel)
                print(f"Data channel '{channel.label}' closed from {client_ip}")
                
                # If no more Pi clients, stop vision processing
                if not pi_clients:
                    vision_service.set_pi_connection_status(False)
                    print(f"Pi glasses disconnected - stopping vision processing")

        @pc.on('track')
        async def on_track(track):
            print(f"📡 Received {track.kind} track from Pi client {client_ip}")
            if track.kind == 'audio':
                print("🎤 Setting up audio frame processing...")
                frame_count = 0
                @track.on('frame')
                async def on_frame(frame):
                    nonlocal frame_count
                    frame_count += 1
                    if frame_count % 100 == 0:  # Log every 100th frame to avoid spam
                        print(f"🎵 Audio frame #{frame_count} received from {client_ip}")
                    pcm = frame.to_ndarray()
                    # Mark this audio as coming from Pi and forward to STT pipeline
                    stt_service.audio_queue.put((pcm, True))  # (audio_data, from_pi)
            elif track.kind == 'video':
                print("📹 Setting up video frame processing...")
                frame_count = 0
                @track.on('frame')
                async def on_frame(frame):
                    nonlocal frame_count
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30th frame
                        print(f"📷 Video frame #{frame_count} received from {client_ip} (size: {frame.width}x{frame.height})")
                    
                    try:
                        # Store frame for vision processing
                        vision_service.store_latest_frame(frame)
                        
                        # Debug: Confirm frame was stored
                        if frame_count == 1:
                            print(f"✅ First video frame successfully stored for vision processing")
                        elif frame_count % 100 == 0:
                            print(f"📊 Frame #{frame_count} stored - vision processing should be active")
                            
                    except Exception as frame_error:
                        print(f"❌ Error storing video frame #{frame_count}: {frame_error}")
                        import traceback
                        traceback.print_exc()
            
            # Add track to recorder safely
            if recorder:
                try:
                    recorder.addTrack(track)
                except Exception as rec_err:
                    logging.error(f"Recorder addTrack failed: {rec_err}")
            else:
                logging.warning("Recorder not initialized, skipping track recording")

        @pc.on('connectionstatechange')
        async def on_connectionstatechange():
            print(f"Connection state changed to {pc.connectionState} for {client_ip}")
            if pc.connectionState in ('failed', 'closed'):  # handle normal closure too
                await pc.close()
                pcs.discard(pc)  # free slot for new connections

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({'sdp': pc.localDescription.sdp,
                                  'type': pc.localDescription.type})
    
    except Exception as e:
        logging.error(f"Error handling offer from {client_ip}: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)

# SocketIO event handlers for receiving responses from main F.R.E.D. server
@sio_client.event
async def connect():
    print("Connected to F.R.E.D. main server")

@sio_client.event
async def voice_response(data):
    """Forward F.R.E.D.'s responses back to Pi clients"""
    if 'response' in data:
        response_text = data['response']
        # Send response to all connected Pi clients
        for channel in data_channels.copy():
            try:
                channel.send(response_text)
                print(f"Sent to Pi: {response_text}")
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

async def cleanup(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    data_channels.clear()
    pi_clients.clear()
    
    # Stop vision processing
    vision_service.set_pi_connection_status(False)
    
    await sio_client.disconnect()

async def init_app():
    """Initialize the WebRTC server and connect to main F.R.E.D. server"""
    try:
        # Connect to main F.R.E.D. server
        await sio_client.connect('http://localhost:5000')
        print("Connected to F.R.E.D. main server for response forwarding")
    except Exception as e:
        print(f"Failed to connect to F.R.E.D. main server: {e}")

app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/offer', offer)
app.on_startup.append(lambda app: asyncio.create_task(init_app()))
app.on_shutdown.append(cleanup)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting F.R.E.D. WebRTC Server")
    print(f"🔐 Authentication: {'Enabled' if FRED_AUTH_TOKEN else 'Disabled'}")
    print(f"🔢 Max connections: {MAX_CONNECTIONS}")
    print("📡 Listening on 0.0.0.0:8080")
    
    web.run_app(app, host=config.WEBRTC_HOST, port=config.WEBRTC_PORT)
