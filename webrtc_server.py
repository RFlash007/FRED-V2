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
import argparse
import ssl
import numpy as np
from scipy.signal import resample_poly
from ollietec_theme import apply_theme, banner
from ollie_print import olliePrint_simple

# Import configuration
from config import config

# Get security configuration from config
FRED_AUTH_TOKEN = config.FRED_AUTH_TOKEN
MAX_CONNECTIONS = config.MAX_PI_CONNECTIONS

apply_theme()

pcs = set()
data_channels = set()  # Store data channels for sending responses back to Pi
pi_clients = set()  # Track Pi clients for vision processing
connection_timestamps = {}  # Track connection times for rate limiting

# Runner instance for graceful shutdown when running as a script
runner = None

async def send_capture_request_to_pi():
    """Send capture request to all connected Pi clients"""
    if not data_channels:
        olliePrint_simple("❌ No Pi clients connected for capture request", 'warning')
        return False
    
    try:
        capture_sent = False
        for channel in list(data_channels):
            try:
                if channel.readyState == 'open':
                    channel.send("[CAPTURE_REQUEST]")
                    capture_sent = True
                    olliePrint_simple("📡 [REQUEST] Sent capture request to Pi")
            except Exception as e:
                olliePrint_simple(f"Failed to send capture request: {e}", 'warning')
        
        return capture_sent
    except Exception as e:
        olliePrint_simple(f"Capture request error: {e}", 'error')
        return False

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
    
    # Rate limiting check
    if not check_rate_limit(client_ip):
        olliePrint_simple(f"Rate limit exceeded: {client_ip}")
        return web.json_response({'error': 'Rate limit exceeded'}, status=429)
    
    # Authentication check
    if not authenticate_request(request):
        olliePrint_simple(f"Unauthorized access: {client_ip}")
        return web.json_response({'error': 'Unauthorized'}, status=401)
    
    try:
        # Get URL query parameters for client_type
        query_params = request.query
        client_type_param = query_params.get('client_type', 'unknown')
        
        # Get JSON body for WebRTC offer
        params = await request.json()
        params['client_type'] = client_type_param
        
        # Store the main event loop for background threads
        main_event_loop = asyncio.get_running_loop()
        
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        olliePrint_simple(f"ArmLink connected: {client_ip}")
        
        # Initialize media recorder
        recorder = MediaBlackhole()
        try:
            await recorder.start()
        except Exception as e:
            olliePrint_simple(f"Recorder start failed: {e}", level='warning')
        
        # STT setup - consolidated
        if not getattr(stt_service, "is_processing", False):
            def process_pi_transcription(text, from_pi=False):
                """Handle text recognised from Pi audio."""
                if not text or not text.strip():
                    return

                # Prepare chat request for main server - use FRED_OLLAMA_MODEL for consistent personality
                payload = {
                    "message": text,
                    "model": config.FRED_OLLAMA_MODEL,  # Use FRED's personality model for Pi transcriptions
                    "mute_fred": False,
                    "from_pi_glasses": True,
                }

                def _call_fred():
                    try:
                        import requests, json as _json
                        resp = requests.post("http://localhost:5000/chat", json=payload, stream=True, timeout=None)
                        if resp.status_code == 200:
                            buffer = ""
                            
                            for line in resp.iter_lines():
                                if not line:
                                    continue
                                try:
                                    data = _json.loads(line.decode("utf-8"))
                                except:
                                    continue

                                if "response" in data:
                                    buffer += data["response"]
                                elif "final_response" in data or "message" in data:
                                    final_text = data.get("final_response") or data.get("message", "")
                                    if final_text:
                                        _send_to_pi_safe(final_text)
                            
                            # Send complete response
                            if buffer:
                                olliePrint_simple(f"Response to Pi: '{buffer[:50]}...'")
                                _send_to_pi_safe(buffer)
                        else:
                            olliePrint_simple(f"Chat request failed: {resp.status_code}", level='error')
                    except Exception as exc:
                        olliePrint_simple(f"Error relaying to F.R.E.D.: {exc}", level='error')

                def _send_to_pi_safe(message):
                    """Thread-safe method to send messages to Pi clients"""
                    try:
                        if main_event_loop and not main_event_loop.is_closed():
                            future = asyncio.run_coroutine_threadsafe(_send_to_pi_async(message), main_event_loop)
                            try:
                                future.result(timeout=5)
                            except Exception as e:
                                olliePrint_simple(f"Send to Pi failed: {e}")
                    
                    except Exception as e:
                        olliePrint_simple(f"Pi communication error: {e}")

                async def _send_to_pi_async(message):
                    """Send message to all connected Pi clients via data channels"""
                    if not data_channels:
                        return
                    
                    for channel in list(data_channels):
                        try:
                            channel.send(message)
                        except Exception as e:
                            olliePrint_simple(f"Channel send failed: {e}")

                # Run in background thread
                import threading
                threading.Thread(target=_call_fred, daemon=True).start()

        @pc.on('datachannel')
        def on_datachannel(channel):
            data_channels.add(channel)
            pi_clients.add(client_ip)
            olliePrint_simple(f"Data channel established: {client_ip}")
            
            # Set Pi connection for vision service
            vision_service.set_pi_connection_status(True)
            
            # Determine if client uses local STT
            is_local_stt = client_type_param == 'local_stt'
            
            @channel.on('message')
            def on_message(message):
                # Consolidated message handling
                if message == "[HEARTBEAT]":
                    channel.send("[HEARTBEAT_ACK]")
                    return
                
                # Handle fresh image data from Pi
                if message.startswith("[IMAGE_DATA:"):
                    try:
                        # Extract image data - format: [IMAGE_DATA:jpeg]base64_data
                        header_end = message.find(']')
                        format_info = message[12:header_end]  # Skip "[IMAGE_DATA:"
                        image_b64 = message[header_end + 1:]
                        
                        olliePrint_simple(f"📸 [RECEIVED] Fresh image from Pi ({format_info}, {len(image_b64)} chars)")
                        
                        # Send to vision service for processing
                        async def process_image_async():
                            await vision_service.process_fresh_image(image_b64, format_info)
                        
                        # Schedule processing in the main event loop
                        asyncio.run_coroutine_threadsafe(process_image_async(), main_event_loop)
                        
                    except Exception as e:
                        olliePrint_simple(f"Image processing error: {e}", 'error')
                    return
                
                if is_local_stt:
                    # Handle pre-transcribed text from Pi STT - Keep original simple format
                    if message.startswith("TRANSCRIPTION:"):
                        text = message.replace("TRANSCRIPTION:", "").strip()
                        olliePrint_simple(f"Pi: '{text}'")
                        process_pi_transcription(text, from_pi=True)
                    else:
                        # Plain text - this is the main working path
                        text = message.strip()
                        if text and len(text) > 2:  # Ignore very short messages
                            olliePrint_simple(f"Pi: '{text}'")
                        process_pi_transcription(text, from_pi=True)
                else:
                    # Server-side STT processing
                    if hasattr(stt_service, 'process_audio_from_webrtc'):
                        stt_service.process_audio_from_webrtc(message, from_pi=True)
            
            @channel.on('close')
            def on_close():
                data_channels.discard(channel)
                pi_clients.discard(client_ip)
                if not pi_clients:
                    vision_service.set_pi_connection_status(False)
                olliePrint_simple(f"Data channel closed: {client_ip}")

        @pc.on('track')
        async def on_track(track):
            olliePrint_simple(f"{track.kind.upper()} track connected: {client_ip}")
            
            if track.kind == "audio":
                # Simplified audio handling
                is_local_stt = client_type_param == 'local_stt'
                
                if is_local_stt:
                    # Skip server audio processing
                    async def consume_audio_frames_minimal():
                        frame_count = 0
                        try:
                            async for frame in track:
                                frame_count += 1
                                if frame_count % 100 == 0:  # Log every 100 frames instead of every frame
                                    olliePrint_simple(f"Audio frames: {frame_count} (local STT)")
                        except Exception as e:
                            pass  # Silent audio end
                    
                    asyncio.create_task(consume_audio_frames_minimal())
                else:
                    # Server STT processing
                    async def consume_audio_frames():
                        try:
                            # Start STT if not already running
                            if not getattr(stt_service, "is_processing", False):
                                stt_service.start_processing(process_pi_transcription)
                                olliePrint_simple("STT processing started")
                            
                            frame_count = 0
                            async for frame in track:
                                frame_count += 1
                                
                                # Convert frame to numpy array
                                audio_array = frame.to_ndarray()
                                
                                # Handle stereo to mono conversion
                                if len(audio_array.shape) == 2 and audio_array.shape[1] == 2:
                                    audio_array = np.mean(audio_array, axis=1)
                                
                                # Resample if needed
                                input_sr = frame.sample_rate
                                target_sr = 16000
                                
                                if input_sr != target_sr:
                                    try:
                                        from scipy.signal import resample_poly
                                        gcd_val = np.gcd(target_sr, input_sr)
                                        up = target_sr // gcd_val
                                        down = input_sr // gcd_val
                                        audio_array = resample_poly(audio_array, up, down)
                                    except Exception:
                                        pass  # Continue without resampling
                                
                                # Send to STT service
                                if hasattr(stt_service, 'process_audio_from_webrtc'):
                                    stt_service.process_audio_from_webrtc(audio_array, from_pi=True)
                                
                                # Reduced logging frequency
                                if frame_count % 50 == 0:
                                    olliePrint_simple(f"Processing audio frames: {frame_count}")
                        
                        except Exception as e:
                            pass  # Silent error handling
                    
                    asyncio.create_task(consume_audio_frames())
                
            elif track.kind == "video":
                # No longer processing continuous video - using on-demand capture instead
                olliePrint_simple("Video track connected (on-demand capture mode)")
                
                # Consume video frames without processing to prevent WebRTC errors
                async def consume_video_frames_minimal():
                    try:
                        frame_count = 0
                        async for frame in track:
                            frame_count += 1
                            # Only log occasionally to reduce noise
                            if frame_count % 500 == 0:
                                olliePrint_simple(f"Video connection maintained ({frame_count} frames discarded)")
                    except Exception as e:
                        pass  # Silent video end
                
                asyncio.create_task(consume_video_frames_minimal())
            
            # Record track
            try:
                recorder.addTrack(track)
            except Exception:
                pass  # Silent recording failure

        @pc.on('connectionstatechange')
        async def on_connectionstatechange():
            state = pc.connectionState
            if state == "connected":
                olliePrint_simple(f"ArmLink operational: {client_ip}")
            elif state in ["disconnected", "failed", "closed"]:
                olliePrint_simple(f"ArmLink offline: {client_ip}")
                pcs.discard(pc)

        # Process the offer and create answer
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })

    except Exception as e:
        olliePrint_simple(f"Connection error {client_ip}: {e}", level='error')
        return web.json_response({'error': 'Internal server error'}, status=500)

# SocketIO event handlers for receiving responses from main F.R.E.D. server
@sio_client.event
async def connect():
    olliePrint_simple("[SHELTER-NET] Established secure link to F.R.E.D. mainframe", 'success')
    # Emit connection confirmation
    await sio_client.emit('webrtc_server_connected')
    olliePrint_simple("[BRIDGE] Wasteland communication network ONLINE - standing by for field operations", 'success')

@sio_client.event
async def disconnect():
    olliePrint_simple("[WARNING] Lost connection to F.R.E.D. mainframe - audio relay offline", 'warning')

@sio_client.event
async def connect_error(data):
    olliePrint_simple(f"[ERROR] SocketIO connection error: {data}", 'error')

@sio_client.event
async def voice_response(data):
    """Forward F.R.E.D.'s responses back to Pi clients"""
    if 'response' in data:
        response_text = data['response']
        # Send response to all connected Pi clients
        for channel in data_channels.copy():
            try:
                channel.send(response_text)
                olliePrint_simple(f"📱 Text sent to Pi: {response_text[:50]}...")
            except Exception as e:
                olliePrint_simple(f"Failed to send to Pi: {e}")
                data_channels.discard(channel)

@sio_client.event
async def fred_acknowledgment(data):
    """Forward acknowledgments back to Pi clients"""
    ack_text = data.get('text', '')
    for channel in data_channels.copy():
        try:
            channel.send(f"[ACK] {ack_text}")
            olliePrint_simple(f"Sent acknowledgment to Pi: {ack_text}")
        except Exception as e:
            olliePrint_simple(f"Failed to send acknowledgment to Pi: {e}")
            data_channels.discard(channel)

@sio_client.event
async def fred_audio(data):
    """Forward F.R.E.D.'s audio responses to Pi clients"""
    audio_b64 = data.get('audio_data', '')
    text = data.get('text', '')
    audio_format = data.get('format', 'wav')
    
    if audio_b64:
        olliePrint_simple(f"[TRANSMISSION] Audio matrix received from F.R.E.D. ({len(audio_b64)} chars) for '{text[:50]}...'")
        olliePrint_simple(f"[NETWORK] {len(data_channels)} ArmLink device(s) in communication range")
        
        # Send audio to all connected Pi clients
        sent_count = 0
        for channel in data_channels.copy():
            try:
                message = f"[AUDIO_BASE64:{audio_format}]{audio_b64}"
                channel.send(message)
                sent_count += 1
                olliePrint_simple(f"[RELAY] Voice data transmitted to ArmLink #{sent_count} for '{text[:50]}...'")
            except Exception as e:
                olliePrint_simple(f"[ERROR] ArmLink #{sent_count+1} transmission failure: {e}")
                data_channels.discard(channel)
        
        if sent_count == 0:
            olliePrint_simple("[WARNING] No ArmLink devices available - audio transmission failed")
        else:
            olliePrint_simple(f"[SUCCESS] Voice transmission complete - {sent_count} field operative(s) reached")
        
        # Inform STT to pause listening during playback
        estimated_bytes = int(len(audio_b64) * 3 / 4)  # rough base64 decode
        playback_seconds = max(1, estimated_bytes // 32000 + 1)  # 32kB ≈ 1s at 16-kHz s16 mono
        stt_service.set_speaking_state(True)
        # Schedule automatic resume after playback finishes
        loop = asyncio.get_event_loop()
        loop.call_later(playback_seconds, lambda: stt_service.set_speaking_state(False))
    else:
        olliePrint_simple("[ERROR] No audio data in transmission from F.R.E.D. mainframe")


async def cleanup(app):
    # This is still valuable for graceful shutdown
    olliePrint_simple("Cleaning up server resources...")
    for pc in pcs:
        if pc.connectionState != "closed":
            await pc.close()
    pcs.clear()
    if 'fred_client_task' in app:
        app['fred_client_task'].cancel()
        await app['fred_client_task']

async def init_app(app):
    """Initialize the WebRTC server and connect to main F.R.E.D. server"""
    max_retries = 10
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            olliePrint_simple(f"[BRIDGE] Attempting SocketIO connection to F.R.E.D. mainframe (attempt {attempt + 1}/{max_retries})")
            # Connect to main F.R.E.D. server
            await sio_client.connect('http://localhost:5000')
            olliePrint_simple("[SUCCESS] Connected to F.R.E.D. main server for response forwarding", 'success')
            return  # Success, exit retry loop
        except Exception as e:
            olliePrint_simple(f"[ATTEMPT {attempt + 1}] Connection failed: {e}", 'warning')
            if attempt < max_retries - 1:
                olliePrint_simple(f"[BRIDGE] Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                olliePrint_simple(f"[CRITICAL] Failed to connect to F.R.E.D. main server after {max_retries} attempts", 'error')
                olliePrint_simple("[WARNING] Audio relay will not function - Pi clients will only receive text responses", 'warning')

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
        olliePrint_simple("Verbose logging enabled", level='info')

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

    global runner
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=args.host, port=args.port, ssl_context=ssl_context)
    await site.start()
    
    olliePrint_simple(f"[NETWORK] WebRTC server ready on http://{args.host}:{args.port}")
    olliePrint_simple(f"🔐 Authentication: {'Enabled' if FRED_AUTH_TOKEN else 'Disabled'}")
    olliePrint_simple(f"🔢 Max connections: {MAX_CONNECTIONS}")

    # Keep server running until interrupted
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        olliePrint_simple("\n⌨️ Server shutting down manually...")
    finally:
        # This cleanup is for when running the file directly
        if runner:
            asyncio.run(runner.cleanup())
        olliePrint_simple("✅ Server shutdown complete.")
