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
client_video_tracks = {}  # Store video tracks for on-demand frame requests
connection_timestamps = {}  # Track connection times for rate limiting

# Runner instance for graceful shutdown when running as a script
runner = None

async def request_frame_from_client(client_ip):
    """Request a fresh frame from a specific client"""
    if client_ip not in client_video_tracks:
        olliePrint_simple(f"⚠️ No video track available for {client_ip}")
        return None
    
    try:
        track = client_video_tracks[client_ip]
        olliePrint_simple(f"📸 Requesting fresh frame from {client_ip}")
        frame = await track.recv()
        olliePrint_simple(f"✅ Fresh frame received from {client_ip} (size: {frame.width}x{frame.height})")
        return frame
    except Exception as e:
        olliePrint_simple(f"❌ Failed to request frame from {client_ip}: {e}")
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
        olliePrint_simple(f"Rate limit exceeded for {client_ip}", level='warning')
        return web.json_response({'error': 'Too many connections'}, status=429)
    
    # Authenticate request
    if not authenticate_request(request):
        olliePrint_simple(f"Unauthorized connection attempt from {client_ip}", level='warning')
        return web.json_response({'error': 'Unauthorized'}, status=401)
    
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        olliePrint_simple(f"[SHELTER-NET] New ArmLink connection from {client_ip}")

        # Initialize media recorder (blackhole – we don't save media)
        recorder = MediaBlackhole()
        # Start the recorder (MediaBlackhole.start is *not* a coroutine)
        try:
            await recorder.start()
        except Exception as e:
            olliePrint_simple(f"Recorder failed to start: {e}", level='warning')
        
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

                olliePrint_simple(f"[ARMLINK COMM] Field operative transmission: '{text}'")

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
                            olliePrint_simple(f"Chat request failed: {resp.status_code}", level='error')
                    except Exception as exc:
                        olliePrint_simple(f"Error relaying message to F.R.E.D.: {exc}", level='error')

                import threading as _threading
                _threading.Thread(target=_call_fred, daemon=True).start()

            # Start STT processing – this will spin up the processing thread
            try:
                stt_service.start_processing(process_pi_transcription)
                olliePrint_simple("STT processing thread started for Pi audio")
            except Exception as stt_err:
                olliePrint_simple(f"Unable to start STT processing: {stt_err}", level='error')
        
        # Store data channel when created
        @pc.on('datachannel')
        def on_datachannel(channel):
            data_channels.add(channel)
            pi_clients.add(channel)  # Track Pi clients
            olliePrint_simple(f"[ARMLINK] Data channel '{channel.label}' established with field operative at {client_ip}")
            
            # Notify vision service that Pi is connected
            vision_service.set_pi_connection_status(True)
            olliePrint_simple(f"[OPTICS] ArmLink visual sensors ONLINE - initiating reconnaissance protocols")
            
            # Check if this is a local STT client
            client_type = params.get('client_type', 'unknown')
            is_local_stt = 'local_stt' in client_type
            
            # Debug logging for client type detection
            olliePrint_simple(f"🔍 [DEBUG] Client type: '{client_type}', is_local_stt: {is_local_stt}")
            
            if is_local_stt:
                olliePrint_simple(f"🧠 [LOCAL STT] Client using on-device transcription - text-only mode")
            
            @channel.on('message')
            def on_message(message):
                # Debug logging for message handling
                olliePrint_simple(f"🔍 [DEBUG] Message received: '{message}' | is_local_stt: {is_local_stt}")
                
                # Handle heartbeat messages
                if message == '[HEARTBEAT]':
                    olliePrint_simple(f"[VITAL-MONITOR] ArmLink heartbeat confirmed from {client_ip}")
                    channel.send('[HEARTBEAT_ACK]')
                elif is_local_stt:
                    olliePrint_simple(f"🔀 [DEBUG] Taking LOCAL STT path")
                    # Handle transcribed text from Pi
                    try:
                        import json
                        data = json.loads(message)
                        if data.get('type') == 'transcription':
                            text = data.get('text', '').strip()
                            if text:
                                olliePrint_simple(f"🗣️ [PI TRANSCRIPTION] '{text}' from {client_ip}")
                                olliePrint_simple(f"📨 [SERVER-RECEIVED] Processing transcription: '{text}'")
                                # Process the transcribed text (same as old audio processing)
                                process_pi_transcription(text, from_pi=True)
                                olliePrint_simple(f"✅ [SERVER-PROCESSED] Transcription sent to F.R.E.D.: '{text}'")
                    except json.JSONDecodeError:
                        # Handle plain text messages
                        if message.strip():
                            olliePrint_simple(f"🗣️ [PI TRANSCRIPTION] '{message.strip()}' from {client_ip}")
                            olliePrint_simple(f"📨 [SERVER-RECEIVED] Processing plain text: '{message.strip()}'")
                            process_pi_transcription(message.strip(), from_pi=True)
                            olliePrint_simple(f"✅ [SERVER-PROCESSED] Plain text sent to F.R.E.D.: '{message.strip()}'")
                else:
                    olliePrint_simple(f"🔀 [DEBUG] Taking ELSE path (not local STT)")
                    olliePrint_simple(f"[ARMLINK COMM] Field operative message: {message}")
            
            @channel.on('close')
            def on_close():
                data_channels.discard(channel)
                pi_clients.discard(channel)
                olliePrint_simple(f"[ARMLINK] Data channel '{channel.label}' closed from {client_ip}")
                
                # If no more Pi clients, stop vision processing
                if not pi_clients:
                    vision_service.set_pi_connection_status(False)
                    olliePrint_simple(f"[OPTICS] All ArmLinks disconnected - reconnaissance protocols OFFLINE")

        @pc.on('track')
        async def on_track(track):
            olliePrint_simple(f"[SIGNAL] {track.kind.upper()} link established with field operative {client_ip}")
            
            if track.kind == 'audio':
                # Check if this is a local STT client
                client_type = params.get('client_type', 'unknown')
                is_local_stt = 'local_stt' in client_type
                
                if is_local_stt:
                    olliePrint_simple("[AUDIO] Client uses local STT - skipping server audio processing")
                    # Just consume frames without processing to prevent buffer buildup
                    async def consume_audio_frames_minimal():
                        try:
                            frame_count = 0
                            while True:
                                frame = await track.recv()
                                frame_count += 1
                                if frame_count == 1:
                                    olliePrint_simple(f"[AUDIO] Receiving audio frames from {client_ip} (not processing - local STT active)")
                                elif frame_count % 5000 == 0:  # Very minimal logging
                                    olliePrint_simple(f"[AUDIO] {frame_count} frames received (local STT mode)")
                        except Exception as e:
                            olliePrint_simple(f"[AUDIO] Audio stream ended: {e}")
                    
                    asyncio.create_task(consume_audio_frames_minimal())
                else:
                    olliePrint_simple("[AUDIO] Voice communication protocols active (server STT)")
                    frame_count = 0
                    
                    # Create a task to consume audio frames from the track
                    async def consume_audio_frames():
                        nonlocal frame_count
                        
                        try:
                            buffer = []
                            total_samples = 0
                            # Pi-audio: use ~5-second chunk (80,000 samples @16 kHz) like old system
                            CHUNK_TARGET = config.STT_SAMPLE_RATE * 5  # 48 000 samples

                            while True:
                                frame = await track.recv()
                                frame_count += 1

                                if frame_count == 1:
                                    olliePrint_simple(f"[AUDIO] First transmission received from {client_ip}")
                                # Reduced frequency of frame logging for conciseness
                                elif frame_count % 2000 == 0:  # Every ~40 seconds instead of every 10
                                    olliePrint_simple(f"[AUDIO] {frame_count} frames processed")

                                pcm = frame.to_ndarray()

                                # Shape (1, samples) -> flatten to samples
                                if pcm.ndim == 2:
                                    pcm = pcm.flatten()

                                # HIGH QUALITY AUDIO: Preserve original 48kHz and let Whisper handle resampling
                                input_sr = getattr(frame, 'sample_rate', 48000)
                                target_sr = config.STT_SAMPLE_RATE
                                
                                # CRITICAL FIX: Use high-quality resampling and preserve float32 precision
                                if input_sr != target_sr:
                                    try:
                                        # Convert to float32 normalized [-1, 1] for high-quality processing
                                        if pcm.dtype == np.int16:
                                            pcm_f32 = pcm.astype(np.float32) / 32768.0
                                        elif pcm.dtype == np.float32:
                                            pcm_f32 = pcm
                                        else:
                                            pcm_f32 = pcm.astype(np.float32)
                                        
                                        # High-quality Kaiser window resampling (like your old system)
                                        from scipy.signal import resample_poly
                                        pcm_resampled = resample_poly(pcm_f32, target_sr, input_sr)
                                        
                                        # Keep as float32 to preserve quality (don't quantize to int16!)
                                        pcm = pcm_resampled.astype(np.float32)
                                        
                                    except Exception as rs_err:
                                        olliePrint_simple(f"[WARNING] Audio resampling error ({input_sr}->{target_sr}): {rs_err}")
                                        # Fallback: simple decimation
                                        if input_sr == 48000 and target_sr == 16000:
                                            pcm = pcm[::3].astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm[::3]
                                else:
                                    # Normalize to float32 even if no resampling needed
                                    if pcm.dtype == np.int16:
                                        pcm = pcm.astype(np.float32) / 32768.0
                                    elif pcm.dtype != np.float32:
                                        pcm = pcm.astype(np.float32)

                                buffer.append(pcm)
                                total_samples += pcm.shape[0]

                                if total_samples >= CHUNK_TARGET:
                                    chunk = np.concatenate(buffer)
                                    stt_service.audio_queue.put((chunk, True))
                                    buffer = []
                                    total_samples = 0
                                    # Reduced frequency of chunk processing logs
                                    if frame_count % 3000 == 0:  # Much less frequent
                                        olliePrint_simple(f"[PROCESSING] Speech data sent to recognition system")

                        except Exception as e:
                            olliePrint_simple(f"[ERROR] Audio transmission ended: {e}")
                        finally:
                            # graceful exit when track ends
                            return
                    
                    asyncio.create_task(consume_audio_frames())
                
            elif track.kind == 'video':
                olliePrint_simple("[OPTICS] Visual reconnaissance feed active")
                frame_count = 0
                
                # Store track for on-demand frame requests
                client_video_tracks[client_ip] = track
                
                # Test: Request first frame immediately
                try:
                    frame = await track.recv()
                    frame_count += 1
                    olliePrint_simple(f"[OPTICS] Visual feed confirmed ({frame.width}x{frame.height})")
                    
                    # Store frame for vision processing
                    vision_service.store_latest_frame(frame)
                    
                except Exception as e:
                    olliePrint_simple(f"[ERROR] Visual feed initialization failed: {e}")
            
            # Add track to recorder safely
            if recorder:
                try:
                    recorder.addTrack(track)
                except Exception as rec_err:
                    olliePrint_simple(f"[WARNING] Track recording failed: {rec_err}", level='error')

        @pc.on('connectionstatechange')
        async def on_connectionstatechange():
            
            if pc.connectionState == 'connected':
                olliePrint_simple(f"[SUCCESS] ArmLink fully operational at {client_ip}")
            elif pc.connectionState in ('failed', 'closed'):
                olliePrint_simple(f"[DISCONNECT] ArmLink {client_ip} offline")
                # Clean up video track reference
                if client_ip in client_video_tracks:
                    del client_video_tracks[client_ip]
                await pc.close()
                pcs.discard(pc)  # free slot for new connections

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({'sdp': pc.localDescription.sdp,
                                  'type': pc.localDescription.type})
    
    except Exception as e:
        olliePrint_simple(f"Error handling offer from {client_ip}: {e}", level='error')
        return web.json_response({'error': 'Internal server error'}, status=500)

# SocketIO event handlers for receiving responses from main F.R.E.D. server
@sio_client.event
async def connect():
    olliePrint_simple("[SHELTER-NET] Established secure link to F.R.E.D. mainframe")
    # Emit connection confirmation
    await sio_client.emit('webrtc_server_connected')
    olliePrint_simple("[BRIDGE] Wasteland communication network ONLINE - standing by for field operations")

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
    try:
        # Connect to main F.R.E.D. server
        await sio_client.connect('http://localhost:5000')
        olliePrint_simple("Connected to F.R.E.D. main server for response forwarding")
    except Exception as e:
        olliePrint_simple(f"Failed to connect to F.R.E.D. main server: {e}")

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
