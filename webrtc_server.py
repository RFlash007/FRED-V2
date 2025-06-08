from aiohttp import web
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from stt_service import stt_service
from vision_service import vision_service
import socketio
import json

pcs = set()
data_channels = set()  # Store data channels for sending responses back to Pi
pi_clients = set()  # Track Pi clients for vision processing

# SocketIO client to connect to main F.R.E.D. server
sio_client = socketio.AsyncClient()

async def index(request):
    return web.Response(text="FRED WebRTC Server Running")

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    pc = RTCPeerConnection()
    pcs.add(pc)

    recorder = MediaBlackhole()
    
    # Store data channel when created
    @pc.on('datachannel')
    def on_datachannel(channel):
        data_channels.add(channel)
        pi_clients.add(channel)  # Track Pi clients
        print(f"Data channel '{channel.label}' opened")
        
        # Notify vision service that Pi is connected
        vision_service.set_pi_connection_status(True)
        print(f"Pi glasses connected - starting vision processing")
        
        @channel.on('close')
        def on_close():
            data_channels.discard(channel)
            pi_clients.discard(channel)
            print(f"Data channel '{channel.label}' closed")
            
            # If no more Pi clients, stop vision processing
            if not pi_clients:
                vision_service.set_pi_connection_status(False)
                print(f"Pi glasses disconnected - stopping vision processing")

    @pc.on('track')
    async def on_track(track):
        print(f"📡 Received {track.kind} track from Pi client")
        if track.kind == 'audio':
            print("🎤 Setting up audio frame processing...")
            frame_count = 0
            @track.on('frame')
            async def on_frame(frame):
                nonlocal frame_count
                frame_count += 1
                if frame_count % 100 == 0:  # Log every 100th frame to avoid spam
                    print(f"🎵 Audio frame #{frame_count} received")
                pcm = frame.to_ndarray()
                # Mark this audio as coming from Pi and forward to STT pipeline
                stt_service.audio_queue.put((pcm, True))  # (audio_data, from_pi)
        elif track.kind == 'video':
            print("📹 Setting up video frame processing...")
            @track.on('frame')
            async def on_frame(frame):
                # Store frame for vision processing
                vision_service.store_latest_frame(frame)
        await recorder.addTrack(track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({'sdp': pc.localDescription.sdp,
                              'type': pc.localDescription.type})

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
    web.run_app(app, port=8080)
