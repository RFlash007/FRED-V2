from aiohttp import web
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from stt_service import stt_service

pcs = set()

async def index(request):
    return web.Response(text="FRED WebRTC Server Running")

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    pc = RTCPeerConnection()
    pcs.add(pc)

    recorder = MediaBlackhole()

    @pc.on('track')
    async def on_track(track):
        if track.kind == 'audio':
            @track.on('frame')
            async def on_frame(frame):
                pcm = frame.to_ndarray()
                # Forward PCM audio to the existing STT pipeline
                stt_service.audio_queue.put(pcm)
        elif track.kind == 'video':
            # Placeholder for future object detection processing
            pass
        await recorder.addTrack(track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({'sdp': pc.localDescription.sdp,
                              'type': pc.localDescription.type})

async def cleanup(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/offer', offer)
app.on_shutdown.append(cleanup)

if __name__ == '__main__':
    web.run_app(app, port=8080)
