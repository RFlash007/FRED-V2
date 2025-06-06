import asyncio
import argparse
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


def create_local_tracks(video=True, audio=True):
    tracks = []
    if video:
        try:
            player = MediaPlayer('/dev/video0', format='v4l2')
            if player.video:
                tracks.append(player.video)
        except Exception:
            pass
    if audio:
        try:
            mic = MediaPlayer('default', format='pulse')
            if mic.audio:
                tracks.append(mic.audio)
        except Exception:
            pass
    return tracks


async def run(server_url):
    pc = RTCPeerConnection()

    channel = pc.createDataChannel('chat')

    @channel.on('open')
    def on_open():
        print('Data channel opened')

    @channel.on('message')
    def on_message(message):
        print('FRED:', message)

    for t in create_local_tracks():
        pc.addTrack(t)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    response = requests.post(f'{server_url}/offer', json={
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    })
    answer = RTCSessionDescription(**response.json())
    await pc.setRemoteDescription(answer)

    await asyncio.Future()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True,
                        help='URL of FRED WebRTC server, e.g. http://fred:8080')
    args = parser.parse_args()

    asyncio.run(run(args.server))
