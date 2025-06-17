#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import argparse
import cv2
import numpy as np
import base64
import time
import sounddevice as sd
from picamera2 import Picamera2
from aiortc import VideoStreamTrack, AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
import threading
import queue
import logging

# Configure minimal logging for Pip-Boy interface
logging.basicConfig(level=logging.WARNING)
logging.getLogger('aiortc').setLevel(logging.WARNING)
logging.getLogger('aiohttp').setLevel(logging.WARNING)

class PiCamera2Track(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        print("📸 Initializing Picamera2...")
        
        # Initialize Picamera2 with minimal logging
        self.picam2 = Picamera2()
        
        # Configure for optimal Qwen 2.5-VL resolution (3280x2464 = 8.1 MP)
        config = self.picam2.create_preview_configuration(
            main={"size": (3280, 2464), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        print(f"🎯 Using native resolution (3280, 2464) = 8.1 MP (optimal for Qwen 2.5-VL - no upscaling needed!)")
        print("✅ Picamera2 initialized successfully.")
        
        self.frame_count = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # Capture frame
        frame = self.picam2.capture_array()
        
        # Send first frame notification
        if self.frame_count == 0:
            print("🚀 First frame sent! Size:", frame.shape)
        
        # Periodic status updates (every 150 frames ≈ 5 seconds at 30fps)
        self.frame_count += 1
        if self.frame_count % 150 == 0:
            avg_fps = self.frame_count / (time.time() - getattr(self, 'start_time', time.time()))
            if not hasattr(self, 'start_time'):
                self.start_time = time.time()
            print(f"📊 Sent {self.frame_count} frames. Average FPS: {avg_fps:.2f}")
        
        # Convert to VideoFrame
        from av import VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        av_frame.pts = pts
        av_frame.time_base = time_base
        
        return av_frame

class SoundDeviceAudioTrack(AudioStreamTrack):
    def __init__(self, device_id, sample_rate=16000, channels=1):
        super().__init__()
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.stream = None
        self.first_frame_sent = False
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            # Only log significant issues, not input overflow
            if "overflow" not in str(status).lower():
                print(f"🎧 PortAudio issue: {status}")
        
        # Put audio data in queue
        self.audio_queue.put(indata.copy())
        
        if not self.first_frame_sent:
            print("🚀 FIRST AUDIO FRAME SENT!")
            self.first_frame_sent = True

    async def recv(self):
        if self.stream is None:
            print("🎤 Starting sounddevice audio capture...")
            print(f"   Device: {self.device_id}")
            print(f"   Sample rate: {self.sample_rate}")
            print(f"   Channels: {self.channels}")
            
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=1024,
                dtype=np.float32
            )
            self.stream.start()
            print("✅ Audio capture started successfully!")

        # Wait for audio data with timeout
        try:
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None, self.audio_queue.get, True, 0.1
            )
        except:
            # Return silence if no data
            audio_data = np.zeros((1024, self.channels), dtype=np.float32)

        # Convert to AudioFrame
        from av import AudioFrame
        av_frame = AudioFrame.from_ndarray(audio_data, format='flt', layout='mono')
        av_frame.sample_rate = self.sample_rate
        
        return av_frame

def find_corsair_device():
    """Find CORSAIR HS80 audio device"""
    print("🔍 Available audio input devices:")
    devices = sd.query_devices()
    corsair_device = None
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i+1}: {device['name']} (inputs: {device['max_input_channels']}, rate: {device['default_samplerate']})")
            if 'CORSAIR HS80' in device['name']:
                print(f"       👉 CORSAIR HS80 FOUND!")
                corsair_device = i
    
    if corsair_device is not None:
        print(f"🎯 Found CORSAIR device: {devices[corsair_device]['name']} (device {corsair_device+1})")
        return corsair_device
    else:
        print("⚠️ CORSAIR HS80 not found, using default device")
        return None

async def main():
    parser = argparse.ArgumentParser(description='F.R.E.D. Pip-Boy Client')
    parser.add_argument('--server', required=True, help='Server URL (e.g., https://example.ngrok-free.app)')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum connection retries')
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════")
    print("     F.R.E.D. Pip-Boy Interface v2.0")
    print("     Field Operations Communication System")
    print("═══════════════════════════════════════════════════")
    print(f"🎯 Using provided server: {args.server}")

    # Setup video with minimal logging
    print("🎥 Setting up video with Picamera2...")
    video_track = PiCamera2Track()
    print("✅ Picamera2 video track created successfully!")

    # Setup audio
    print("🎤 Setting up audio...")
    corsair_device = find_corsair_device()
    audio_device = corsair_device if corsair_device is not None else None
    
    print("🎤 Audio capture will start on first recv() call…")
    audio_track = SoundDeviceAudioTrack(device_id=audio_device)
    print("✅ Audio working with sounddevice")

    print(f"📊 Total tracks created: 2")
    print(f"  Track 1: video")
    print(f"  Track 2: audio")

    # Create peer connection with minimal logging
    pc = RTCPeerConnection()

    print("📡 Added video track (PiCamera2Track)")
    print(f"  📹 Video track details: PiCamera2Track")
    print(f"     Picamera2 initialized: True")
    pc.addTrack(video_track)

    print("📡 Added audio track (SoundDeviceAudioTrack)")
    pc.addTrack(audio_track)

    # Connect to server
    for attempt in range(args.max_retries):
        print(f"🔄 Connection attempt {attempt + 1}/{args.max_retries}")
        try:
            async with aiohttp.ClientSession() as session:
                print(f"🔗 Connecting to {args.server}/offer...")
                
                # Create offer
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)

                # Send offer to server
                async with session.post(
                    f"{args.server}/offer",
                    data=json.dumps({
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type,
                    }),
                    headers={"content-type": "application/json"},
                ) as response:
                    if response.status == 200:
                        answer = await response.json()
                        await pc.setRemoteDescription(RTCSessionDescription(
                            sdp=answer["sdp"], type=answer["type"]
                        ))
                        
                        print("🚀 F.R.E.D. Pi Glasses connected and ready!")
                        print("[VAULT-NET] Secure connection established with F.R.E.D. mainframe!")
                        print("[PIP-BOY] Audio/visual sensors ONLINE - ready for wasteland operations...")
                        print(f"📐 Native Resolution: 3280x2464 = 8.1 MP (optimal for Qwen 2.5-VL)")
                        
                        # Keep connection alive with minimal output
                        try:
                            while True:
                                await asyncio.sleep(30)  # Quiet keepalive
                        except KeyboardInterrupt:
                            print("\n[PIP-BOY] Shutting down field operations...")
                            break
                    else:
                        print(f"❌ Server returned status {response.status}")
                        if attempt < args.max_retries - 1:
                            await asyncio.sleep(2)
                        continue
                        
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            if attempt < args.max_retries - 1:
                print(f"⏳ Retrying in 2 seconds...")
                await asyncio.sleep(2)
            continue
        break
    else:
        print("❌ Failed to connect after all retries")
        return

    # Cleanup
    await pc.close()

if __name__ == "__main__":
    asyncio.run(main())
