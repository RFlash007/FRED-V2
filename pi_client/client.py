#!/usr/bin/env python3

"""
F.R.E.D. Pip-Boy Interface v2.0
Field Operations Communication System

Advanced AI-powered reconnaissance and communication device
for wasteland operations and mainframe connectivity.
"""

import asyncio
import json
import time
import argparse
import contextlib
import io
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from aiortc.contrib.media import MediaRecorder
import aiohttp
import cv2
import numpy as np
import logging

# Configure logging to suppress verbose output
logging.getLogger("libav").setLevel(logging.CRITICAL)
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)

# Suppress Picamera2 verbose logging
import sys
old_stdout = sys.stdout
old_stderr = sys.stderr

def print_vault_header():
    """Display the Pip-Boy startup sequence"""
    print("═══════════════════════════════════════════════════")
    print("     F.R.E.D. Pip-Boy Interface v2.0")
    print("     Field Operations Communication System")
    print("═══════════════════════════════════════════════════")

try:
    # Suppress Picamera2 initialization messages
    sys.stdout = contextlib.redirect_stdout(io.StringIO())
    sys.stderr = contextlib.redirect_stderr(io.StringIO())
    
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import CircularOutput
    import sounddevice as sd
    
    # Restore stdout/stderr
    sys.stdout = old_stdout  
    sys.stderr = old_stderr
    
except ImportError as e:
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print(f"[ERROR] Required dependencies not found: {e}")
    print("[INFO] Please run: pip install picamera2 sounddevice")
    exit(1)

class PiCamera2Track(VideoStreamTrack):
    """
    Pip-Boy Visual Reconnaissance System
    Advanced camera interface for wasteland surveillance
    """
    kind = "video"

    def __init__(self):
        super().__init__()
        self.picam2 = None
        self.camera_ready = False
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize_camera(self):
        """Initialize Pip-Boy visual sensors"""
        try:
            # Suppress Picamera2 verbose output during initialization
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                
                self.picam2 = Picamera2()
                
                # Configure for optimal Qwen 2.5-VL performance (native 3280x2464)
                config = self.picam2.create_video_configuration(
                    main={"size": (3280, 2464), "format": "RGB888"},
                    raw={"size": (3280, 2464)},
                    lores={"size": (640, 480), "format": "YUV420"}
                )
                self.picam2.configure(config)
                
                print("🎯 Pip-Boy visual sensors: OPTIMAL resolution (8.1 MP)")
                
                self.picam2.start()
                self.camera_ready = True
                
                print("✅ Visual reconnaissance systems ONLINE")
                return True
                
        except Exception as e:
            print(f"[CRITICAL] Pip-Boy visual systems OFFLINE: {e}")
            return False
    
    async def recv(self):
        """Capture frame from Pip-Boy visual sensors"""
        if not self.camera_ready:
            if not self.initialize_camera():
                # Return a black frame if camera fails
                pts, time_base = await self.next_timestamp()
                frame = VideoFrame.from_ndarray(
                    np.zeros((480, 640, 3), dtype=np.uint8), format="rgb24"
                )
                frame.pts = pts
                frame.time_base = time_base
                return frame
        
        try:
            array = self.picam2.capture_array()
            
            self.frame_count += 1
            
            # Status update every 150 frames (much less frequent)
            if self.frame_count % 150 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                print(f"📊 Pip-Boy recon data: {self.frame_count} frames @ {fps:.1f} FPS")
            
            # Convert to VideoFrame
            from av import VideoFrame
            pts, time_base = await self.next_timestamp()
            frame = VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            
            if self.frame_count == 1:
                print("🚀 FIRST VISUAL TRANSMISSION sent to mainframe!")
                h, w = array.shape[:2]
                print(f"📐 Visual data specs: {w}x{h} = {(w*h)/1e6:.1f} MP")
            
            return frame
            
        except Exception as e:
            print(f"[WARNING] Visual sensor glitch: {e}")
            # Return black frame on error
            pts, time_base = await self.next_timestamp()
            frame = VideoFrame.from_ndarray(
                np.zeros((480, 640, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = pts
            frame.time_base = time_base
            return frame

class SoundDeviceAudioTrack(AudioStreamTrack):
    """
    Pip-Boy Audio Communication Array
    Military-grade voice transmission system for field operations
    """
    kind = "audio"
    
    def __init__(self, device_id=None):
        super().__init__()
        self.device_id = device_id
        self.sample_rate = 16000  # Optimized for mainframe compatibility
        self.channels = 1
        self.frame_size = 480  # 30ms frames
        self.audio_buffer = asyncio.Queue()
        self.stream = None
        self.audio_started = False
        self.overflow_count = 0
        self.last_overflow_warning = 0
        
    def find_corsair_device(self):
        """Locate CORSAIR communication array"""
        devices = sd.query_devices()
        print("🔍 Scanning for communication devices:")
        
        corsair_device = None
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_name = device['name']
                print(f"  {i+1}: {device_name}")
                
                # Prefer CORSAIR devices for field operations
                if 'CORSAIR' in device_name.upper():
                    corsair_device = i
                    print(f"       👉 CORSAIR ARRAY DETECTED!")
        
        if corsair_device is not None:
            print(f"🎯 Primary comms: {devices[corsair_device]['name']} (device {corsair_device+1})")
            return corsair_device
        else:
            print("🎯 Using default communication array")
            return None
    
    def audio_callback(self, indata, frames, time, status):
        """Process incoming audio from field operative"""
        if status:
            # Only warn about overflows occasionally to avoid spam
            if status.input_overflow:
                self.overflow_count += 1
                current_time = time.time()
                if current_time - self.last_overflow_warning > 10:  # Only every 10 seconds
                    print("⚠️ Pip-Boy audio buffer overflow - optimizing...")
                    self.last_overflow_warning = current_time
        
        # Queue audio data for transmission
        audio_data = indata.copy()
        try:
            self.audio_buffer.put_nowait(audio_data)
        except asyncio.QueueFull:
            pass  # Drop frames if queue is full (silent handling)
    
    async def recv(self):
        """Receive audio from Pip-Boy communication array"""
        if not self.audio_started:
            device = self.find_corsair_device()
            
            try:
                self.stream = sd.InputStream(
                    device=device,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=self.audio_callback,
                    blocksize=self.frame_size,
                    dtype='float32'
                )
                self.stream.start()
                self.audio_started = True
                print("🎤 Pip-Boy audio array ONLINE")
                print("✅ Voice transmission ready!")
                print("🚀 FIRST AUDIO TRANSMISSION sent to mainframe!")
                
            except Exception as e:
                print(f"[CRITICAL] Audio array malfunction: {e}")
                return None
        
        # Wait for audio data
        try:
            audio_data = await asyncio.wait_for(self.audio_buffer.get(), timeout=1.0)
            
            # Convert to AudioFrame
            from av import AudioFrame
            frame = AudioFrame.from_ndarray(
                audio_data.T,  # Transpose for av format
                layout='mono',
                format='flt'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self.next_timestamp()
            
            return frame
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"[ERROR] Audio transmission error: {e}")
            return None

class PiClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        self.pc = None
        self.connection_attempts = 0
        self.max_attempts = 5
        
    async def connect_to_mainframe(self):
        """Establish secure connection to F.R.E.D. mainframe"""
        print(f"🎯 Target mainframe: {self.server_url}")
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"🔄 Connection attempt {attempt}/{self.max_attempts}")
            
            try:
                # Initialize Pip-Boy systems
                print("🎥 Initializing Pip-Boy visual systems...")
                
                # Suppress Picamera2 verbose initialization
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    video_track = PiCamera2Track()
                
                print("🎤 Initializing Pip-Boy audio systems...")
                audio_track = SoundDeviceAudioTrack()
                
                # Create secure peer connection
                self.pc = RTCPeerConnection()
                
                # Add tracks for mainframe transmission
                self.pc.addTrack(video_track)
                self.pc.addTrack(audio_track)
                
                print(f"📊 Pip-Boy systems: 2 transmission channels ready")
                print(f"  📹 Visual reconnaissance: {video_track.__class__.__name__}")
                print(f"  📡 Audio communication: {audio_track.__class__.__name__}")
                
                # Create WebRTC offer
                offer = await self.pc.createOffer()
                await self.pc.setLocalDescription(offer)
                
                # Send offer to mainframe
                print(f"🔗 Establishing quantum entanglement with mainframe...")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_url}/offer",
                        json={"sdp": offer.sdp, "type": offer.type},
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            answer = RTCSessionDescription(
                                sdp=data["sdp"], 
                                type=data["type"]
                            )
                            await self.pc.setRemoteDescription(answer)
                            
                            print("🚀 F.R.E.D. Pip-Boy connected and OPERATIONAL!")
                            print("[VAULT-NET] Secure connection established with F.R.E.D. mainframe!")
                            print("[PIP-BOY] Audio/visual sensors ONLINE - ready for wasteland operations...")
                            return True
                        else:
                            print(f"[ERROR] Mainframe rejected connection: {response.status}")
                            
            except Exception as e:
                print(f"[ERROR] Connection attempt {attempt} failed: {e}")
                if attempt < self.max_attempts:
                    await asyncio.sleep(2)
        
        print("[CRITICAL] Unable to establish mainframe connection")
        return False
    
    async def run(self):
        """Main Pip-Boy operational loop"""
        if await self.connect_to_mainframe():
            try:
                print("\n🎯 Pip-Boy field operations ACTIVE")
                print("⚡ Press Ctrl+C to terminate field mission")
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Field mission terminated by operator")
            except Exception as e:
                print(f"[CRITICAL] Pip-Boy system error: {e}")
            finally:
                if self.pc:
                    await self.pc.close()
                    print("[SHUTDOWN] Pip-Boy systems powered down")

async def main():
    print_vault_header()
    
    parser = argparse.ArgumentParser(description='F.R.E.D. Pip-Boy Interface')
    parser.add_argument('--server', required=True, help='F.R.E.D. mainframe URL')
    args = parser.parse_args()
    
    client = PiClient(args.server)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())
