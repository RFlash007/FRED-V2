#!/usr/bin/env python3
"""
F.R.E.D. Pi Client with Local Speech-to-Text
by OllieTec
Processes voice locally using tiny.en model, sends transcribed text to server
"""

import os
# Shut up libcamera: only errors
os.environ.setdefault('LIBCAMERA_LOG_LEVELS', '*:3')

import asyncio
import argparse
import requests
import json
import time
import sys
import base64
import tempfile
import subprocess
import threading
import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaPlayer

from ollietec_theme import apply_theme, banner
from pi_stt_service import pi_stt_service

# Apply theming to all prints
apply_theme()

def play_audio_from_base64(audio_b64, format_type='wav'):
    """Decode base64 audio and play it on the Pi in a non-blocking thread."""
    
    def _playback_thread():
        temp_file_path = None
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(audio_b64)
            print(f"[AUDIO] F.R.E.D. transmission received ({len(audio_data)} bytes)")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Play using aplay (ALSA) - most reliable on Pi
            try:
                subprocess.run(['aplay', temp_file_path], check=True, capture_output=True, timeout=30)
                print("[SUCCESS] F.R.E.D. voice transmission complete")
            except subprocess.CalledProcessError:
                # Fallback to paplay (PulseAudio)
                try:
                    subprocess.run(['paplay', temp_file_path], check=True, capture_output=True, timeout=30)
                    print("[SUCCESS] F.R.E.D. voice transmission complete (PulseAudio)")
                except subprocess.CalledProcessError:
                    # Last resort: mpv
                    try:
                        subprocess.run(['mpv', '--no-video', temp_file_path], check=True, capture_output=True, timeout=30)
                        print("[SUCCESS] F.R.E.D. voice transmission complete (mpv)")
                    except subprocess.CalledProcessError:
                        print("[CRITICAL] All audio protocols failed - check Pip-Boy speakers")
            
        except Exception as e:
            print(f"[CRITICAL] Audio playback system failure: {e}")
        finally:
            # Clean up temporary file
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_err:
                    print(f"[WARNING] Failed to purge audio cache: {cleanup_err}")

    # Run playback in a separate daemon thread to avoid blocking the main asyncio loop
    playback_thread = threading.Thread(target=_playback_thread, daemon=True)
    playback_thread.start()

class LocalAudioProcessor:
    """Handles local audio capture and processing for STT"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_thread = None
        self.sample_rate = 16000
        self.channels = 1
        self.blocksize = int(0.5 * self.sample_rate)  # 0.5 second blocks
        
    def start_recording(self, callback):
        """Start recording audio and processing locally"""
        if self.is_recording:
            return
            
        print("🎤 [PIP-BOY] Starting local audio processing...")
        self.is_recording = True
        
        # Initialize STT service
        if not pi_stt_service.start_processing(callback):
            print("❌ [CRITICAL] Failed to start voice recognition")
            self.is_recording = False
            return False
            
        # Start audio capture thread
        self.audio_thread = threading.Thread(
            target=self._audio_capture_loop, 
            daemon=True
        )
        self.audio_thread.start()
        return True
        
    def stop_recording(self):
        """Stop recording and processing"""
        print("🔇 [PIP-BOY] Stopping audio processing...")
        self.is_recording = False
        pi_stt_service.stop_processing()
        
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
    
    def _audio_capture_loop(self):
        """Capture audio using sounddevice and feed to STT"""
        try:
            print(f"🎧 [AUDIO] Capturing at {self.sample_rate}Hz, {self.channels} channel(s)")
            
            def audio_callback(indata, frames, time, status):
                """Callback for each audio block"""
                if status:
                    print(f"[WARNING] Audio status: {status}")
                
                # The STT service expects a 1D float32 numpy array.
                audio_data = indata[:, 0] if indata.ndim > 1 else indata
                pi_stt_service.add_audio_chunk(audio_data)
            
            # Start recording with sounddevice
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype='float32'
            ):
                print("✅ [AUDIO] Recording started")
                while self.is_recording:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ [AUDIO] Recording failed: {e}")
            # Fallback to ALSA if sounddevice fails
            self._alsa_fallback()
    
    def _alsa_fallback(self):
        """Fallback audio capture using ALSA"""
        print("🔄 [AUDIO] Falling back to ALSA capture...")
        # Implementation would go here if needed
        pass

def create_video_track():
    """Create video track with Picamera2 - EXACT COPY from working client.py"""
    try:
        from picamera2 import Picamera2
        import libcamera
        from aiortc import VideoStreamTrack
        import av

        class PiCamera2Track(VideoStreamTrack):
            """
            A video track that streams video from a Picamera2 camera.
            This is the modern, recommended approach for Raspberry Pi.
            """
            def __init__(self):
                super().__init__()
                print("📸 Initializing Picamera2...")
                self.picam2 = Picamera2()
                
                # Configure for Qwen 2.5-VL 7B - Maximum quality approach
                # Capture at maximum available resolution for full field of view
                sensor_modes = self.picam2.sensor_modes
                max_mode = max(sensor_modes, key=lambda x: x['size'][0] * x['size'][1])
                max_res = max_mode['size']
                print(f"🎯 Using native resolution {max_res} = {(max_res[0] * max_res[1] / 1_000_000):.1f} MP (optimal for Qwen 2.5-VL - no upscaling needed!)")
                
                config = self.picam2.create_video_configuration(
                    main={"size": max_res, "format": "RGB888"},  # Full sensor resolution for maximum FOV
                    controls={
                        "FrameRate": 5,  # Lower FPS for on-demand processing
                        "Brightness": 0.1,  # Slightly brighter for better AI analysis
                        "Contrast": 1.1,    # Enhanced contrast
                        "Saturation": 1.0,  # Natural colors
                        # "NoiseReductionMode": libcamera.controls.NoiseReductionModeEnum.Off, # Removed: Causes crash on newer libcamera
                    },
                    buffer_count=2  # Minimize buffer for low latency
                )
                self.picam2.configure(config)
                self.picam2.start()
                
                self.frame_count = 0
                self.start_time = time.time()
                print("✅ Picamera2 initialized successfully.")

            async def recv(self):
                """Receive video frames from the camera."""
                pts, time_base = await self.next_timestamp()
                
                # Get the frame from Picamera2
                try:
                    array = self.picam2.capture_array("main")
                except Exception as e:
                    print(f"💥 Failed to capture frame from Picamera2: {e}")
                    # As a fallback, create a black frame at native resolution
                    array = np.zeros((2464, 3280, 3), dtype=np.uint8)
                
                # Use native camera resolution - no resizing needed!
                # Native 3280x2464 = 8.1 MP is within Qwen 2.5-VL's 12.8 MP budget
                if self.frame_count == 1:
                    print(f"📐 Native Resolution: {array.shape[1]}x{array.shape[0]} = {(array.shape[0] * array.shape[1] / 1_000_000):.1f} MP (optimal for Qwen 2.5-VL)")
                
                # Convert to video frame for aiortc
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = pts
                frame.time_base = time_base

                self.frame_count += 1
                if self.frame_count == 1:
                    print(f"🚀 First frame sent! Size: {array.shape}")
                elif self.frame_count % 150 == 0: # Log every ~10 seconds
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        fps = self.frame_count / elapsed
                        print(f"📊 Sent {self.frame_count} frames. Average FPS: {fps:.2f}")

                return frame

            def stop(self):
                """Release camera resources cleanly and synchronously when the track is stopped."""
                super().stop()
                if hasattr(self, 'picam2') and self.picam2.is_open:
                    print("🎥 Releasing camera resources...")
                    self.picam2.close()
                    print("🛑 Picamera2 camera closed.")

        return PiCamera2Track()
        
    except ImportError:
        print("❌ Picamera2 library not found. Please run: pip install picamera2")
        return None
    except Exception as e:
        print(f"❌ Picamera2 video setup failed: {e}")
        print("   Ensure libcamera is working. You can test with 'libcamera-hello'.")
        import traceback
        traceback.print_exc()
        return None

def get_server_url(provided_url=None):
    """Get server URL from various sources"""
    # Check command line argument
    if provided_url:
        if not provided_url.startswith(('http://', 'https://')):
            provided_url = f"https://{provided_url}"
        return provided_url
    
    # Check tunnel_info.json
    try:
        with open('tunnel_info.json', 'r') as f:
            tunnel_data = json.load(f)
            server_url = tunnel_data.get('webrtc_server')
            if server_url:
                print(f"📁 Found server URL in tunnel_info.json: {server_url}")
                return server_url
    except FileNotFoundError:
        print("📁 No tunnel_info.json found")
    except Exception as e:
        print(f"❌ Error reading tunnel_info.json: {e}")
    
    # Default to localhost
    default_url = "http://localhost:8000"
    print(f"🏠 Using default URL: {default_url}")
    return default_url

# Global variables for managing connection state
data_channel = None
audio_processor = None

async def run_with_reconnection(server_url, max_retries=5):
    """Run the client with automatic reconnection"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            await run(server_url)
            # If we get here, connection was successful but ended
            print(f"🔄 Connection ended. Retrying... ({retry_count + 1}/{max_retries})")
            await asyncio.sleep(2)  # Wait before retrying
            retry_count += 1
        except KeyboardInterrupt:
            print("\n👋 Graceful shutdown requested")
            break
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"🔄 Retrying in 3 seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(3)
    
    if retry_count >= max_retries:
        print(f"💀 Maximum retries ({max_retries}) exceeded. Giving up.")

async def run(server_url):
    """Main client function with local STT"""
    global data_channel, audio_processor
    
    print(f"🚀 Connecting to F.R.E.D. server: {server_url}")
    
    # Create peer connection
    pc = RTCPeerConnection()
    
    # Add video track
    video_track = create_video_track()
    if video_track:
        pc.addTrack(video_track)
        print("📹 Video track added")
    
    # Create data channel for text communication
    data_channel = pc.createDataChannel('text', ordered=True)
    
    @data_channel.on('open')
    def on_data_channel_open():
        print("📡 [DATA CHANNEL] Connected to F.R.E.D. mainframe")
        
        # Start local audio processing
        global audio_processor
        audio_processor = LocalAudioProcessor()
        
        # Get the running asyncio event loop to safely call async functions from the STT thread
        loop = asyncio.get_running_loop()

        def on_transcription(text):
            """Handle transcribed text by safely sending it from the main event loop."""
            print(f"🗣️ [TRANSCRIBED] '{text}'")
            
            async def send_to_server():
                """Coroutine to send text over the data channel."""
                if data_channel and data_channel.readyState == 'open':
                    # Send transcribed text to server
                    message = {
                        'type': 'transcription',
                        'text': text,
                        'timestamp': time.time()
                    }
                    data_channel.send(json.dumps(message))
                    print(f"📤 [SENT] Text to F.R.E.D.: '{text}'")

            # Schedule the coroutine to be executed on the main event loop
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(send_to_server(), loop)
        
        # Start audio processing with callback
        if audio_processor.start_recording(on_transcription):
            print("🎤 [SUCCESS] Local voice recognition active")
        else:
            print("❌ [CRITICAL] Failed to start voice recognition")
    
    @data_channel.on('message')
    def on_data_channel_message(message):
        """Handle incoming messages from server"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'audio':
                # Server is sending audio response
                audio_b64 = data.get('audio')
                if audio_b64:
                    print("🔊 [FRED RESPONSE] Processing audio...")
                    play_audio_from_base64(audio_b64)
            
            elif message_type == 'text':
                # Server is sending text response  
                text = data.get('text', '')
                print(f"💬 [FRED SAYS] {text}")
                
            elif message_type == 'status':
                # Server status update
                status = data.get('status', '')
                print(f"📊 [SERVER STATUS] {status}")
                
        except json.JSONDecodeError:
            print(f"📝 [RAW MESSAGE] {message}")
        except Exception as e:
            print(f"❌ [ERROR] Processing message: {e}")
    
    @data_channel.on('close')
    def on_data_channel_close():
        print("📡 [DATA CHANNEL] Disconnected from F.R.E.D.")
        if audio_processor:
            audio_processor.stop_recording()
    
    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        print(f"🔗 [CONNECTION] State: {pc.connectionState}")
        if pc.connectionState == 'failed':
            await pc.close()
            if audio_processor:
                audio_processor.stop_recording()
    
    # Create offer and get answer from server
    try:
        # Create offer
        await pc.setLocalDescription(await pc.createOffer())
        
        # Send offer to server with authentication
        headers = {
            'Authorization': 'Bearer fred_pi_glasses_2024',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{server_url}/offer",
            json={
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type,
                'client_type': 'pi_glasses_with_local_stt'
            },
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"Server returned {response.status_code}: {response.text}")
        
        answer = response.json()
        
        # Set remote description
        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=answer['sdp'],
            type=answer['type']
        ))
        
        print("✅ [SUCCESS] WebRTC connection established with local STT")
        
        # Keep connection alive
        try:
            while pc.connectionState != 'closed':
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested")
        
    except Exception as e:
        print(f"❌ [CONNECTION ERROR] {e}")
        raise
    
    finally:
        print("🧹 [CLEANUP] Closing connections...")
        if audio_processor:
            audio_processor.stop_recording()
        await pc.close()

def main():
    """Main entry point"""
    print(banner("PIP-BOY LOCAL STT"))
    
    parser = argparse.ArgumentParser(description='F.R.E.D. Pi Client with Local STT')
    parser.add_argument('--server', type=str, help='Server URL (e.g., https://example.ngrok.io)')
    args = parser.parse_args()
    
    server_url = get_server_url(args.server)
    
    print("🤖 [F.R.E.D. GLASSES] Initializing Pip-Boy interface...")
    print("🧠 [LOCAL STT] Using tiny.en model with int8 quantization")
    print("📡 [NETWORK] Sending transcribed text instead of raw audio")
    
    try:
        asyncio.run(run_with_reconnection(server_url))
    except KeyboardInterrupt:
        print("\n👋 [SHUTDOWN] F.R.E.D. Pip-Boy interface offline")
    except Exception as e:
        print(f"❌ [CRITICAL ERROR] {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 