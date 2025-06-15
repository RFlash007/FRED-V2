#!/usr/bin/env python3
"""
Alternative Audio Capture for F.R.E.D. Pi Glasses using sounddevice
This provides a backup method if aiortc MediaPlayer audio fails
"""

import asyncio
import sounddevice as sd
import numpy as np
from aiortc import RTCPeerConnection, MediaStreamTrack
from av import AudioFrame
import threading
import queue
import collections
from collections import deque
from fractions import Fraction


class SoundDeviceAudioTrack(MediaStreamTrack):
    """
    Custom audio track using sounddevice for reliable Pi audio capture
    """
    kind = "audio"
    
    def __init__(self, device=None, sample_rate=16000, channels=1, blocksize=320):
        super().__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        
        # Use a circular buffer for incoming audio samples. We store raw float32 mono
        # samples so 1 item = 1 sample. At 16 kHz, 1 second ≈ 16_000 samples. Keep
        # a few seconds of history to absorb scheduling jitter but not too much
        # to consume memory unnecessarily.
        self.buffer_size = self.sample_rate * 2  # 2-second rolling buffer is sufficient
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Keep track of how many samples we have already sent so PTS values are
        # monotonic and WebRTC can sync correctly.
        self.samples_sent = 0
        self.time_base = Fraction(1, self.sample_rate)
        
        self.stream = None
        self._running = False
        
        # We will start capturing on-demand the moment the first WebRTC
        # `recv()` call arrives.  This avoids filling the ring-buffer before
        # the peer connection is ready and prevents the initial overflow we
        # have observed.
        print("🎤 Audio capture will start on first recv() call…")
        
    def start(self):
        """Start audio capture"""
        if self._running:
            return
            
        try:
            print(f"🎤 Starting sounddevice audio capture...")
            print(f"   Device: {self.device}")
            print(f"   Sample rate: {self.sample_rate}")
            print(f"   Channels: {self.channels}")
            
            def audio_callback(indata, frames, time, status):
                try:
                    if status and ('overflow' in str(status).lower()):
                        # Throttle spam – print every 20th overflow warning
                        if (self.samples_sent // (self.sample_rate * 0.02)) % 20 == 0:
                            print(f"🎧 PortAudio status: {status}")

                    # Flatten to mono float32 regardless of channel count
                    mono = indata.mean(axis=1).astype(np.float32)

                    with self.buffer_lock:
                        self.audio_buffer.extend(mono)
                except Exception as cb_err:
                    # Catch ANY error so the stream doesn't silently abort
                    print(f"💥 Audio callback exception: {cb_err}")
            
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                callback=audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            self._running = True
            print("✅ Audio capture started successfully!")
            
        except Exception as e:
            print(f"❌ Failed to start audio capture: {e}")
            raise
    
    def stop(self):
        """Stop audio capture"""
        print("⚠️ SoundDeviceAudioTrack.stop() invoked – cleaning up stream")
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"⚠️ Error closing stream: {e}")
        self._running = False
        print("🛑 Audio capture stopped")
    
    async def recv(self):
        """Receive audio frames for WebRTC"""
        if not self._running:  
            self.start()
        
        try:
            # We will produce 20 ms of audio per frame which is commonly used by
            # WebRTC. At 16 kHz that is 0.02 * 16000 = 320 samples.
            frame_samples = int(self.sample_rate * 0.02)

            # Wait until we have enough samples. This yields control so other
            # tasks can run without blocking the event loop for long periods.
            while True:
                with self.buffer_lock:
                    if len(self.audio_buffer) >= frame_samples:
                        break
                await asyncio.sleep(0.001)  # 1 ms back-off

            with self.buffer_lock:
                # Collect exactly `frame_samples` to maintain constant frame
                # size. Convert to numpy array for AudioFrame.
                samples = [self.audio_buffer.popleft() for _ in range(frame_samples)]

            # Convert to 16-bit signed PCM which is the most widely supported
            # format for WebRTC / Opus encoding in aiortc.
            pcm_f32 = np.array(samples, dtype=np.float32)
            pcm_i16 = (np.clip(pcm_f32, -1.0, 1.0) * 32767).astype(np.int16)

            # For packed (interleaved) formats, PyAV expects shape (channels, samples)
            # so we provide (1, samples) for mono.
            pcm_i16 = pcm_i16.reshape(1, -1)  # (channels, samples)

            frame = AudioFrame.from_ndarray(pcm_i16, format='s16', layout='mono')
            frame.sample_rate = self.sample_rate
            frame.pts = self.samples_sent
            frame.time_base = self.time_base

            self.samples_sent += frame_samples

            if self.samples_sent == frame_samples:
                print("🚀 FIRST AUDIO FRAME SENT!")
            elif self.samples_sent % (frame_samples * 50) == 0:  # every second
                print(f"📢 Sent {self.samples_sent // frame_samples} audio frames")

            return frame
            
        except Exception as e:
            print(f"Error receiving audio frame: {e}")
            # Return silence on error to keep the stream going
            silence = np.zeros((1, 1), dtype=np.float32)
            frame = AudioFrame.from_ndarray(
                silence,
                format='flt',
                layout='mono' if self.channels == 1 else 'stereo'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = None
            return frame


def list_audio_devices():
    """List available audio input devices"""
    print("🔍 Available audio input devices:")
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']}, rate: {device['default_samplerate']})")
            
            # Highlight CORSAIR device
            if 'CORSAIR' in device['name'] or 'HS80' in device['name']:
                print(f"       👉 CORSAIR HS80 FOUND!")
    
    return input_devices


def find_corsair_device():
    """Find CORSAIR HS80 device automatically"""
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            if 'CORSAIR' in device['name'] or 'HS80' in device['name']:
                print(f"🎯 Found CORSAIR device: {device['name']} (device {i})")
                return i
    
    print("⚠️  CORSAIR HS80 not found, using default device")
    return None


def create_sounddevice_audio_track():
    """Create audio track using sounddevice as fallback"""
    try:
        # List available devices
        input_devices = list_audio_devices()
        
        if not input_devices:
            print("❌ No audio input devices found!")
            return None
        
        # Try to find CORSAIR device first
        corsair_device = find_corsair_device()
        device_id = corsair_device if corsair_device is not None else input_devices[0][0]
        
        # Create and test the audio track
        audio_track = SoundDeviceAudioTrack(
            device=device_id,
            sample_rate=16000,
            channels=1,
            blocksize=0  # Let PortAudio choose optimal blocksize; we still frame at 20 ms
        )
        
        return audio_track
        
    except Exception as e:
        print(f"❌ Failed to create sounddevice audio track: {e}")
        return None


def test_audio_capture():
    """Test audio capture with sounddevice"""
    print("🧪 Testing sounddevice audio capture...")
    
    try:
        device_id = find_corsair_device()
        
        def test_callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            
            # Check audio levels
            volume = np.sqrt(np.mean(indata**2))
            if volume > 0.01:  # Threshold for detecting audio
                print(f"🔊 Audio detected! Volume: {volume:.4f}")
        
        # Record for 5 seconds
        print("🎤 Recording for 5 seconds... (speak into microphone)")
        with sd.InputStream(device=device_id, channels=1, samplerate=16000, callback=test_callback):
            sd.sleep(5000)  # 5 seconds
        
        print("✅ Audio test completed!")
        
    except Exception as e:
        print(f"❌ Audio test failed: {e}")


if __name__ == "__main__":
    print("🍇 F.R.E.D. Alternative Audio Capture Test")
    print("=" * 50)
    
    # Test sounddevice audio capture
    test_audio_capture()
    
    # Test creating audio track
    audio_track = create_sounddevice_audio_track()
    if audio_track:
        print("✅ Audio track created successfully!")
    else:
        print("❌ Failed to create audio track!") 