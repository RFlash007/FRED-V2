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


class SoundDeviceAudioTrack(MediaStreamTrack):
    """
    Custom audio track using sounddevice for reliable Pi audio capture
    """
    kind = "audio"
    
    def __init__(self, device=None, sample_rate=16000, channels=1, blocksize=1024):
        super().__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.audio_queue = queue.Queue(maxsize=10)
        self.stream = None
        self._running = False
        
        # Start audio capture immediately when track is created
        print("🎤 Auto-starting audio capture for WebRTC...")
        self.start()
        
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
                if status:
                    print(f"Audio status: {status}")
                # Convert to the format expected by aiortc
                audio_data = indata.copy().astype(np.float32)
                if not self.audio_queue.full():
                    self.audio_queue.put(audio_data)
            
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
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self._running = False
            print("🛑 Audio capture stopped")
    
    async def recv(self):
        """Receive audio frames for WebRTC"""
        if not self._running:  
            self.start()
        
        # We need to store partial blocks between calls
        if not hasattr(self, '_current_block'):
            self._current_block = None
            self._block_index = 0
        
        try:
            # If we don't have a current block, get a new one
            if self._current_block is None:
                audio_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.audio_queue.get, True, 0.1
                )
                self._current_block = audio_data
                self._block_index = 0
            
            # Send one sample from the current block
            if self._block_index < len(self._current_block):
                # Get single sample and reshape to (1, channels)
                sample = self._current_block[self._block_index:self._block_index+1]
                
                frame = AudioFrame.from_ndarray(
                    sample,
                    format='flt', 
                    layout='mono' if self.channels == 1 else 'stereo'
                )
                frame.sample_rate = self.sample_rate
                frame.pts = None
                
                self._block_index += 1
                
                # If we've sent all samples from this block, get ready for next block
                if self._block_index >= len(self._current_block):
                    self._current_block = None
                    self._block_index = 0
                
                return frame
            else:
                # This shouldn't happen, but just in case
                self._current_block = None
                self._block_index = 0
                raise Exception("Block index out of range")
            
        except queue.Empty:
            # Return silence if no audio available
            silence = np.zeros((1, self.channels), dtype=np.float32)
            frame = AudioFrame.from_ndarray(
                silence,
                format='flt',
                layout='mono' if self.channels == 1 else 'stereo'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = None
            return frame
        except Exception as e:
            print(f"Error receiving audio frame: {e}")
            # Return silence on error to keep the stream going
            silence = np.zeros((1, self.channels), dtype=np.float32)
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
            blocksize=1024
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