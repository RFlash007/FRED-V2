#!/usr/bin/env python3
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
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

import builtins as _b
# === Vault-Tec themed print wrapper ===
_RESET = '\033[0m'
_COLOR_MAP = {
    'CRITICAL': '\033[91m',
    'ERROR': '\033[91m',
    'WARNING': '\033[93m',
    'SUCCESS': '\033[92m',
    'AUDIO': '\033[95m',
    'NETWORK': '\033[96m',
    'VAULT': '\033[96m',
    'PIP-BOY': '\033[96m',
    'MAINFRAME': '\033[95m',
}
_orig_print = _b.print

def _vault_print(*args, **kwargs):
    colored_args = []
    for arg in args:
        text = str(arg)
        color = ''
        for token, col in _COLOR_MAP.items():
            if token in text:
                color = col
                break
        if color:
            text = f"{color}{text}{_RESET}"
        colored_args.append(text)
    _orig_print(*colored_args, **kwargs)

_b.print = _vault_print

def play_audio_from_base64(audio_b64, format_type='wav'):
    """Decode base64 audio and play it on the Pi."""
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(audio_b64)
        _vault_print(f"[AUDIO] F.R.E.D. transmission received ({len(audio_data)} bytes)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Play using aplay (ALSA) - most reliable on Pi
        try:
            subprocess.run(['aplay', temp_file_path], check=True, capture_output=True)
            _vault_print("[SUCCESS] F.R.E.D. voice transmission complete")
        except subprocess.CalledProcessError as e:
            # Fallback to paplay (PulseAudio)
            try:
                subprocess.run(['paplay', temp_file_path], check=True, capture_output=True)
                _vault_print("[SUCCESS] F.R.E.D. voice transmission complete (PulseAudio)")
            except subprocess.CalledProcessError as e2:
                # Last resort: mpv
                try:
                    subprocess.run(['mpv', '--no-video', temp_file_path], check=True, capture_output=True)
                    _vault_print("[SUCCESS] F.R.E.D. voice transmission complete (mpv)")
                except subprocess.CalledProcessError as e3:
                    _vault_print(f"[CRITICAL] All audio protocols failed - check Pip-Boy speakers")
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as cleanup_err:
            _vault_print(f"[WARNING] Failed to purge audio cache: {cleanup_err}")
            
    except Exception as e:
        _vault_print(f"[CRITICAL] Audio playback system failure: {e}")

import numpy as np


def create_local_tracks(video=True, audio=True):
    tracks = []
    
    if video:
        _vault_print("🎥 Setting up video with Picamera2...")
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
                    _vault_print("📸 Initializing Picamera2...")
                    self.picam2 = Picamera2()
                    
                    # Configure for Qwen 2.5-VL 7B - Maximum quality approach
                    # Capture at maximum available resolution for full field of view
                    sensor_modes = self.picam2.sensor_modes
                    max_mode = max(sensor_modes, key=lambda x: x['size'][0] * x['size'][1])
                    max_res = max_mode['size']
                    _vault_print(f"🎯 Using native resolution {max_res} = {(max_res[0] * max_res[1] / 1_000_000):.1f} MP (optimal for Qwen 2.5-VL - no upscaling needed!)")
                    
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
                    _vault_print("✅ Picamera2 initialized successfully.")

                async def recv(self):
                    """Receive video frames from the camera."""
                    pts, time_base = await self.next_timestamp()
                    
                    # Get the frame from Picamera2
                    try:
                        array = self.picam2.capture_array("main")
                    except Exception as e:
                        _vault_print(f"💥 Failed to capture frame from Picamera2: {e}")
                        # As a fallback, create a black frame at native resolution
                        array = np.zeros((2464, 3280, 3), dtype=np.uint8)
                    
                    # Use native camera resolution - no resizing needed!
                    # Native 3280x2464 = 8.1 MP is within Qwen 2.5-VL's 12.8 MP budget
                    if self.frame_count == 1:
                        _vault_print(f"📐 Native Resolution: {array.shape[1]}x{array.shape[0]} = {(array.shape[0] * array.shape[1] / 1_000_000):.1f} MP (optimal for Qwen 2.5-VL)")
                    
                    # Convert to video frame for aiortc
                    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                    frame.pts = pts
                    frame.time_base = time_base

                    self.frame_count += 1
                    if self.frame_count == 1:
                        _vault_print(f"🚀 First frame sent! Size: {array.shape}")
                    elif self.frame_count % 150 == 0: # Log every ~10 seconds
                        elapsed = time.time() - self.start_time
                        if elapsed > 0:
                            fps = self.frame_count / elapsed
                            _vault_print(f"📊 Sent {self.frame_count} frames. Average FPS: {fps:.2f}")

                    return frame

                def __del__(self):
                    """Cleanup camera resources."""
                    try:
                        if hasattr(self, 'picam2') and getattr(self, 'picam2', None):
                            if self.picam2.is_open:
                                self.picam2.stop()
                                _vault_print('🛑 Picamera2 stopped (cleanup).')
                    except Exception as e:
                        # Silently ignore cleanup exceptions to avoid noisy traces
                        pass

            tracks.append(PiCamera2Track())
            _vault_print("✅ Picamera2 video track created successfully!")
            
        except ImportError:
            _vault_print("❌ Picamera2 library not found. Please run: pip install picamera2")
        except Exception as e:
            _vault_print(f"❌ Picamera2 video setup failed: {e}")
            _vault_print("   Ensure libcamera is working. You can test with 'libcamera-hello'.")
            import traceback
            traceback.print_exc()
    
    if audio:
        _vault_print("🎤 Setting up audio...")
        
        # Try sounddevice first if available
        try:
            from audio_capture_alternative import create_sounddevice_audio_track
            audio_track = create_sounddevice_audio_track()
            if audio_track:
                tracks.append(audio_track)
                _vault_print("✅ Audio working with sounddevice")
                # Don't return early - continue to add video track if requested
        except ImportError:
            _vault_print("  sounddevice not available, trying ALSA methods")
        except Exception as e:
            _vault_print(f"  sounddevice failed: {e}")
        
        # Basic ALSA approaches with correct format syntax (only if sounddevice didn't work)
        if not any(hasattr(t, 'kind') and getattr(t, 'kind', None) == 'audio' for t in tracks):
            audio_configs = [
                ('hw:3,0', None, {'sample_rate': '16000', 'channels': '1'}),  # Your CORSAIR card
                ('hw:1,0', None, {'sample_rate': '16000', 'channels': '1'}),
                ('hw:0,0', None, {'sample_rate': '16000', 'channels': '1'}),
                ('default', None, {'sample_rate': '16000', 'channels': '1'}),
            ]
            
            for device, fmt, options in audio_configs:
                try:
                    _vault_print(f"  Trying ALSA device: {device}")
                    player = MediaPlayer(device, format='alsa', options=options)
                    if player.audio:
                        tracks.append(player.audio)
                        _vault_print(f"✅ Audio working: {device}")
                        break
                except Exception as e:
                    _vault_print(f"  Failed {device}: {e}")
                    continue
            
            if not any(hasattr(t, 'kind') and getattr(t, 'kind', None) == 'audio' for t in tracks):
                _vault_print("⚠️  No audio capture working - continuing without audio")
    
    _vault_print(f"📊 Total tracks created: {len(tracks)}")
    for i, track in enumerate(tracks):
        if hasattr(track, 'kind'):
            _vault_print(f"  Track {i+1}: {track.kind}")
        else:
            _vault_print(f"  Track {i+1}: {type(track).__name__}")
    
    return tracks


def get_server_url(provided_url=None):
    """Auto-discover server URL with multiple fallback methods"""
    
    if provided_url:
        _vault_print(f"🎯 Using provided server: {provided_url}")
        return provided_url
    
    _vault_print("🔍 Auto-discovering F.R.E.D. server...")
    
    # Method 1: Check for tunnel info file (for remote access)
    tunnel_files = [
        os.path.expanduser("~/tunnel_info.json"),      # Home directory
        "tunnel_info.json",                            # Current directory
        "../tunnel_info.json"                         # Parent directory
    ]
    
    for tunnel_file in tunnel_files:
        if os.path.exists(tunnel_file):
            _vault_print(f"📡 Found tunnel file: {tunnel_file}")
            break
    else:
        tunnel_file = None
    
    if tunnel_file:
        try:
            with open(tunnel_file, 'r') as f:
                tunnel_data = json.load(f)
                if tunnel_data.get('webrtc_server'):
                    url = tunnel_data['webrtc_server']
                    _vault_print(f"📡 Found remote tunnel: {url}")
                    return url
        except Exception as e:
            _vault_print(f"⚠️  Could not read tunnel file: {e}")
    
    # Method 2: Try common local network addresses
    local_candidates = [
        "http://192.168.50.65:8080",  # Your current setup
        "http://192.168.1.100:8080",  # Common router range  
        "http://192.168.0.100:8080",  # Alternative range
        "http://192.168.1.65:8080",   # Alternative for your setup
        "http://localhost:8080",       # Local development
    ]
    
    _vault_print("🔍 Trying local network candidates...")
    for candidate in local_candidates:
        try:
            _vault_print(f"  Testing: {candidate}")
            response = requests.get(candidate, timeout=5)
            if response.status_code == 200 and "FRED" in response.text:
                _vault_print(f"✅ Found local server: {candidate}")
                return candidate
            else:
                _vault_print(f"  Response: {response.status_code} - Not F.R.E.D. server")
        except requests.exceptions.ConnectionError:
            _vault_print(f"  Connection refused")
        except requests.exceptions.Timeout:
            _vault_print(f"  Timeout")
        except Exception as e:
            _vault_print(f"  Error: {e}")
            continue
    
    # Method 3: Basic network scan for F.R.E.D. servers
    _vault_print("🌐 Scanning local network for F.R.E.D. servers...")
    try:
        # Get local network range
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        _vault_print(f"  Pi IP: {local_ip}")
        
        # Extract network base (assumes /24 subnet)
        network_base = '.'.join(local_ip.split('.')[:-1])
        _vault_print(f"  Scanning network: {network_base}.x")
        
        # Common server IPs to try in your network range
        for host_ip in [65, 100, 1, 10, 50]:
            candidate = f"http://{network_base}.{host_ip}:8080"
            try:
                _vault_print(f"  Testing: {candidate}")
                response = requests.get(candidate, timeout=2)
                if response.status_code == 200 and "FRED" in response.text:
                    _vault_print(f"✅ Found F.R.E.D. server: {candidate}")
                    return candidate
            except:
                continue
                
    except Exception as e:
        _vault_print(f"  Network scan failed: {e}")
    
    _vault_print("\n❌ Auto-discovery failed. Please:")
    _vault_print("1. Make sure F.R.E.D. server is running on your home computer")
    _vault_print("2. Check that both devices are on the same network")
    _vault_print("3. Try: python client.py --server http://YOUR_COMPUTER_IP:8080")
    _vault_print("4. Or use ngrok URL if connecting remotely")
    
    raise Exception("❌ No F.R.E.D. server found. Please specify --server URL")


async def run_with_reconnection(server_url, max_retries=5):
    """Run client with automatic reconnection logic"""
    
    for attempt in range(max_retries):
        try:
            _vault_print(f"🔄 Connection attempt {attempt + 1}/{max_retries}")
            await run(server_url)
            
        except Exception as e:
            _vault_print(f"❌ Connection failed: {e}")
            
            # Clean up any lingering resources
            _vault_print("🧹 Cleaning up resources...")
            try:
                # Give time for resources to be released.
                # The camera track's __del__ method will handle its own cleanup.
                await asyncio.sleep(1)
                    
            except Exception as cleanup_error:
                _vault_print(f"Warning: Cleanup error: {cleanup_error}")
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                _vault_print(f"⏳ Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                _vault_print("💀 Max retries exceeded. Please check your connection.")
                raise


async def run(server_url):
    # Configure ICE servers for better NAT traversal
    from aiortc import RTCConfiguration, RTCIceServer
    
    try:
        # Try newer aiortc API
        ice_servers = [
            RTCIceServer("stun:stun.l.google.com:19302"),
            RTCIceServer("stun:stun1.l.google.com:19302")
        ]
        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(configuration=config)
    except Exception:
        # Fallback for older aiortc versions
        _vault_print("🔧 Using legacy aiortc configuration")
        pc = RTCPeerConnection()
    
    # Set up data channel for communication
    channel = pc.createDataChannel('chat')
    
    @channel.on('open')
    def on_open():
        _vault_print('[VAULT-NET] Secure connection established with F.R.E.D. mainframe!')
        _vault_print('[PIP-BOY] Audio/visual sensors ONLINE - ready for wasteland operations...')
    
    @channel.on('message')
    def on_message(message):
        if message.startswith('[HEARTBEAT_ACK]'):
            # Silent acknowledgment - no logging needed for routine heartbeats
            pass
        elif message.startswith('[ACK]'):
            ack_text = message.replace('[ACK] ', '')
            _vault_print(f"[F.R.E.D.] {ack_text}")
        elif message.startswith('[AUDIO_BASE64:'):
            # Handle incoming audio from F.R.E.D.
            try:
                # Extract format and audio data
                header_end = message.find(']')
                format_info = message[14:header_end]  # Skip '[AUDIO_BASE64:'
                audio_b64 = message[header_end + 1:]
                
                _vault_print(f"[TRANSMISSION] Incoming voice data from F.R.E.D. ({format_info})")
                
                # Decode and play audio
                play_audio_from_base64(audio_b64, format_info)
                
            except Exception as e:
                _vault_print(f"[ERROR] Audio processing failure: {e}")
        else:
            _vault_print(f'\n[F.R.E.D.] {message}')
        
        # Only show listening status occasionally to reduce clutter
        if not message.startswith('[HEARTBEAT_ACK]'):
            _vault_print('[PIP-BOY] Standing by for commands...')
    
    @channel.on('close')
    def on_close():
        _vault_print('[CRITICAL] Connection to F.R.E.D. mainframe terminated')
        # This will cause the connection to fail and trigger reconnection
        raise Exception("Data channel closed")
    
    # Add tracks (video and audio)
    tracks = create_local_tracks(video=True, audio=True)
    
    if not tracks:
        _vault_print("⚠️  No media tracks available - connecting with data channel only")
    
    for track in tracks:
        pc.addTrack(track)
        track_kind = getattr(track, 'kind', 'unknown')
        track_type = type(track).__name__
        _vault_print(f"📡 Added {track_kind} track ({track_type})")
        
        # Add extra logging for video tracks
        if hasattr(track, 'kind') and track.kind == 'video':
            _vault_print(f"  📹 Video track details: {track_type}")
            if hasattr(track, 'picam2'):
                _vault_print(f"     Picamera2 initialized: {track.picam2 is not None}")
    
    # Create offer and connect
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    try:
        _vault_print(f"🔗 Connecting to {server_url}/offer...")
        
        # Add authentication header
        headers = {
            'Authorization': 'Bearer fred_pi_glasses_2024',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(f'{server_url}/offer', json={
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }, headers=headers, timeout=15)  # Increased timeout for remote connections
        
        if response.status_code == 200:
            answer = RTCSessionDescription(**response.json())
            await pc.setRemoteDescription(answer)
            _vault_print("🚀 F.R.E.D. Pi Glasses connected and ready!")
        else:
            _vault_print(f"❌ Server error: {response.status_code}")
            raise Exception(f"Server returned {response.status_code}")
            
    except requests.exceptions.Timeout:
        raise Exception("Connection timeout - server may be unreachable")
    except requests.exceptions.ConnectionError:
        raise Exception("Connection refused - server may be down")
    except Exception as e:
        raise Exception(f"Connection failed: {e}")
    
    # Keep connection alive with heartbeat
    start_time = time.time()
    heartbeat_interval = 30  # seconds
    last_heartbeat = start_time
    
    while True:
        try:
            await asyncio.sleep(1)
            
            # Send heartbeat periodically (silent unless there's an issue)
            current_time = time.time()
            if current_time - last_heartbeat > heartbeat_interval:
                if channel.readyState == 'open':
                    channel.send('[HEARTBEAT]')
                    last_heartbeat = current_time
                    # Only log heartbeat occasionally for conciseness
                    if int(current_time) % 120 == 0:  # Every 2 minutes
                        _vault_print("[VITAL-MONITOR] Pip-Boy status confirmed")
                else:
                    raise Exception("Data channel not open")
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            _vault_print(f"[CRITICAL] Connection to mainframe lost: {e}")
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', help='F.R.E.D. server URL (auto-discovery if not provided)')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum connection retry attempts')
    args = parser.parse_args()
    
    banner = (\
        '\033[92m' + '═'*60 + '\033[0m\n' +\
        '\033[96m  🤖  OLLIE-TEC™ Pip-Boy Interface v2.0  🤖\033[0m\n' +\
        '\033[96m  Field Operations Communication Subsystem   \033[0m\n' +\
        f'\033[93m  Boot: {time.strftime("%Y-%m-%d %H:%M:%S")}\033[0m\n' +\
        '\033[92m' + '═'*60 + '\033[0m'\
    )
    _vault_print(banner)
    
    try:
        # Auto-discover or use provided server URL
        server_url = get_server_url(args.server)
        
        # Run with automatic reconnection
        asyncio.run(run_with_reconnection(server_url, args.max_retries))
        
    except KeyboardInterrupt:
        _vault_print("\n[SHUTDOWN] Field operative terminating connection")
    except Exception as e:
        _vault_print(f"\n[CRITICAL] System failure: {e}")
        _vault_print("\n[VAULT-TEC] Troubleshooting protocols:")
        _vault_print("1. Verify F.R.E.D. mainframe is operational")
        _vault_print("2. Check wasteland communication network")
        _vault_print("3. Try manual server specification with --server")
        sys.exit(1)
