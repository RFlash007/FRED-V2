#!/usr/bin/env python3
"""
F.R.E.D. Pi Client with Local, High-Accuracy Speech-to-Text

This single-file client handles audio capture, local transcription, and communication
with the F.R.E.D. server. It incorporates advanced audio processing techniques
from previous versions for improved accuracy, including ambient noise calibration
and complete utterance detection before transcription.
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
import queue
import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaPlayer
from faster_whisper import WhisperModel
import torch

from ollietec_theme import apply_theme, banner

# Apply theming to all prints
apply_theme()


class VoiceProcessor:
    """
    Handles local voice processing: audio capture, VAD, wake word, and transcription.
    Merges logic from pi_stt_service.py and OLD FRED TRANSCRIBE.py for best results.
    """

    def __init__(self, transcription_callback):
        self.transcription_callback = transcription_callback
        self.model = None
        self.is_initialized = False

        # Audio capture settings
        self.samplerate = 16000
        self.channels = 1

        # Voice Activity Detection (VAD) & Buffering
        self.silence_threshold = 0.0015  # Initial threshold, will be calibrated
        self.calibration_duration = 2  # seconds
        self.silence_duration = 0.7    # Seconds of silence to mark end of speech
        self.last_speech_time = time.time()
        self.speech_buffer = []

        # Processing control
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.terminate_event = threading.Event()
        self.is_listening = False  # Is F.R.E.D. actively listening for a command?

        # Wake/Stop words
        self.wake_words = ["fred", "hey fred", "okay fred", "hi fred", "excuse me fred"]
        self.stop_words = ["goodbye", "bye fred", "stop listening", "that's all", "thank you fred"]

    def initialize(self):
        """Initialize the Whisper model with optimal settings."""
        if self.is_initialized:
            return True
        try:
            print("[PIP-BOY STT] Initializing voice recognition systems...")
            model_size = "medium.en"  # Upgraded for better accuracy
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "int8" if device == "cpu" else "float16" # int8 for Pi CPU

            print(f"🔧 Loading {model_size} model ({compute_type} on {device})...")

            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=4, # Use all available cores
                num_workers=1
            )

            print("✅ [PIP-BOY STT] Voice recognition ONLINE")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"❌ [CRITICAL] STT initialization failed: {e}")
            return False

    def _audio_capture_callback(self, indata, frames, time_info, status):
        """Callback for the audio stream, puts data in a queue."""
        if status:
            print(f"[WARNING] Audio status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def calibrate_silence_threshold(self):
        """Calibrate the silence threshold based on ambient noise."""
        print("🎤 Calibrating microphone... Please remain quiet for a few seconds.")
        
        # Drain queue to ensure we're using current audio
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        calibration_samples = []
        start_time = time.time()
        while time.time() - start_time < self.calibration_duration:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                audio_level = np.abs(audio_data).mean()
                calibration_samples.append(audio_level)
            except queue.Empty:
                continue

        if calibration_samples:
            avg_noise = np.mean(calibration_samples)
            self.silence_threshold = avg_noise * 1.5 # Set threshold 50% above ambient noise
            print(f"✅ Calibration complete. Silence threshold set to: {self.silence_threshold:.6f}")
        else:
            print(f"⚠️ Calibration failed. Using default threshold: {self.silence_threshold:.6f}")

    def _process_audio_loop(self):
        """
        Processes audio from the queue, detects speech, and handles transcription.
        This is the core logic adapted from 'OLD FRED TRANSCRIBE.py'.
        """
        while not self.terminate_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_data_flat = audio_data.flatten().astype(np.float32)
                audio_level = np.abs(audio_data_flat).mean()

                is_speech = audio_level > self.silence_threshold

                if is_speech:
                    # print(f"\rAudio level: {audio_level:.6f} (Speaking)", end="")
                    self.last_speech_time = time.time()
                    self.speech_buffer.append(audio_data_flat)
                else:
                    # print(f"\rAudio level: {audio_level:.6f} (Silent)  ", end="")
                    # If silence follows speech, process the buffer
                    if self.speech_buffer and time.time() - self.last_speech_time > self.silence_duration:
                        complete_audio = np.concatenate(self.speech_buffer)
                        self.speech_buffer = []
                        self._transcribe_and_handle(complete_audio)

            except queue.Empty:
                # If queue is empty and there's buffered speech after a timeout, process it
                if self.speech_buffer and time.time() - self.last_speech_time > self.silence_duration:
                    complete_audio = np.concatenate(self.speech_buffer)
                    self.speech_buffer = []
                    self._transcribe_and_handle(complete_audio)
                continue
            except Exception as e:
                print(f"\n[ERROR] Error in audio processing loop: {e}")
                time.sleep(0.1)

    def _transcribe_and_handle(self, audio_data: np.ndarray):
        """Transcribes a complete audio segment and handles the resulting text."""
        try:
            print(f"\n🎤 Transcribing {len(audio_data)/self.samplerate:.2f}s of audio...")
            segments, _ = self.model.transcribe(
                audio_data,
                language="en",
                beam_size=5,  # Increased for better accuracy
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}
            )

            transcribed_text = "".join(seg.text for seg in segments).strip().lower()

            if not transcribed_text or transcribed_text == "thanks for watching!":
                return

            print(f"🎙️ [DETECTED] '{transcribed_text}'")

            # --- Wake/Stop Word Logic ---
            if not self.is_listening:
                if any(wake_word in transcribed_text for wake_word in self.wake_words):
                    print("👋 [WAKE] Wake word detected! Listening...")
                    self.is_listening = True
                    # Optional: play an acknowledgment sound
            else: # We are actively listening
                if any(stop_word in transcribed_text for stop_word in self.stop_words):
                    print("💤 [SLEEP] Stop word detected.")
                    self.is_listening = False
                    self.transcription_callback("goodbye") # Notify server
                else:
                    # This is a command
                    print(f"🗣️ [COMMAND] Sending to F.R.E.D.: '{transcribed_text}'")
                    self.transcription_callback(transcribed_text)
                    # Stay listening for the next command until a stop word is heard
                    # or a timeout (which is handled by the server logic).

        except Exception as e:
            print(f"\n[ERROR] Error during transcription: {e}")

    def start(self):
        """Starts the voice processing system."""
        if self.processing_thread is not None:
            return False # Already running

        if not self.is_initialized:
            if not self.initialize():
                return False

        self.terminate_event.clear()

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._audio_capture_callback,
            blocksize=int(0.2 * self.samplerate) # Smaller blocks for responsiveness
        )
        self.stream.start()
        print("✅ [AUDIO] Audio stream started.")

        # Give the stream a moment to buffer some audio for calibration
        time.sleep(1) 
        self.calibrate_silence_threshold()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        print("👂 Listening for wake word...")
        return True

    def stop(self):
        """Stops the voice processing system."""
        if self.processing_thread is None:
            return

        print("🔇 Stopping audio processing...")
        self.terminate_event.set()
        if self.stream.active:
            self.stream.stop()
            self.stream.close()
        
        self.processing_thread.join(timeout=2.0)
        self.processing_thread = None
        print("✅ Audio processing stopped.")


def play_audio_from_base64(audio_b64, format_type='wav'):
    """Decode base64 audio and play it on the Pi."""
    try:
        audio_data = base64.b64decode(audio_b64)
        print(f"[AUDIO] F.R.E.D. transmission received ({len(audio_data)} bytes)")
        
        with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            subprocess.run(['aplay', temp_file_path], check=True, capture_output=True)
            print("[SUCCESS] F.R.E.D. voice transmission complete")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(['paplay', temp_file_path], check=True, capture_output=True)
                print("[SUCCESS] F.R.E.D. voice transmission complete (PulseAudio)")
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(['mpv', '--no-video', temp_file_path], check=True, capture_output=True)
                    print("[SUCCESS] F.R.E.D. voice transmission complete (mpv)")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print(f"[CRITICAL] All audio protocols failed - check Pip-Boy speakers")
        
        os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"[CRITICAL] Audio playback system failure: {e}")

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
                sensor_modes = self.picam2.sensor_modes
                max_mode = max(sensor_modes, key=lambda x: x['size'][0] * x['size'][1])
                max_res = max_mode['size']
                print(f"🎯 Using native resolution {max_res} = {(max_res[0] * max_res[1] / 1_000_000):.1f} MP (optimal for Qwen 2.5-VL - no upscaling needed!)")
                
                config = self.picam2.create_video_configuration(
                    main={"size": max_res, "format": "RGB888"},
                    controls={"FrameRate": 5, "Brightness": 0.1, "Contrast": 1.1},
                    buffer_count=2
                )
                self.picam2.configure(config)
                self.picam2.start()
                
                self.frame_count = 0
                self.start_time = time.time()
                print("✅ Picamera2 initialized successfully.")

            async def recv(self):
                """Receive video frames from the camera."""
                pts, time_base = await self.next_timestamp()
                
                try:
                    array = self.picam2.capture_array("main")
                except Exception as e:
                    print(f"💥 Failed to capture frame from Picamera2: {e}")
                    array = np.zeros((2464, 3280, 3), dtype=np.uint8)
                
                if self.frame_count == 1:
                    print(f"📐 Native Resolution: {array.shape[1]}x{array.shape[0]} = {(array.shape[0] * array.shape[1] / 1_000_000):.1f} MP (optimal for Qwen 2.5-VL)")
                
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = pts
                frame.time_base = time_base

                self.frame_count += 1
                if self.frame_count == 1:
                    print(f"🚀 First frame sent! Size: {array.shape}")
                
                return frame

            def __del__(self):
                """Cleanup camera resources."""
                try:
                    if hasattr(self, 'picam2') and self.picam2 and self.picam2.is_open:
                        self.picam2.stop()
                        print('🛑 Picamera2 stopped (cleanup).')
                except Exception:
                    pass

        return PiCamera2Track()
        
    except ImportError:
        print("❌ Picamera2 library not found. Please run: pip install picamera2")
        return None
    except Exception as e:
        print(f"❌ Picamera2 video setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_server_url(provided_url=None):
    """
    Get server URL from argument, tunnel_info.json, or prompt user.
    Handles both ngrok and localtunnel formats.
    """
    if provided_url:
        return provided_url

    try:
        with open("../tunnel_info.json", "r") as f:
            data = json.load(f)
            # Support for localtunnel
            if "url" in data:
                return data["url"]
            # Support for ngrok
            if "tunnels" in data and data["tunnels"]:
                for tunnel in data["tunnels"]:
                    if tunnel.get("proto") == "https" and "public_url" in tunnel:
                        print(f"Found ngrok tunnel: {tunnel['public_url']}")
                        return tunnel["public_url"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"[WARNING] Could not read tunnel info: {e}")
    
    # Fallback to user input
    return input("Enter F.R.E.D. Server URL (e.g., https://your-url.ngrok-free.app): ")

async def run(server_url):
    """Main function to run the client, connect to server, and handle communication."""
    pc = RTCPeerConnection()
    data_channel = None
    voice_processor = None
    
    @pc.on("datachannel")
    def on_datachannel(channel):
        nonlocal data_channel
        data_channel = channel
        print(f"[SUCCESS] Pip-Boy data link established with F.R.E.D. mainframe ({channel.label})")

        @channel.on("open")
        def on_open():
            nonlocal voice_processor
            print("✅ Data channel open. Starting local voice processing.")
            
            def send_transcription_to_server(text):
                """Callback function to send transcribed text over the data channel."""
                if channel.readyState == "open":
                    print(f"📡 Sending to server: {text}")
                    message = json.dumps({"type": "stt", "text": text})
                    channel.send(message)
                else:
                    print("[WARNING] Cannot send text, data channel is not open.")

            voice_processor = VoiceProcessor(send_transcription_to_server)
            if not voice_processor.start():
                print("[CRITICAL] Failed to start voice processor. Text input disabled.")

        @channel.on("message")
        def on_message(message):
            """Handle incoming messages from the server."""
            try:
                data = json.loads(message)
                
                if data.get("type") == "audio_response":
                    print("🔊 F.R.E.D. is responding...")
                    play_audio_from_base64(data["audio"])
                
                elif data.get("type") == "text_response":
                    print(f"\n[F.R.E.D.]> {data['text']}\n")

                elif data.get("type") == "system_message":
                    print(f"\n[SYSTEM]> {data['message']}\n")

            except json.JSONDecodeError:
                print(f"[RAW_DATA] < {message}")
            except Exception as e:
                print(f"[ERROR] Error handling message: {e}")

        @channel.on("close")
        def on_close():
            nonlocal voice_processor
            print("❌ Data channel closed.")
            if voice_processor:
                voice_processor.stop()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            print("Disconnected.")

    # Add video track
    video_track = create_video_track()
    if video_track:
        pc.addTrack(video_track)

    try:
        # Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # Post offer to server
        print(f"Attempting to connect to F.R.E.D. at {server_url}...")
        response = requests.post(
            f"{server_url}/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        answer_data = response.json()
        
        # Set remote description
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
        )
        print("[SUCCESS] WebRTC handshake complete. Awaiting data channel...")

        # Keep the connection alive
        await asyncio.Event().wait()

    except requests.exceptions.RequestException as e:
        print(f"[CRITICAL] Failed to connect to F.R.E.D. server: {e}")
        print("Please ensure the server is running and the URL is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Closing connection...")
        if voice_processor:
            voice_processor.stop()
        await pc.close()


def main():
    banner()
    parser = argparse.ArgumentParser(description="F.R.E.D. Pi Client")
    parser.add_argument("--url", help="The URL of the F.R.E.D. server.")
    args = parser.parse_args()

    server_url = get_server_url(args.url)
    if not server_url:
        print("[CRITICAL] Server URL not found. Exiting.")
        sys.exit(1)

    try:
        asyncio.run(run(server_url))
    except KeyboardInterrupt:
        print("\nShutting down F.R.E.D. client.")
    finally:
        print("Cleanup complete. Pip-Boy is now offline.")

if __name__ == "__main__":
    main() 