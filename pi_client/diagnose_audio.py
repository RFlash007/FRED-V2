#!/usr/bin/env python3
"""
F.R.E.D. Pi Glasses Audio Diagnostic Tool
This script helps diagnose microphone issues step by step
"""

import subprocess
import sys
import os
import time
import numpy as np

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run_command(cmd, description):
    """Run a system command and return result"""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Success:")
            print(result.stdout)
            return True, result.stdout
        else:
            print(f"❌ Failed (return code: {result.returncode}):")
            print(result.stderr)
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False, "Timeout"
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False, "Command not found"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, str(e)

def test_system_audio():
    """Test basic system audio setup"""
    print_section("1. SYSTEM AUDIO HARDWARE DETECTION")
    
    # Check ALSA devices
    run_command(['arecord', '-l'], "List ALSA recording devices")
    
    # Check PulseAudio devices (if available)
    run_command(['pactl', 'list', 'sources', 'short'], "List PulseAudio sources")
    
    # Check USB devices (for CORSAIR HS80)
    run_command(['lsusb'], "List USB devices")
    
    # Check sound cards in /proc
    print(f"\n🔍 /proc/asound/cards:")
    try:
        with open('/proc/asound/cards', 'r') as f:
            cards = f.read()
            print(cards)
            if 'CORSAIR' in cards or 'HS80' in cards:
                print("👉 CORSAIR HS80 detected in sound cards!")
    except Exception as e:
        print(f"❌ Could not read /proc/asound/cards: {e}")

def test_alsa_recording():
    """Test ALSA recording directly"""
    print_section("2. ALSA RECORDING TEST")
    
    # Test various ALSA devices
    devices_to_test = [
        'hw:3,0',  # CORSAIR specific
        'hw:1,0',
        'hw:0,0', 
        'default',
        'plughw:3,0'  # Plugin version for CORSAIR
    ]
    
    for device in devices_to_test:
        print(f"\n🎤 Testing ALSA device: {device}")
        success, output = run_command([
            'arecord', '-D', device, '-f', 'S16_LE', '-r', '16000', '-c', '1', 
            '-d', '2', '/tmp/test_audio.wav'
        ], f"Record 2 seconds from {device}")
        
        if success:
            # Check if file was created and has content
            try:
                size = os.path.getsize('/tmp/test_audio.wav')
                print(f"✅ Recording successful! File size: {size} bytes")
                if size > 1000:  # Should be much larger for real audio
                    print(f"🎵 Audio file looks good - likely contains real audio data")
                    return device  # Return working device
                else:
                    print(f"⚠️  File too small - might be silence")
            except FileNotFoundError:
                print(f"❌ No output file created")
        
        # Clean up
        try:
            os.unlink('/tmp/test_audio.wav')
        except:
            pass
    
    return None

def test_python_libraries():
    """Test Python audio libraries"""
    print_section("3. PYTHON AUDIO LIBRARIES")
    
    # Test sounddevice
    print(f"\n📦 Testing sounddevice library...")
    try:
        import sounddevice as sd
        print("✅ sounddevice imported successfully")
        
        # List devices
        print("\n🔍 sounddevice devices:")
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device))
                marker = "👉 CORSAIR!" if 'CORSAIR' in device['name'] or 'HS80' in device['name'] else ""
                print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']}) {marker}")
        
        if input_devices:
            print(f"\n✅ Found {len(input_devices)} input devices")
            return input_devices
        else:
            print(f"❌ No input devices found!")
            return []
            
    except ImportError:
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        return []
    except Exception as e:
        print(f"❌ sounddevice error: {e}")
        return []

def test_sounddevice_recording():
    """Test sounddevice recording"""
    print_section("4. SOUNDDEVICE RECORDING TEST")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Find CORSAIR device
        corsair_device = None
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if 'CORSAIR' in device['name'] or 'HS80' in device['name']:
                    corsair_device = i
                    print(f"🎯 Found CORSAIR device: {device['name']} (device {i})")
                    break
        
        # Test device (CORSAIR if found, else default)
        device_id = corsair_device
        device_name = "CORSAIR" if corsair_device else "default"
        
        print(f"\n🎤 Testing sounddevice recording with {device_name} device...")
        print("Recording 3 seconds... SPEAK NOW!")
        
        # Record audio
        recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1, device=device_id)
        sd.wait()  # Wait until recording is finished
        
        # Analyze recording
        volume = np.sqrt(np.mean(recording**2))
        max_amplitude = np.max(np.abs(recording))
        
        print(f"📊 Recording analysis:")
        print(f"   RMS Volume: {volume:.6f}")
        print(f"   Max amplitude: {max_amplitude:.6f}")
        
        if volume > 0.001:  # Threshold for real audio
            print(f"✅ Audio detected! Microphone is working with sounddevice")
            return True
        else:
            print(f"❌ No audio detected - recording is silent")
            return False
            
    except ImportError:
        print("❌ sounddevice not available")
        return False
    except Exception as e:
        print(f"❌ sounddevice recording failed: {e}")
        return False

def test_aiortc_media_player():
    """Test aiortc MediaPlayer for audio"""
    print_section("5. AIORTC MEDIAPLAYER TEST")
    
    try:
        from aiortc.contrib.media import MediaPlayer
        
        devices_to_test = [
            ('hw:3,0', {'sample_rate': '16000', 'channels': '1'}),  # CORSAIR
            ('hw:1,0', {'sample_rate': '16000', 'channels': '1'}),
            ('hw:0,0', {'sample_rate': '16000', 'channels': '1'}),
            ('default', {'sample_rate': '16000', 'channels': '1'}),
        ]
        
        for device, options in devices_to_test:
            print(f"\n🎤 Testing MediaPlayer with device: {device}")
            try:
                player = MediaPlayer(device, format='alsa', options=options)
                if player.audio:
                    print(f"✅ MediaPlayer created successfully for {device}")
                    print(f"   Audio track: {player.audio}")
                    return device
                else:
                    print(f"❌ No audio track created for {device}")
            except Exception as e:
                print(f"❌ MediaPlayer failed for {device}: {e}")
        
        return None
        
    except ImportError:
        print("❌ aiortc not available")
        return None
    except Exception as e:
        print(f"❌ aiortc test failed: {e}")
        return None

def test_webrtc_integration():
    """Test the actual WebRTC audio integration"""
    print_section("6. WEBRTC INTEGRATION TEST")
    
    try:
        # Import the actual audio capture code
        sys.path.append(os.path.dirname(__file__))
        from audio_capture_alternative import create_sounddevice_audio_track, test_audio_capture
        
        print("\n🧪 Running audio_capture_alternative test...")
        test_audio_capture()
        
        print("\n🧪 Creating sounddevice audio track...")
        audio_track = create_sounddevice_audio_track()
        
        if audio_track:
            print("✅ Audio track created successfully!")
            print(f"   Track type: {type(audio_track)}")
            print(f"   Track kind: {getattr(audio_track, 'kind', 'unknown')}")
            return True
        else:
            print("❌ Failed to create audio track")
            return False
            
    except Exception as e:
        print(f"❌ WebRTC integration test failed: {e}")
        return False

def print_recommendations():
    """Print troubleshooting recommendations"""
    print_section("TROUBLESHOOTING RECOMMENDATIONS")
    
    print("🔧 Based on the tests above, try these solutions:")
    print()
    print("1. If CORSAIR HS80 not detected:")
    print("   - Check USB connection: lsusb | grep CORSAIR")
    print("   - Try different USB port")
    print("   - Check dmesg | grep -i corsair for driver messages")
    print()
    print("2. If ALSA recording fails:")
    print("   - Install ALSA utilities: sudo apt install alsa-utils")
    print("   - Configure audio: sudo raspi-config (Advanced -> Audio)")
    print("   - Check permissions: sudo usermod -a -G audio $USER")
    print("   - Reboot after permission changes")
    print()
    print("3. If sounddevice fails:")
    print("   - Install: pip install sounddevice")
    print("   - Install system deps: sudo apt install libportaudio2 libportaudiocpp0 portaudio19-dev")
    print()
    print("4. If MediaPlayer fails:")
    print("   - Install aiortc: pip install aiortc")
    print("   - Check ALSA device exists: arecord -l")
    print()
    print("5. For WebRTC integration:")
    print("   - Ensure client.py uses working audio device")
    print("   - Check server logs for audio processing")
    print("   - Monitor network connectivity during audio streaming")

def main():
    """Run complete audio diagnostic"""
    print("🍇 F.R.E.D. Pi Glasses Audio Diagnostic")
    print("🎤 This will test your microphone setup step by step")
    print()
    
    results = {}
    
    # Run all tests
    test_system_audio()
    
    working_alsa_device = test_alsa_recording()
    results['alsa_device'] = working_alsa_device
    
    input_devices = test_python_libraries()
    results['sounddevice_devices'] = input_devices
    
    sounddevice_works = test_sounddevice_recording()
    results['sounddevice_recording'] = sounddevice_works
    
    working_mediaplayer = test_aiortc_media_player()
    results['mediaplayer_device'] = working_mediaplayer
    
    webrtc_works = test_webrtc_integration()
    results['webrtc_integration'] = webrtc_works
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    print("📊 Test Results:")
    print(f"   ALSA Recording: {'✅' if results['alsa_device'] else '❌'} {results['alsa_device'] or 'Failed'}")
    print(f"   sounddevice: {'✅' if results['sounddevice_recording'] else '❌'}")
    print(f"   MediaPlayer: {'✅' if results['mediaplayer_device'] else '❌'} {results['mediaplayer_device'] or 'Failed'}")
    print(f"   WebRTC Integration: {'✅' if results['webrtc_integration'] else '❌'}")
    
    if all([results['alsa_device'], results['sounddevice_recording'], results['webrtc_integration']]):
        print("\n🎉 All tests passed! Your microphone should work with F.R.E.D.")
    else:
        print("\n❌ Some tests failed. See recommendations below.")
        print_recommendations()

if __name__ == "__main__":
    main() 