#!/usr/bin/env python3
"""
Video Device Release Utility for F.R.E.D. Pi Glasses
Helps release video devices that might be stuck from previous sessions
"""

import os
import subprocess
import time

def release_video_devices():
    """Release any stuck video devices"""
    print("🎥 Releasing video devices...")
    
    # Kill any processes that might be using the camera
    processes_to_kill = [
        'python3',  # Other Python camera processes
        'python',
        'libcamera',
        'raspistill',
        'raspivid',
        'opencv',
        'gstreamer'
    ]
    
    for process in processes_to_kill:
        try:
            result = subprocess.run(['pkill', '-f', process], capture_output=True)
            if result.returncode == 0:
                print(f"  Killed {process} processes")
        except:
            continue
    
    # Wait a moment for processes to die
    time.sleep(2)
    
    # Try to reset USB video devices
    video_devices = ['/dev/video0', '/dev/video1', '/dev/video2']
    
    for device in video_devices:
        if os.path.exists(device):
            try:
                print(f"  Checking {device}...")
                
                # Try to briefly open and close the device to reset it
                fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
                os.close(fd)
                print(f"  ✅ {device} is available")
                
            except OSError as e:
                if e.errno == 16:  # Device busy
                    print(f"  ⚠️  {device} still busy")
                elif e.errno == 13:  # Permission denied
                    print(f"  ⚠️  {device} permission denied - try sudo")
                else:
                    print(f"  ❌ {device} error: {e}")
            except Exception as e:
                print(f"  ❌ {device} error: {e}")

def reset_usb_devices():
    """Reset USB devices (requires sudo)"""
    print("🔌 Attempting USB device reset...")
    
    try:
        # Find USB video devices
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'camera' in line.lower() or 'video' in line.lower():
                    print(f"  Found: {line}")
        
        # You can add specific USB reset commands here if needed
        print("  USB reset requires manual intervention if needed")
        
    except Exception as e:
        print(f"  USB reset failed: {e}")

def main():
    print("🍇 F.R.E.D. Video Device Release Utility")
    print("=" * 40)
    
    release_video_devices()
    
    print("\n🧪 Testing video device availability...")
    
    # Test each video device
    for i in range(3):
        device = f'/dev/video{i}'
        if os.path.exists(device):
            try:
                # Quick test with v4l2-ctl
                result = subprocess.run([
                    'v4l2-ctl', '--device', device, '--list-formats-ext'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    print(f"✅ {device} is working")
                else:
                    print(f"❌ {device} failed v4l2 test")
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {device} test timed out (device may be busy)")
            except FileNotFoundError:
                print(f"⚠️  v4l2-ctl not found - install with: sudo apt install v4l-utils")
            except Exception as e:
                print(f"❌ {device} test error: {e}")
    
    print("\n✅ Video device release complete!")
    print("Now try: python3 client.py --server YOUR_SERVER_URL")

if __name__ == "__main__":
    main() 