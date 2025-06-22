#!/usr/bin/env python3
"""
Test Script for Stewie Voice Pipeline
Tests the complete Stewie voice generation and transmission pipeline
"""

import os
import sys
import time
from config import config
from ollietec_theme import apply_theme, banner
from ollie_print import olliePrint, olliePrint_simple

apply_theme()

def test_stewie_voice_initialization():
    """Test Stewie voice cloning initialization"""
    try:
        from stewie_voice_clone import initialize_stewie_voice, validate_stewie_samples
        
        olliePrint_simple("[TEST] Testing Stewie voice initialization...", 'audio')
        
        # Initialize the system
        success = initialize_stewie_voice()
        if success:
            olliePrint_simple("[TEST] ✅ Stewie voice initialization successful!", 'success')
            
            # Validate samples
            stats = validate_stewie_samples()
            olliePrint_simple(f"[TEST] Found {stats['total_samples']} voice samples", 'success')
            olliePrint_simple(f"[TEST] Total duration: {stats['total_duration']:.1f}s", 'success')
            
            return True
        else:
            olliePrint_simple("[TEST] ❌ Stewie voice initialization failed", 'error')
            return False
            
    except Exception as e:
        olliePrint_simple(f"[TEST] ❌ Initialization error: {e}", 'error')
        return False

def test_stewie_voice_generation():
    """Test Stewie voice generation"""
    try:
        from stewie_voice_clone import generate_stewie_speech
        
        test_text = "What the devil? This is a test of F.R.E.D.'s voice cloning system."
        output_file = "test_stewie_output.wav"
        
        olliePrint_simple(f"[TEST] Generating test speech: '{test_text}'", 'audio')
        
        start_time = time.time()
        success = generate_stewie_speech(test_text, output_file)
        generation_time = time.time() - start_time
        
        if success and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            olliePrint_simple(f"[TEST] ✅ Voice generation successful!", 'success')
            olliePrint_simple(f"[TEST] Generation time: {generation_time:.1f}s", 'success')
            olliePrint_simple(f"[TEST] Output file: {output_file} ({file_size} bytes)", 'success')
            
            # Cleanup test file
            try:
                os.remove(output_file)
                olliePrint_simple("[TEST] Test file cleaned up", 'success')
            except:
                pass
                
            return True
        else:
            olliePrint_simple("[TEST] ❌ Voice generation failed", 'error')
            return False
            
    except Exception as e:
        olliePrint_simple(f"[TEST] ❌ Generation error: {e}", 'error')
        return False

def test_fred_speak_function():
    """Test the main fred_speak function with Pi targeting"""
    try:
        # Import app module
        from app import fred_speak, initialize_tts, fred_state
        
        olliePrint_simple("[TEST] Testing F.R.E.D. speak function...", 'audio')
        
        # Initialize TTS first
        if not fred_state.get_tts_engine():
            initialize_tts()
        
        test_text = "Testing F.R.E.D.'s integrated voice pipeline."
        
        # Test with Pi target
        olliePrint_simple("[TEST] Testing with Pi target device...", 'audio')
        fred_speak(test_text, mute_fred=True, target_device='pi')  # Muted to avoid actual playback
        
        olliePrint_simple("[TEST] ✅ F.R.E.D. speak function test completed", 'success')
        return True
        
    except Exception as e:
        olliePrint_simple(f"[TEST] ❌ F.R.E.D. speak test error: {e}", 'error')
        return False

def test_configuration():
    """Test configuration settings"""
    olliePrint_simple("[TEST] Checking configuration...", 'audio')
    
    # Check if Stewie voice is enabled
    if config.STEWIE_VOICE_ENABLED:
        olliePrint_simple("[TEST] ✅ Stewie voice cloning is enabled", 'success')
    else:
        olliePrint_simple("[TEST] ⚠️  Stewie voice cloning is disabled in config", 'warning')
    
    # Check voice samples directory
    if os.path.exists(config.STEWIE_VOICE_SAMPLES_DIR):
        samples = os.listdir(config.STEWIE_VOICE_SAMPLES_DIR)
        audio_files = [f for f in samples if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
        olliePrint_simple(f"[TEST] ✅ Voice samples directory exists with {len(audio_files)} audio files", 'success')
    else:
        olliePrint_simple(f"[TEST] ❌ Voice samples directory not found: {config.STEWIE_VOICE_SAMPLES_DIR}", 'error')
        return False
    
    return True

def main():
    """Main test function"""
    olliePrint(banner("Stewie Voice Pipeline Test"))
    olliePrint_simple("[TEST] Starting comprehensive pipeline test...\n", 'audio')
    
    tests = [
        ("Configuration Check", test_configuration),
        ("Stewie Voice Initialization", test_stewie_voice_initialization),
        ("Stewie Voice Generation", test_stewie_voice_generation),
        ("F.R.E.D. Speak Function", test_fred_speak_function),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        olliePrint_simple(f"\n{'='*50}")
        olliePrint_simple(f"[TEST] Running: {test_name}")
        olliePrint_simple(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                olliePrint_simple(f"[TEST] ✅ {test_name} PASSED", 'success')
            else:
                olliePrint_simple(f"[TEST] ❌ {test_name} FAILED", 'error')
        except Exception as e:
            olliePrint_simple(f"[TEST] ❌ {test_name} FAILED with exception: {e}", 'error')
    
    olliePrint_simple(f"\n{'='*50}")
    olliePrint_simple(f"[TEST] Test Results: {passed}/{total} tests passed")
    olliePrint_simple(f"{'='*50}")
    
    if passed == total:
        olliePrint_simple("[TEST] 🎉 All tests passed! Stewie voice pipeline is ready!", 'success')
        return 0
    else:
        olliePrint_simple(f"[TEST] ⚠️  {total - passed} test(s) failed. Check the issues above.", 'warning')
        return 1

if __name__ == "__main__":
    sys.exit(main()) 