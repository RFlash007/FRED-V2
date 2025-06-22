#!/usr/bin/env python3
"""
Simple Stewie Speech Generator
Generate custom speech in Stewie's voice
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from ollietec_theme import apply_theme, banner
from ollie_print import olliePrint, olliePrint_simple

# Apply theming
apply_theme()

def generate_custom_stewie_speech(text, output_filename=None):
    """Generate Stewie saying custom text"""
    
    if not output_filename:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c for c in text[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')
        output_filename = f"stewie_{safe_text}_{timestamp}.wav"
    
    try:
        # Import voice cloning system
        from stewie_voice_clone import initialize_stewie_voice, generate_stewie_speech
        
        # Initialize the system
        olliePrint_simple("[INIT] Initializing Stewie voice cloning...", 'audio')
        if not initialize_stewie_voice():
            olliePrint_simple("❌ Failed to initialize voice cloning system!", 'error')
            return False
        
        # Generate the speech
        olliePrint_simple(f"[GENERATE] Creating Stewie voice for: '{text}'", 'audio')
        olliePrint_simple("[PROCESS] Generating speech...", 'warning')
        
        start_time = time.time()
        success = generate_stewie_speech(text, output_filename)
        generation_time = time.time() - start_time
        
        if success:
            if os.path.exists(output_filename):
                file_size = os.path.getsize(output_filename)
                olliePrint_simple(f"🎉 SUCCESS! Generated in {generation_time:.1f}s", 'success')
                olliePrint_simple(f"📁 Output: {output_filename} ({file_size:,} bytes)", 'success')
                olliePrint_simple(f"🎵 Stewie says: '{text}'", 'success')
                
                # Try to play it automatically (Windows)
                try:
                    import subprocess
                    olliePrint_simple("🔊 Playing audio...", 'audio')
                    subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filename}").PlaySync()'], 
                                 check=True, capture_output=True)
                    olliePrint_simple("✅ Playback complete!", 'success')
                except Exception as e:
                    olliePrint_simple(f"⚠️ Auto-play failed: {e}", 'warning')
                    olliePrint_simple(f"💡 Manually play: {output_filename}", 'warning')
                
                return True, output_filename
            else:
                olliePrint_simple("❌ Output file was not created", 'error')
                return False, None
        else:
            olliePrint_simple("❌ Voice generation failed", 'error')
            return False, None
            
    except ImportError as e:
        olliePrint_simple(f"❌ Import error: {e}", 'error')
        olliePrint_simple("💡 Make sure voice cloning system is set up", 'warning')
        return False, None
    except Exception as e:
        olliePrint_simple(f"❌ Unexpected error: {e}", 'error')
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main function with interactive mode"""
    
    print()
    olliePrint(banner("Stewie Speech Generator"))
    olliePrint_simple("[OLLIE-TEC] Advanced Voice Synthesis Division", 'audio')
    olliePrint_simple("[STEWIE] Ready to generate custom speech...\n", 'success')
    
    # Check for command line argument
    if len(sys.argv) > 1:
        # Use command line text
        text = " ".join(sys.argv[1:])
        olliePrint_simple(f"[INPUT] Using command line text: '{text}'", 'audio')
        success, filename = generate_custom_stewie_speech(text)
    else:
        # Interactive mode
        try:
            text = input("Enter text for Stewie to say: ").strip()
            if not text:
                olliePrint_simple("❌ No text provided!", 'error')
                return 1
            
            # Optional custom filename
            custom_filename = input("Custom filename (press Enter for auto): ").strip()
            if custom_filename and not custom_filename.endswith('.wav'):
                custom_filename += '.wav'
            
            filename = custom_filename if custom_filename else None
            success, output_file = generate_custom_stewie_speech(text, filename)
            
        except KeyboardInterrupt:
            olliePrint_simple("\n[CANCELLED] Generation cancelled by user", 'warning')
            return 1
    
    if success:
        olliePrint_simple(f"\n🎭 [COMPLETE] Stewie speech generated successfully!", 'success')
        olliePrint_simple(f"📁 File saved as: {output_file if 'output_file' in locals() else filename}", 'success')
    else:
        olliePrint_simple("\n❌ [FAILED] Generation unsuccessful", 'error')
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 