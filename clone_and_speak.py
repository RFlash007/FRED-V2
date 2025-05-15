import os
import torch
from TTS.api import TTS

# --- Configuration ---
# 1. Select a voice cloning model
# Other options: "tts_models/multilingual/multi-dataset/your_tts"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# 2. Set the path to your voice sample (WAV file, a few seconds long is good)
# IMPORTANT: Replace this with the actual path to your WAV file.
# For example: "my_voice_sample.wav" if it's in the same directory as the script.
SPEAKER_WAV_PATH = "path/to/your/voice_sample.wav" 

# 3. Set the language for the speech (ISO 639-1 code)
LANGUAGE = "en"

# 4. Set the text you want to convert to speech
TEXT_TO_SPEAK = "Hello, this is a test of voice cloning with F.R.E.D. I hope this sounds like the original speaker."

# 5. Output audio file name
OUTPUT_WAV_PATH = "cloned_speech.wav"

# --- Main Script ---
def main():
    # Check if the speaker WAV file exists
    if not os.path.exists(SPEAKER_WAV_PATH):
        print(f"Error: Speaker WAV file not found at '{SPEAKER_WAV_PATH}'")
        print("Please update the SPEAKER_WAV_PATH variable in the script.")
        return

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Initializing TTS with model: {MODEL_NAME}...")
    try:
        tts = TTS(MODEL_NAME).to(device)
    except Exception as e:
        print(f"Error initializing TTS model: {e}")
        print("This might be due to a missing model or internet connection issues during model download.")
        print("You can try listing available models with: print(TTS().list_models())")
        return

    print("Starting TTS synthesis with voice cloning...")
    try:
        tts.tts_to_file(
            text=TEXT_TO_SPEAK,
            speaker_wav=SPEAKER_WAV_PATH,
            language=LANGUAGE,
            file_path=OUTPUT_WAV_PATH
        )
        print(f"Speech saved to {OUTPUT_WAV_PATH}")

        # Play the generated audio file (Windows specific command)
        # For macOS, use: os.system(f"afplay {OUTPUT_WAV_PATH}")
        # For Linux, use: os.system(f"aplay {OUTPUT_WAV_PATH}") or os.system(f"xdg-open {OUTPUT_WAV_PATH}")
        print(f"Playing {OUTPUT_WAV_PATH}...")
        if os.name == 'nt': # For Windows
            os.system(f"start {OUTPUT_WAV_PATH}")
        elif os.uname().sysname == 'Darwin': # For macOS
             os.system(f"afplay {OUTPUT_WAV_PATH}")
        else: # For Linux and other POSIX systems
            # Try aplay first, then xdg-open as a fallback
            if os.system(f"aplay {OUTPUT_WAV_PATH} > /dev/null 2>&1") != 0:
                os.system(f"xdg-open {OUTPUT_WAV_PATH}")
        
        print("Playback initiated.")

    except RuntimeError as e:
        if "Mot Implemented" in str(e) and "get_conditioning_latents" in str(e):
             print(f"Error during TTS synthesis: {e}")
             print("This specific model might require a GPU. If you are on CPU, try a different model or ensure CUDA is available if you have an NVIDIA GPU.")
        elif "xtts_cloner.cuda()" in str(e) :
            print(f"Error during TTS synthesis: {e}")
            print("XTTS models often require CUDA. Ensure you have an NVIDIA GPU and CUDA properly set up, or try running on CPU (though it will be slower).")
            print("If running on CPU was intended, ensure your PyTorch installation supports your CPU architecture correctly.")
        else:
            print(f"An unexpected error occurred during TTS synthesis: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 