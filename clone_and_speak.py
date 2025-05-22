import os
import torch
from TTS.api import TTS
import playsound # Added for playing sound

# --- Configuration ---
# Text to convert to speech
TEXT_TO_SPEAK = "Spit on me Brian"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"  # Specify the model for quality voice cloning/referencing
SPEAKER_WAV_PATH = "new_voice_sample.wav"  # Reference WAV for the desired voice
LANGUAGE = "en"  # Language of the text and speaker_wav
OUTPUT_FILE = "output.wav" # Output file name remains the same

# --- Main Script ---
def main():
    # Check if the speaker WAV file exists
    if not os.path.exists(SPEAKER_WAV_PATH):
        print(f"Error: Speaker WAV file not found at '{SPEAKER_WAV_PATH}'")
        print(f"Please ensure '{SPEAKER_WAV_PATH}' is in the same directory as the script.")
        return

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Initializing TTS with model: {MODEL_NAME}...")
    try:
        # Initialize with the specified model
        tts = TTS(model_name=MODEL_NAME).to(device)
    except Exception as e:
        print(f"Error initializing TTS: {e}")
        return

    print(f"Starting TTS synthesis with voice from {SPEAKER_WAV_PATH}...")
    try:
        # Generate speech using the specified model and speaker WAV
        tts.tts_to_file(
            text=TEXT_TO_SPEAK,
            speaker_wav=SPEAKER_WAV_PATH,
            language=LANGUAGE,
            file_path=OUTPUT_FILE
        )
        print(f"Speech generated to {OUTPUT_FILE}")

        # Play the generated audio file
        print(f"Playing {OUTPUT_FILE}...")
        try:
            playsound.playsound(OUTPUT_FILE, block=True) # Use playsound, block until done
            print("Playback finished.")
        except Exception as e:
            print(f"Error playing sound with playsound: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()