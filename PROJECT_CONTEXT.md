# F.R.E.D. Project Context & Environment Notes

## I. Project Overview: F.R.E.D. (Funny Rude Educated Droid)

*   **Goal:** To create a sophisticated, locally-hosted, personalized AI assistant codenamed F.R.E.D.
*   **Inspiration:** Concepts like Iron Man's Jarvis, aiming for an ambient intelligence integrated into the user's home and personal space.
*   **Primary Interaction:** Smart glasses for egocentric perception (real-time visual/audio) and AR overlays; natural language voice commands.
*   **Core Pillars:**
    *   Environmental Awareness (multi-sensor integration).
    *   Contextual & Conversational AI.
    *   Autonomous Task Execution (smart home, scheduling, info retrieval).
    *   Hyper-Personalized Memory (private knowledge graph).
    *   Ubiquitous Interface (voice, AR, environmental cues).
*   **Architectural Cornerstone:** **Local-first, privacy-centric design.** A powerful local home server for sensitive data/AI computation, with edge devices for lighter tasks.

## II. Python Virtual Environment (venv) Setup

This project uses a Python virtual environment to manage its dependencies.

*   **Location:** The virtual environment for this project is located at `C:\Users\ianjc\Documents\GitHub\F.R.E.D.-V2\.venv`
*   **Activation (in PowerShell):**
    ```powershell
    cd C:\Users\ianjc\Documents\GitHub\F.R.E.D.-V2
    .\.venv\Scripts\activate
    ```
    After activation, your PowerShell prompt should begin with `(.venv)`.

*   **Running Python Scripts:**
    Once the venv is active, you can typically run scripts using `python your_script_name.py`.
    For maximum certainty, especially if encountering issues, use the full path to the venv's Python interpreter:
    ```powershell
    .\.venv\Scripts\python.exe your_script_name.py
    ```

*   **Installing Packages (with pip):**
    When the venv is active, install packages using `pip install package_name`.
    For maximum certainty, or if `pip` seems to point to the wrong environment, use the full path to the venv's Python interpreter to run pip:
    ```powershell
    .\.venv\Scripts\python.exe -m pip install package_name
    ```

## III. Important Note on Potential PATH Conflicts (Resolved)

*   **Issue Encountered:** Previously, the system's `PATH` environment variable (and the `sys.path` for Python interpreters used by some tools) was prioritizing a different virtual environment located at `C:\Users\ianjc\Desktop\Fred v2\.venv`. This caused `ModuleNotFoundError` issues when running scripts or installing packages if not explicitly targeting the correct venv.
*   **Resolution for Local Terminal:** The issue was resolved for local terminal sessions by explicitly calling the correct Python interpreter and pip module within the `C:\Users\ianjc\Documents\GitHub\F.R.E.D.-V2\.venv` (e.g., `.\.venv\Scripts\python.exe -m pip install ...`).
*   **Resolution for AI Assistant Tools:** The AI assistant will now endeavor to use explicit paths to the `C:\Users\ianjc\Documents\GitHub\F.R.E.D.-V2\.venv\Scripts\python.exe` when executing terminal commands to ensure the correct environment is used.
*   **Recommendation for System Cleanliness (User Action):** If the `Fred v2` project on the Desktop is no longer needed, consider removing its Scripts directory (`C:\Users\ianjc\Desktop\Fred v2\.venv\Scripts`) from your User PATH environment variable (via System Properties -> Environment Variables) or deleting the old Desktop project folder entirely to prevent future conflicts. Remember to restart terminals and development tools after making PATH changes.

## IV. Key Project Files & Current Status

*   `app.py`: Main Flask application for F.R.E.D.'s web interface and core logic.
    *   TTS is integrated.
    *   Speaker WAV for F.R.E.D.'s voice is set to: `new_voice_sample.wav`.
*   `clone_and_speak.py`: Utility script for testing voice cloning and TTS with Coqui TTS.
    *   Currently configured to use `new_voice_sample.wav` as the reference voice.
*   `Tools.py`: Manages callable tools for the LLM.
*   `memory/librarian.py`: Handles knowledge base interactions.
*   `templates/index.html` & `static/script.js`: Frontend for the chat interface.
    *   Mute button functionality has been added.

This document serves as a quick reference for the project's context and critical environment details. 