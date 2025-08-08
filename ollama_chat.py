import sys

# --- Project Integration ---
# This script now uses the centralized configuration and connection manager
# from your F.R.E.D. project. This ensures that all model parameters
# and connection settings are consistent with the main application.

try:
    from config import config, ollama_manager
except ImportError:
    # Silenced console output; preserve exit behavior
    sys.exit(1)

# The model and its parameters are loaded from config
MODEL_NAME = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"

# --- Main Chat Logic ---
def main():
    """Main function to run the terminal chat client."""
    # Silenced startup console output

    messages = []

    while True:
        try:
            prompt = input("\n>>> You: ")

            if prompt.lower() in ['quit', 'exit']:
                break

            messages.append({
                'role': 'user',
                'content': prompt,
            })

            # Silenced thinking notice

            full_response = ""
            # Use the project's concurrent-safe chat method
            # This automatically applies the options from config.py
            stream = ollama_manager.chat_concurrent_safe(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                response_piece = chunk['message']['content']
                full_response += response_piece

            # Silenced model response output

            messages.append({
                'role': 'assistant',
                'content': full_response,
            })

        except KeyboardInterrupt:
            break
        except Exception as e:
            # Silenced error details; exit silently
            break

if __name__ == "__main__":
    main()
