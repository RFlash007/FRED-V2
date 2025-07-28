import sys

# --- Project Integration ---
# This script now uses the centralized configuration and connection manager
# from your F.R.E.D. project. This ensures that all model parameters
# and connection settings are consistent with the main application.

try:
    from config import config, ollama_manager
except ImportError:
    print("Error: Could not import project modules ('config', 'ollama_manager').")
    print("Please run this script from the root of your F.R.E.D. project directory.")
    sys.exit(1)

# The model and its parameters (num_gpu, num_ctx, etc.) are now loaded from config
MODEL_NAME = "awaescher/qwen3-235b-thinking-2507-unsloth-q3-k-xl:latest"

# --- Main Chat Logic ---
def main():
    """Main function to run the terminal chat client."""
    print(f"--- Starting chat with '{MODEL_NAME}' ---")
    print(f"(Parameters: num_gpu={config.THINKING_MODE_OPTIONS.get('num_gpu')}, num_ctx={config.THINKING_MODE_OPTIONS.get('num_ctx')})")
    print("Type 'quit', 'exit', or press Ctrl+C to end the chat.")
    print("--------------------------------------------------")

    messages = []

    while True:
        try:
            prompt = input("\n>>> You: ")

            if prompt.lower() in ['quit', 'exit']:
                print("--- Exiting chat. Goodbye! ---")
                break

            messages.append({
                'role': 'user',
                'content': prompt,
            })

            print(f"... {MODEL_NAME} is thinking ...", end="\r")
            sys.stdout.flush()

            full_response = ""
            # Use the project's concurrent-safe chat method
            # This automatically applies the options from config.py
            stream = ollama_manager.chat_concurrent_safe(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )
            
            print(" " * 50, end="\r") 
            print(f"<<< AI: ", end="")
            sys.stdout.flush()

            for chunk in stream:
                response_piece = chunk['message']['content']
                print(response_piece, end="", flush=True)
                full_response += response_piece

            messages.append({
                'role': 'assistant',
                'content': full_response,
            })

        except KeyboardInterrupt:
            print("\n--- Exiting chat. Goodbye! ---")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure the Ollama server is running.")
            break

if __name__ == "__main__":
    main()
