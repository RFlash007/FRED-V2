import sys

# --- Project Integration ---
# This script now uses the centralized configuration and connection manager
# from your F.R.E.D. project. This ensures that all model parameters
# and connection settings are consistent with the main application.

try:
    from config import config, ollama_manager
    from ollie_print import olliePrint_simple
except ImportError:
    print("Error: Could not import project modules ('config', 'ollama_manager').")
    print("Please run this script from the root of your F.R.E.D. project directory.")
    sys.exit(1)

# The model and its parameters are loaded from config
MODEL_NAME = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"

# --- Main Chat Logic ---
def main():
    """Main function to run the terminal chat client."""
    olliePrint_simple(f"Starting chat with '{MODEL_NAME}'")
    olliePrint_simple(f"(Parameters: num_ctx={config.LLM_GENERATION_OPTIONS.get('num_ctx')})")
    olliePrint_simple("Type 'quit', 'exit', or press Ctrl+C to end the chat.")
    olliePrint_simple("--------------------------------------------------")

    messages = []

    while True:
        try:
            prompt = input("\n>>> You: ")

            if prompt.lower() in ['quit', 'exit']:
                olliePrint_simple("Exiting chat. Goodbye!")
                break

            messages.append({
                'role': 'user',
                'content': prompt,
            })

            olliePrint_simple(f"... {MODEL_NAME} is thinking ...")

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

            olliePrint_simple(full_response)

            messages.append({
                'role': 'assistant',
                'content': full_response,
            })

        except KeyboardInterrupt:
            olliePrint_simple("Exiting chat. Goodbye!")
            break
        except Exception as e:
            olliePrint_simple(f"An error occurred: {e}", level='error')
            olliePrint_simple("Please ensure the Ollama server is running.")
            break

if __name__ == "__main__":
    main()
