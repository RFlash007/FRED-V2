import os
import json5
import playwright.sync_api
import re
import io
import sys
import ollama # Import ollama directly for manual interaction

# --- Configuration (Consistent with config.py style) ---
OLLAMA_BASE_URL = 'http://localhost:11434'
DEFAULT_MODEL = 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M'
LLM_GENERATION_OPTIONS = {
    'temperature': 0.6,
    'top_p': 0.95,
    'top_k': 20,
    'num_ctx': 4096, # Context window
}

SYSTEM_PROMPT = """You are a helpful AI assistant. You have access to the following tools: headless_browser and code_interpreter. Use them to answer questions and perform tasks. Always prioritize using the most appropriate tool for the task."""

# --- Custom Tools ---

class HeadlessBrowser:
    """Headless browser to navigate and extract content based on prompt."""
    description = 'Headless browser to navigate and extract content based on prompt.'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Prompt like "go to URL and extract title"',
        'required': True
    }]

    def call(self, prompt: str, **kwargs) -> str:
        """Executes the browser action and returns the result."""
        url_match = re.search(r'https?://\S+', prompt)
        url = url_match.group() if url_match else 'https://example.com'
        print(f"[LOG] HeadlessBrowser: Prompt: {prompt}")
        print(f"[LOG] HeadlessBrowser: Navigating to: {url}")
        
        with playwright.sync_api.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, timeout=30000)
                title = page.title()
                # Capture up to 1000 characters of visible text content for better context
                content_snippet = page.inner_text('body')[:1000] 
                result = f"Title: {title}\nContent Snippet: {content_snippet}"
            except Exception as e:
                result = f"Error: {str(e)}"
            finally:
                browser.close()
        
        print(f"[LOG] HeadlessBrowser: Result: {result}")
        return json5.dumps({'result': result})

class CodeInterpreterTool:
    """Executes Python code and returns the output."""
    description = 'Executes Python code and returns the output.'
    parameters = [{
        'name': 'code',
        'type': 'string',
        'description': 'The Python code to execute.',
        'required': True
    }]

    def call(self, code: str, **kwargs) -> str:
        """Executes the provided Python code and captures its output/errors."""
        print(f"[LOG] CodeInterpreterTool: Executing Code:\n{code}")
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            exec(code, globals())
            output = redirected_output.getvalue()
            print(f"[LOG] CodeInterpreterTool: Output:\n{output}")
            return json5.dumps({'output': output})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[LOG] CodeInterpreterTool: Error:\n{error_msg}")
            return json5.dumps({'error': error_msg})
        finally:
            sys.stdout = old_stdout

# --- Tool Schemas for Ollama ---
# Dynamically generate tool schemas from the tool classes
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "headless_browser",
            "description": HeadlessBrowser.description,
            "parameters": {
                "type": "object",
                "properties": {param['name']: {k:v for k,v in param.items() if k != 'required'} for param in HeadlessBrowser.parameters},
                "required": [param['name'] for param in HeadlessBrowser.parameters if param.get('required', False)]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": CodeInterpreterTool.description,
            "parameters": {
                "type": "object",
                "properties": {param['name']: {k:v for k,v in param.items() if k != 'required'} for param in CodeInterpreterTool.parameters},
                "required": [param['name'] for param in CodeInterpreterTool.parameters if param.get('required', False)]
            }
        }
    }
]

# Map tool names to their instances for manual execution
TOOL_FUNCTIONS_MAP = {
    "headless_browser": HeadlessBrowser(),
    "code_interpreter": CodeInterpreterTool()
}

# --- Main Prototyping Function ---

def test_ollama_with_custom_tools():
    """Tests Ollama's tool-calling capabilities with custom headless browser and code interpreter tools."""
    print("\n=== Ollama with Custom Tools Test Environment ===")
    print("Configuration:")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  Ollama URL: {OLLAMA_BASE_URL}")
    print(f"  Generation Options: {LLM_GENERATION_OPTIONS}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Plot y = x^2 using matplotlib and save it to 'parabola.png', then browse to https://www.python.org/ and tell me the title and a brief summary of the content."}
    ]

    max_iterations = 5
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        print("Sending messages to Ollama...")
        
        try:
            # Make the chat call to Ollama with our custom tools
            response = ollama.chat(
                model=DEFAULT_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS, # Pass our custom tool schemas
                options=LLM_GENERATION_OPTIONS
            )
            
            response_message = response.get('message', {})
            model_content = response_message.get('content')
            tool_calls = response_message.get('tool_calls')

            # If the model provides content, add it to history and print
            if model_content:
                print(f"Model Response: {model_content}")
                messages.append({"role": "assistant", "content": model_content})

            # If the model requested tool calls, execute them
            if tool_calls:
                print(f"Model requested {len(tool_calls)} tool(s).")
                tool_outputs = []
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    tool_arguments = tool_call['function']['arguments']
                    tool_call_id = tool_call['id'] # Capture tool call ID for response

                    if function_name in TOOL_FUNCTIONS_MAP:
                        tool_instance = TOOL_FUNCTIONS_MAP[function_name]
                        print(f"Executing tool: {function_name} with args: {tool_arguments}")
                        result_content = tool_instance.call(**tool_arguments)
                        tool_outputs.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id, 
                            "content": result_content
                        })
                    else:
                        error_msg = f"Tool not found: {function_name}"
                        print(f"Error: {error_msg}")
                        tool_outputs.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json5.dumps({'error': error_msg})
                        })
                # Add tool outputs to messages for the next turn
                messages.extend(tool_outputs)
            else:
                print("No more tool calls. Conversation complete.")
                break

        except Exception as e:
            print(f"An error occurred during Ollama interaction: {e}")
            break

# --- Script Entry Point ---
if __name__ == "__main__":
    test_ollama_with_custom_tools()
