import os
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
import playwright.sync_api
import re

# Ollama config from your setup
llm_cfg = {
    'model': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M',
    'model_server': 'http://localhost:11434/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 20,
        'max_tokens': 2048
    }
}

# Custom Headless Browser Tool
@register_tool('headless_browser')
class HeadlessBrowser(BaseTool):
    description = 'Headless browser to navigate and extract content based on prompt.'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Prompt like "go to URL and extract title"',
        'required': True
    }]

    def call(self, prompt: str, **kwargs) -> str:
        url = re.search(r'https?://\S+', prompt).group() if re.search(r'https?://\S+', prompt) else 'https://example.com'
        print(f"[LOG] Browser Prompt: {prompt}")
        
        with playwright.sync_api.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                # Simple example: assume prompt is "browse to {url}"
                print(f"[LOG] Navigating to: {url}")
                page.goto(url, timeout=30000)
                title = page.title()
                content = page.content()[:500]  # Truncate for logging
                result = f"Title: {title}\nContent Snippet: {content}"
            except Exception as e:
                result = f"Error: {str(e)}"
            finally:
                browser.close()
        
        print(f"[LOG] Browser Result: {result}")
        return json5.dumps({'result': result})

# Generate bigger sample file for RAG
def create_sample_doc():
    sample_text = "This is placeholder text for a long document. " * 2000  # ~10k words
    with open('sample_long_doc.txt', 'w') as f:
        f.write(sample_text)
    print("[LOG] Created sample_long_doc.txt")

# Test RAG
def test_rag():
    print("\n=== RAG Test ===")
    create_sample_doc()
    tools = []
    files = ['sample_long_doc.txt']
    bot = Assistant(llm=llm_cfg, function_list=tools, files=files)
    
    messages = [{'role': 'user', 'content': 'Summarize the document content.'}]
    print(f"[LOG] RAG User Prompt: {messages}")
    full_response = []
    for chunk in bot.run(messages=messages):
        full_response.append(chunk)
    print(f"[LOG] Full RAG Response: {full_response}")
    response_plain = ''
    if full_response and full_response[-1] and full_response[-1][-1].get('content'):
        response_plain = typewriter_print(full_response[-1][-1]['content'], response_plain)
    else:
        print("[LOG] No valid content in response.")

# Test Code Interpreter
def test_code_interpreter():
    print("\n=== Code Interpreter Test ===")
    tools = ['code_interpreter']
    bot = Assistant(llm=llm_cfg, function_list=tools)
    
    messages = [{'role': 'user', 'content': 'Plot and save a line graph for y = 2x + 1.'}]
    print(f"[LOG] Code Interpreter User Prompt: {messages}")
    full_response = []
    for chunk in bot.run(messages=messages):
        full_response.append(chunk)
    print(f"[LOG] Full Code Interpreter Response: {full_response}")
    response_plain = ''
    if full_response and full_response[-1] and full_response[-1][-1].get('content'):
        response_plain = typewriter_print(full_response[-1][-1]['content'], response_plain)
    else:
        print("[LOG] No valid content in response.")

# Test Headless Browser
def test_browser():
    print("\n=== Headless Browser Test ===")
    tools = ['headless_browser']
    bot = Assistant(llm=llm_cfg, function_list=tools)
    
    user_prompt = input("Enter a browser prompt (e.g., 'browse to https://github.com'): ")
    messages = [{'role': 'user', 'content': user_prompt}]
    print(f"[LOG] Browser User Prompt: {messages}")
    full_response = []
    try:
        for chunk in bot.run(messages=messages):
            full_response.append(chunk)
        print(f"[LOG] Full Browser Response: {full_response}")
        response_plain = ''
        if full_response and full_response[-1] and full_response[-1][-1].get('content'):
            response_plain = typewriter_print(full_response[-1][-1]['content'], response_plain)
        else:
            print("[LOG] No valid content in response.")
    except Exception as e:
        print(f"[LOG] Browser Error: {str(e)}. Consider using async Playwright for compatibility.")

if __name__ == "__main__":
    #test_rag()
    #test_code_interpreter()
    test_browser()
