import os
import threading
from typing import Optional, Dict, Any

import ollama

try:
    from ollie_print import olliePrint_simple, log_model_io
except ImportError:
    def olliePrint_simple(msg, level='info'):
        print(f"[{level.upper()}] {msg}")

    def log_model_io(model, inputs, outputs):
        if isinstance(inputs, list):
            for m in reversed(inputs):
                if isinstance(m, dict) and m.get('role') == 'user':
                    inputs = m.get('content', '')
                    break
        if isinstance(outputs, dict) and 'embedding' in outputs:
            emb = outputs['embedding']
            if isinstance(emb, (list, tuple)):
                outputs = f"[{len(emb)}-dimensional embedding]"
        elif isinstance(outputs, (list, tuple)) and all(isinstance(x, (int, float)) for x in outputs):
            outputs = f"[{len(outputs)}-dimensional vector]"
        print(f"[MODEL {model} INPUT]: {inputs}")
        print(f"[MODEL {model} OUTPUT]: {outputs}")


class OllamaConnectionManager:
    """
    Optimized Ollama connection manager with memory-efficient model loading.
    
    This class provides a single, reusable connection to the Ollama server and configures
    Ollama environment variables to prevent unnecessary model loading/unloading cycles.
    """
    
    def __init__(self, base_url: str, thinking_options: Dict):
        self._client = None
        self._lock = threading.Lock()
        self.base_url = base_url
        
        # MEMORY OPTIMIZATION: Configure Ollama environment variables for efficient model management
        self._configure_ollama_environment()
        
        # Optimized defaults for Qwen model compatibility
        self.default_options = thinking_options.copy()
        # Keep model loaded during tool execution delays
        self.default_options['keep_alive'] = '30m'  # Keep model resident to avoid repeated loads
    
    def _configure_ollama_environment(self):
        """Configure Ollama environment variables for optimal memory usage."""
        # Assertively set environment variables to ensure memory-safe execution for this script
        os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Force only one model in memory
        os.environ['OLLAMA_NUM_PARALLEL'] = '1'      # Force single-file processing
        os.environ['OLLAMA_KEEP_ALIVE'] = '30m'      # Keep model resident for 30 minutes to prevent thrashing
            
        # Use safe printing to avoid import issues during config initialization
        self._safe_print("[OLLAMA CONFIG] Memory optimization settings applied:")
        self._safe_print(f"  MAX_LOADED_MODELS: {os.environ.get('OLLAMA_MAX_LOADED_MODELS', 'default')}")
        self._safe_print(f"  NUM_PARALLEL: {os.environ.get('OLLAMA_NUM_PARALLEL', 'default')}")
        self._safe_print(f"  KEEP_ALIVE: {os.environ.get('OLLAMA_KEEP_ALIVE', 'default')}")
    
    def _safe_print(self, message: str):
        """Safe printing method that works during config initialization."""
        olliePrint_simple(message)
    
    def get_client(self, host: Optional[str] = None) -> ollama.Client:
        """
        Get or create the single Ollama client.
        
        Args:
            host: Ollama host URL (defaults to config.OLLAMA_BASE_URL)
            
        Returns:
            ollama.Client: The single configured client
        """
        if host is None:
            host = self.base_url
        
        with self._lock:
            if self._client is None:
                self._client = ollama.Client(host=host)
                self._safe_print(f"[OLLAMA] Created optimized client for {host}")
            return self._client
    
    def chat_concurrent_safe(self, host: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Make a chat call using the single connection with memory optimization.
        
        Args:
            host: Ollama host URL (optional, defaults to config)
            **kwargs: All other arguments passed to ollama.chat()
        
        Returns:
            Dict: Response from Ollama chat API
        """
        client = self.get_client(host)
        
        # MEMORY OPTIMIZATION: Merge optimized options with provided options
        if 'options' in kwargs:
            merged_options = self.default_options.copy()
            merged_options.update(kwargs['options'])
            kwargs['options'] = merged_options
        else:
            kwargs['options'] = self.default_options.copy()
        
        # Remove timeout-related options to prevent timeouts during long research cycles
        if 'timeout' in kwargs:
            del kwargs['timeout']

        model_name = kwargs.get('model', 'unknown')
        messages = kwargs.get('messages')

        if kwargs.get('stream'):
            stream = client.chat(**kwargs)

            def generator():
                output_text = ""
                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        output_text += content
                    yield chunk
                log_model_io(model_name, messages, output_text)

            return generator()
        else:
            response = client.chat(**kwargs)
            output_text = response.get('message', {}).get('content', response)
            log_model_io(model_name, messages, output_text)
            return response
    
    def embeddings(self, model: str, prompt: str, host: Optional[str] = None) -> Dict[str, Any]:
        """
        Get embeddings using the single connection with consistent configuration.
        
        Args:
            model: Embedding model name
            prompt: Text to embed
            host: Ollama host URL (optional, defaults to config)
            
        Returns:
            Dict: Response from Ollama embeddings API
        """
        client = self.get_client(host)
        response = client.embeddings(model=model, prompt=prompt)
        log_model_io(model, prompt, response)
        return response
    
    def preload_model(self, model_name: str) -> bool:
        """
        Preload a model to keep it resident in memory for the research pipeline.
        
        Args:
            model_name: Name of the model to preload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._safe_print(f"[OLLAMA] Preloading model to prevent unloading: {model_name}")
            
            # Simple ping to load the model with keep_alive
            self.chat_concurrent_safe(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                options={'keep_alive': '30m'}  # Keep loaded for 30 minutes
            )
            
            self._safe_print(f"[OLLAMA] ✅ Model {model_name} preloaded and will stay resident")
            return True
            
        except Exception as e:
            self._safe_print(f"[OLLAMA] ❌ Failed to preload model {model_name}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection and configuration statistics."""
        with self._lock:
            return {
                'has_connection': self._client is not None,
                'single_connection_mode': True,
                'memory_optimizations': {
                    'max_loaded_models': os.environ.get('OLLAMA_MAX_LOADED_MODELS', 'not_set'),
                    'num_parallel': os.environ.get('OLLAMA_NUM_PARALLEL', 'not_set'),
                    'keep_alive': os.environ.get('OLLAMA_KEEP_ALIVE', 'not_set'),
                    'max_queue': os.environ.get('OLLAMA_MAX_QUEUE', 'not_set')
                }
            }
