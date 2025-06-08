"""
F.R.E.D. Configuration Management
Centralized configuration for all F.R.E.D. components
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration class for F.R.E.D."""
    
    # Flask Configuration
    SECRET_KEY = 'fred_secret_key_2024'
    PORT = int(os.environ.get('PORT', 5000))
    HOST = '0.0.0.0'
    DEBUG = False
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_EMBED_URL = os.getenv('OLLAMA_EMBED_URL', 'http://localhost:11434/api/embeddings')
    OLLAMA_GENERATE_URL = os.getenv('OLLAMA_GENERATE_URL', 'http://localhost:11434/api/generate')
    OLLAMA_TIMEOUT = 20
    
    # Model Configuration
    DEFAULT_MODEL = 'huihui_ai/qwen3-abliterated:8b'
    EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
    LLM_DECISION_MODEL = os.getenv('LLM_DECISION_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M')
    THINKING_MODE_OPTIONS = {"temperature": 0.6, "min_p": 0.0, "top_p": 0.95, "top_k": 20}
    
    # TTS Configuration
    FRED_SPEAKER_WAV_PATH = "new_voice_sample.wav"
    XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    FRED_LANGUAGE = "en"
    TTS_CLEANUP_DELAY = 2  # seconds
    
    # STT Configuration
    STT_SAMPLE_RATE = 16000
    STT_CHANNELS = 1
    STT_BLOCK_DURATION = 5  # seconds
    STT_SILENCE_THRESHOLD = 0.0015
    STT_CALIBRATION_DURATION = 2  # seconds
    STT_SILENCE_DURATION = 0.7  # seconds
    STT_MODEL_SIZE = "medium"
    
    # Database Configuration
    DB_PATH = os.path.join('memory', 'memory.db')
    EMBEDDING_DIM = 768
    AUTO_EDGE_SIMILARITY_CHECK_LIMIT = 3
    
    # Tool Configuration
    MAX_TOOL_ITERATIONS = 5
    WEB_SEARCH_MAX_RESULTS = 3
    WEB_SEARCH_NEWS_MAX_RESULTS = 2
    WEB_SEARCH_TIMEOUT = 60
    
    # Memory Configuration
    MEMORY_SEARCH_LIMIT = 10
    GRAPH_VISUALIZATION_LIMIT = 10
    
    # STM Configuration
    STM_TRIGGER_INTERVAL = 5                    # conversation turns
    STM_ANALYSIS_WINDOW = 15                    # context messages for analysis  
    STM_MAX_MEMORIES = 100                      # total capacity
    STM_SIMILARITY_THRESHOLD = 0.85             # deduplication threshold
    STM_RETRIEVAL_LIMIT = 2                     # results per query
    STM_RETRIEVAL_THRESHOLD = 0.3               # similarity threshold for context retrieval
    STM_ANALYSIS_MODEL = "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"  # configurable model for analysis
    
    # Vision Configuration
    VISION_PROCESSING_INTERVAL = 5              # seconds between vision processing
    VISION_MODEL = "gemma3:4b"                  # multimodal model for vision
    VISION_ENABLED = True                       # enable/disable vision processing
    VISION_FRAME_QUALITY = 0.7                 # JPEG compression quality (0.5-1.0)
    VISION_MAX_DESCRIPTION_LENGTH = 500         # max chars for scene description
    
    # Wake Words and Commands
    WAKE_WORDS = [
        "fred", "hey fred", "okay fred", 
        "hi fred", "excuse me fred", "fred are you there"
    ]
    
    STOP_WORDS = [
        "goodbye", "bye fred", "stop listening", 
        "that's all", "thank you fred", "sleep now"
    ]
    
    ACKNOWLEDGMENTS = [
        "Yes, I'm here.",
        "How can I help?", 
        "I'm listening.",
        "What can I do for you?",
        "At your service."
    ]
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    @classmethod
    def get_db_path(cls, app_root):
        """Get the full database path."""
        return os.path.join(app_root, cls.DB_PATH)
    
    @classmethod
    def get_stt_blocksize(cls):
        """Calculate STT blocksize."""
        return int(cls.STT_BLOCK_DURATION * cls.STT_SAMPLE_RATE)

# Global config instance
config = Config() 