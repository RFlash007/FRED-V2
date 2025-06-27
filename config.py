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
    
    # WebRTC Configuration
    WEBRTC_PORT = int(os.environ.get('WEBRTC_PORT', 8080))
    WEBRTC_HOST = '0.0.0.0'
    
    # Security Configuration
    FRED_AUTH_TOKEN = os.environ.get('FRED_AUTH_TOKEN', 'fred_pi_glasses_2024')
    MAX_PI_CONNECTIONS = int(os.environ.get('MAX_PI_CONNECTIONS', 3))
    
    # ngrok Configuration
    NGROK_ENABLED = os.environ.get('NGROK_ENABLED', 'true').lower() == 'true'
    NGROK_AUTH_TOKEN = os.environ.get('NGROK_AUTH_TOKEN', '2yCKXFFreg1EEaQK6RGb3Kbdt6f_4owEF1Xji51DMheaKDV5U')
    
    # ICE/STUN Configuration for WebRTC
    ICE_SERVERS = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"}
    ]
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_EMBED_URL = os.getenv('OLLAMA_EMBED_URL', 'http://localhost:11434/api/embeddings')
    OLLAMA_GENERATE_URL = os.getenv('OLLAMA_GENERATE_URL', 'http://localhost:11434/api/generate')
    OLLAMA_TIMEOUT = None  # No timeout - let operations complete
    
    # Model Configuration
    DEFAULT_MODEL = 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M'
    EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
    LLM_DECISION_MODEL = os.getenv('LLM_DECISION_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M')
    THINKING_MODE_OPTIONS = {"temperature": 0.6, "min_p": 0.0, "top_p": 0.95, "top_k": 20}
    
    # TTS Configuration
    FRED_SPEAKER_WAV_PATH = "new_voice_sample.wav"
    XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    FRED_LANGUAGE = "en"
    TTS_CLEANUP_DELAY = 2  # seconds
    
    # Voice Cloning Configuration for Stewie
    STEWIE_VOICE_ENABLED = True
    STEWIE_VOICE_SAMPLES_DIR = "voice_samples/stewie"  # Directory containing Stewie voice clips
    STEWIE_VOICE_SAMPLES = [
        "stewie_sample_1.wav",
        "stewie_sample_2.wav", 
        "stewie_sample_3.wav",
        "stewie_sample_4.wav",
        "stewie_sample_5.wav"
    ]
    STEWIE_VOICE_CLONE_TEMP_DIR = "voice_cloning_temp"  # Temporary directory for processing
    STEWIE_VOICE_CLONE_MODEL_CACHE = "voice_cloning_cache"  # Cache directory for voice model
    STEWIE_VOICE_CLONE_QUALITY = "premium"  # Higher quality synthesis (options: ultra_fast, fast, standard, high, premium)
    STEWIE_VOICE_CLONE_SPEED = 0.9  # Slightly slower for natural pacing
    STEWIE_VOICE_CLONE_TEMPERATURE = 0.3  # Balanced variation for naturalness
    STEWIE_VOICE_CLONE_REPETITION_PENALTY = 1.05  # Optimal for XTTS-v2 (1.0-1.1 range)
    STEWIE_VOICE_CLONE_LENGTH_PENALTY = 1.0  # Controls output length consistency
    STEWIE_VOICE_CLONE_ENABLE_TEXT_SPLITTING = True  # Better handling of longer texts
    
    # STT Configuration (Vosk-based)
    STT_SAMPLE_RATE = 16000
    STT_CHANNELS = 1
    STT_BLOCK_DURATION = 5  # Audio processing block duration
    STT_SILENCE_THRESHOLD = 0.002   # Increased for better noise rejection
    # Separate default VAD threshold for Raspberry Pi glasses audio (mono, 16 kHz)
    STT_PI_SILENCE_THRESHOLD = 0.0015  # Slightly higher for Pi noise rejection
    STT_CALIBRATION_DURATION = 3  # seconds - longer calibration for better accuracy
    STT_SILENCE_DURATION = 0.8  # seconds - slightly longer pause detection
    
    # Vosk Model Configuration
    VOSK_MODEL_PATHS = [
        "models/vosk-model-en-us-0.22",              # Large, accurate model (preferred for main server)
        "models/vosk-model-en-us-0.21",              # Alternative large model
        "models/vosk-model-small-en-us-0.15",        # Fallback to small model
        "../models/vosk-model-en-us-0.22",
        "../models/vosk-model-en-us-0.21", 
        "../models/vosk-model-small-en-us-0.15",
        "./vosk-model-en-us-0.22", 
        "./vosk-model-en-us-0.21",
        "./vosk-model-small-en-us-0.15",
        "/opt/vosk/models/vosk-model-en-us-0.22",
        "/opt/vosk/models/vosk-model-en-us-0.21",
        "/opt/vosk/models/vosk-model-small-en-us-0.15"
    ]

    # Pi Client Specific Model Paths (prioritize small model for resource efficiency)
    VOSK_PI_MODEL_PATHS = [
        "models/vosk-model-small-en-us-0.15",         # Optimized for Pi resources (preferred)
        "models/vosk-model-en-us-0.22",               # Larger model if Pi has sufficient resources
        "../models/vosk-model-small-en-us-0.15",
        "../models/vosk-model-en-us-0.22",
        "./vosk-model-small-en-us-0.15",
        "./vosk-model-en-us-0.22",
        "/home/raspberry/FRED-V2/pi_client/models/vosk-model-small-en-us-0.15",
        "/opt/vosk/models/vosk-model-small-en-us-0.15"
    ]
    
    VOSK_ENABLE_WORDS = True        # Enable word-level results for better accuracy
    VOSK_ENABLE_PARTIAL_WORDS = True # Enable partial word results for responsiveness
    VOSK_LOG_LEVEL = -1             # Disable Vosk logging (-1 = silent)
    
    # Enhanced Speech Processing Settings
    STT_MIN_WORD_LENGTH = 2         # Minimum word length to consider
    STT_MIN_PHRASE_LENGTH = 3       # Minimum phrase length to process
    STT_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for accepting results
    
    # Database Configuration
    DB_PATH = os.path.join('memory', 'memory.db')
    EMBEDDING_DIM = 768
    AUTO_EDGE_SIMILARITY_CHECK_LIMIT = 3
    
    # Tool Configuration
    MAX_TOOL_ITERATIONS = 5
    WEB_SEARCH_MAX_RESULTS = 3
    WEB_SEARCH_NEWS_MAX_RESULTS = 2
    WEB_SEARCH_TIMEOUT = None  # No timeout for web searches
    
    # Memory Configuration
    MEMORY_SEARCH_LIMIT = 10
    GRAPH_VISUALIZATION_LIMIT = 10
    MAX_CONVERSATION_MESSAGES = 50              # Maximum messages in conversation history
    
    # STM Configuration
    STM_TRIGGER_INTERVAL = 5                    # conversation turns
    STM_ANALYSIS_WINDOW = 15                    # context messages for analysis  
    STM_MAX_MEMORIES = 100                      # total capacity
    STM_SIMILARITY_THRESHOLD = 0.85             # deduplication threshold
    STM_RETRIEVAL_LIMIT = 2                     # results per query
    STM_RETRIEVAL_THRESHOLD = 0.3               # similarity threshold for context retrieval
    STM_ANALYSIS_MODEL = "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"  # configurable model for analysis
    
    # Vision Configuration
    VISION_PROCESSING_INTERVAL = 10              # seconds between vision processing
    VISION_MODEL = "qwen2.5vl:7b"               # multimodal model for vision (3584x3584 maximum)
    VISION_ENABLED = True                       # enable/disable vision processing
    VISION_FRAME_QUALITY = 1.0                 # JPEG compression quality - Maximum for Qwen 2.5-VL (3584x3584 maximum)
    VISION_MAX_DESCRIPTION_LENGTH = 0           # max chars for scene description (0 = unlimited)
    VISION_RESOLUTION = 3584                    # maximum resolution for Qwen 2.5-VL (3584x3584 = 12.8 MP)
    
    # Pi Glasses Configuration
    PI_HEARTBEAT_INTERVAL = 30                  # seconds between heartbeats
    PI_CONNECTION_TIMEOUT = 60                  # seconds before considering Pi disconnected
    PI_RECONNECT_MAX_RETRIES = 5               # maximum reconnection attempts
    PI_RECONNECT_BACKOFF_MAX = 30              # maximum backoff time in seconds
    
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
    
    @classmethod
    def get_webrtc_config(cls):
        """Get WebRTC configuration dictionary."""
        return {
            'host': cls.WEBRTC_HOST,
            'port': cls.WEBRTC_PORT,
            'auth_token': cls.FRED_AUTH_TOKEN,
            'max_connections': cls.MAX_PI_CONNECTIONS,
            'ice_servers': cls.ICE_SERVERS
        }
    
    @classmethod
    def get_ngrok_config(cls):
        """Get ngrok configuration dictionary."""
        return {
            'enabled': cls.NGROK_ENABLED,
            'auth_token': cls.NGROK_AUTH_TOKEN,
            'port': cls.WEBRTC_PORT
        }

# Global config instance
config = Config() 