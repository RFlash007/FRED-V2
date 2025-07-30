"""
F.R.E.D. Comprehensive Configuration Management
==============================================

This module contains the complete configuration for F.R.E.D. (Funny Rude Educated Droid),
Ian's personal AI assistant developed by Ollie-Tech. All configurations are organized 
logically by functional area with comprehensive documentation.

Configuration Hierarchy:
1. Core System Configuration (Flask, WebRTC, Security)
2. AI & Model Configuration (Ollama, models, reasoning parameters)
3. Audio Processing Configuration (TTS, STT, voice systems)
4. Data & Memory Configuration (Database, embeddings, knowledge graph)
5. External APIs & Services (Research APIs, web services)
6. Research System Configuration (ARCH/DELVE/VET/SAGE pipeline)
7. Agent System Configuration (Limits, thresholds, timeouts)
8. Hardware Integration (Pi glasses, vision, IoT)
9. User Interface Configuration (Wake words, commands)
10. System Prompts (AI agent instructions organized by function)
11. Utility Classes & Helper Methods

Environment Variables:
- All sensitive data (API keys, tokens) should be set via .env file
- Default values are provided for development but should be overridden in production
- Use .env.example as template for required environment variables
"""

import os
from dotenv import load_dotenv
from prompts import *
from tool_schemas import *
from ollama_manager import OllamaConnectionManager

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration class for F.R.E.D.
    
    This class contains all configuration variables organized by functional area.
    Each section includes detailed documentation explaining the purpose and
    impact of each configuration variable.
    """
    
    # ============================================================================
    # 1. CORE SYSTEM CONFIGURATION
    # ============================================================================
    # Flask web server, WebRTC signaling, security, and networking settings
    
    # --- Flask Web Server Configuration ---
    # Flask serves the main web interface and REST API endpoints
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fred_secret_key_2024_dev_only')
    """
    Flask secret key for session management and CSRF protection.
    SECURITY: Must be set via environment variable in production.
    Used for: Session encryption, form validation, cookie signing
    """
    
    PORT = int(os.environ.get('PORT', 5000))
    """
    Port number for Flask web server.
    Default: 5000 (standard Flask development port)
    Production: Often set to 80 (HTTP) or 443 (HTTPS)
    """
    
    HOST = '0.0.0.0'
    """
    Flask server bind address.
    '0.0.0.0': Binds to all available network interfaces (default)
    '127.0.0.1': Localhost only (more secure for development)
    Affects: Which network interfaces can access the web server
    """
    
    DEBUG = False
    """
    Flask debug mode toggle.
    False: Production mode (default) - secure, no debug info exposed
    True: Development mode - auto-reload, detailed error pages, debug toolbar
    SECURITY: Must be False in production to prevent information disclosure
    """
    
    # --- WebRTC Real-Time Communication Configuration ---
    # WebRTC enables real-time audio/video communication with Pi glasses
    WEBRTC_PORT = int(os.environ.get('WEBRTC_PORT', 8080))
    """
    WebRTC signaling server port.
    Default: 8080 (separate from Flask to avoid conflicts)
    Used for: Real-time communication with Raspberry Pi glasses
    """
    
    WEBRTC_HOST = '0.0.0.0'
    """
    WebRTC server bind address.
    Same as Flask HOST but for WebRTC signaling server.
    Enables Pi glasses to connect from any network location.
    """
    
    ICE_SERVERS = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"}
    ]
    """
    STUN/TURN servers for WebRTC NAT traversal.
    STUN: Helps discover public IP address behind NAT/firewall
    Google STUN servers are free and reliable for most use cases
    For corporate networks, may need private TURN servers
    """
    
    # --- Security & Authentication Configuration ---
    FRED_AUTH_TOKEN = os.environ.get('FRED_AUTH_TOKEN', 'fred_pi_glasses_2024_dev_only')
    """
    Authentication token for Pi glasses communication.
    SECURITY: Must be set via environment variable in production.
    Used for: Authenticating Pi glasses connections to main server
    """
    
    MAX_PI_CONNECTIONS = int(os.environ.get('MAX_PI_CONNECTIONS', 3))
    """
    Maximum simultaneous Pi glasses connections.
    Prevents resource exhaustion from too many concurrent connections.
    Typical use: 1-2 glasses per user, allows some buffer for reconnections
    """
    
    # --- Network Tunneling Configuration ---
    NGROK_ENABLED = os.environ.get('NGROK_ENABLED', 'true').lower() == 'true'
    """
    Enable ngrok tunneling for external access.
    True: Creates secure tunnel for remote access (useful for development)
    False: Local network only (more secure for production)
    Use case: Access F.R.E.D. from outside local network
    """
    
    NGROK_AUTH_TOKEN = os.environ.get('NGROK_AUTH_TOKEN', '')
    """
    ngrok authentication token.
    Required for ngrok tunneling if NGROK_ENABLED=True
    Get from: https://dashboard.ngrok.com/get-started/your-authtoken
    """
    
    # ============================================================================
    # 2. AI & MODEL CONFIGURATION  
    # ============================================================================
    # Ollama server, language models, and AI reasoning parameters
    
    # --- Ollama Server Configuration ---
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    """
    Ollama API server endpoint.
    Default: http://localhost:11434 (standard Ollama port)
    Remote: http://your-server:11434 (for remote Ollama instance)
    Used for: All language model inference and embedding generation
    """
    
    OLLAMA_TIMEOUT = None
    """
    Timeout for Ollama API requests in seconds.
    None: No timeout (recommended for large models that may take time)
    Number: Timeout in seconds (use for production with SLA requirements)
    """
    
    # --- Language Model Assignments ---
    FRED_MODEL = 'hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_XL'
    """
    Primary language model for F.R.E.D.'s personality and general responses.
    This is the main model that defines F.R.E.D.'s personality and conversational style.
    Current: Qwen3-30B-A3B-Instruct-2507 (high-quality, instruction-tuned model)
    Affects: Response quality, personality, and conversational style
    """
    
    DEFAULT_MODEL = FRED_MODEL  # Backward compatibility
    """
    Legacy reference to FRED's personality model.
    Using FRED_MODEL is preferred for clarity.
    """
    
    EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
    """
    Text embedding model for semantic search and memory operations.
    Current: nomic-embed-text (768-dim, good quality/speed balance)
    Alternative: all-minilm (smaller, faster), sentence-transformers (larger)
    Affects: Memory search quality and semantic understanding
    """
    
    DEFAULT_EMBEDDING_MODEL = os.getenv('DEFAULT_EMBEDDING_MODEL', 'nomic-embed-text')
    """
    Default embedding model for agenda system and general embedding tasks.
    Should match EMBED_MODEL for consistency across the system.
    """
    
    LLM_DECISION_MODEL = os.getenv('LLM_DECISION_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL')
    THINKING_MODEL = DEFAULT_MODEL
    """
    Model for complex decision-making and reasoning tasks.
    Should be high-quality model with strong reasoning capabilities.
    Used for: Agent routing, complex analysis, strategic decisions
    """
    
    # --- Consolidated Research Model ---
    CONSOLIDATED_MODEL = os.getenv('CONSOLIDATED_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL')
    """
    Unified model for all research agents (ARCH, DELVE, VET, SAGE) to prevent multiple model loads.
    Using same model with different system prompts for different agent personalities.
    Memory optimization: Single model instance shared across all research components.
    """
    
    # --- AI Reasoning Parameters ---
    THINKING_MODE_OPTIONS = {
        "temperature": 0.6, 
        "min_p": 0.0, 
        "top_p": 0.95, 
        "top_k": 20,
        "num_ctx": 4096,
        "num_thread": 16
    }
    """
    Optimized parameters for Qwen3 thinking mode.
    - temperature: 0.6 (balanced creativity/consistency)
    - min_p: 0.0 (no minimum probability threshold)
    - top_p: 0.95 (nucleus sampling for quality)
    - top_k: 20 (limit consideration to top 20 tokens)
    Source: Official Qwen3 documentation recommendations
    """
    
    # --- Specialized Model Assignments ---
    CRAP_MODEL = 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL'
    """
    Model for C.R.A.P. (Context Retrieval for Augmented Prompts) memory analysis.
    Q4_K_M quantization: Slightly compressed for faster memory operations
    while maintaining quality for context retrieval tasks.
    """
    
    # ============================================================================
    # 3. AUDIO PROCESSING CONFIGURATION
    # ============================================================================
    # Text-to-Speech, Speech-to-Text, voice cloning, and audio processing
    
    # --- Text-to-Speech (TTS) Configuration ---
    FRED_SPEAKER_WAV_PATH = "new_voice_sample.wav"
    """
    Base voice sample file for F.R.E.D.'s TTS voice.
    Should be high-quality recording of desired voice characteristics.
    Affects: Voice quality, speaking style, and personality expression
    """
    
    XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    """
    XTTS v2 model for advanced text-to-speech synthesis.
    Supports: Voice cloning, multilingual, high-quality synthesis
    Alternative: Faster models available but with quality trade-offs
    """
    
    FRED_LANGUAGE = "en"
    """
    Primary language for F.R.E.D.'s speech output.
    en: English (default)
    Supported: Many languages via XTTS v2 multilingual support
    """
    
    TTS_CLEANUP_DELAY = 2
    """
    Seconds to wait before cleaning up temporary TTS audio files.
    Prevents deletion of files still being played or processed.
    Increase if experiencing audio playback issues.
    """
    
    # --- Voice Cloning Configuration (Stewie Griffin) ---
    STEWIE_VOICE_ENABLED = True
    """
    Enable/disable Stewie Griffin voice cloning feature.
    True: F.R.E.D. can use Stewie's voice for humorous responses
    False: Use only default voice (faster, less resource intensive)
    """
    
    STEWIE_VOICE_SAMPLES_DIR = "voice_samples/stewie"
    """
    Directory containing Stewie voice training samples.
    Should contain multiple high-quality WAV files for voice cloning.
    More samples = better voice cloning quality
    """
    
    STEWIE_VOICE_SAMPLES = [
        "stewie_sample_1.wav", "stewie_sample_2.wav", "stewie_sample_3.wav",
        "stewie_sample_4.wav", "stewie_sample_5.wav"
    ]
    """
    Specific Stewie voice sample files for cloning.
    Each file should be clear, noise-free recording of Stewie's voice.
    Recommended: 3-10 seconds each, different emotional tones
    """
    
    STEWIE_VOICE_CLONE_TEMP_DIR = "voice_cloning_temp"
    """Temporary directory for voice cloning intermediate files."""
    
    STEWIE_VOICE_CLONE_MODEL_CACHE = "voice_cloning_cache"
    """Cache directory for downloaded voice cloning models (speeds up processing)."""
    
    STEWIE_VOICE_CLONE_QUALITY = "premium"
    """
    Voice synthesis quality setting.
    Options: ultra_fast, fast, standard, high, premium
    premium: Best quality (slower processing)
    fast: Good quality, faster processing
    """
    
    STEWIE_VOICE_CLONE_SPEED = 0.9
    """
    Speaking speed multiplier for cloned voice.
    1.0: Normal speed
    <1.0: Slower (more deliberate, easier to understand)
    >1.0: Faster (more energetic, may reduce clarity)
    """
    
    STEWIE_VOICE_CLONE_TEMPERATURE = 0.3
    """
    Voice variation/randomness control.
    Lower values (0.1-0.3): More consistent, predictable voice
    Higher values (0.5-0.8): More natural variation, less predictable
    """
    
    STEWIE_VOICE_CLONE_REPETITION_PENALTY = 1.05
    """
    Prevents repetitive speech patterns in voice synthesis.
    Optimal range: 1.0-1.1 for XTTS-v2
    Higher values reduce repetition but may affect naturalness
    """
    
    STEWIE_VOICE_CLONE_LENGTH_PENALTY = 1.0
    """
    Influences length of generated speech segments.
    1.0: Standard length
    >1.0: Encourages longer segments
    <1.0: Encourages shorter segments
    """
    
    STEWIE_VOICE_CLONE_ENABLE_TEXT_SPLITTING = True
    """
    Automatically split long texts for better synthesis quality.
    True: Split long texts into smaller chunks (recommended)
    False: Process entire text at once (may reduce quality for long texts)
    """
    
    # --- Speech-to-Text (STT) Configuration ---
    STT_SAMPLE_RATE = 16000
    """
    Audio sample rate for speech recognition in Hz.
    16000: Standard for Vosk models (good quality/performance balance)
    44100: Higher quality but more processing overhead
    8000: Lower quality, faster processing
    """
    
    STT_CHANNELS = 1
    """
    Number of audio channels for STT processing.
    1: Mono (recommended for speech recognition)
    2: Stereo (not typically needed for speech)
    """
    
    STT_BLOCK_DURATION = 5
    """
    Duration in seconds of audio chunks processed by STT at once.
    Longer blocks: Better accuracy for continuous speech
    Shorter blocks: Lower latency, more responsive
    """
    
    STT_SILENCE_THRESHOLD = 0.002
    """
    General silence detection threshold (amplitude level).
    Lower values: More sensitive (detect quieter speech)
    Higher values: Less sensitive (ignore background noise)
    Adjust based on microphone and environment
    """
    
    STT_PI_SILENCE_THRESHOLD = 0.0015
    """
    Specialized silence threshold for Raspberry Pi audio.
    Pi microphones often have different noise characteristics.
    Typically lower than general threshold due to Pi audio quality
    """
    
    STT_CALIBRATION_DURATION = 3
    """
    Duration in seconds for initial microphone calibration.
    Measures ambient noise to adapt silence detection.
    Longer calibration: Better noise adaptation
    """
    
    STT_SILENCE_DURATION = 0.8
    """
    Duration of continuous silence required to end speech segment.
    Longer duration: Less interruption of natural speech pauses
    Shorter duration: More responsive to quick commands
    """
    
    # --- Vosk Model Configuration ---
    VOSK_MODEL_PATHS = [
        "models/vosk-model-en-us-0.22",              # Large, high-accuracy model (preferred)
        "models/vosk-model-en-us-0.21",              # Alternative large model
        "models/vosk-model-small-en-us-0.15",        # Fallback small model
        "../models/vosk-model-en-us-0.22",           # Parent directory variants
        "../models/vosk-model-en-us-0.21", 
        "../models/vosk-model-small-en-us-0.15",
        "./vosk-model-en-us-0.22",                   # Current directory variants
        "./vosk-model-en-us-0.21",
        "./vosk-model-small-en-us-0.15",
        "/opt/vosk/models/vosk-model-en-us-0.22",    # System-wide installation
        "/opt/vosk/models/vosk-model-en-us-0.21",
        "/opt/vosk/models/vosk-model-small-en-us-0.15"
    ]
    """
    Ordered list of Vosk STT model paths (searched in preference order).
    Large models: Better accuracy, more CPU/memory usage
    Small models: Faster processing, good for real-time applications
    Path search allows flexible deployment scenarios
    """

    VOSK_PI_MODEL_PATHS = [
        "models/vosk-model-small-en-us-0.15",         # Optimized for Pi resources
        "models/vosk-model-en-us-0.22",               # Larger model if Pi can handle it
        "../models/vosk-model-small-en-us-0.15",
        "../models/vosk-model-en-us-0.22",
        "./vosk-model-small-en-us-0.15",
        "./vosk-model-en-us-0.22",
        "/home/raspberry/FRED-V2/pi_client/models/vosk-model-small-en-us-0.15",
        "/opt/vosk/models/vosk-model-small-en-us-0.15"
    ]
    """
    Raspberry Pi specific model paths (prioritizes small models for resource efficiency).
    Pi hardware constraints require careful model selection for real-time performance.
    """
    
    VOSK_ENABLE_WORDS = True
    """Enable word-level timestamps and confidence scores in Vosk results."""
    
    VOSK_ENABLE_PARTIAL_WORDS = True
    """Enable partial word recognition for real-time speech feedback."""
    
    VOSK_LOG_LEVEL = -1
    """Vosk internal logging level (-1 = disabled for cleaner output)."""
    
    # --- Enhanced Speech Processing ---
    STT_MIN_WORD_LENGTH = 2
    """Minimum character length for recognized words to be considered valid."""
    
    STT_MIN_PHRASE_LENGTH = 3
    """Minimum number of words required in a phrase to be processed."""
    
    STT_CONFIDENCE_THRESHOLD = 0.5
    """Minimum confidence score (0.0-1.0) for speech recognition acceptance."""
    
    # ============================================================================
    # 4. DATA & MEMORY CONFIGURATION
    # ============================================================================
    # Database, embeddings, knowledge graph, and memory system settings
    
    # --- Database Configuration ---
    DB_PATH = os.path.join('memory', 'memory.db')
    """
    Path to SQLite database file for F.R.E.D.'s memory and knowledge graph.
    Contains: L2 episodic cache, L3 long-term memory, user preferences, relationships
    Backup recommended: This file contains F.R.E.D.'s learned knowledge
    """
    
    EMBEDDING_DIM = 768
    """
    Dimensionality of text embeddings for semantic search.
    768: Standard for nomic-embed-text model
    Must match embedding model dimensions
    Affects: Memory storage size and search performance
    """
    
    AUTO_EDGE_SIMILARITY_CHECK_LIMIT = 3
    """
    Maximum number of similar nodes to check for automatic knowledge graph edge creation.
    Higher values: More comprehensive relationship detection (slower)
    Lower values: Faster processing, may miss some relationships
    """
    
    # --- Memory System Configuration ---
    MEMORY_SEARCH_LIMIT = 10
    """Default number of memories retrieved in semantic searches."""
    
    GRAPH_VISUALIZATION_LIMIT = 10
    """Maximum nodes to include in knowledge graph visualizations."""
    
    FRED_MAX_CONVERSATION_MESSAGES = 50
    """Maximum conversation history length for F.R.E.D.'s context."""
    
    CRAP_MAX_CONVERSATION_MESSAGES = 10
    """Maximum conversation history for C.R.A.P. memory analysis."""
    
    GATE_MAX_CONVERSATION_MESSAGES = 5
    """Maximum recent history for G.A.T.E. routing analysis."""
    
    # --- L2 Episodic Memory Configuration ---
    L2_TRIGGER_INTERVAL = 5
    """Conversation turns between L2 memory creation checks."""
    
    L2_SIMILARITY_THRESHOLD = 0.6
    """Semantic similarity threshold for detecting topic changes."""
    
    L2_ROLLING_AVERAGE_WINDOW = 6
    """Number of recent turns for topic change detection."""
    
    L2_ANALYSIS_WINDOW = 15
    """Context messages used for L2 memory analysis."""
    
    L2_MAX_MEMORIES = 1000
    """Total capacity of L2 episodic cache."""
    
    L2_CONSOLIDATION_DAYS = 14
    """Days before L2 memories are considered for L3 consolidation."""
    
    L2_RETRIEVAL_LIMIT = 2
    """Maximum L2 memories retrieved per query."""
    
    L2_RETRIEVAL_THRESHOLD = 0.3
    """Semantic similarity threshold for L2 memory retrieval."""
    
    L2_ANALYSIS_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"
    """Model for analyzing conversation segments into L2 memories."""
    
    L2_FALLBACK_TURN_LIMIT = 15
    """Auto-create L2 memory if single topic exceeds this turn limit."""
    
    L2_MIN_CREATION_GAP = 3
    """
    Minimum turns between consecutive L2 memory creations.
    """
    
    # --- MCP-Inspired L3 Memory Enhancement Configuration ---
    # Advanced memory system capabilities inspired by Model Context Protocol standards
    
    MAX_SUBGRAPH_DEPTH = 5
    """
    Maximum depth for subgraph retrieval operations.
    Higher values: More comprehensive relationship mapping (slower, more memory)
    Lower values: Faster queries, may miss distant connections
    Used by: get_subgraph() for advanced memory network analysis
    """
    
    MAX_SUBGRAPH_NODES = 100
    """
    Maximum nodes in a subgraph to prevent memory issues.
    Prevents overwhelming responses and system resource exhaustion.
    Balances comprehensive analysis with performance constraints.
    """
    
    MAX_SEARCH_LIMIT = 50
    """
    Maximum search results to prevent overwhelming responses.
    Caps advanced search operations while maintaining usability.
    Higher than standard limit for comprehensive research queries.
    """
    
    RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.7
    """
    Minimum confidence for auto-created relationships (0.0-1.0).
    Higher values: More conservative, fewer but higher-quality relationships
    Lower values: More relationships detected, may include some false positives
    Used by: Enhanced relationship discovery system
    """
    
    ADVANCED_SEARCH_DEFAULT_LIMIT = 10
    """
    Default search result limit for advanced search operations.
    Balances comprehensive results with response time.
    Can be overridden per query up to MAX_SEARCH_LIMIT.
    """
    
    RELATIONSHIP_DISCOVERY_CONTEXT_WINDOW = 5
    """
    Default number of related nodes to consider for relationship context.
    Larger windows: Better relationship accuracy (more processing)
    Smaller windows: Faster analysis, may miss contextual relationships
    """
    
    OBSERVATION_METADATA_ENABLED = True
    """
    Enable MCP-style observation and metadata support in memory nodes.
    True: Allows structured observations and metadata attachment
    False: Standard memory node creation only
    Enhances memory richness and searchability.
    """
    
    L2_MIN_CHUNK_SIZE = 4
    """Minimum messages required to form L2 processing chunk."""
    
    # ============================================================================
    # 5. EXTERNAL APIS & SERVICES
    # ============================================================================
    # Research APIs, web services, and external integrations
    
    # --- Basic Tool Configuration ---
    MAX_TOOL_ITERATIONS = 5
    """Maximum consecutive tool calls in a single reasoning chain."""
    
    CRAP_MAX_TOOL_ITERATIONS = 3
    """Maximum tool iterations for C.R.A.P. memory analysis."""
    
    WEB_SEARCH_MAX_RESULTS = 3
    """Maximum web search results per query."""
    
    WEB_SEARCH_NEWS_MAX_RESULTS = 2
    """Maximum news search results per query."""
    
    WEB_SEARCH_TIMEOUT = 120
    """Timeout in seconds for web search requests."""
    
    # --- Research Source Limits ---
    # Optimized for quality over quantity in research operations
    RESEARCH_MAX_ACADEMIC_PAPERS = 6
    """Maximum academic papers to retrieve per research query."""
    
    RESEARCH_MAX_WEB_ARTICLES = 8
    """Maximum web articles from general search per query."""
    
    RESEARCH_MAX_NEWS_ARTICLES = 4
    """Maximum news articles per research query."""
    
    RESEARCH_MAX_FORUM_POSTS = 5
    """Maximum forum posts (Reddit, Stack Overflow) per query."""
    
    RESEARCH_MAX_SOCIAL_POSTS = 5
    """Maximum social media posts per query."""
    
    RESEARCH_MAX_VIDEO_TRANSCRIPTS = 3
    """Maximum video transcripts from YouTube per query."""
    
    RESEARCH_MAX_DOCUMENTATION = 3
    """Maximum documentation links per query."""
    
    RESEARCH_MAX_LINKS_PER_CATEGORY = 5
    """Maximum links to extract per category for each search."""
    
    # --- YouTube Data API Configuration ---
    YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyAG1DQIrpE2nUbzOzYJ8_oS7f8AcU4Thvc')
    """
    YouTube Data API v3 key for video transcript access.
    Get from: https://console.developers.google.com/
    1. Create/select project
    2. Enable "YouTube Data API v3"
    3. Create API Key credentials
    Free tier: 10,000 quota units/day
    """
    
    YOUTUBE_API_QUOTA_LIMIT = 10000
    """Daily quota units for YouTube API (Google default limit)."""
    
    # --- Reddit API Configuration ---
    REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID', 'qbap3jhtPoKF-jwrl-4HMA')
    """
    Reddit API client ID for forum/social content access.
    Get from: https://www.reddit.com/prefs/apps
    1. Create "script" type app
    2. Copy client ID and secret
    """
    
    REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET', 'UDN2ZasREkEfEtFbvGHhqgGKBMOIuA')
    """Reddit API client secret (pairs with client ID)."""
    
    REDDIT_USER_AGENT = 'F.R.E.D. Research Assistant v2.0'
    """User agent string for Reddit API requests."""
    
    REDDIT_REQUEST_LIMIT = 100
    """Requests per minute limit for Reddit API."""
    
    # --- Stack Overflow API Configuration ---
    STACKOVERFLOW_API_KEY = os.environ.get('STACKOVERFLOW_API_KEY', '')
    """
    Stack Overflow API key (optional, increases rate limits).
    Get from: https://stackapps.com/apps/oauth/register
    Without key: 300 requests/day
    With key: 10,000 requests/day
    """
    
    STACKOVERFLOW_REQUEST_LIMIT = 300
    """Default rate limit per day for Stack Overflow API."""
    
    # --- Jina AI Reader API Configuration ---
    JINA_API_KEY = os.environ.get('JINA_API_KEY', '')
    """
    Jina AI Reader API for webpage content extraction.
    Get from: https://jina.ai/reader/
    Free tier: 1,000 requests/month
    Alternative: Can use without key (public endpoint) with rate limits
    """
    
    JINA_REQUEST_LIMIT = 1000
    """Monthly request limit for Jina AI free tier."""
    
    # --- Academic Research APIs (Free, No Keys Required) ---
    ARXIV_API_BASE = 'http://export.arxiv.org/api/query'
    """ArXiv API endpoint for academic paper search."""
    
    SEMANTIC_SCHOLAR_API_BASE = 'https://api.semanticscholar.org/graph/v1'
    """Semantic Scholar API for academic research papers."""
    
    PUBMED_API_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
    """PubMed API for medical/biological research papers."""
    
    # --- Brave Search API Configuration ---
    BRAVE_SEARCH_API_KEY = os.environ.get('BRAVE_SEARCH_API_KEY', 'BSAXrrHvaG5TWPIaJeCGJXhxVRXGuRH')
    """
    Brave Search API for independent web search.
    Get from: https://api.search.brave.com/
    Free tier: 2,000 queries/month, 1 query/second
    Upgrade: $3/1000 additional queries
    """
    
    BRAVE_SEARCH_API_URL = 'https://api.search.brave.com/res/v1/web/search'
    """Brave web search endpoint."""
    
    BRAVE_NEWS_API_URL = 'https://api.search.brave.com/res/v1/news/search'
    """Brave news search endpoint."""
    
    BRAVE_SEARCH_REQUEST_LIMIT = 2000
    """Monthly request limit for Brave Search free tier."""
    
    # --- SearchAPI.io Configuration ---
    SEARCHAPI_API_KEY = os.environ.get('SEARCHAPI_API_KEY', 'UpyF9L72UJ1xQw1tvVCVsUdv')
    """
    SearchAPI.io proxy for DuckDuckGo and other search engines.
    Get from: https://www.searchapi.io/
    Free tier: 100 searches/month
    Helps avoid rate limits on direct search engine access
    """
    
    SEARCHAPI_BASE_URL = 'https://www.searchapi.io/api/v1/search'
    """SearchAPI.io endpoint for proxied searches."""
    
    SEARCHAPI_REQUEST_LIMIT = 100
    """Monthly request limit for SearchAPI.io free tier."""
    
    # --- News API Configuration ---
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '6ef0134a37c64189aa9cda119fc8f1a1')
    """
    NewsAPI.org key for current news access.
    Get from: https://newsapi.org/register
    Free tier: 100 requests/day, 1000/month
    """
    
    NEWS_API_REQUEST_LIMIT = 100
    """Daily request limit for NewsAPI.org free tier."""
    
    # --- Research System Behavior Controls ---
    RESEARCH_ENABLE_PARALLEL_CORE = True
    """Enable parallel execution of core research sources (faster)."""
    
    RESEARCH_ENABLE_SOCIAL_SOURCES = True
    """Enable social media/forum searches (Reddit, forums)."""
    
    RESEARCH_ENABLE_VIDEO_TRANSCRIPTS = True
    """Enable YouTube transcript searches."""
    
    RESEARCH_REQUEST_DELAY = 1
    """Delay in seconds between sequential social source requests."""
    
    RESEARCH_CORE_TIMEOUT = 120
    """Timeout in seconds for core research sources."""
    
    RESEARCH_SOCIAL_TIMEOUT = 120
    """Timeout in seconds for social research sources."""
    
    # ============================================================================
    # 6. RESEARCH SYSTEM CONFIGURATION
    # ============================================================================
    # ARCH/DELVE/VET/SAGE research pipeline and enhanced research settings
    
    # --- Legacy Research System ---
    ARCH_DELVE_MAX_RESEARCH_ITERATIONS = 20
    """Maximum conversation turns in ARCH/DELVE research loop."""
    
    ARCH_DELVE_MAX_CONVERSATION_MESSAGES = 5
    """Maximum messages kept in ARCH/DELVE context memory."""
    
    ARCH_DELVE_CONVERSATION_STORAGE_PATH = "memory/agenda_conversations"
    """Directory for storing research conversation logs."""
    
    # --- Model Assignments for Research Components ---
    ARCH_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"
    """Model for A.R.C.H. (research direction and strategic thinking)."""
    
    DELVE_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"
    """Model for D.E.L.V.E. (research execution and analysis)."""
    
    SAGE_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"
    """Model for S.A.G.E. (synthesis and memory optimization)."""
    
    # --- Enhanced Research System Configuration ---
    ENHANCED_RESEARCH_MODELS = {
        'arch': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL',      # Strategic planning
        'delve': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL',     # Data gathering  
        'vet': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL',       # Quality assessment
        'sage': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL'       # Final synthesis
    }
    """
    Model assignments for enhanced research pipeline components.
    All use same high-quality model for consistency and performance.
    Could be optimized with smaller models for DELVE if needed.
    """
    
    ENHANCED_RESEARCH_CONFIG = {
        'max_iterations': 6,                    # Maximum research iterations
        'max_delve_sub_iterations': 8,          # Maximum DELVE tool iterations
        'citation_deduplication_threshold': 0.95,  # Similarity threshold for duplicate URLs
        'consensus_high_threshold': 0.7,        # High confidence consensus threshold
        'consensus_medium_threshold': 0.4,      # Medium confidence consensus threshold
        'contradiction_tolerance': 2,           # Maximum contradictions before low confidence
        'fresh_context_enabled': True,          # Enable fresh context for DELVE/VET
        'rag_database_enabled': True,           # Enable RAG database for SAGE
        'truth_determination_enabled': True     # Enable truth determination algorithm
    }
    """
    Configuration parameters for enhanced research system with fresh context,
    global citation management, and advanced truth determination.
    """
    
    QUALITY_ASSESSMENT_CONFIG = {
        'credibility_weights': {'high': 3, 'medium': 2, 'low': 1},  # Source reliability weights
        'source_diversity_target': 0.6,        # Target mix of source types
        'data_balance_target': 0.5,            # Target quantitative/qualitative balance
        'contradiction_indicators': [          # Keywords indicating contradictions
            'contradicts', 'disagrees', 'conflicting', 'however', 'but', 
            'although', 'different from', 'opposes', 'disputes'
        ]
    }
    """
    Quality assessment framework configuration for multi-layer research evaluation.
    Used by Phase 4 quality assessment system for comprehensive confidence scoring.
    """
    
    # --- Specialized Agent Models ---
    GIST_SUMMARY_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"
    """Model for G.I.S.T. (Global Information Sanitation Tool) content filtering."""
    
    REFLEX_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_XL"
    """Model for R.E.F.L.E.X. (Research Executive For Learning EXtraction) processing."""
    
    # ============================================================================
    # 7. AGENT SYSTEM CONFIGURATION
    # ============================================================================
    # Agent limits, thresholds, timeouts, and system behavior
    
    # --- General Agent Configuration ---
    USE_MAD = True
    """
    Master toggle for the M.A.D. (Memory Addition Daemon) agent.
    True: M.A.D. runs in parallel to analyze conversation turns for new memories.
    False: Disables the M.A.D. agent completely.
    """
    
    MAX_CONCURRENT_AGENTS = 1
    """Maximum number of agents that can run simultaneously."""
    
    AGENT_ERRORS = {
        "memory_failure": "My memory isn't working right now.",
        "search_failure": "The web search failed.",
        "reminder_failure": "The reminder system is unavailable.",
        "pi_tools_failure": "Pi tools are not responding.",
        "synthesis_failure": "Context synthesis failed."
    }
    """Standardized error messages shown to users when agent operations fail."""
    
    # --- Agent-Specific Thresholds ---
    SCOUT_CONFIDENCE_THRESHOLD = 70
    """S.C.O.U.T. confidence threshold - below this triggers deep research escalation."""
    
    L2_RETRIEVAL_THRESHOLD = 0.6
    """Minimum similarity threshold for L2 memory retrieval."""
    
    SYNAPSE_MAX_BULLETS = 8
    """Maximum bullet points in S.Y.N.A.P.S.E. neural processing core output."""
    
    # --- Agenda System Configuration ---
    AGENDA_PRIORITY_IMPORTANT = 1
    """Priority level for important agenda tasks (lower = higher priority)."""
    
    AGENDA_PRIORITY_NORMAL = 2
    """Priority level for normal agenda tasks."""
    
    AGENDA_MAX_CONCURRENT_TASKS = 10
    """Maximum number of pending tasks in agenda system."""
    
    # --- Sleep Cycle Configuration ---
    SLEEP_CYCLE_BLOCKING = True
    """Whether F.R.E.D.'s main loop blocks during sleep cycle processing."""
    
    SLEEP_CYCLE_MAX_AGENDA_TASKS = 5
    """Maximum agenda tasks to process in single sleep cycle."""
    
    SLEEP_CYCLE_L2_CONSOLIDATION_BATCH = 10
    """Number of L2 memories to consolidate into L3 per sleep cycle."""
    
    SLEEP_CYCLE_MESSAGE = "Initiating sleep cycle... (processing offline tasks)"
    """Status message displayed when F.R.E.D. enters sleep cycle."""
    
    # ============================================================================
    # 8. HARDWARE INTEGRATION
    # ============================================================================
    # Pi glasses, vision processing, and IoT device configuration
    
    # --- Vision Processing Configuration ---
    VISION_PROCESSING_INTERVAL = 10
    """Time interval in seconds between visual processing cycles."""
    
    VISION_MODEL = "qwen2.5vl:7b"
    """Multimodal model for analyzing visual input from smart glasses."""
    
    VISION_ENABLED = True
    """Enable/disable visual processing system."""
    
    VISION_FRAME_QUALITY = 1.0
    """JPEG compression quality for vision frames (1.0 = highest quality)."""
    
    VISION_MAX_DESCRIPTION_LENGTH = 0
    """Maximum character length for scene descriptions (0 = unlimited)."""
    
    VISION_RESOLUTION = 3584
    """Target resolution for visual input frames (optimized for Qwen 2.5-VL)."""
    
    # --- Pi Glasses Configuration ---
    PI_HEARTBEAT_INTERVAL = 30
    """Interval in seconds for Pi client heartbeat signals."""
    
    PI_CONNECTION_TIMEOUT = 60
    """Timeout in seconds for Pi connection before considered disconnected."""
    
    PI_RECONNECT_MAX_RETRIES = 5
    """Maximum reconnection attempts for Pi client."""
    
    PI_RECONNECT_BACKOFF_MAX = 30
    """Maximum delay in seconds for exponential backoff during reconnection."""
    
    # ============================================================================
    # 9. USER INTERFACE CONFIGURATION
    # ============================================================================
    # Wake words, commands, responses, and user interaction settings
    
    # --- Wake Words & Commands ---
    WAKE_WORDS = [
        "fred", "hey fred", "okay fred", 
        "hi fred", "excuse me fred", "fred are you there"
    ]
    """Keywords/phrases that activate F.R.E.D.'s listening mode."""
    
    STOP_WORDS = [
        "goodbye", "bye fred", "stop listening", 
        "that's all", "thank you fred", "sleep now"
    ]
    """Keywords/phrases that signal F.R.E.D. to stop listening or end conversation."""
    
    ACKNOWLEDGMENTS = [
        "Yes, I'm here.",
        "How can I help?", 
        "I'm listening.",
        "What can I do for you?",
        "At your service."
    ]
    """Predefined responses F.R.E.D. uses to acknowledge wake words."""
    
    # --- Reminder System Configuration ---
    REMINDER_KEYWORDS = [
        "remind me", "schedule", "appointment", "meeting", "deadline",
        "tomorrow", "next week", "later", "don't forget", "remember to"
    ]
    """Keywords that trigger reminder system activation."""
    
    REMINDER_ACKNOWLEDGMENT_PHRASES = [
        "thanks", "got it", "okay", "ok", "sure", "alright", "understood",
        "will do", "noted", "roger", "copy that"
    ]
    """Phrases that confirm reminder acceptance."""
    
    # --- Logging Configuration ---
    LOG_LEVEL = 'INFO'
    """Minimum logging level for console and file output."""
    
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    """Format string for log messages."""
    
    # ============================================================================
    # 11. UTILITY CLASSES & HELPER METHODS
    # ============================================================================

    @classmethod
    def get_db_path(cls, app_root):
        """Get the full database path."""
        return os.path.join(app_root, cls.DB_PATH)
    
    @classmethod
    def get_stt_blocksize(cls):
        """Calculate STT blocksize based on sample rate and duration."""
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


# Global Ollama connection manager instance
ollama_manager = OllamaConnectionManager(Config.OLLAMA_BASE_URL, Config.THINKING_MODE_OPTIONS)

# Global config instance
config = Config()

# Make CRAP_MODEL available for direct import
CRAP_MODEL = Config.CRAP_MODEL

# Re-attach prompts and tools to the Config class to maintain the config.VARIABLE access pattern
config.FRED_SYSTEM_PROMPT = FRED_SYSTEM_PROMPT
config.GATE_SYSTEM_PROMPT = GATE_SYSTEM_PROMPT
config.GATE_USER_PROMPT = GATE_USER_PROMPT
config.ARCH_SYSTEM_PROMPT = ARCH_SYSTEM_PROMPT
config.ARCH_TASK_PROMPT = ARCH_TASK_PROMPT
config.DELVE_SYSTEM_PROMPT = DELVE_SYSTEM_PROMPT
config.VET_SYSTEM_PROMPT = VET_SYSTEM_PROMPT
config.SAGE_FINAL_REPORT_SYSTEM_PROMPT = SAGE_FINAL_REPORT_SYSTEM_PROMPT
config.SAGE_FINAL_REPORT_USER_PROMPT = SAGE_FINAL_REPORT_USER_PROMPT
config.SAGE_L3_MEMORY_SYSTEM_PROMPT = SAGE_L3_MEMORY_SYSTEM_PROMPT
config.SAGE_L3_MEMORY_USER_PROMPT = SAGE_L3_MEMORY_USER_PROMPT
config.GIST_SYSTEM_PROMPT = GIST_SYSTEM_PROMPT
config.GIST_USER_PROMPT = GIST_USER_PROMPT
config.REFLEX_SYSTEM_PROMPT = REFLEX_SYSTEM_PROMPT
config.REFLEX_USER_PROMPT = REFLEX_USER_PROMPT
config.SYNAPSE_SYSTEM_PROMPT = SYNAPSE_SYSTEM_PROMPT
config.SCOUT_CONFIDENCE_PROMPT = SCOUT_CONFIDENCE_PROMPT
config.VISION_SYSTEM_PROMPT = VISION_SYSTEM_PROMPT
config.VISION_USER_PROMPT = VISION_USER_PROMPT
config.CRAP_SYSTEM_PROMPT = CRAP_SYSTEM_PROMPT
config.CRAP_USER_PROMPT = CRAP_USER_PROMPT
config.CRAP_MODEL = Config.CRAP_MODEL

config.MEMORY_TOOLS = MEMORY_TOOLS
config.CRAP_TOOLS = CRAP_TOOLS
config.RESEARCH_TOOLS = RESEARCH_TOOLS
config.AGENT_MANAGEMENT_TOOLS = AGENT_MANAGEMENT_TOOLS
config.PIPELINE_CONTROL_TOOLS = PIPELINE_CONTROL_TOOLS
config.UTILITY_TOOLS = UTILITY_TOOLS
config.FRED_TOOLS = FRED_TOOLS
config.DELVE_TOOLS = DELVE_TOOLS
config.ARCH_TOOLS = ARCH_TOOLS
config.AVAILABLE_TOOLS = AVAILABLE_TOOLS