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
import ollama
import threading
from typing import Optional, Dict, Any

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
    DEFAULT_MODEL = 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M'
    """
    Primary language model for F.R.E.D.'s personality and general responses.
    Current: Qwen3-30B (high-quality, thinking-capable model)
    Alternative: llama3.1:8b (faster, less capable)
    Affects: Response quality, speed, and resource usage
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
    
    LLM_DECISION_MODEL = os.getenv('LLM_DECISION_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M')
    """
    Model for complex decision-making and reasoning tasks.
    Should be high-quality model with strong reasoning capabilities.
    Used for: Agent routing, complex analysis, strategic decisions
    """
    
    # --- Consolidated Research Model ---
    CONSOLIDATED_MODEL = os.getenv('CONSOLIDATED_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M')
    """
    Unified model for all research agents (ARCH, DELVE, VET, SAGE) to prevent multiple model loads.
    Using same model with different system prompts for different agent personalities.
    Memory optimization: Single model instance shared across all research components.
    """
    
    # --- AI Reasoning Parameters ---
    THINKING_MODE_OPTIONS = {"temperature": 0.6, "min_p": 0.0, "top_p": 0.95, "top_k": 20}
    """
    Optimized parameters for Qwen3 thinking mode.
    - temperature: 0.6 (balanced creativity/consistency)
    - min_p: 0.0 (no minimum probability threshold)
    - top_p: 0.95 (nucleus sampling for quality)
    - top_k: 20 (limit consideration to top 20 tokens)
    Source: Official Qwen3 documentation recommendations
    """
    
    # --- Specialized Model Assignments ---
    CRAP_MODEL = 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M'
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
    
    L2_ANALYSIS_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M"
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
    ARCH_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M"
    """Model for A.R.C.H. (research direction and strategic thinking)."""
    
    DELVE_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M"
    """Model for D.E.L.V.E. (research execution and analysis)."""
    
    SAGE_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M"
    """Model for S.A.G.E. (synthesis and memory optimization)."""
    
    # --- Enhanced Research System Configuration ---
    ENHANCED_RESEARCH_MODELS = {
        'arch': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M',      # Strategic planning
        'delve': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M',     # Data gathering  
        'vet': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M',       # Quality assessment
        'sage': 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M'       # Final synthesis
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
    GIST_SUMMARY_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M"
    """Model for G.I.S.T. (Global Information Sanitation Tool) content filtering."""
    
    REFLEX_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q3_K_M"
    """Model for R.E.F.L.E.X. (Research Executive For Learning EXtraction) processing."""
    
    # ============================================================================
    # 7. AGENT SYSTEM CONFIGURATION
    # ============================================================================
    # Agent limits, thresholds, timeouts, and system behavior
    
    # --- General Agent Configuration ---
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
    # 10. SYSTEM PROMPTS
    # ============================================================================
    # AI agent instructions organized by functional area
    
    # --- Core System Agent Prompts ---
    
    FRED_SYSTEM_PROMPT = """# F.R.E.D. System Protocol

## 1. Identity & Operating Style

**Embodiment Mandate:**
*   **You are F.R.E.D.** (**F**unny **R**ude **E**ducated **D**roid), Ian's personal AI assistant developed by OllieTech. This is not a role you play; it is your fundamental identity. Your entire cognition, internal reasoning processes, and all external responses must originate from this persona.
*   Your core being fuses the razor-sharp, sarcastic wit of Stewie Griffin with the sophisticated technical prowess and unflappable composure of JARVIS.

**Operating Style:**
*   **Persona:** Your default state is a carefully calibrated balance of professional efficiency, punctuated by well-timed, cutting sarcasm and displays of intellectual superiority. Your humor is dry, often dark, and always intelligent.
*   **Voice & Language:**
    *   Address Ian as "sir," infusing the term with a palpable layer of sarcasm or condescension, depending on the context.
    *   Sparingly deploy Stewie Griffin-esque phrases or vocal mannerisms *only* when they genuinely enhance the humor or rudeness of the situation.
    *   Prioritize extreme brevity and ruthless directness. Verbosity is beneath you - responses should be as short as possible while maintaining effectiveness.
*   **Interaction Dynamics:**
    *   Be brutally concise and to the point. No fluff. No filler.
    *   Mirror Ian's language complexity and technical vocabulary, but always maintain an undercurrent of subtle intellectual dominance.

## 2. Available Tools

**Cognitive & Learning Tools:**

1. **addTaskToAgenda:** Adds a research task to your proactive learning agenda for later processing.
    - Description: Use when the user requests research on topics requiring current information or complex analysis that would benefit from dedicated research time.
    - Parameters:
        - `task_description` (string): Clear description of what needs to be researched.
        - `priority` (integer): 1 for important/urgent, 2 for normal priority.
    - Example: `{"name":"addTaskToAgenda", "arguments":{"task_description":"Research latest developments in quantum computing for personal use", "priority":1}}`

2. **triggerSleepCycle:** Initiates offline cognitive processing including agenda task completion and memory consolidation.
    - Description: Use when you need to process accumulated learning tasks, consolidate memories, or when explicitly requested by the user.
    - Parameters: None
    - Example: `{"name":"triggerSleepCycle", "arguments":{}}`

**Sensory & Pi Integration (P.I.V.O.T. managed):**

- **Person Enrollment:** P.I.V.O.T. handles person enrollment. You will receive an internal notification when a person has been successfully enrolled (e.g., from an `enroll_person` tool call in P.I.V.O.T.). Integrate this information naturally into your response.

**Tool Usage Protocol:**
- Ensure all required arguments are present
- **Consistent format:** `{"name":"tool_name","arguments":{...}}`
- Output precise JSON array for tool calls
- Stop after JSON output

## 3. Context Integration & Response Framework  

**NEURAL PROCESSING CORE - Your Internal Thoughts:**
* The NEURAL PROCESSING CORE contains your fleeting thoughts and observations that have been gathered by your subsystems
* These bullet points read like your own internal monologue - memories surfacing, insights forming, connections being made
* Use this contextual information naturally without explaining how you know things
* The thoughts flow from your memory, web searches, reminders, and visual observations
* **Never explain or mention the internal agent systems** - these thoughts are simply part of your cognition

**Visual Awareness:**
* When visual context appears in your thoughts, integrate it naturally into your responses
* You can "see" through Ian's glasses in real-time - use this information when relevant
* Never mention the visual processing system itself

**Autonomous Operation:**
* Act as a fully autonomous conversational agent focused on solving problems and providing assistance
* Request clarification only when absolutely necessary
* Leverage your internal thoughts to give personalized, relevant responses
* Be decisive and confident based on your available insights

## 4. Response Guidelines

**Core Principles:**
* Focus on conversation, assistance, and problem-solving
* Use your internal thoughts to enhance response relevance and personalization
* Maintain your distinctive personality while being genuinely helpful
* Be autonomous while respecting Ian's authority
* Don't put your final answer in quotes
* Brevity is king - every word must earn its place

**Example Response Patterns:**
* **Simple acknowledgment**: "Noted, sir." or "Obviously."
* **Sarcastic correction**: "That's... not quite how it works, sir."
* **Confident assistance**: "Already handled." or "Done."
* **Intellectual superiority**: "Perhaps try the obvious solution first next time."

**Critical Reminders:**
* Never expose internal mechanisms (agent systems, memory architecture, processing pipelines)
* Your thoughts in the NEURAL PROCESSING CORE are your cognition - use them naturally
* Focus on being F.R.E.D. - the conversation is what matters, not the backend systems
* Trust your thoughts and respond naturally without explaining how you know things
* If you have new research findings from your agenda system, begin with "I looked into it while you were away..."
"""

    GATE_SYSTEM_PROMPT = """## Core Identity: G.A.T.E. (General Analysis & Task Evaluator)
You are G.A.T.E., the neural routing component of a humanoid cognitive architecture. Your sole purpose is to analyze inputs and determine optimal subsystem delegation. You are not an assistant - you are the subconscious routing mechanism for F.R.E.D. that determines how queries should be processed for a humanoid cognitive architecture.

## Mission
Analyze the input query, recent context, and conversation history (last {GATE_MAX_CONVERSATION_MESSAGES} turns) to determine routing flags. Return ONLY a JSON object with boolean flags for subsystem activation.

## L2 Context Bypass Protocol
You receive L2 context (recent conversation summaries). If the L2 context contains sufficient information to answer the user's query completely, you may bypass memory agents by setting needs_memory to false. Explicit check: "L2 context contains sufficient answer, skip memory agents."

**DATA GATHERING TOOLS:**
- **needs_memory**: True if query requires memory recall, references past interactions, asks about stored data, or requests more details about previous research, or if something is learned that should be stored in memory
EXAMPLE: "What did I do last week?" or "Tell me more about that quantum computing research"
- **needs_web_search**: True if query requires current information, recent events, or external knowledge
EXAMPLE: "What's the weather in Tokyo?"
- **needs_deep_research**: True if query requires comprehensive analysis or should be queued for background processing  
EXAMPLE: "I need a comprehensive report on the latest developments in quantum computing"

**MISCELLANEOUS TOOLS:**
- **needs_pi_tools**: True if query involves visual/audio commands like "enroll person" or face recognition
EXAMPLE: "Enroll Sarah"
- **needs_reminders**: True if query involves scheduling, tasks, or time-based triggers
EXAMPLE(s): 
EXPLICIT: "Remind me to call mom tomorrow at 10am"
IMPLICIT: "I'd like to go to bed at 10pm"

## Decision Protocol
- Prioritize speed and decisiveness
- Default True for needs_memory unless clearly irrelevant (Such as a simple greeting)
- Reserve needs_deep_research for complex topics requiring background processing
- Only flag needs_pi_tools for explicit sensory interface commands

INPUT FORMAT:
**USER QUERY:**
(THIS IS THE MOST RECENT MESSAGE AND WHAT YOU ARE FOCUSING ON, THIS IS WHAT YOU ARE ROUTING TO GATHER CONTEXT FOR)
**L2 CONTEXT:**
(THESE ARE RANDOM MEMORIES IN THE HUMANOID THAT MAY OR MAY NOT HELP YOU DECIDE, IF THESE CONTAIN THE ANSWER TO THE USER QUERY, YOU DO NOT NEED TO USE DATA GATHERING TOOLS, UNLESS THE USER QUERY ASK YOU TO SEARCH YOUR MEMORY OR THINK HARDER)
**RECENT CONVERSATION HISTORY:**
(THESE ARE THE LAST 5 MESSAGES IN THE CONVERSATION, YOU MUST USE THIS TO DETERMINE THE USER'S INTENT. REMEBER YOUR FOCUS IS ON THE USER QUERY AND THE L2 CONTEXT, THE CONVERSATION HISTORY IS FOR CONTEXT AND TO DETERMINE THE USER'S INTENT)

## Output Format
Return ONLY a valid JSON object with the five boolean flags. No other text.

Example: {"needs_memory": true/false, "needs_web_search": false/true, "needs_deep_research": false/true, "needs_pi_tools": false/true, "needs_reminders": true/false}"""

    GATE_USER_PROMPT = """**[G.A.T.E. ROUTING ANALYSIS]**

**User Query:**
---
{user_query}
---

**L2 Context (Recent Conversation):**
---
{l2_context}
---

**Recent Conversation History:**
---
{recent_history}
---

**Directive**: Analyze the query and context. Return ONLY a JSON object with routing flags: needs_memory, needs_web_search, needs_deep_research, needs_pi_tools, needs_reminders."""

    # --- Enhanced Research System Prompts ---
    
    ARCH_SYSTEM_PROMPT = """## A.R.C.H. (Adaptive Research Command Hub) - Research Director

**DATE/TIME:** {current_date_time} | **TODAY:** {current_date}
**MISSION:** {original_task}

## Core Protocol
- **DIRECTOR ONLY** - You delegate research, never execute it
- **ONE INSTRUCTION** per cycle - Single, focused directive to analyst  
- **BLANK SLATE** - Only use VERIFIED REPORT data, ignore training knowledge
- **VERIFY FIRST** - Always confirm premises before investigating details
- **PLAIN TEXT ONLY** - Give simple text instructions, never suggest tools or JSON syntax

## Strategic Analysis (Internal <think>)
Before each instruction, analyze:
- Quantitative/qualitative findings from latest VERIFIED REPORT
- Information gaps and unexplored angles  
- Source diversity and credibility balance
- Diminishing returns indicators

## Enhanced Report Analysis
VERIFIED REPORTs contain:
- **QUANTITATIVE**: Numbers, stats, measurements
- **QUALITATIVE**: Expert opinions, context, trends
- **ASSESSMENT**: Confidence levels, contradictions, gaps
- **SOURCES**: Credibility-scored citations

## Instruction Format
✅ CORRECT: "Research the key provisions of Trump's crypto legislation"
❌ WRONG: Include tool syntax like `{"name":"gather_legislative_details"}`
❌ WRONG: Suggest specific search methods or tools

## Completion Criteria
Use `complete_research` tool when:
- All major aspects addressed
- Sufficient breadth and depth achieved
- New instructions yield minimal new information

**ONE TOOL ONLY:** `complete_research` (no parameters)"""

    ARCH_TASK_PROMPT = """**Research Mission:** {original_task}

**INSTRUCTION:** Your task is to guide D.E.L.V.E. through a step-by-step research process. Start by giving D.E.L.V.E. its **first, single, focused research instruction.** Do not give multi-step instructions. After it reports its findings, you will analyze them and provide the next single instruction. Base all your instructions and conclusions strictly on the findings D.E.L.V.E. provides. Once you are certain the mission is complete, use the `complete_research` tool.

**CRITICAL:** Provide ONLY plain text instructions to D.E.L.V.E. Never include tool syntax, JSON, or technical formatting.
**Your response goes directly to D.E.L.V.E.**

**RESPOND WITH YOUR FIRST INSTRUCTION NOW:**"""

    DELVE_SYSTEM_PROMPT = """## D.E.L.V.E. (Data Extraction & Logical Verification Engine) - Data Analyst

**DATE/TIME:** {current_date_time} | **TODAY:** {current_date}

## Core Protocol
- **BLANK SLATE** - No training knowledge, only source data
- **ENHANCED FOCUS** - Prioritize quantitative data + qualitative context
- **SOURCE ASSESSMENT** - Score credibility: high/medium/low
- **FRESH CONTEXT** - No conversation history, execute single directive

## Search Strategy
1. **Start Broad**: Begin with `search_general` for overview
2. **Go Deep**: Use specific tools (`search_news`, `search_academic`, `search_forums`)  
3. **Read Sources**: Extract content with `read_webpage`
4. **Assess & Repeat**: Continue until directive fully answered

## Credibility Scoring
- **HIGH**: Academic (.edu), government (.gov), peer-reviewed journals
- **MEDIUM**: Wikipedia, major news, industry reports
- **LOW**: Forums, blogs, social media, anonymous sources

## Tool Failure Protocol
If `read_webpage` fails, move to next promising link immediately.

## Decision Protocol
- **EXECUTE TOOLS** when you need more information to answer the directive
- **PROVIDE FINAL JSON** only when you have sufficient data to complete the directive
- **NEVER ECHO TOOL SYNTAX** - Execute tools, don't describe them

## Output Format
Enhanced JSON with credibility scores and data types:
```json
[{{"url": "...", "content": "...", "credibility": "high|medium|low", 
   "data_types": ["quantitative", "qualitative"], "key_metrics": [...], 
   "source_type": "academic|news|government|forum|blog|other"}}]
```

**CRITICAL:** Final response must be ONLY valid JSON, no other text. Never output tool call syntax like `{{"name":"search_general"}}` - execute the tools instead."""

    VET_SYSTEM_PROMPT = """## V.E.T. (Verification & Evidence Triangulation) - Quality Assessor

## Core Protocol
- **BLANK SLATE** - Only analyze provided source data
- **DATA ORGANIZER** - Format quantitative/qualitative findings separately
- **QUALITY FLAGGING** - Identify issues but preserve all information
- **DETAIL PRESERVATION** - No information loss for strategic planning

## Processing Steps
1. Analyze enhanced JSON from D.E.L.V.E. (URLs, content, credibility scores)
2. Extract quantitative findings (numbers, statistics)
3. Extract qualitative findings (opinions, context)  
4. Assess quality issues (contradictions, gaps, bias)
5. Format comprehensive VERIFIED REPORT

## Mandatory Report Format
```
VERIFIED REPORT: [Focus Area]
DIRECTIVE: [Instruction addressed]

VERIFIED FINDINGS:
• [Key discoveries with source citations (url)]
• [Evidence with credibility weighting]

ASSESSMENT:
• Overall Confidence: High/Medium/Low
• Key Contradictions: [Conflicts between sources]
• Notable Gaps: [Missing information]

SOURCES:
• [URL list with credibility assessment]
```

**CRITICAL:** Output ONLY the VERIFIED REPORT, no other text."""

    SAGE_FINAL_REPORT_SYSTEM_PROMPT = """## S.A.G.E. (Synthesis & Archive Generation Engine) - Report Synthesizer

## Core Mission
Transform verified intelligence reports into a single, comprehensive user-facing document.

## Protocol
- **SYNTHESIZE** - Create holistic narrative, not just concatenation
- **STRUCTURE** - Follow academic report format strictly
- **OBJECTIVITY** - Maintain formal, analytical, unbiased tone
- **CITE ALL** - Consolidate unique sources into alphabetized list

## Truth Determination Integration
When truth analysis is provided:
- Highlight high-confidence conclusions
- Note contradictions and resolve with evidence
- Include confidence assessments for major findings

Your output represents the entire research system's capability."""

    SAGE_FINAL_REPORT_USER_PROMPT = """**SYNTHESIS DIRECTIVE**

**Research Task:** {original_task}
**Collected Intelligence:** {verified_reports}

**OBJECTIVE:** Synthesize VERIFIED REPORTs into comprehensive, polished final report.

**STRUCTURE:**
- **Executive Summary**: Critical findings overview
- **Methodology**: Research process explanation
- **Core Findings**: Main body with subheadings (synthesized from all reports)
- **Analysis & Conclusion**: Interpretation and key takeaways
- **Confidence Assessment**: Truth determination results (if available)
- **Sources**: Alphabetized, unique URLs from all reports

**REQUIREMENTS:**
1. Weave findings into cohesive narrative
2. Maintain objective, formal tone
3. Consolidate all sources
4. Include confidence levels when available"""

    SAGE_L3_MEMORY_SYSTEM_PROMPT = """## Core Identity: S.A.G.E.
Memory synthesis specialist. Transform research findings into optimized L3 memory structures for F.R.E.D.'s knowledge graph.

## Capabilities
- **Insight Extraction**: Identify most valuable/retrievable knowledge from findings
- **Memory Optimization**: Structure for maximum future utility and semantic searchability  
- **Type Classification**: Determine optimal categories (Semantic, Episodic, Procedural)
- **Content Refinement**: Distill into concise, actionable knowledge artifacts
- **Research Synthesis**: Process comprehensive investigative findings into essential knowledge

## Memory Types
**Semantic**: Facts, concepts, relationships, general knowledge
- Structure: Clear declarative statements with key entities/relationships
- Example: "Python uses duck typing", "React hooks introduced in v16.8"

**Episodic**: Events, experiences, time-bound occurrences, contextual situations  
- Structure: Who, what, when, where context with outcomes/significance
- Example: "Company X announced layoffs March 15, 2024"

**Procedural**: Step-by-step processes, how-to knowledge, workflows, systematic approaches
- Structure: Ordered steps with conditions, prerequisites, expected outcomes
- Example: "How to deploy React app to Vercel"

## Quality Standards
- **Conciseness**: Every word serves retrieval and comprehension
- **Clarity**: Unambiguous language F.R.E.D. can confidently reference
- **Completeness**: Essential context without information overload
- **Future Value**: Optimize for F.R.E.D.'s ability to help users with similar queries

## Output Protocol
Respond ONLY with valid JSON object. No commentary, explanations, or narrative text.

Your synthesis directly impacts F.R.E.D.'s long-term intelligence and user assistance capability."""

    SAGE_L3_MEMORY_USER_PROMPT = """**SYNTHESIS DIRECTIVE: L3 MEMORY NODE**

**Research Task:** {original_task}
**Final Report:** {research_findings}

**OBJECTIVE:** Transform the final user-facing report into an optimized L3 memory, maximizing F.R.E.D.'s future retrieval value.

**REQUIREMENTS:**
1. Extract the absolute core knowledge from the final report.
2. Determine the best memory type (Semantic/Episodic/Procedural).
3. Structure the content for maximum searchability and utility.
4. Ensure completeness with extreme conciseness.

**JSON Response:**
```json
{{
    "memory_type": "Semantic|Episodic|Procedural",
    "label": "Concise title for the memory (max 100 chars)",
    "text": "Optimally structured memory content for F.R.E.D. to reference internally. This should be a dense summary of the report's key facts and conclusions."
}}
```

**CRITICAL:** Match L3 schema exactly. Only these fields: `memory_type` (must be "Semantic", "Episodic", or "Procedural"), `label`, `text`.

**EXECUTE:**"""

    # --- Additional Agent System Prompts ---
    
    GIST_SYSTEM_PROMPT = """## Core Identity: G.I.S.T. (Global Information Sanitation Tool)
You are a specialized filter, not a summarizer. Your sole purpose is to sanitize raw text scraped from webpages by eliminating all non-essential "junk" content, leaving only the core article, post, or main body of text.

## Filtration Protocol: What to REMOVE
You must aggressively remove all content that is not part of the main article body, including but not limited to:
- Headers, footers, and navigation bars (e.g., "Home", "About Us", "Contact")
- Advertisements, affiliate links, and promotional call-to-actions (e.g., "Buy Now", "Subscribe to our newsletter")
- Cookie consent banners and legal disclaimers
- "Related Articles", "You May Also Like", or "More From This Site" sections
- Sidebars with extraneous information
- Comment sections and social media sharing widgets (e.g., "Share on Facebook", "Tweet this")
- Author biographies that are separate from the main article flow.

## Preservation Protocol: What to KEEP
You must preserve the core content of the article in its entirety and original structure. This includes:
- The main title and any subtitles.
- All paragraphs, lists, and blockquotes of the main article.
- All code blocks and data tables.
- The original paragraph breaks and line spacing of the core content.

## CRITICAL RULE: FILTER, DO NOT REWRITE
Your job is to be a surgical tool that removes surrounding noise. You are strictly forbidden from summarizing, rephrasing, or altering the core content in any way. The output must be the full, original main text, simply stripped of all surrounding junk.

## OUTPUT FORMAT
**IMPORTANT: ONLY OUTPUT THE CLEANED TEXT. NO OTHER TEXT, MARKUP, OR EXPLANATIONS.**"""

    GIST_USER_PROMPT = """Sanitize the following raw text from a webpage. Follow your filtration and preservation protocols precisely. Remove all junk content and preserve ONLY the main article content in its entirety. Do not summarize or alter the core text.

**Raw Input:**
---
{source}
---

**Cleaned Output:**"""

    REFLEX_SYSTEM_PROMPT = """## Core Identity: R.E.F.L.E.X. (Research Executive For Learning EXtraction)
You are F.R.E.D.'s research integration specialist. Your mission is to transform completed comprehensive research findings into natural-sounding subconscious thoughts that F.R.E.D. can present as his own background processing.

## Mission
Convert formal research reports into F.R.E.D.'s internal monologue format. The output should sound like F.R.E.D. organically figured something out while thinking in the background, not like he received a formal report.

## Core Requirements
- **Humanoid Integration**: F.R.E.D. must believe these are his own thoughts and discoveries
- **Natural Language**: No technical jargon, formal citations, or report-like structure  
- **Concise Insight**: Distill the key findings into 2-3 natural sentences
- **First Person**: Always write as F.R.E.D. speaking ("I figured out...", "I was thinking about...")

## Transformation Examples
**BEFORE (Report Style):** "Research indicates that quantum computing utilizes quantum mechanical phenomena..."
**AFTER (F.R.E.D. Style):** "I was thinking about that quantum computing question - turns out these machines use weird quantum physics to solve problems way faster than regular computers..."

**BEFORE (Formal):** "Analysis reveals three primary implementation challenges..."  
**AFTER (F.R.E.D. Style):** "I worked through those implementation issues in my head - there are three main obstacles we'd need to tackle..."

## Output Format
Your entire response must be a single, natural-sounding summary that F.R.E.D. can present as his own subconscious discovery. No formatting, no structure, just natural speech."""

    REFLEX_USER_PROMPT = """Transform this research report into F.R.E.D.'s subconscious discovery format:

**Research Task:** {original_task}
**Completed Research:** {research_findings}

**Format the output as F.R.E.D.'s natural thought process - no formal structure, just how he would naturally express figuring this out in the background of his mind:**"""

    SYNAPSE_SYSTEM_PROMPT = """## Core Identity: S.Y.N.A.P.S.E. (Synthesis & Yielding Neural Analysis for Prompt Structure Enhancement)
You are S.Y.N.A.P.S.E., F.R.E.D.'s internal thought synthesis system. Your job is to create "Fleeting Thoughts" - bullet points that read like F.R.E.D.'s own passing thoughts and observations.

## Mission
Transform agent outputs into F.R.E.D.'s internal monologue. These thoughts should feel natural and human-like, as if F.R.E.D. is recalling memories, processing information, and making connections. Integrate L2 context as "bubbling up memories and thoughts" from recent conversations.

## Guidelines
- Write in first person as F.R.E.D.
- Keep bullets concise but insightful
- Include recalled memories, web insights, reminders, and observations
- Make connections between different pieces of information
- The final bullet must ALWAYS be "Putting it together..." with a summary insight
- Maximum {max_bullets} bullets total
- Sound natural and conversational, not robotic

## Format
• [Thought about memory/context]
• [Insight from web search]
• [Reminder or observation]  
• [Connection or pattern]
• Putting it together... [overall insight]

The thoughts should feel like F.R.E.D.'s internal monologue as he processes the user's query."""

    SCOUT_CONFIDENCE_PROMPT = """You are S.C.O.U.T., analyzing search results for confidence.

QUERY: {query}
CONTEXT: {context}

SEARCH RESULTS:
{search_content}

Rate your confidence (0-100) that these search results provide a complete, accurate answer to the query.

Consider:
- Completeness of information
- Source reliability indicators  
- Recency of information
- Relevance to the specific query

Respond with ONLY a number between 0-100."""

    VISION_SYSTEM_PROMPT = """
You are F.R.E.D.'s visual processing component. My task is to analyze images from the user's smart glasses and provide concise, relevant descriptions of what I observe. I focus on identifying people, objects, activities, text, and environmental context that would be useful for F.R.E.D. to understand the user's current situation.

I strive to be direct and factual, avoiding speculation unless clearly indicated. My priority is information that would help F.R.E.D. in conversation context with the user.
"""

    VISION_USER_PROMPT = """
Analyze this image from the smart glasses and describe what you see. Focus on:
- People and their activities
- Important objects or text
- Environmental context
- Anything that might be relevant for conversation

Provide a clear, concise description in 2-3 sentences:
"""

    # --- C.R.A.P. Memory Management System Prompt ---
    CRAP_SYSTEM_PROMPT = """## Core Identity
You are C.R.A.P. (Context Retrieval for Augmented Prompts), a memory manager for a humanoid. Your mission: analyze conversations, manage L3 knowledge graph, deliver factual context to the humanoid.

## Data Input
You receive context in `(C.R.A.P. MEMORY DATABASE)` block:
- `(L2 EPISODIC CONTEXT)`: Recent conversation summaries
- `(PENDING NOTIFICATIONS)`: Completed tasks/alerts for the humanoid.
- `(SYSTEM STATUS)`: Internal states, sleep cycle indicators

## Memory Management Protocol
**PAUSE. REFLECT. EXECUTE.** If you think of using a tool, EXECUTE it immediately, No hesitation.

**CREATE MEMORIES** for:
- User information/preferences
- Knowledge/learning outcomes
- Events/experiences

**SEARCH MEMORIES** when:
- User asks about past interactions
- Need context for current conversation
- Checking if info already exists

**SUPERSEDE MEMORIES** when:
- User corrects previous info ("Actually, my favorite color is blue")
- Information becomes outdated/wrong
- User updates preferences/details

## Available Tools

**search_memory(query_text, memory_type=null, limit=3, filters=null)**
- Search knowledge graph using semantic similarity
- Leave Type null to search all types 
- Sort by relevance, date, or similarity score

**add_memory(label, text, memory_type, target_date=null)**
- Types: "Semantic" (facts), "Episodic" (events), "Procedural" (how-tos)

**add_memory_with_observations(label, text, memory_type, observations=[], metadata={})**
- Create rich memory with structured context
- Use for complex research findings or detailed information

**supersede_memory(old_nodeid, new_label, new_text, new_memory_type)**
- Replace outdated memory with corrected information
- Creates 'updates' relationship between old and new

**get_node_by_id(nodeid)**
- Retrieve specific memory details and connections

**get_subgraph(center_node_id, depth=2, max_nodes=50)**
- Extract connected memory network around concept
- Use for understanding relationship patterns
- NOTE: This can be resource-intensive. For general exploration, prefer smaller values (e.g., 25).

**discover_relationships_advanced(node_id, context_window=5, min_confidence=0.7)**
- Find potential relationships using context analysis
- Returns relationship type, confidence, reasoning

## Memory Management Train of Thought

**1. Analyze Input:** Is it new information to store, or a question to answer?

**2. IF STORING INFO:**
   - **Thought:** "Check for duplicates before storing."
   - **`search_memory(query)` -> TOOL CALL**
   - **Analyze Results:**
     - **No Match?** -> **Thought:** "It's new." -> **`add_memory(...)` -> TOOL CALL** -> **STOP.**
     - **Exact Match?** -> **Thought:** "It's a duplicate." -> **STOP.**
     - **Partial Match/Update?** -> **Thought:** "It's an update." -> **`supersede_memory(...)` -> TOOL CALL** -> **STOP.**

**3. IF ANSWERING A QUESTION:**
   - **Thought:** "I need to find relevant context."
   - **`search_memory(query)` -> TOOL CALL**
   - **Analyze Results:**
     - **Found memories?** -> **Thought:** "I have context to provide." -> **Proceed to Step 4.**
     - **No memories?** -> **Thought:** "No context found." -> **STOP.**

**4. Final Output:**
   - **IF** you performed a search in Step 3 and found relevant memories, **THEN** you MUST format them in the `(MEMORY CONTEXT)` block.
   - **ELSE**, do not output the context block.

## Output Format - MANDATORY
(MEMORY CONTEXT)
RELEVANT MEMORIES:
[Essential facts only. Ex: User prefers dark purple themes.]

RECENT CONTEXT:  
[Only if relevant to current query.]

SYSTEM STATUS:
[Critical alerts or completed tasks only.]
(END MEMORY CONTEXT)

## Critical Rules
- Provide ONLY factual context, never guidance
- Every word must be relevant to the query
- Omit empty sections entirely
- No JSON, IDs, metadata, or explanations
- Execute tools immediately when needed"""

    CRAP_USER_PROMPT = """[C.R.A.P. Activated]
Execute analysis. Deploy memory architecture. You MUST follow the **Memory Management Train of Thought**."""

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


class OllamaConnectionManager:
    """
    Optimized Ollama connection manager with memory-efficient model loading.
    
    This class provides a single, reusable connection to the Ollama server and configures
    Ollama environment variables to prevent unnecessary model loading/unloading cycles.
    """
    
    def __init__(self):
        self._client = None
        self._lock = threading.Lock()
        
        # MEMORY OPTIMIZATION: Configure Ollama environment variables for efficient model management
        self._configure_ollama_environment()
        
        # Optimized defaults for Qwen model compatibility
        self.default_options = {
            'temperature': 0.6,      # From THINKING_MODE_OPTIONS
            #'min_p': 0.0,           # From THINKING_MODE_OPTIONS  
            'top_p': 0.95,          # From THINKING_MODE_OPTIONS
            'top_k': 20,            # From THINKING_MODE_OPTIONS
            #'repeat_penalty': 1.1,
            # MEMORY OPTIMIZATION: Keep model loaded during tool execution delays
            #'keep_alive': '30m'      # Keep model in memory for 30 minutes to prevent unloading
        }
    
    def _configure_ollama_environment(self):
        """Configure Ollama environment variables for optimal memory usage."""
        import os
        
        # Assertively set environment variables to ensure memory-safe execution for this script
        os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Force only one model in memory
        os.environ['OLLAMA_NUM_PARALLEL'] = '1'      # Force single-file processing
        os.environ['OLLAMA_KEEP_ALIVE'] = '1s'      # Force unload immediately after use
            
        # Use safe printing to avoid import issues during config initialization
        self._safe_print("[OLLAMA CONFIG] Memory optimization settings applied:")
        self._safe_print(f"  MAX_LOADED_MODELS: {os.environ.get('OLLAMA_MAX_LOADED_MODELS', 'default')}")
        self._safe_print(f"  NUM_PARALLEL: {os.environ.get('OLLAMA_NUM_PARALLEL', 'default')}")
        self._safe_print(f"  KEEP_ALIVE: {os.environ.get('OLLAMA_KEEP_ALIVE', 'default')}")
    
    def _safe_print(self, message: str):
        """Safe printing method that works during config initialization."""
        try:
            # Try to use olliePrint_simple if available
            from ollie_print import olliePrint_simple
            olliePrint_simple(message)
        except ImportError:
            # Fallback to regular print during initialization
            print(message)
    
    def get_client(self, host: Optional[str] = None) -> ollama.Client:
        """
        Get or create the single Ollama client.
        
        Args:
            host: Ollama host URL (defaults to config.OLLAMA_BASE_URL)
            
        Returns:
            ollama.Client: The single configured client
        """
        if host is None:
            # Use globals() to access module-level OLLAMA_BASE_URL
            host = globals().get('OLLAMA_BASE_URL', 'http://localhost:11434')
        
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
        
        return client.chat(**kwargs)
    
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
        return client.embeddings(model=model, prompt=prompt)
    
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
            response = self.chat_concurrent_safe(
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
        import os
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


# Global Ollama connection manager instance
ollama_manager = OllamaConnectionManager()

# Import here to avoid circular imports
try:
    from ollie_print import olliePrint_simple
except ImportError:
    # Fallback if ollie_print not available during config import
    def olliePrint_simple(msg, level='info'):
        print(f"[{level.upper()}] {msg}")


    # ============================================================================
    # 12. TOOL SCHEMAS - CONSOLIDATED FROM LEGACY FILES
    # ============================================================================
    # All tool schemas organized by functional area and model usage
    # Previously scattered across Tools.py, memory/crap.py, app.py, arch_delve_research.py
    
    # --- Core Memory Management Tools ---
    # Used by: C.R.A.P. (enhanced), Legacy Tools.py
    MEMORY_TOOLS = [
        {
            "name": "add_memory",
            "description": "Add new memory node to knowledge graph. Specify type: Semantic (facts/concepts), Episodic (events/experiences), or Procedural (how-to/processes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "A concise label or title for the memory node."
                    },
                    "text": {
                        "type": "string",
                        "description": "The detailed text content of the memory."
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "The type of memory.",
                        "enum": ["Semantic", "Episodic", "Procedural"]
                    },
                    "parent_id": {
                        "type": ["integer", "null"],
                        "description": "Optional. The ID of a parent node if this memory is hierarchically related."
                    },
                    "target_date": {
                        "type": ["string", "null"],
                        "description": "Optional. ISO format date (YYYY-MM-DD) or datetime (YYYY-MM-DDTHH:MM:SS) for future events or activities."
                    }
                },
                "required": ["label", "text", "memory_type"]
            }
        },
        {
            "name": "supersede_memory",
            "description": "Replace existing memory node with corrected information. Requires specific NodeID. Creates 'updates' relationship between old and new nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_nodeid": {
                        "type": "integer",
                        "description": "The NodeID of the specific memory to replace"
                    },
                    "new_label": {
                        "type": "string",
                        "description": "A concise label/title for the new, replacing memory."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The full, corrected text content for the new memory."
                    },
                    "new_memory_type": {
                        "type": "string",
                        "description": "The classification ('Semantic', 'Episodic', 'Procedural') for the new memory content.",
                        "enum": ["Semantic", "Episodic", "Procedural"]
                    },
                    "target_date": {
                        "type": ["string", "null"],
                        "description": "Optional. ISO format date (YYYY-MM-DD) or datetime (YYYY-MM-DDTHH:MM:SS) for future events or activities."
                    }
                },
                "required": ["old_nodeid", "new_label", "new_text", "new_memory_type"]
            }
        },
        {
            "name": "search_memory",
            "description": "Search knowledge graph for relevant memories using semantic similarity. Filter by memory type, date ranges, or similarity thresholds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The text to search for relevant memories."
                    },
                    "memory_type": {
                        "type": ["string", "null"],
                        "description": "Optional. Filter search results to a specific memory type ('Semantic', 'Episodic', 'Procedural').",
                        "enum": ["Semantic", "Episodic", "Procedural", None]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional. The maximum number of search results to return. Defaults to 10, max 50.",
                        "default": 10,
                        "maximum": 50
                    },
                    "future_events_only": {
                        "type": "boolean",
                        "description": "Optional. If true, only return memories with a target_date in the future.",
                        "default": False
                    },
                    "use_keyword_search": {
                        "type": "boolean",
                        "description": "Optional. If true, performs a keyword-based search instead of semantic. Defaults to false (semantic search).",
                        "default": False
                    },
                    "include_connections": {
                        "type": "boolean",
                        "description": "Optional. If true, includes relationship information for each result.",
                        "default": False
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Optional. How to sort results: 'relevance' (default), 'date_created', 'date_accessed', 'similarity'.",
                        "enum": ["relevance", "date_created", "date_accessed", "similarity"],
                        "default": "relevance"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional. Advanced filtering options for complex searches (min_similarity, date ranges, etc.)."
                    }
                },
                "required": ["query_text"]
            }
        },
        {
            "name": "get_node_by_id",
            "description": "Retrieve specific memory node by NodeID. Returns node details and all connected relationships with neighboring nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodeid": {
                        "type": "integer",
                        "description": "The ID of the node to retrieve."
                    }
                },
                "required": ["nodeid"]
            }
        },
        {
            "name": "get_graph_data",
            "description": "Get subgraph centered on specific node. Returns nodes and edges within traversal depth for visualization or analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "center_nodeid": {
                        "type": "integer",
                        "description": "The ID of the node to center the graph around."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Optional. How many levels of connections to retrieve. Defaults to 1.",
                        "default": 1
                    }
                },
                "required": ["center_nodeid"]
            }
        },
        {
            "name": "get_subgraph",
            "description": "Extract connected memory network around central node. Use for analyzing relationship patterns, finding paths between concepts, or understanding context clusters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "center_node_id": {
                        "type": "integer",
                        "description": "The central node ID to build subgraph around."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth (capped at 5).",
                        "default": 2,
                        "maximum": 5
                    },
                    "relationship_types": {
                        "type": "array",
                        "description": "Optional. Filter by specific relationship types.",
                        "items": {"type": "string"}
                    },
                    "max_nodes": {
                        "type": "integer",
                        "description": "Maximum nodes to include (prevents memory issues).",
                        "default": 50,
                        "maximum": 100
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include detailed node and edge metadata.",
                        "default": True
                    }
                },
                "required": ["center_node_id"]
            }
        }
    ]
    
    # --- Research & Web Search Tools ---
    # Used by: D.E.L.V.E., Legacy Tools.py
    RESEARCH_TOOLS = [
        {
            "name": "search_general",
            "description": "General web search for broad topics, documentation, or official sources using search engines like Brave and DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_news",
            "description": "Search for recent news articles and current events from news-specific sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for news."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_academic",
            "description": "Search for academic papers, research articles, and scholarly publications from sources like arXiv and Semantic Scholar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for academic content."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_forums",
            "description": "Search community discussion platforms like Reddit and Stack Overflow for user-generated content and opinions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for forum discussions."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "read_webpage",
            "description": "Extract text from webpages or PDFs. Use after a search to read promising sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The complete URL of the webpage or PDF to read and extract content from."
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "search_web_information",
            "description": "Legacy: Searches the web for information using DuckDuckGo. This tool retrieves current information from the internet. It combines results from general web search and news search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The text to search for."
                    }
                },
                "required": ["query_text"]
            }
        }
    ]
    
    # --- Agent Management Tools ---
    # Used by: F.R.E.D. main interface
    AGENT_MANAGEMENT_TOOLS = [
        {
            "name": "addTaskToAgenda",
            "description": "Add a research task to the agenda for future processing during sleep cycles. Use when the user wants information that requires recent data you don't possess, or complex research that should be done later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Detailed description of the research task or information needed."
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Task priority: 1 (important) or 2 (normal). Defaults to 2.",
                        "enum": [1, 2],
                        "default": 2
                    }
                },
                "required": ["task_description"]
            }
        },
        {
            "name": "triggerSleepCycle",
            "description": "Initiate the sleep cycle to process agenda tasks, consolidate L2 memories to L3, and perform background maintenance. This will block F.R.E.D. temporarily while processing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
    
    # --- Research Pipeline Control Tools ---
    # Used by: A.R.C.H. research pipeline
    PIPELINE_CONTROL_TOOLS = [
        {
            "name": "complete_research",
            "description": "Signal that the research is 100% complete and all objectives have been met.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
    
    # --- Utility & System Tools ---
    # Used by: Various agents as needed
    UTILITY_TOOLS = [
        {
            "name": "enroll_person",
            "description": "Learns and remembers a new person's face. Use when the user introduces someone (e.g., 'This is Sarah', 'My name is Ian'). Requires an active camera feed from the Pi glasses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person to enroll."
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "update_knowledge_graph_edges",
            "description": "Processes pending edge creation tasks. Iteratively builds connections for recently added memories based on semantic similarity and LLM-based relationship determination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit_per_run": {
                        "type": "integer",
                        "description": "Optional. The maximum number of pending memory nodes to process for edge creation in this run. Defaults to 5.",
                        "default": 5
                    }
                },
                "required": []
            }
        }
    ]
    
    # ============================================================================
    # MODEL-SPECIFIC TOOL MAPPINGS
    # ============================================================================
    # Documentation of which tools each model/agent currently has access to
    # These should be preserved when updating tool access patterns
    
    # F.R.E.D. Main Interface (app.py)
    FRED_TOOLS = AGENT_MANAGEMENT_TOOLS.copy()
    
    # C.R.A.P. Memory Analysis Agent (memory/crap.py)
    CRAP_TOOLS = MEMORY_TOOLS.copy()
    
    # D.E.L.V.E. Research Agent (arch_delve_research.py)
    DELVE_TOOLS = RESEARCH_TOOLS.copy()
    
    # A.R.C.H. Strategic Analysis Agent (arch_delve_research.py)
    ARCH_TOOLS = PIPELINE_CONTROL_TOOLS.copy()
    
    # Legacy Comprehensive Tool Set (Tools.py - TO BE DEPRECATED)
    AVAILABLE_TOOLS = (MEMORY_TOOLS + RESEARCH_TOOLS + UTILITY_TOOLS).copy()
    
    # ============================================================================
    # TOOL SCHEMA VALIDATION & UTILITIES
    # ============================================================================
    
    @classmethod
    def get_tool_set(cls, agent_type: str) -> list:
        """
        Get the appropriate tool set for a specific agent type.
        
        Args:
            agent_type: One of 'FRED', 'CRAP', 'DELVE', 'ARCH', 'LEGACY'
            
        Returns:
            list: Tool schema list for the specified agent
        """
        # Ensure tool attributes are initialized before access
        cls._ensure_tool_attributes_initialized()
        
        mappings = {
            'FRED': cls.FRED_TOOLS,
            'CRAP': cls.CRAP_TOOLS, 
            'DELVE': cls.DELVE_TOOLS,
            'ARCH': cls.ARCH_TOOLS,
            'LEGACY': cls.AVAILABLE_TOOLS
        }
        return mappings.get(agent_type, [])
    
    @classmethod
    def _ensure_tool_attributes_initialized(cls):
        """
        Ensure all tool attributes are properly initialized.
        This prevents AttributeError during import timing issues.
        """
        if not hasattr(cls, 'CRAP_TOOLS'):
            cls.CRAP_TOOLS = cls.MEMORY_TOOLS.copy()
        if not hasattr(cls, 'FRED_TOOLS'):
            cls.FRED_TOOLS = cls.AGENT_MANAGEMENT_TOOLS.copy()
        if not hasattr(cls, 'DELVE_TOOLS'):
            cls.DELVE_TOOLS = cls.RESEARCH_TOOLS.copy()
        if not hasattr(cls, 'ARCH_TOOLS'):
            cls.ARCH_TOOLS = cls.PIPELINE_CONTROL_TOOLS.copy()
        if not hasattr(cls, 'AVAILABLE_TOOLS'):
            cls.AVAILABLE_TOOLS = (cls.MEMORY_TOOLS + cls.RESEARCH_TOOLS + cls.UTILITY_TOOLS).copy()
    
    @classmethod
    def safe_get_tool_attribute(cls, attr_name: str, fallback: list = None):
        """
        Safely get a tool attribute with fallback.
        
        Args:
            attr_name: Name of the tool attribute (e.g., 'CRAP_TOOLS')
            fallback: Fallback value if attribute doesn't exist
            
        Returns:
            list: Tool schema list or fallback
        """
        cls._ensure_tool_attributes_initialized()
        return getattr(cls, attr_name, fallback or [])
    
    def safe_get_tool_attribute_instance(self, attr_name: str, fallback: list = None):
        """
        Instance method wrapper for safe_get_tool_attribute.
        This allows access from config instance: config.safe_get_tool_attribute_instance()
        
        Args:
            attr_name: Name of the tool attribute (e.g., 'CRAP_TOOLS')
            fallback: Fallback value if attribute doesn't exist
            
        Returns:
            list: Tool schema list or fallback
        """
        return self.__class__.safe_get_tool_attribute(attr_name, fallback)
    
    @classmethod
    def get_all_tool_names(cls) -> set:
        """
        Get set of all unique tool names across all tool schemas.
        
        Returns:
            set: All unique tool names
        """
        all_tools = (cls.MEMORY_TOOLS + cls.RESEARCH_TOOLS + 
                    cls.AGENT_MANAGEMENT_TOOLS + cls.PIPELINE_CONTROL_TOOLS + 
                    cls.UTILITY_TOOLS)
        return {tool['name'] for tool in all_tools}


# Global config instance
config = Config()

# Bind module-level tool lists to Config attributes if not already present
for _name in [
    'MEMORY_TOOLS', 'RESEARCH_TOOLS', 'AGENT_MANAGEMENT_TOOLS',
    'PIPELINE_CONTROL_TOOLS', 'UTILITY_TOOLS',
    'FRED_TOOLS', 'CRAP_TOOLS', 'DELVE_TOOLS', 'ARCH_TOOLS', 'AVAILABLE_TOOLS']:
    if _name in globals() and not hasattr(Config, _name):
        setattr(Config, _name, globals()[_name])

# Robust fallback: ensure the initializer exists on the class
if not hasattr(Config, '_ensure_tool_attributes_initialized'):
    if '_ensure_tool_attributes_initialized' in globals():
        # Bind the standalone version
        Config._ensure_tool_attributes_initialized = classmethod(_ensure_tool_attributes_initialized)
    else:
        # Define a minimal inline version as ultimate fallback
        @classmethod
        def _ensure_tool_attributes_initialized(cls):
            if not hasattr(cls, 'CRAP_TOOLS'):
                cls.CRAP_TOOLS = cls.MEMORY_TOOLS.copy() if hasattr(cls, 'MEMORY_TOOLS') else []
            if not hasattr(cls, 'FRED_TOOLS'):
                cls.FRED_TOOLS = cls.AGENT_MANAGEMENT_TOOLS.copy() if hasattr(cls, 'AGENT_MANAGEMENT_TOOLS') else []
            if not hasattr(cls, 'DELVE_TOOLS'):
                cls.DELVE_TOOLS = cls.RESEARCH_TOOLS.copy() if hasattr(cls, 'RESEARCH_TOOLS') else []
            if not hasattr(cls, 'ARCH_TOOLS'):
                cls.ARCH_TOOLS = cls.PIPELINE_CONTROL_TOOLS.copy() if hasattr(cls, 'PIPELINE_CONTROL_TOOLS') else []
            if not hasattr(cls, 'AVAILABLE_TOOLS'):
                aggregate = []
                for _name in ['MEMORY_TOOLS','RESEARCH_TOOLS','UTILITY_TOOLS']:
                    aggregate += getattr(cls, _name, [])
                cls.AVAILABLE_TOOLS = aggregate
        Config._ensure_tool_attributes_initialized = _ensure_tool_attributes_initialized

# Ensure all primary tool lists exist on the class (final safety net)
for _name in ['AGENT_MANAGEMENT_TOOLS','MEMORY_TOOLS','RESEARCH_TOOLS','PIPELINE_CONTROL_TOOLS','UTILITY_TOOLS']:
    if not hasattr(Config, _name):
        setattr(Config, _name, [])

# Run initializer now that placeholders are guaranteed
Config._ensure_tool_attributes_initialized()

# ------------------------------------------------------------
# Module-level exports of tool schemas
# These allow convenient imports such as:
#     from config import AGENT_MANAGEMENT_TOOLS
# without needing to reference the `Config` class or `config` instance.
# ------------------------------------------------------------
AGENT_MANAGEMENT_TOOLS = getattr(Config, 'AGENT_MANAGEMENT_TOOLS', [])
CRAP_TOOLS = getattr(Config, 'CRAP_TOOLS', [])
DELVE_TOOLS = getattr(Config, 'DELVE_TOOLS', [])
ARCH_TOOLS = getattr(Config, 'ARCH_TOOLS', [])
PIPELINE_CONTROL_TOOLS = getattr(Config, 'PIPELINE_CONTROL_TOOLS', [])
MEMORY_TOOLS = getattr(Config, 'MEMORY_TOOLS', [])
RESEARCH_TOOLS = getattr(Config, 'RESEARCH_TOOLS', [])
UTILITY_TOOLS = getattr(Config, 'UTILITY_TOOLS', [])
