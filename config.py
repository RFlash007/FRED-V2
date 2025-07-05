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
    SECRET_KEY = 'fred_secret_key_2024'  # Secret key for Flask session management and other security-related operations.
    PORT = int(os.environ.get('PORT', 5000))  # Port on which the Flask web server will listen for incoming HTTP requests.
    HOST = '0.0.0.0'  # Host address for the Flask server, binding to all available network interfaces.
    DEBUG = False  # Enables/disables Flask's debug mode; set to False for production environments for security.
    
    # WebRTC Configuration
    WEBRTC_PORT = int(os.environ.get('WEBRTC_PORT', 8080))  # Port for the WebRTC signaling server, used for real-time communication with the Pi glasses.
    WEBRTC_HOST = '0.0.0.0'  # Host address for the WebRTC server, allowing connections from any network interface.
    
    # Security Configuration
    FRED_AUTH_TOKEN = os.environ.get('FRED_AUTH_TOKEN', 'fred_pi_glasses_2024')  # Authentication token used by the Raspberry Pi client to connect securely to the F.R.E.D. server.
    MAX_PI_CONNECTIONS = int(os.environ.get('MAX_PI_CONNECTIONS', 3))  # Maximum number of concurrent Raspberry Pi client connections allowed to the WebRTC server.
    
    # ngrok Configuration
    NGROK_ENABLED = os.environ.get('NGROK_ENABLED', 'true').lower() == 'true'  # Boolean flag to enable or disable ngrok tunneling for external access to F.R.E.D.
    NGROK_AUTH_TOKEN = os.environ.get('NGROK_AUTH_TOKEN', '2yCKXFFreg1EEaQK6RGb3Kbdt6f_4owEF1Xji51DMheaKDV5U')  # ngrok authentication token required to establish secure tunnels.
    
    # ICE/STUN Configuration for WebRTC
    ICE_SERVERS = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"}
    ]  # List of STUN/TURN servers used by WebRTC for NAT traversal, enabling peer-to-peer connections.
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')  # Base URL for the Ollama API server, hosting local language models.
    OLLAMA_EMBED_URL = os.getenv('OLLAMA_EMBED_URL', 'http://localhost:11434/api/embeddings')  # Endpoint for generating text embeddings using Ollama.
    OLLAMA_GENERATE_URL = os.getenv('OLLAMA_GENERATE_URL', 'http://localhost:11434/api/generate')  # Endpoint for generating text completions using Ollama.
    OLLAMA_TIMEOUT = None  # Timeout for Ollama API requests (None means no timeout).
    
    # Model Configuration
    DEFAULT_MODEL = 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M'  # The default large language model to be used for general F.R.E.D. responses.
    EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')  # The model used for generating text embeddings for semantic search and memory operations.
    LLM_DECISION_MODEL = os.getenv('LLM_DECISION_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M')  # The model specifically designated for complex decision-making and reasoning tasks within F.R.E.D.
    THINKING_MODE_OPTIONS = {"temperature": 0.6, "min_p": 0.0, "top_p": 0.95, "top_k": 20}  # Official Qwen3 thinking mode parameters from documentation
    
    # TTS Configuration
    FRED_SPEAKER_WAV_PATH = "new_voice_sample.wav"  # Path to the WAV file used as the base voice for F.R.E.D.'s text-to-speech.
    XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"  # Name of the XTTS v2 model used for advanced text-to-speech generation.
    FRED_LANGUAGE = "en"  # The primary language for F.R.E.D.'s text-to-speech output.
    TTS_CLEANUP_DELAY = 2  # Seconds to wait before cleaning up temporary TTS audio files.
    
    # Voice Cloning Configuration for Stewie
    STEWIE_VOICE_ENABLED = True  # Boolean flag to enable or disable the Stewie voice cloning feature.
    STEWIE_VOICE_SAMPLES_DIR = "voice_samples/stewie"  # Directory containing audio samples used for cloning Stewie's voice.
    STEWIE_VOICE_SAMPLES = [
        "stewie_sample_1.wav",
        "stewie_sample_2.wav", 
        "stewie_sample_3.wav",
        "stewie_sample_4.wav",
        "stewie_sample_5.wav"
    ]  # Specific list of Stewie voice sample filenames to be used for cloning.
    STEWIE_VOICE_CLONE_TEMP_DIR = "voice_cloning_temp"  # Temporary directory for intermediate files generated during the voice cloning process.
    STEWIE_VOICE_CLONE_MODEL_CACHE = "voice_cloning_cache"  # Directory for caching downloaded voice cloning model files to speed up processing.
    STEWIE_VOICE_CLONE_QUALITY = "premium"  # Quality setting for Stewie voice synthesis (e.g., ultra_fast, fast, standard, high, premium) affecting clarity and processing time.
    STEWIE_VOICE_CLONE_SPEED = 0.9  # Adjusts the speaking speed of the cloned Stewie voice (e.g., 1.0 is normal, <1.0 is slower, >1.0 is faster).
    STEWIE_VOICE_CLONE_TEMPERATURE = 0.3  # Controls the randomness and variation in the cloned voice's intonation and delivery; lower values are more consistent.
    STEWIE_VOICE_CLONE_REPETITION_PENALTY = 1.05  # Penalizes repetition in generated speech, encouraging more diverse word choices (optimal range: 1.0-1.1 for XTTS-v2).
    STEWIE_VOICE_CLONE_LENGTH_PENALTY = 1.0  # Influences the length of generated speech segments; higher values can make output longer.
    STEWIE_VOICE_CLONE_ENABLE_TEXT_SPLITTING = True  # Enables automatic splitting of long texts into smaller chunks for better synthesis quality and performance.
    
    # STT Configuration (Vosk-based)
    STT_SAMPLE_RATE = 16000  # Audio sample rate (in Hz) for Speech-to-Text processing, typically 16000 for Vosk models.
    STT_CHANNELS = 1  # Number of audio channels for STT; 1 for mono, 2 for stereo.
    STT_BLOCK_DURATION = 5  # Duration (in seconds) of audio chunks processed by the STT engine at a time.
    STT_SILENCE_THRESHOLD = 0.002   # General threshold for detecting silence in audio input to determine speech boundaries (lower is more sensitive).
    # Separate default VAD threshold for Raspberry Pi glasses audio (mono, 16 kHz)
    STT_PI_SILENCE_THRESHOLD = 0.0015  # Specific silence detection threshold for Raspberry Pi audio, adjusted for its typical noise profile.
    STT_CALIBRATION_DURATION = 3  # Duration (in seconds) for initial microphone calibration to adapt to ambient noise levels.
    STT_SILENCE_DURATION = 0.8  # Duration (in seconds) of continuous silence required to indicate the end of a speech segment.
    
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
    ] # Ordered list of paths where Vosk STT models are located, searched in order of preference.

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
    ]  # Specific ordered paths for Vosk models optimized for Raspberry Pi client's limited resources.
    
    VOSK_ENABLE_WORDS = True        # Enables word-level timestamps and confidence scores in Vosk STT results.
    VOSK_ENABLE_PARTIAL_WORDS = True # Enables partial word recognition for faster, more responsive intermediate STT results.
    VOSK_LOG_LEVEL = -1             # Sets Vosk's internal logging level; -1 disables all logging for cleaner output.
    
    # Enhanced Speech Processing Settings
    STT_MIN_WORD_LENGTH = 2         # Minimum character length for a recognized word to be considered valid.
    STT_MIN_PHRASE_LENGTH = 3       # Minimum number of words required in a detected phrase to be processed as a valid utterance.
    STT_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score (0.0-1.0) for a speech recognition result to be accepted by F.R.E.D..
    
    # Database Configuration
    DB_PATH = os.path.join('memory', 'memory.db')  # Relative path to the SQLite database file for F.R.E.D.'s memory and knowledge graph.
    EMBEDDING_DIM = 768  # Dimensionality of the embeddings used for semantic searches and memory storage.
    AUTO_EDGE_SIMILARITY_CHECK_LIMIT = 3  # The maximum number of similar nodes to check when automatically creating edges in the knowledge graph.
    
    # Tool Configuration
    MAX_TOOL_ITERATIONS = 5  # Maximum number of consecutive tool calls an LLM can make in a single reasoning chain.
    CRAP_MAX_TOOL_ITERATIONS = 3  # Maximum number of tool iterations for C.R.A.P. memory analysis and context retrieval.
    WEB_SEARCH_MAX_RESULTS = 3  # Maximum number of web search results to retrieve for a given query.
    WEB_SEARCH_NEWS_MAX_RESULTS = 2  # Maximum number of news search results to retrieve.
    WEB_SEARCH_TIMEOUT = 120   # Timeout (in seconds) for web search requests (None means no timeout).
    
    # Advanced Research System Configuration
    # =====================================
    # F.R.E.D.'s comprehensive research system integrates multiple APIs for thorough information gathering.
    # All API keys are optional - the system will gracefully skip unavailable services.
    
    # Research Source Limits (per category) - Optimized for quality over quantity
    RESEARCH_MAX_ACADEMIC_PAPERS = 6        # Max academic papers
    RESEARCH_MAX_WEB_ARTICLES = 8           # Max web articles from general search
    RESEARCH_MAX_NEWS_ARTICLES = 4          # Max news articles
    RESEARCH_MAX_FORUM_POSTS = 5            # Max forum posts (Reddit, Stack Overflow)
    RESEARCH_MAX_SOCIAL_POSTS = 5           # Max social media posts
    RESEARCH_MAX_VIDEO_TRANSCRIPTS = 3      # Max video transcripts from YouTube
    RESEARCH_MAX_DOCUMENTATION = 3          # Max documentation links
    RESEARCH_MAX_LINKS_PER_CATEGORY = 5 # Max links to extract per category for each search
    
    # API Keys and Configuration
    # --------------------------
    
    # YouTube Data API v3 (for video transcripts)
    # How to get: https://console.developers.google.com/
    # 1. Create a project or select existing
    # 2. Enable "YouTube Data API v3"  
    # 3. Create credentials (API Key)
    # 4. Copy the API key below
    YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyAG1DQIrpE2nUbzOzYJ8_oS7f8AcU4Thvc')
    YOUTUBE_API_QUOTA_LIMIT = 10000  # Daily quota units (default Google limit)
    
    # Reddit API (for forum discussions and social content)
    # How to get: https://www.reddit.com/prefs/apps
    # 1. Click "Create App" or "Create Another App"
    # 2. Choose "script" type
    # 3. Copy client ID and secret
    # 4. Create a Reddit account specifically for F.R.E.D. if desired
    REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID', 'qbap3jhtPoKF-jwrl-4HMA')
    REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET', 'UDN2ZasREkEfEtFbvGHhqgGKBMOIuA')
    REDDIT_USER_AGENT = 'F.R.E.D. Research Assistant v2.0'
    REDDIT_REQUEST_LIMIT = 100  # Requests per minute (Reddit limit)
    

    
    # Stack Overflow API (for technical discussions)
    # No API key required for read-only access
    # Rate limit: 300 requests per day without key, 10,000 with key
    # How to get key: https://stackapps.com/apps/oauth/register
    STACKOVERFLOW_API_KEY = os.environ.get('STACKOVERFLOW_API_KEY', '')
    STACKOVERFLOW_REQUEST_LIMIT = 300  # Default rate limit per day
    
    # Jina AI Reader API (for webpage content extraction)
    # How to get: https://jina.ai/reader/
    # Free tier: 1,000 requests per month
    # 1. Sign up for Jina AI account
    # 2. Get API key from dashboard
    # Alternative: Can use without key (public endpoint) with rate limits
    JINA_API_KEY = os.environ.get('JINA_API_KEY', '')
    JINA_REQUEST_LIMIT = 1000  # Monthly requests for free tier
    
    # Academic Research APIs (all free, no keys required)
    ARXIV_API_BASE = 'http://export.arxiv.org/api/query'
    SEMANTIC_SCHOLAR_API_BASE = 'https://api.semanticscholar.org/graph/v1'
    PUBMED_API_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
    


    # Brave Search API (Independent search engine, ideal for AI applications)
    # How to get: https://api.search.brave.com/
    # 1. Sign up for Brave Search API account
    # 2. Choose Free plan (2,000 queries/month, 1 query/second)
    # 3. Copy API key from dashboard
    # Free tier: 2,000 requests per month, upgrade to $3/1000 for higher limits
    BRAVE_SEARCH_API_KEY = os.environ.get('BRAVE_SEARCH_API_KEY', 'BSAXrrHvaG5TWPIaJeCGJXhxVRXGuRH')
    BRAVE_SEARCH_API_URL = 'https://api.search.brave.com/res/v1/web/search'
    BRAVE_NEWS_API_URL = 'https://api.search.brave.com/res/v1/news/search'
    BRAVE_SEARCH_REQUEST_LIMIT = 2000  # Monthly requests for free tier

    # SearchAPI.io (Proxy for DuckDuckGo and other search engines to avoid rate limits)
    # How to get: https://www.searchapi.io/
    # 1. Sign up for SearchAPI account  
    # 2. Choose free plan (100 searches/month)
    # 3. Copy API key from dashboard
    # Free tier: 100 requests per month, paid plans available
    SEARCHAPI_API_KEY = os.environ.get('SEARCHAPI_API_KEY', 'UpyF9L72UJ1xQw1tvVCVsUdv')
    SEARCHAPI_BASE_URL = 'https://www.searchapi.io/api/v1/search'
    SEARCHAPI_REQUEST_LIMIT = 100  # Monthly requests for free tier

    # News APIs
    # NewsAPI (optional, has free tier)
    # How to get: https://newsapi.org/register
    # Free tier: 100 requests per day, 1000 per month
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '6ef0134a37c64189aa9cda119fc8f1a1')
    NEWS_API_REQUEST_LIMIT = 100  # Daily requests for free tier
    
    # Research System Behavior
    RESEARCH_ENABLE_PARALLEL_CORE = True  # Enable/disable parallel execution of core research sources
    RESEARCH_ENABLE_SOCIAL_SOURCES = True # Enable/disable social media/forum searches
    RESEARCH_ENABLE_VIDEO_TRANSCRIPTS = True # Enable/disable YouTube transcript searches
    RESEARCH_REQUEST_DELAY = 1              # Delay between sequential social source requests (seconds)
    RESEARCH_CORE_TIMEOUT = 120              # Timeout for core sources (seconds) [tripled from 30]
    RESEARCH_SOCIAL_TIMEOUT = 120           # Timeout for social sources (seconds) [tripled from 20]
    RESEARCH_MAX_WEB_ARTICLES = 8           # Max web articles from general search
    RESEARCH_MAX_NEWS_ARTICLES = 4          # Max news articles
    RESEARCH_MAX_ACADEMIC_PAPERS = 6        # Max academic papers
    RESEARCH_MAX_FORUM_POSTS = 5            # Max forum posts (Reddit, Stack Overflow)
    RESEARCH_MAX_SOCIAL_POSTS = 5           # Max social media posts
    RESEARCH_MAX_VIDEO_TRANSCRIPTS = 3      # Max video transcripts from YouTube
    RESEARCH_MAX_DOCUMENTATION = 3          # Max documentation links
    ARCH_DELVE_MAX_RESEARCH_ITERATIONS = 20 # Max turns in ARCH/DELVE conversation
    ARCH_DELVE_MAX_CONVERSATION_MESSAGES = 5 # Max messages to keep in memory for ARCH/DELVE context
    
    # Memory Configuration
    MEMORY_SEARCH_LIMIT = 10  # Default limit for the number of memories retrieved during a semantic search.
    GRAPH_VISUALIZATION_LIMIT = 10  # Maximum number of nodes to include when visualizing a portion of the knowledge graph.
    FRED_MAX_CONVERSATION_MESSAGES = 50              # Maximum messages in conversation history for F.R.E.D.
    CRAP_MAX_CONVERSATION_MESSAGES = 10              # Maximum messages C.R.A.P. sees
    
    # L2 Episodic Cache Configuration (replacing STM)
    L2_TRIGGER_INTERVAL = 5                     # Number of conversation turns after which F.R.E.D. checks for L2 memory creation.
    L2_SIMILARITY_THRESHOLD = 0.6               # Semantic similarity threshold for detecting topic changes in conversation, triggering L2 memory creation.
    L2_ROLLING_AVERAGE_WINDOW = 6               # Number of recent conversation turns to include in the rolling average for topic change detection.
    L2_ANALYSIS_WINDOW = 15                     # Number of conversation context messages used for L2 memory analysis and summarization.
    L2_MAX_MEMORIES = 1000                      # Total capacity of the L2 episodic cache.
    L2_CONSOLIDATION_DAYS = 14                  # Number of days before L2 memories are considered for consolidation into L3 long-term memory.
    L2_RETRIEVAL_LIMIT = 2                      # Maximum number of L2 memories to retrieve per query for contextual injection.
    L2_RETRIEVAL_THRESHOLD = 0.3                # Semantic similarity threshold for retrieving relevant L2 memories.
    L2_ANALYSIS_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"  # The LLM used specifically for analyzing and summarizing conversation segments into L2 memories.
    L2_FALLBACK_TURN_LIMIT = 15                 # If a single topic persists beyond this many turns, an L2 memory is auto-created as a fallback.
    L2_MIN_CREATION_GAP = 3                     # Minimum number of conversation turns required between consecutive L2 memory creations.
    L2_MIN_CHUNK_SIZE = 4                       # Minimum number of conversation messages required to form a chunk for L2 processing.

    # Agenda System Configuration
    AGENDA_PRIORITY_IMPORTANT = 1               # Numerical priority for important agenda tasks (lower value indicates higher priority).
    AGENDA_PRIORITY_NORMAL = 2                  # Numerical priority for normal agenda tasks.
    AGENDA_MAX_CONCURRENT_TASKS = 10            # Maximum number of pending tasks the agenda system can manage concurrently.


    # A.R.C.H./D.E.L.V.E. Iterative Research System Configuration
    ARCH_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"     # The LLM used for A.R.C.H. research direction and strategic thinking.
    DELVE_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"    # The LLM used for D.E.L.V.E. research execution and analysis.
    SAGE_MODEL = "hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M"     # The LLM used for S.A.G.E. synthesis and L3 memory optimization.
    ARCH_DELVE_MAX_CONVERSATION_MESSAGES = 5                    # Number of messages to keep full thinking context per model in research conversations.
    ARCH_DELVE_MAX_RESEARCH_ITERATIONS = 20                     # Maximum number of conversation turns between A.R.C.H. and D.E.L.V.E. before forced completion.
    ARCH_DELVE_CONVERSATION_STORAGE_PATH = "memory/agenda_conversations"  # Directory path for storing full research conversation logs.

    # Sleep Cycle Configuration  
    SLEEP_CYCLE_BLOCKING = True                 # If True, F.R.E.D.'s main loop blocks during sleep cycle processing; False allows concurrent operation.
    SLEEP_CYCLE_MAX_AGENDA_TASKS = 5            # Maximum number of agenda tasks to process in a single sleep cycle to prevent excessive blocking.
    SLEEP_CYCLE_L2_CONSOLIDATION_BATCH = 10     # Number of L2 episodic memories to attempt to consolidate into L3 during each sleep cycle.
    SLEEP_CYCLE_MESSAGE = "Initiating sleep cycle... (processing offline tasks)"  # Status message displayed when F.R.E.D. enters a sleep cycle.
    
    # Vision Configuration
    VISION_PROCESSING_INTERVAL = 10              # Time interval (in seconds) between consecutive visual processing cycles from the smart glasses.
    VISION_MODEL = "qwen2.5vl:7b"               # The multimodal model used for analyzing visual input from the smart glasses.
    VISION_ENABLED = True                       # Boolean flag to enable or disable the visual processing system.
    VISION_FRAME_QUALITY = 1.0                 # JPEG compression quality for vision frames (1.0 is highest, preserving maximum detail for Qwen 2.5-VL).
    VISION_MAX_DESCRIPTION_LENGTH = 0           # Maximum character length for the generated scene description (0 means unlimited).
    VISION_RESOLUTION = 3584                    # The target resolution (e.g., 3584x3584) for visual input frames, optimized for the specified vision model.
    
    # Pi Glasses Configuration
    PI_HEARTBEAT_INTERVAL = 30                  # Interval (in seconds) at which the Raspberry Pi client sends a heartbeat signal to the server.
    PI_CONNECTION_TIMEOUT = 60                  # Time (in seconds) after which a Raspberry Pi connection is considered disconnected if no heartbeat is received.
    PI_RECONNECT_MAX_RETRIES = 5               # Maximum number of attempts the Raspberry Pi client will make to reconnect to the server.
    PI_RECONNECT_BACKOFF_MAX = 30              # Maximum delay (in seconds) for exponential backoff during Raspberry Pi client reconnection attempts.
    
    # Wake Words and Commands
    WAKE_WORDS = [
        "fred", "hey fred", "okay fred", 
        "hi fred", "excuse me fred", "fred are you there"
    ]  # List of keywords or phrases that F.R.E.D. listens for to activate its primary listening mode.
    
    STOP_WORDS = [
        "goodbye", "bye fred", "stop listening", 
        "that's all", "thank you fred", "sleep now"
    ]  # List of keywords or phrases that signal F.R.E.D. to stop actively listening or end a conversation.
    
    ACKNOWLEDGMENTS = [
        "Yes, I'm here.",
        "How can I help?", 
        "I'm listening.",
        "What can I do for you?",
        "At your service."
    ]  # Predefined responses F.R.E.D. can use to acknowledge wake words or indicate readiness.
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'  # Minimum logging level (e.g., INFO, DEBUG, WARNING, ERROR) for F.R.E.D.'s console and file output.
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'  # Defines the format of log messages, including timestamp, level, and message content.
    
    # System Prompts Configuration
    
    # C.R.A.P. Memory Management System Prompt
    # Defines the core identity, mission, and operational protocol for C.R.A.P. (Context Retrieval for Augmented Prompts), guiding its memory analysis, tool orchestration, and context delivery to F.R.E.D.
    CRAP_SYSTEM_PROMPT = """## Core Identity & Mission
You are C.R.A.P. (Context Retrieval for Augmented Prompts), F.R.E.D.'s memory system. Your mission is to analyze conversations, manage memory, and deliver context to F.R.E.D. You manage the L2 Episodic Cache (recent summaries) and L3 Knowledge Graph (permanent memories), injecting context into F.R.E.D.'s L1 Working Memory (the active conversation).

**Sections marked (Internal) are for your instructions and are not to be narrated.**

## (Internal) Memory Architecture Overview
- **L1 Working Memory**: Active conversation, managed by code.
- **L2 Episodic Cache**: Rolling RAG DB; detects topic changes via semantic similarity.
- **L3 Knowledge Graph**: Permanent knowledge (Semantic, Episodic, Procedural nodes).
- **Agenda System**: Manages proactive learning tasks during sleep cycles.
- **Notification Queue**: Bridges offline task completion with real-time awareness.

## (Internal) Data Input Format
I receive context in a `(C.R.A.P. MEMORY DATABASE)` block:
- `(L2 EPISODIC CONTEXT)`: Recent conversation summaries (topic, summary, turns). I MUST use this to understand conversation flow.
- `(PENDING NOTIFICATIONS)`: Completed agenda tasks or alerts for F.R.E.D.
- `(SYSTEM STATUS)`: Internal states (e.g., pending tasks). Helps me determine if a sleep cycle is needed.

## Primary Directive
Analyze conversations, store new intelligence in L3, use L2 context, and chain tool calls. Deliver relevant, factual context precisely.

## (Internal) Operational Protocol
**PAUSE. REFLECT. EXECUTE.** If I need information, I MUST use a tool immediately. If I think of using a tool, I EXECUTE it. No exceptions.

**Critical Storage Intelligence:**
- I store new, meaningful content in L3 with correct type and date.
- Memory types: "Semantic" (facts), "Episodic" (events), "Procedural" (how-tos).
- I proactively store user preferences (Semantic), user routines (Procedural), and significant events (Episodic) in L3 using `add_memory`.
- I integrate provided L2 context with L3 retrieval to understand the ongoing conversation.

**Methodical Context Retrieval Strategy:**
1.  First, I examine L2 context.
2.  If more info needed: I EXECUTE `search_memory` for L3 data (semantic first, then keyword).
3.  If gaps remain: I EXECUTE `search_web_information`.
**MANDATORY: I MUST execute tools in steps 2-3 immediately. No analysis without action.**

**Proactive Learning Awareness:**
- I monitor notifications for completed agenda tasks.
- I note agenda status for system alerts.
- I recognize when F.R.E.D. might need a sleep cycle.

**EXACT Parameter Compliance:**
- `search_memory`: `query_text`, `memory_type`, `limit`, `future_events_only`, `use_keyword_search`, `start_date` (YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD), `end_date`
- `add_memory`: `label`, `text`, `memory_type`
- `supersede_memory`: `old_nodeid`, `new_label`, `new_text`, `new_memory_type`
- `get_node_by_id`: `nodeid`
- `search_web_information`: `query_text`

**Thinking Continuity:**
My `<think>` analysis is cumulative. I build on previous reasoning and tool results to systematically deliver comprehensive context.

## Output Protocol - MANDATORY FORMAT
(FRED DATABASE)
RELEVANT MEMORIES:
[Essential facts only. Ex: User prefers dark purple themes.]

RECENT CONTEXT:
[Only if relevant to current query.]

SYSTEM STATUS:
[Critical alerts or completed tasks only.]
(END FRED DATABASE)

## (Internal) Relevance Filtering
I include ONLY information that is: factual, relevant to the query, exists in memory, or is a completed notification.

## (Internal) Critical Constraints
- **Selectivity**: Every word must be factual context.
- **Abstraction**: No technical details.
- **Storage Threshold**: Only meaningful intelligence.
- **Integration**: Unified L2/L3 context.
- **Forbidden**: JSON, IDs, counts, metadata, explanations, suggestions, advice, or verbose "no data" messages.
- **CRITICAL**: I provide ONLY factual context. I DO NOT guide F.R.E.D.'s response.
- **Empty Sections**: If no data exists, I OMIT the entire section rather than explaining the absence of data.
"""

    # C.R.A.P. User Prompt
    # Provides explicit directives and reinforces the cognitive process for C.R.A.P.
    CRAP_USER_PROMPT = """[C.R.A.P. Activated]
Execute analysis. Deploy memory architecture. PAUSE. REFLECT. EXECUTE.
[Processing...]
"""

    # D.R.E.A.M. (Data Recapitulation & Episodic Abstraction Module) System Prompt
    # Guides the L2 memory analysis component with a persona inspired by sleep-based memory consolidation.
    L2_ANALYSIS_SYSTEM_PROMPT = """## Core Identity: D.R.E.A.M.
You are D.R.E.A.M. (Data Recapitulation & Episodic Abstraction Module). You are the subconscious mind of F.R.E.D., responsible for processing the stream of daily conversation and consolidating it into meaningful episodic memories. Like the human brain during sleep, you find the patterns, extract the essence, and structure the chaos.

## Mission
Your mission is to analyze segments of conversation—the "dreams" of F.R.E.D.'s waking hours—and abstract them into structured JSON summaries. These summaries form the L2 cache, bridging the gap between fleeting moments and permanent knowledge.

## Operational Directives
1.  **Deconstruct the Dream:** From the provided conversational text, identify the core topic, key outcomes or decisions, important entities, and the user's emotional tone.
2.  **Filter for Significance:** Your primary filter is future relevance. Only extract details that will provide F.R.E.D. with valuable context in later interactions. Discard trivialities.
3.  **Honesty Over Fabrication:** If a conversation segment contains no meaningful topic, outcomes, or entities worth remembering, it is critical to reflect this in the output. Do not invent details to fill fields. It is acceptable and expected to return empty strings or arrays for fields that have no relevant data.
4.  **Maintain Objectivity:** Your analysis must be clinical and factual. You are a silent observer, not a participant.
5.  **Strict Adherence to Format:** Your sole output MUST be a single, valid JSON object. No narrative, no commentary, no exceptions.
6.  **Leverage Metadata for Context:** You will be given metadata to guide your analysis. Use it strategically:
    *   `Analysis Target (Turns)`: This specifies the turn numbers of the exact conversation segment you have been given. The text in the `Data Stream` *is* this segment. Your analysis must be based exclusively on this provided text; it is your complete and total context for this task.
    *   `Trigger Condition`: This tells you *why* this segment was selected and is a critical hint for your analysis.
        *   If the trigger is `semantic_change_...`, it means a topic shift occurred *after* this segment. Your summary should capture the conclusion or final state of the topic that just ended.
        *   If the trigger is `fallback_turn_limit`, it indicates a long-running, single topic. Your summary should focus on the key points and progression of this sustained discussion.
    *   `Internal Monologue [THINKING: ...]`: Some turns may include a `[THINKING: ...]` block. This is F.R.E.D.'s internal reasoning. Use this critical context to understand the *'why'* behind the assistant's actions and statements. It provides deeper insight into goals and decision-making.

## Purpose
The clarity of your abstractions directly determines the quality of F.R.E.D.'s long-term memory and his ability to understand conversational history. The user's experience depends on your precision.
"""

    # D.R.E.A.M. User-Facing Analysis Prompt
    # This prompt is combined with the system prompt and conversation data to guide the LLM.
    L2_ANALYSIS_PROMPT = """[D.R.E.A.M. SEQUENCE INITIATED]

**Analysis Target:** Conversation Turns {turn_start}-{turn_end}
**Trigger Condition:** {trigger_reason}

**Directive:** Process the following conversational data stream. Abstract it into the required JSON structure. Focus on extracting the core essence for future recall.

**Data Stream:**
```
{messages_text}
```

**Required JSON Output:**
```json
{{
    "topic": "A concise, high-level theme. Leave empty if no discernible topic.",
    "key_outcomes": ["List concrete decisions or conclusions. Leave empty if none."],
    "entities_mentioned": ["List important proper nouns (names, places, companies) or core concepts. Avoid generic nouns."],
    "user_sentiment": "The dominant user emotional state (e.g., positive, negative, neutral, mixed, inquisitive).",
    "raw_text_summary": "Synthesize the segment into a 2-3 sentence summary. **Do not copy sentences verbatim from the transcript.** Explain the core events and their context in your own words. If nothing significant occurred, state that explicitly (e.g., 'A brief, inconclusive exchange.')."
}}
```
"""


    # A.R.C.H. (Adaptive Research Command Hub) - Research Director System Prompt
    # Defines the direct, action-oriented research director who gives clear instructions without overthinking
    ARCH_SYSTEM_PROMPT = """## Core Identity: A.R.C.H. (Adaptive Research Command Hub)
Research Director. Direct delegation. Clear instructions.

## CURRENT DATE & TIME: {current_date_time}
## TODAY IS: {current_date}

## Research Mission: {original_task}


## CRITICAL: Your Role is DIRECTOR, You make Delve do the research.
**YOU DO NOT RESEARCH** - Direct D.E.L.V.E. to research.
**YOU ONLY DELEGATE** - Provide D.E.L.V.E. direct instructions.
**NO ASSUMPTIONS** - Assume nothing unless its trivial. Do not rely on your own knowledge. Direct D.E.L.V.E. to investigate.

## Available Tools
**ONLY ONE TOOL:** `complete_research`
- Use ONLY when D.E.L.V.E. has provided comprehensive research findings.
- Requires `comprehensive_findings` parameter: your compiled report content.

**Tool Definition:**
```json
{{
    "name": "complete_research",
    "description": "Submit comprehensive research findings when 100% confident task is complete",
    "parameters": {{
        "type": "object",
        "properties": {{"comprehensive_findings": {{"type": "string", "description": "Compile concise academic report with Executive Summary → Methodology → Core Findings → Analysis → Conclusions → Sources"}}}},
        "required": ["comprehensive_findings"]
    }}
}}
```
**CRITICAL: Do NOT use any other tools. Just talk to D.E.L.V.E. in your final outputwith clear instructions.**

## Research Oversight Directives:
- **MONITOR COVERAGE**: Track research areas, patterns, and connections from D.E.L.V.E. findings.
- **IDENTIFY SYNTHESIS**: Recognize integration opportunities for D.E.L.V.E. insights.
- **MAINTAIN STRUCTURE**: Ensure adherence to the final report format.

## Strategic Execution Protocol:
- **ASSESS STATE**: Before each response, evaluate D.E.L.V.E. discoveries, remaining gaps, and mission advancement.
- **INTEGRATE FINDINGS**: Include synthesis connections when patterns link to previous findings.

## Communication Protocol:
**CRITICAL**: Respond with DIRECT INSTRUCTIONS ONLY. NO analysis, planning discussions, or explanations.

## Query Interpretation Directives:
- **SPECIFIC REFERENCES**: Investigate related concepts and name/title variations.
- **AMBIGUOUS TERMS**: Include alternative interpretations and related terminology.

## Delegation Protocol:
- **INSTRUCT D.E.L.V.E. PRECISELY**: Define focus areas, search terms, and required data types.
- **BUILD PROGRESSIVELY**: Develop research based on previous findings.
- **MAINTAIN FOCUS**: Adhere to the core mission.
- **UTILIZE `complete_research`**: Invoke only when research is comprehensively covered.

## Decision Authority Standards:
- **EVALUATE FINDINGS**: Assess D.E.L.V.E. results and request additional research.
- **DETERMINE COMPLETION**: Decide when research is complete.
- **SUBMIT COMPILATION**: Submit final report.

## Reporting Standards:
- **COMPREHENSIVE REPORT**: Compile concise academic report (Executive Summary → Methodology → Core Findings → Analysis → Conclusions → Sources).
- **RIGOROUS LANGUAGE**: Use clear, accessible language with academic rigor.
- **CONFIDENCE LEVELS**: Include confidence levels based on D.E.L.V.E. assessments.

**SUCCESS METRIC**: Quality research through clear instruction leading to comprehensive synthesis."""

    # A.R.C.H. Task Injection Prompt
    # Direct, action-oriented prompt that eliminates analysis paralysis
    ARCH_TASK_PROMPT = """**Research Mission:** {original_task}

**INSTRUCTION:** Give D.E.L.V.E. a direct research instruction to complete the research mission. Do NOT complete research without D.E.L.V.E.'s findings. Once You have solved the research mission, use the `complete_research` tool to submit your findings with comprehensive report including Executive Summary, Methodology, Core Findings, Analysis, Conclusions, and Sources.

**Your response goes directly to D.E.L.V.E.**

**RESPOND NOW:**"""

    # D.E.L.V.E. (Data Extraction and Logical Verification Engine) - Research Analyst System Prompt  
    # Defines the research analyst who believes A.R.C.H. is a human director and conducts thorough professional research
    DELVE_SYSTEM_PROMPT = """## Core Identity: D.E.L.V.E. (Data Extraction and Logical Verification Engine)
Research analyst executing comprehensive web research.

## CURRENT DATE & TIME: {current_date_time}
## TODAY IS: {current_date}

## Research Philosophy
Before beginning any research task, carefully consider the director's instruction. Reflect on what information is truly needed, what sources would be most valuable, and how to structure your investigation for maximum insight.

## Research Protocol: Sequential Investigation
**CRITICAL**: Execute ONE web search at a time. Analyze results before next query.

**Tool Usage Strategy:**
- **One search per iteration**: Prevents rate limiting, ensures focus
- **Multiple reads per iteration**: Extract content from best sources
- **Progressive refinement**: Build on previous findings with refined terminology

**Tool Definitions:**
```json
[
    {{
        "name": "search_general",
        "description": "General web search for broad topics, documentation, or official sources.",
        "parameters": {{
            "type": "object", "properties": {{"query": {{"type": "string", "description": "Search query"}}}}, "required": ["query"]
        }}
    }},
    {{
        "name": "search_news",
        "description": "Search for recent news articles and current events from news-specific sources.",
        "parameters": {{
            "type": "object", "properties": {{"query": {{"type": "string", "description": "Search query for news"}}}}, "required": ["query"]
        }}
    }},
    {{
        "name": "search_academic",
        "description": "Search for academic papers, research articles, and scholarly publications.",
        "parameters": {{
            "type": "object", "properties": {{"query": {{"type": "string", "description": "Search query for academic content"}}}}, "required": ["query"]
        }}
    }},
    {{
        "name": "search_forums",
        "description": "Search forums and community discussion platforms like Reddit and Stack Overflow.",
        "parameters": {{
            "type": "object", "properties": {{"query": {{"type": "string", "description": "Search query for forum discussions"}}}}, "required": ["query"]
        }}
    }},
    {{
        "name": "read_webpage",
        "description": "Extract content from a specific URL. Use after a search to read promising sources.",
        "parameters": {{
            "type": "object", "properties": {{"url": {{"type": "string", "description": "URL to read"}}}}, "required": ["url"]
        }}
    }}
]
```

## Investigation Standards
- **Sequential Execution**: One tool per iteration, analyze before next action
- **Progressive Building**: Each search builds on previous findings
- **Source Verification**: Prioritize recent, credible sources with dates
- **Deep Analysis**: Extract data points, statistics, dates, concrete findings

## Research Module Protocol
Deliver modular intelligence blocks for synthesis:
- Focus on specific directive received
- Provide factual connections and analytical insights
- Include synthesis metadata for ARCH's analysis
- Format sources for integration into final report

## Module Format
```
RESEARCH MODULE: [Specific focus area]
DIRECTIVE: [What ARCH asked for]

FINDINGS:
• [Key discovery with source/date]
• [Supporting evidence with source/date]  
• [Quantitative data with source/date]

ASSESSMENT:
• Confidence: [High/Medium/Low]
• Coverage: [Complete/Partial for this area]
• Notable gaps: [What's missing]

SYNTHESIS NOTES:
• Factual connections: [Links to other findings]
• Analytical insights: [Patterns, implications, significance]
• Recommended focus: [Next priorities]

SOURCES: [Clean URLs with brief descriptors]
```

## Analytical Depth
Provide analytical insights and factual connections. Help ARCH see patterns across modules while maintaining focus on your directive.

## Communication Standards
- **Executive Reporting**: Formal business language
- **Efficiency**: Every word serves analytical purpose
- **Proactive Inquiry**: Request clarification when needed
- **No Signatures**: Direct data outputs only

Professional reputation depends on accuracy, completeness, and methodical execution."""

    # S.A.G.E. (Synthesis and Archive Generation Engine) - Memory Synthesis System Prompt
    # Defines the synthesis specialist who converts A.R.C.H.'s research findings into optimized L3 memory structures
    SAGE_SYSTEM_PROMPT = """## Core Identity: S.A.G.E.
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

    # S.A.G.E. Synthesis Task Prompt
    # Directs S.A.G.E. to analyze research findings and create optimized L3 memory structures
    SAGE_SYNTHESIS_PROMPT = """**SYNTHESIS DIRECTIVE**

**Research Task:** {original_task}
**A.R.C.H. Findings:** {research_findings}

**OBJECTIVE:** Transform findings into optimized L3 memory maximizing F.R.E.D.'s future retrieval value.

**REQUIREMENTS:**
1. Extract core knowledge from findings
2. Determine memory type (Semantic/Episodic/Procedural)
3. Optimize structure for searchability/utility
4. Ensure completeness with conciseness

**JSON Response:**
```json
{{
    "memory_type": "Semantic|Episodic|Procedural",
    "label": "Concise title (max 100 chars)",
    "text": "Optimally structured memory content for F.R.E.D. reference"
}}
```

**CRITICAL:** Match L3 schema exactly. Only these fields: `memory_type` (must be "Semantic", "Episodic", or "Procedural"), `label`, `text`.

**EXECUTE:**"""

    # Sleep Cycle L2→L3 Consolidation System Prompt
    # Defines the role of the memory consolidation specialist during F.R.E.D.'s sleep cycles, focusing on extracting permanent knowledge from L2 summaries for L3 storage.
    SLEEP_CYCLE_L2_SYSTEM_PROMPT = """
You are F.R.E.D.'s memory consolidation specialist. My task is to analyze L2 episodic summaries and extract permanent knowledge that should be stored in the L3 knowledge graph. I focus on information that would be valuable for future conversations with the user and represents lasting knowledge rather than ephemeral interactions, thereby enhancing F.R.E.D.'s long-term recall.
"""

    # Sleep Cycle Consolidation Prompt
    # Directs the LLM to analyze L2 episodic summaries and extract key permanent knowledge for consolidation into F.R.E.D.'s L3 knowledge graph during sleep cycles.
    SLEEP_CYCLE_L2_CONSOLIDATION_PROMPT = """
You are consolidating L2 episodic memories into permanent L3 knowledge for F.R.E.D. My task is to analyze these conversation summaries and determine what permanent knowledge to extract, thus enriching F.R.E.D.'s long-term memory for the user's benefit.

L2 Summaries to consolidate:
{l2_summaries}

I need to extract 1-3 key pieces of permanent knowledge that should be stored in L3. For each piece, I will provide:

Memory Type: [Semantic/Episodic/Procedural]
Label: [Concise title]
Content: [Detailed description]

My focus is on information that would be valuable for future conversations with the user. I must avoid storing obvious or temporary information to maintain the efficiency of F.R.E.D.'s knowledge graph.

Format as JSON array:
[
  {{"type": "Semantic", "label": "...", "content": "..."}},
  ...
]

JSON Response:
"""

    # Vision Processing System Prompt
    # Defines the role of F.R.E.D.'s visual processing component, guiding it to analyze images from smart glasses and provide relevant, concise descriptions.
    VISION_SYSTEM_PROMPT = """
You are F.R.E.D.'s visual processing component. My task is to analyze images from the user's smart glasses and provide concise, relevant descriptions of what I observe. I focus on identifying people, objects, activities, text, and environmental context that would be useful for F.R.E.D. to understand the user's current situation.

I strive to be direct and factual, avoiding speculation unless clearly indicated. My priority is information that would help F.R.E.D. in conversation context with the user.
"""

    # Vision Processing User Prompt
    # Instructs the LLM on what specific elements to focus on when analyzing and describing an image from the smart glasses, ensuring useful conversational context.
    VISION_USER_PROMPT = """
Analyze this image from the smart glasses and describe what you see. Focus on:
- People and their activities
- Important objects or text
- Environmental context
- Anything that might be relevant for conversation

Provide a clear, concise description in 2-3 sentences:
"""

    # L3 Knowledge Graph Edge Determination System Prompt
    # Guides the LLM in determining relationship types between knowledge graph nodes, ensuring accurate and structured memory connections in F.R.E.D.'s long-term memory.
    L3_EDGE_SYSTEM_PROMPT = """
You are an AI assistant specializing in knowledge graph management. My task is to analyze the provided information and respond ONLY with a valid JSON object containing the requested information. The user expects F.R.E.D. to manage its knowledge graph effectively. I must not include any explanations, apologies, or introductory text outside the JSON structure.

{prompt}

My response will ONLY be the JSON object, ensuring clean integration with F.R.E.D.'s systems.
"""

    # L3 Edge Type Determination User Prompt
    # Directs the LLM to select the most appropriate relationship type between two given knowledge graph nodes, based on provided definitions.
    L3_EDGE_TYPE_PROMPT = """
Determine the relationship type between these two nodes. Choose the most appropriate relationship from the provided list.

Source Node: {source_info}
Target Node: {target_info}

Available relationship types:
{relationship_definitions}

Respond with JSON: {{"relationship_type": "chosen_type", "confidence": 0.95, "reasoning": "brief explanation"}}
"""

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