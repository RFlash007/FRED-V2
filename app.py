import os
import ollama
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import json
import requests
import memory.librarian as lib
import re
import playsound
import uuid
import subprocess
import platform
import glob
from Tools import AVAILABLE_TOOLS, handle_tool_calls
from datetime import datetime
import torch
from TTS.api import TTS
import traceback
from stt_service import stt_service
from vision_service import vision_service
import threading
from ollietec_theme import apply_theme, banner
from ollie_print import olliePrint, olliePrint_simple

apply_theme()
import time
from config import config

# Import STM module
import memory.short_term_memory as stm

# State Management Class
class FREDState:
    """Centralized state management for F.R.E.D."""
    
    def __init__(self):
        self.conversation_history = []
        self.tts_engine = None
        self.last_played_wav = None
        self.stt_enabled = True
        self.total_conversation_turns = 0  # Track conversation turns for STM
        self.last_analyzed_message_index = 0  # Track last analyzed message for STM
        self.stewie_voice_available = False  # Track if Stewie voice cloning is available
        self._lock = threading.Lock()
    
    def add_conversation_turn(self, role, content, thinking=None):
        """Thread-safe conversation history management with automatic cleanup."""
        with self._lock:
            turn = {'role': role, 'content': content}
            if thinking:
                turn['thinking'] = thinking
            self.conversation_history.append(turn)
            
            # Check if we need to remove old messages
            if len(self.conversation_history) > config.MAX_CONVERSATION_MESSAGES:
                messages_to_remove = len(self.conversation_history) - config.MAX_CONVERSATION_MESSAGES
                
                # Log the cleanup action
                olliePrint_simple(f"Conversation history cleanup: removing {messages_to_remove} old messages (keeping {config.MAX_CONVERSATION_MESSAGES})")
                
                # Remove oldest messages
                self.conversation_history = self.conversation_history[messages_to_remove:]
                
                # Adjust STM tracking indices to account for removed messages
                self.last_analyzed_message_index = max(0, self.last_analyzed_message_index - messages_to_remove)
                
                olliePrint_simple(f"Adjusted last_analyzed_message_index to {self.last_analyzed_message_index}")
            
            # Increment turn counter and check STM trigger
            if role == 'assistant':  # Complete turn (user + assistant)
                self.total_conversation_turns += 1
                if self.total_conversation_turns % config.STM_TRIGGER_INTERVAL == 0:
                    # Trigger STM analysis in background
                    threading.Thread(
                        target=stm.process_stm_analysis,
                        args=(
                            self.conversation_history.copy(), 
                            self.last_analyzed_message_index,
                            len(self.conversation_history)
                        ),
                        daemon=True
                    ).start()
                    # Update analyzed index to current position
                    self.last_analyzed_message_index = len(self.conversation_history)
    
    def get_conversation_history(self):
        """Get a copy of conversation history."""
        with self._lock:
            return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        with self._lock:
            self.conversation_history.clear()
    
    def set_tts_engine(self, engine):
        """Thread-safe TTS engine setting."""
        with self._lock:
            self.tts_engine = engine
    
    def get_tts_engine(self):
        """Get TTS engine."""
        with self._lock:
            return self.tts_engine
    
    def get_conversation_stats(self):
        """Get conversation statistics for monitoring."""
        with self._lock:
            return {
                'current_messages': len(self.conversation_history),
                'max_messages': config.MAX_CONVERSATION_MESSAGES,
                'total_turns': self.total_conversation_turns,
                'last_analyzed_index': self.last_analyzed_message_index
            }

# Global state instance
fred_state = FREDState()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration handled via olliePrint; reduce Flask request noise
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = config.get_db_path(APP_ROOT)
lib.DB_FILE = DB_PATH
FRED_CORE_NODE_ID = "FRED_CORE"

def initialize_tts():
    """Initialize TTS engine once during startup."""
    try:
        olliePrint_simple("Voice synthesis initializing...", 'audio')
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Stewie voice cloning if enabled
        if config.STEWIE_VOICE_ENABLED:
            from stewie_voice_clone import initialize_stewie_voice, validate_stewie_samples
            
            stewie_success = initialize_stewie_voice()
            if stewie_success:
                # Validate voice samples
                sample_stats = validate_stewie_samples()
                olliePrint_simple(f"[STEWIE-CLONE] Found {sample_stats['total_samples']} voice samples", 'audio')
                olliePrint_simple(f"[STEWIE-CLONE] Total duration: {sample_stats['total_duration']:.1f}s", 'audio')
                olliePrint_simple(f"[STEWIE-CLONE] Stewie voice cloning ACTIVE!", 'success')
                
                # Set a flag to indicate Stewie voice is available
                fred_state.stewie_voice_available = True
            else:
                olliePrint_simple("[STEWIE-CLONE] Failed to initialize - falling back to standard TTS", 'warning')
                fred_state.stewie_voice_available = False
        else:
            fred_state.stewie_voice_available = False
        
        # Initialize standard TTS as fallback
        tts_engine = TTS(config.XTTS_MODEL_NAME).to(device)
        fred_state.set_tts_engine(tts_engine)
        
        if config.STEWIE_VOICE_ENABLED and fred_state.stewie_voice_available:
            olliePrint_simple(f"Voice synthesis ready on {device.upper()} with STEWIE VOICE CLONING", 'success')
        elif os.path.exists(config.FRED_SPEAKER_WAV_PATH):
            olliePrint_simple(f"Voice synthesis ready on {device.upper()} with custom voice", 'audio')
        else:
            olliePrint_simple(f"Voice synthesis ready on {device.upper()} with default voice", 'audio')
            
    except Exception as e:
        olliePrint_simple(f"Voice synthesis failed: {e}", 'critical')
        fred_state.set_tts_engine(None)
        fred_state.stewie_voice_available = False

# Load System Prompt
SYSTEM_PROMPT_FILE = os.path.join(APP_ROOT, 'system_prompt.txt')
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()
except Exception as e:
    olliePrint_simple(f"Error loading system prompt: {e}", 'error')
    SYSTEM_PROMPT = "You are F.R.E.D., a helpful AI assistant."

def extract_think_content(text):
    """Extract thinking content from <think>...</think> tags."""
    if not text:
        return ""
    matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    return '\n'.join(matches).strip()

def strip_think_tags(text):
    """Remove <think>...</think> blocks from text."""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def summarize_thinking(thinking_content, ollama_client):
    """Summarize thinking content using Qwen3-4B model."""
    if not thinking_content.strip():
        return ""
    
    try:
        prompt = f"""Condense the following Language model reasoning into exactly 2-3 sentences. Focus only on key insights and decisions. Do NOT use thinking tags or meta-commentary, maintain the original perspective:

{thinking_content}

Condensed reasoning (2-3 sentences only):"""

        # Remove verbose logging - just show result
        response = ollama_client.chat(
            model="hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.5, "max_tokens": 150}
        )
        raw_summary_response = response.get('message', {}).get('content', '').strip()
        summary = strip_think_tags(raw_summary_response).strip()
        
        # Only log if there's an actual summary generated
        if summary and len(summary) > 10:
            olliePrint_simple(f"Thinking condensed: {summary[:100]}{'...' if len(summary) > 100 else ''}")
        
        return summary
    except Exception as e:
        olliePrint_simple(f"Thinking summary failed: {e}", 'warning')
        return ""

def prepare_messages_with_thinking(system_prompt, user_message, ollama_client):
    """Prepare messages with appropriate thinking context based on recency."""
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history with thinking context
    for i, turn in enumerate(fred_state.get_conversation_history()):
        age = len(fred_state.get_conversation_history()) - i
        
        if turn['role'] == 'user':
            messages.append({"role": "user", "content": turn['content']})
        elif turn['role'] == 'assistant':
            content = turn['content']
            thinking = turn.get('thinking', '')
            
            if age <= 3 and thinking:
                # Recent messages: include full thinking
                olliePrint_simple(f"[THINKING] Including full thinking for recent message (age {age})")
                full_content = f"<think>\n{thinking}\n</think>\n{content}"
                olliePrint_simple("[THINKING CONTEXT] Message with full thinking being sent to model:")
                olliePrint_simple(f"{'='*50}")
                olliePrint_simple(full_content)
                olliePrint_simple(f"{'='*50}\n")
                messages.append({"role": "assistant", "content": full_content})
            elif age <= 6 and thinking:
                # Older messages: include summarized thinking
                olliePrint_simple(f"[THINKING] Including summarized thinking for older message (age {age})")
                if not hasattr(prepare_messages_with_thinking, '_thinking_cache'):
                    prepare_messages_with_thinking._thinking_cache = {}
                
                cache_key = hash(thinking)
                if cache_key not in prepare_messages_with_thinking._thinking_cache:
                    prepare_messages_with_thinking._thinking_cache[cache_key] = summarize_thinking(thinking, ollama_client)
                
                summarized = prepare_messages_with_thinking._thinking_cache[cache_key]
                if summarized:
                    full_content = f"<think>\n{summarized}\n</think>\n{content}"
                    olliePrint_simple("[THINKING CONTEXT] Message with summarized thinking being sent to model:")
                    olliePrint_simple(f"{'='*50}")
                    olliePrint_simple(full_content)
                    olliePrint_simple(f"{'='*50}\n")
                    messages.append({"role": "assistant", "content": full_content})
                else:
                    messages.append({"role": "assistant", "content": content})
            else:
                # Oldest messages: no thinking context
                if thinking:
                    olliePrint_simple(f"[THINKING] Excluding thinking for old message (age {age})")
                messages.append({"role": "assistant", "content": content})
    
    # Add current user message
    pending_tasks_alert = get_pending_tasks_alert()
    
    # Get STM context
    stm_context = stm.query_stm_context(user_message)
    stm_section = f"\n{stm_context}\n" if stm_context else ""
    
    formatted_input = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
{pending_tasks_alert}{stm_section}The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END OF FRED DATABASE)"""
    
    messages.append({"role": "user", "content": formatted_input})
    return messages

def prepare_messages_with_visual_context(system_prompt, user_message, ollama_client):
    """Prepare messages with visual context for Pi glasses input."""
    messages = prepare_messages_with_thinking(system_prompt, user_message, ollama_client)
    
    # Get current visual context
    visual_context = vision_service.get_current_visual_context()
    
    # Inject visual context into the user message
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user':
            existing_content = messages[i]['content']
            
            # Find the FRED DATABASE section and inject visual context
            if "(FRED DATABASE)" in existing_content:
                # Insert visual context after the database header
                parts = existing_content.split("(FRED DATABASE)", 1)
                if len(parts) == 2:
                    enhanced_content = f"{parts[0]}(FRED DATABASE)\nCurrent Visual Context (Pi Glasses): {visual_context}\n{parts[1]}"
                    messages[i]['content'] = enhanced_content
            break
    
    return messages

def get_pending_tasks_alert():
    """Get pending tasks alert if needed."""
    try:
        with lib.duckdb.connect(lib.DB_FILE) as con:
            count = con.execute("SELECT COUNT(*) FROM pending_edge_creation_tasks WHERE status = 'pending';").fetchone()
            if count and count[0] > 20:
                return f"Alert: {count[0]} memory nodes awaiting connection processing.\n"
    except Exception:
        pass
    return ""

def play_audio_locally(audio_file_path):
    """Robust cross-platform audio playback with Windows optimization."""
    if not os.path.exists(audio_file_path):
        olliePrint_simple(f"[ERROR] Audio file not found: {audio_file_path}")
        return
    
    try:
        system = platform.system().lower()
        olliePrint_simple(f"[AUDIO] Attempting playback on {system.upper()}: {os.path.basename(audio_file_path)}")
        
        if system == "windows":
            # Windows: Use multiple fallback methods
            try:
                # Method 1: PowerShell with Windows Media Format SDK
                cmd = [
                    "powershell", "-Command", 
                    f"(New-Object Media.SoundPlayer '{audio_file_path}').PlaySync()"
                ]
                subprocess.run(cmd, check=True, capture_output=True, timeout=None)
                olliePrint_simple(f"[SUCCESS] Audio playback via PowerShell")
                return
            except Exception as e1:
                olliePrint_simple(f"[FALLBACK] PowerShell failed ({e1}), trying playsound...")
                
                try:
                    # Method 2: playsound library
                    playsound.playsound(audio_file_path, block=False)
                    olliePrint_simple(f"[SUCCESS] Audio playback via playsound")
                    return
                except Exception as e2:
                    olliePrint_simple(f"[FALLBACK] playsound failed ({e2}), trying system call...")
                    
                    try:
                        # Method 3: Direct system call to start with associated program
                        subprocess.run(['start', '/wait', audio_file_path], shell=True, check=True, timeout=None)
                        olliePrint_simple(f"[SUCCESS] Audio playback via system start")
                        return
                    except Exception as e3:
                        olliePrint_simple(f"[ERROR] All Windows audio methods failed: {e3}")
        
        elif system == "linux":
            # Linux: Try multiple audio players
            players = ['aplay', 'paplay', 'mpg123', 'mpv', 'ffplay']
            for player in players:
                try:
                    subprocess.run([player, audio_file_path], check=True, capture_output=True, timeout=None)
                    olliePrint_simple(f"[SUCCESS] Audio playback via {player}")
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            olliePrint_simple(f"[ERROR] No working Linux audio player found")
        
        elif system == "darwin":  # macOS
            try:
                subprocess.run(['afplay', audio_file_path], check=True, capture_output=True, timeout=None)
                olliePrint_simple(f"[SUCCESS] Audio playback via afplay")
                return
            except Exception as e:
                olliePrint_simple(f"[ERROR] macOS audio playback failed: {e}")
        
        else:
            # Fallback: Try playsound for unknown systems
            try:
                playsound.playsound(audio_file_path, block=False)
                olliePrint_simple(f"[SUCCESS] Audio playback via playsound fallback")
                return
            except Exception as e:
                olliePrint_simple(f"[ERROR] Unknown system audio playback failed: {e}")
    
    except Exception as e:
        olliePrint_simple(f"[CRITICAL] Audio playback system failure: {e}")
        # Last resort: show audio file location
        olliePrint_simple(f"[INFO] Generated audio file: {audio_file_path}")
        olliePrint_simple(f"[INFO] You can manually play this file to hear F.R.E.D.'s response")

def fred_speak(text, mute_fred=False, target_device='local'):
    """Generate and play speech using TTS.
    
    Args:
        text: Text to speak
        mute_fred: Whether to mute output
        target_device: 'local' for main computer, 'pi' for Pi glasses, 'all' for both
    """
    if mute_fred:
        olliePrint_simple(f"[SHELTER-NET] Audio protocols disabled - transmission suppressed: '{text[:50]}...'")
        return
        
    if not text.strip():
        olliePrint_simple("[SHELTER-NET] Warning: Empty transmission detected - aborting voice synthesis")
        return

    olliePrint_simple(f"[F.R.E.D.] Initializing voice synthesis - Target: {target_device.upper()} | '{text[:50]}...'")

    # Cleanup previous file
    if fred_state.last_played_wav and os.path.exists(fred_state.last_played_wav):
        try:
            os.remove(fred_state.last_played_wav)
        except Exception:
            pass

    # Generate speech
    unique_id = uuid.uuid4()
    output_path = os.path.join(APP_ROOT, f"fred_speech_output_{unique_id}.wav")

    try:
        olliePrint_simple(f"[ARC-MODE] Synthesizing neural voice patterns: {output_path}")
        
        # Priority 1: Use Stewie voice cloning if available
        if config.STEWIE_VOICE_ENABLED and hasattr(fred_state, 'stewie_voice_available') and fred_state.stewie_voice_available:
            from stewie_voice_clone import generate_stewie_speech
            
            olliePrint_simple(f"[STEWIE-CLONE] Generating speech with Stewie's voice", 'audio')
            stewie_success = generate_stewie_speech(text, output_path)
            
            if stewie_success:
                olliePrint_simple(f"[STEWIE-CLONE] ✅ Voice generation successful!", 'success')
            else:
                olliePrint_simple(f"[STEWIE-CLONE] ❌ Failed - falling back to standard TTS", 'warning')
                # Fall through to standard TTS
                stewie_success = False
        else:
            stewie_success = False
        
        # Fallback: Use standard TTS if Stewie cloning failed or is disabled
        if not stewie_success:
            tts_engine = fred_state.get_tts_engine()
            if tts_engine is None:
                olliePrint_simple("TTS engine not initialized, skipping speech generation", 'warning')
                return

            # Check if we have a voice sample for cloning
            if os.path.exists(config.FRED_SPEAKER_WAV_PATH):
                olliePrint_simple(f"[VOICE-CLONE] Using custom voice sample: {config.FRED_SPEAKER_WAV_PATH}")
                tts_engine.tts_to_file(
                    text=text,
                    speaker_wav=config.FRED_SPEAKER_WAV_PATH,
                    language=config.FRED_LANGUAGE,
                    file_path=output_path
                )
            else:
                olliePrint_simple(f"[VOICE-DEFAULT] No voice sample found - using default XTTS voice")
                # Use default XTTS speaker instead of voice cloning
                tts_engine.tts_to_file(
                    text=text,
                    speaker="Ana Florence",  # Default XTTS speaker
                    language=config.FRED_LANGUAGE,
                    file_path=output_path
                )
        
        olliePrint_simple(f"[SUCCESS] Voice synthesis complete - audio matrix ready")

        # Route audio based on target device
        if target_device in ['local', 'all']:
            # Play locally on main computer
            olliePrint_simple(f"[LOCAL-COMM] Broadcasting to main terminal: '{text[:50]}...'")
            play_audio_locally(output_path)

        if target_device in ['pi', 'all']:
            # Send audio to Pi glasses
            olliePrint_simple(f"[TRANSMISSION] Routing audio to ArmLink interface...")
            send_audio_to_pi(output_path, text)

        if target_device not in ['local', 'pi', 'all']:
            olliePrint_simple(f"[ERROR] Unknown device '{target_device}' - defaulting to local broadcast")
            play_audio_locally(output_path)

        # Schedule cleanup after a delay
        def delayed_cleanup():
            time.sleep(config.TTS_CLEANUP_DELAY)  # Wait for playback to complete
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    olliePrint_simple(f"[CLEANUP] Voice file purged from memory banks")
            except Exception:
                pass

        threading.Thread(target=delayed_cleanup, daemon=True).start()
        fred_state.last_played_wav = output_path

    except Exception as e:
        olliePrint_simple(f"TTS error: {e}", level='error')
        olliePrint_simple(f"[CRITICAL] Voice synthesis failure: {e}")

def send_audio_to_pi(audio_file_path, text):
    """Send audio file to connected Pi clients."""
    try:
        import base64
        
        olliePrint_simple(f"[ARMLINK] Initiating field comm protocol...")
        olliePrint_simple(f"   Audio matrix: {audio_file_path}")
        olliePrint_simple(f"   Message: '{text[:50]}...'")
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            olliePrint_simple(f"[ERROR] Audio matrix not found in data banks: {audio_file_path}")
            return
        
        # Read audio file and encode as base64
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
        
        olliePrint_simple(f"   Data size: {len(audio_data)} bytes")
        
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        olliePrint_simple(f"   Encoded for transmission: {len(audio_b64)} chars")
        
        # Send via SocketIO to WebRTC server
        payload = {
            'audio_data': audio_b64,
            'text': text,
            'format': 'wav'
        }
        
        olliePrint_simple(f"   Broadcasting via secure channel...")
        socketio.emit('fred_audio', payload)
        
        olliePrint_simple(f"[SUCCESS] Transmission complete - audio routed to field operative")
        olliePrint_simple(f"   Payload: {len(audio_data)} bytes for '{text[:30]}...')")
        
    except Exception as e:
        olliePrint_simple(f"Error sending audio to Pi: {e}", level='error')
        olliePrint_simple(f"[CRITICAL] ArmLink transmission failure: {e}")
        import traceback
        traceback.print_exc()

def cleanup_wav_files():
    """Clean up old WAV files."""
    pattern = os.path.join(APP_ROOT, "fred_speech_output_*.wav")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except Exception:
            pass

# Initialize
try:
    lib.init_db()
    olliePrint_simple(f"[INFO] Memory DB initialized at: {DB_PATH}")
except Exception as e:
    olliePrint_simple(f"Database initialization failed: {e}", level='error')

if not os.path.exists(app.static_folder):
    os.makedirs(app.static_folder)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/graph')
def get_graph():
    """Get graph data for visualization."""
    fred_core_node = {
        "id": FRED_CORE_NODE_ID,
        "label": "F.R.E.D.",
        "type": "SystemCore",
        "text": "The Central Processing Core of F.R.E.D.",
        "isFredCore": True,
        "size": 30
    }
    
    nodes = [fred_core_node]
    edges = []
    
    try:
        with lib.duckdb.connect(lib.DB_FILE) as con:
            # Get most connected memories
            memories = con.execute("""
                SELECT n.nodeid, n.label, n.text, n.type, n.created_at, n.last_access,
                       (SELECT COUNT(*) FROM edges e WHERE e.sourceid = n.nodeid OR e.targetid = n.nodeid) AS degree
                FROM nodes n
                WHERE n.superseded_at IS NULL
                ORDER BY degree DESC
                LIMIT 10
            """).fetchall()
            
            for mem in memories:
                node_id = str(mem[0])
                nodes.append({
                    "id": node_id,
                    "label": mem[1],
                    "text": mem[2],
                    "type": mem[3],
                    "created_at": mem[4].isoformat() if mem[4] else None,
                    "last_access": mem[5].isoformat() if mem[5] else None
                })
                
                edges.append({
                    "source": node_id,
                    "target": FRED_CORE_NODE_ID,
                    "rel_type": "connectsTo"
                })
    
    except Exception as e:
        olliePrint_simple(f"Graph generation error: {e}", level='error')
    
    return jsonify({"nodes": nodes, "edges": edges})

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Main chat endpoint with enhanced thinking preservation."""
    global fred_state
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400
        
        user_message = data.get('message')
        model_name = data.get('model', config.DEFAULT_MODEL)
        ollama_base_url = data.get('ollama_url', config.OLLAMA_BASE_URL).strip()
        max_tool_iterations = data.get('max_tool_iterations', config.MAX_TOOL_ITERATIONS)
        mute_fred = data.get('mute_fred', False)
        from_pi_glasses = data.get('from_pi_glasses', False)
        
        if not user_message or not ollama_base_url:
            return jsonify({'error': 'Missing required fields'}), 400
        
        def event_stream():
            # Add user message to history
            fred_state.add_conversation_turn('user', user_message)
            
            client = ollama.Client(host=ollama_base_url)
            
            # Enhanced message preparation with visual context
            if from_pi_glasses:
                messages = prepare_messages_with_visual_context(SYSTEM_PROMPT, user_message, client)
            else:
                messages = prepare_messages_with_thinking(SYSTEM_PROMPT, user_message, client)
            
            assistant_response = ""
            raw_thinking = ""
            
            # Tool iteration loop
            for iteration in range(max_tool_iterations):
                response = client.chat(
                    model=model_name,
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    stream=False,
                    options=config.THINKING_MODE_OPTIONS
                )
                
                response_message = response.get('message', {})
                raw_content = response_message.get('content', '')
                
                # Print the model's full response including thinking
                if raw_content:
                    olliePrint_simple(f"\n{'='*60}")
                    olliePrint_simple(f"[MODEL RESPONSE] Full response from iteration {iteration + 1}:")
                    olliePrint_simple(f"{'='*60}")
                    olliePrint_simple(raw_content)
                    olliePrint_simple(f"{'='*60}\n")
                
                # Extract and store thinking
                current_thinking = extract_think_content(raw_content)
                if current_thinking:
                    olliePrint_simple(f"[THINKING] Extracted {len(current_thinking)} chars of thinking from iteration {iteration + 1}")
                    raw_thinking += current_thinking + "\n"
                
                clean_content = strip_think_tags(raw_content)
                tool_calls = response_message.get('tool_calls')
                
                # Ensure assistant role
                if 'role' not in response_message:
                    response_message['role'] = 'assistant'
                messages.append(response_message)
                
                if tool_calls:
                    # Execute tools
                    olliePrint_simple(f"\n[TOOL CALLS] Model requested {len(tool_calls)} tool(s):")
                    for tc in tool_calls:
                        tool_name = tc.get('function', {}).get('name', 'unknown')
                        tool_args = tc.get('function', {}).get('arguments', {})
                        olliePrint_simple(f"[TOOL CALLS] - {tool_name} with args: {tool_args}")
                        yield json.dumps({
                            "type": "tool_activity",
                            "content": f"Executing {tool_name}..."
                        }) + '\n'
                    
                    tool_outputs = handle_tool_calls(tool_calls)
                    olliePrint_simple(f"[TOOL CALLS] Executed {len(tool_outputs) if tool_outputs else 0} tools successfully")
                    
                    # Print tool results
                    if tool_outputs:
                        olliePrint_simple("\n[TOOL RESULTS] Tool execution results:")
                        olliePrint_simple(f"{'='*50}")
                        for i, output in enumerate(tool_outputs):
                            tool_call_id = output.get('tool_call_id', 'unknown')
                            content = output.get('content', '{}')
                            olliePrint_simple(f"[TOOL RESULTS] Result {i+1} (ID: {tool_call_id}):")
                            try:
                                # Try to format JSON nicely
                                import json as json_module
                                parsed = json_module.loads(content)
                                formatted = json_module.dumps(parsed, indent=2)
                                olliePrint_simple(formatted)
                            except:
                                # If not JSON, print as-is
                                olliePrint_simple(content)
                            olliePrint_simple(f"{'-'*30}")
                        olliePrint_simple(f"{'='*50}\n")
                    
                    if tool_outputs:
                        messages.extend(tool_outputs)
                        
                        # Update user message with tool results
                        tool_results = []
                        for output in tool_outputs:
                            tool_results.append(f"Tool result: {output.get('content', '{}')}")
                        
                        enhanced_message = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
{chr(10).join(tool_results)}
The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END OF FRED DATABASE)"""
                        
                        # Update last user message
                        for i in range(len(messages) - 1, -1, -1):
                            if messages[i].get('role') == 'user':
                                messages[i]['content'] = enhanced_message
                                break
                    else:
                        yield json.dumps({"type": "error", "content": "Tool execution failed"}) + '\n'
                        break
                else:
                    # No tools, we have final response
                    if clean_content:
                        assistant_response = clean_content
                    break
            
            # Generate final response if needed
            if not assistant_response:
                final_stream = client.chat(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    options=config.THINKING_MODE_OPTIONS
                )
                
                streaming_response = ""
                for chunk in final_stream:
                    content_chunk = chunk.get('message', {}).get('content')
                    if content_chunk:
                        streaming_response += content_chunk
                        # Extract thinking from streaming chunks
                        chunk_thinking = extract_think_content(content_chunk)
                        if chunk_thinking:
                            raw_thinking += chunk_thinking + "\n"
                        
                        clean_chunk = strip_think_tags(content_chunk)
                        if clean_chunk:
                            assistant_response += clean_chunk
                            yield json.dumps({'response': clean_chunk}) + '\n'
                    
                    if chunk.get('done', False):
                        # Print full streaming response when done
                        if streaming_response:
                            olliePrint_simple(f"\n{'='*60}")
                            olliePrint_simple("[MODEL RESPONSE] Full streaming response:")
                            olliePrint_simple(f"{'='*60}")
                            olliePrint_simple(streaming_response)
                            olliePrint_simple(f"{'='*60}\n")
                        break
            else:
                # Direct response
                yield json.dumps({'response': assistant_response}) + '\n'
            
            # Store response with thinking in history
            fred_state.add_conversation_turn('assistant', assistant_response, raw_thinking.strip())
            
            # TTS - route audio to appropriate device (use outer-scope flag)
            target_device = 'pi' if from_pi_glasses else 'local'
            fred_speak(assistant_response, mute_fred, target_device)
            yield json.dumps({'type': 'done'}) + '\n'
        
        return Response(event_stream(), headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        })
    
    except Exception as e:
        olliePrint_simple(f"Chat error: {e}", level='error')
        if fred_state.get_conversation_history() and fred_state.get_conversation_history()[-1].get('role') == 'user':
            fred_state.clear_conversation_history()
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory_visualization_data')
def get_memory_viz_data():
    """Get memory visualization data."""
    try:
        return jsonify(lib.get_all_active_nodes_for_viz())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/memory/<int:node_id>/connections')
def get_memory_connections(node_id):
    """Get memory connections."""
    try:
        import Tools
        result = Tools.tool_get_node_by_id(node_id)
        
        if result.get('success'):
            return jsonify({
                'node_id': node_id,
                'connections': result.get('result', {}).get('connections', [])
            })
        else:
            return jsonify({'error': result.get('error', 'Not found')}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/stats')
def get_conversation_stats():
    """Get conversation statistics for debugging."""
    try:
        return jsonify(fred_state.get_conversation_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# STT Functions
def process_transcription(text, from_pi=False):
    """Process transcribed text with optional visual context."""
    if not text or not text.strip():
        return
    
    text = text.strip()
    
    if text.startswith("_acknowledge_"):
        acknowledgment = text.replace("_acknowledge_", "")
        socketio.emit('fred_acknowledgment', {'text': acknowledgment})
        # Route acknowledgment to appropriate device
        target_device = 'pi' if from_pi else 'local'
        fred_speak(acknowledgment, target_device=target_device)
        return
    
    socketio.emit('transcription_result', {'text': text})
    
    def process_voice():
        try:
            # Prepare request data
            request_data = {
                'message': text,
                'model': config.DEFAULT_MODEL,  # Use configured default model, not hardcoded
                'mute_fred': False
            }
            
            # Add visual context flag for glasses input
            if from_pi:
                request_data['from_pi_glasses'] = True
            
            url = f"http://localhost:{os.environ.get('PORT', 5000)}/chat"
            response = requests.post(url, json=request_data, stream=True)
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                full_response += data['response']
                            socketio.emit('voice_response', data)
                        except:
                            continue
                
                # Response handled by main chat system
            else:
                socketio.emit('error', {'message': 'Failed to process voice command'})
        except Exception as e:
            olliePrint_simple(f"Voice processing error: {e}", level='error')
            socketio.emit('error', {'message': f'Error: {str(e)}'})
    
    threading.Thread(target=process_voice, daemon=True).start()

# SocketIO Handlers
@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to F.R.E.D.', 'stt_enabled': fred_state.stt_enabled})

@socketio.on('disconnect')
def handle_disconnect():
    stt_service.stop_processing()

@socketio.on('webrtc_server_connected')
def handle_webrtc_server_connect():
    olliePrint_simple("[BRIDGE] WebRTC communication bridge established")
    emit('status', {'message': 'WebRTC bridge online'})

@socketio.on('start_stt')
def handle_start_stt():
    if not fred_state.stt_enabled:
        emit('status', {'message': 'STT disabled'})
        return
    
    if stt_service.start_processing(process_transcription):
        emit('status', {'message': 'Speech recognition started'})
    else:
        emit('error', {'message': 'Failed to start speech recognition'})

@socketio.on('stop_stt')
def handle_stop_stt():
    stt_service.stop_processing()
    emit('status', {'message': 'Speech recognition stopped'})

@socketio.on('toggle_stt')
def handle_toggle_stt(data):
    global fred_state
    fred_state.stt_enabled = data.get('enabled', True)
    
    if not fred_state.stt_enabled:
        stt_service.stop_processing()
    
    emit('status', {'message': f'STT {"enabled" if fred_state.stt_enabled else "disabled"}', 'stt_enabled': fred_state.stt_enabled})

@socketio.on('voice_message')
def handle_voice_message(data):
    text = data.get('text', '').strip()
    if text:
        socketio.emit('voice_chat_message', {
            'text': text,
            'timestamp': datetime.now().isoformat()
        })

# --- Main Execution ---
def run_app():
    """Starts the F.R.E.D. main server using Flask-SocketIO."""
    olliePrint_simple("[MAINFRAME] F.R.E.D. intelligence core starting...")
    
    # Initialize TTS if not already initialized
    if fred_state.get_tts_engine() is None:
        olliePrint_simple("[INIT] Initializing voice synthesis systems...")
        initialize_tts()
    
    # Initialize STT if not already initialized  
    if not getattr(stt_service, 'is_initialized', False):
        olliePrint_simple("[INIT] Initializing speech recognition systems...")
        stt_service.initialize()
    
    try:
        socketio.run(app, host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
    except Exception as e:
        olliePrint_simple(f"[CRITICAL] Mainframe startup failure: {e}")

if __name__ == '__main__':
    cleanup_wav_files()
    
    # Check Ollama connection
    try:
        requests.get(config.OLLAMA_BASE_URL, timeout=None)
        olliePrint_simple(f"[NEURAL-NET] AI model interface connected")
    except:
        olliePrint_simple("[WARNING] AI model interface not responding", level='warning')
    
    # Initialize TTS
    initialize_tts()
    
    # Initialize STT
    stt_service.initialize()
    
    # Start server
    run_app()