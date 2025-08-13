import os
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import json
import requests
import memory.L3_memory as L3
import re
import playsound
import uuid
import subprocess
import platform
import glob
# Import only non-memory tools for F.R.E.D.
from Tools import handle_tool_calls

# F.R.E.D.'s tool set - imported from consolidated config
# Tool schemas are now centrally managed in config.py
from datetime import datetime
import torch
from TTS.api import TTS
import traceback
from stt_service import stt_service
from vision_service import vision_service
import threading
from utils import strip_think_tags

import time
from config import config, ollama_manager, AGENT_MANAGEMENT_TOOLS

# Conversation orchestration for barge-in and sentence queue TTS
from conversation_orchestrator import (
    InteractionOrchestrator,
    AudioPlaybackController,
    SpeechQueue,
)
# Import L2 and Agenda modules
import memory.L2_memory as L2
import memory.agenda_system as agenda

# State Management Class
class FREDState:
    """Centralized state management for F.R.E.D."""
    
    def __init__(self):
        self.conversation_history = []
        self.tts_engine = None
        self.last_played_wav = None
        self.stt_enabled = True
        self.total_conversation_turns = 0  # Track conversation turns for L2
        self.last_analyzed_message_index = 0  # Track last analyzed message for L2
        self.stewie_voice_available = False  # Track if Stewie voice cloning is available
        self._lock = threading.Lock()
    
    def add_conversation_turn(self, role, content):
        """Thread-safe conversation history management without thinking content."""
        with self._lock:
            turn = {'role': role, 'content': content}
            self.conversation_history.append(turn)
            
            # Check if we need to remove old messages
            if len(self.conversation_history) > config.FRED_MAX_CONVERSATION_MESSAGES:
                messages_to_remove = len(self.conversation_history) - config.FRED_MAX_CONVERSATION_MESSAGES
                
                # Remove oldest messages
                self.conversation_history = self.conversation_history[messages_to_remove:]
                
                # Adjust L2 tracking indices to account for removed messages
                self.last_analyzed_message_index = max(0, self.last_analyzed_message_index - messages_to_remove)
            
            # Increment turn counter and check L2 trigger
            if role == 'assistant':  # Complete turn (user + assistant)
                self.total_conversation_turns += 1
                if self.total_conversation_turns % config.L2_TRIGGER_INTERVAL == 0:
                    # Trigger L2 analysis in background (after F.R.E.D. responds)
                    def delayed_l2_processing():
                        try:
                            # Get the user message that triggered this turn
                            if len(self.conversation_history) >= 2:
                                user_msg = self.conversation_history[-2].get('content', '')
                                L2.process_l2_creation(
                                    self.conversation_history.copy(),
                                    len(self.conversation_history),
                                    user_msg
                                )
                        except Exception as e:
                            pass
                    
                    threading.Thread(
                        target=delayed_l2_processing,
                        daemon=True
                    ).start()
    
    def get_conversation_history(self):
        """Get a copy of conversation history."""
        with self._lock:
            return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        with self._lock:
            self.conversation_history.clear()
            self.total_conversation_turns = 0
            self.last_analyzed_message_index = 0
    
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
                'max_messages': config.FRED_MAX_CONVERSATION_MESSAGES,
                'total_turns': self.total_conversation_turns,
                'last_analyzed_index': self.last_analyzed_message_index
            }

from agents.dispatcher import AgentDispatcher

# Global state instance
fred_state = FREDState()
agent_dispatcher = AgentDispatcher()
interaction_orchestrator = InteractionOrchestrator()
local_playback_controller = AudioPlaybackController()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = config.get_db_path(APP_ROOT)
L3.DB_FILE = DB_PATH
FRED_CORE_NODE_ID = "FRED_CORE"

def initialize_tts():
    """Initialize TTS engine once during startup."""
    try:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Stewie voice cloning if enabled
        if config.STEWIE_VOICE_ENABLED:
            from stewie_voice_clone import initialize_stewie_voice, validate_stewie_samples
            
            stewie_success = initialize_stewie_voice()
            if stewie_success:
                # Validate voice samples
                sample_stats = validate_stewie_samples()
                
                # Set a flag to indicate Stewie voice is available
                fred_state.stewie_voice_available = True
            else:
                fred_state.stewie_voice_available = False
        else:
            fred_state.stewie_voice_available = False
        
        # Initialize standard TTS as fallback
        tts_engine = TTS(config.XTTS_MODEL_NAME).to(device)
        fred_state.set_tts_engine(tts_engine)
        
    except Exception as e:
        fred_state.set_tts_engine(None)
        fred_state.stewie_voice_available = False

# Load System Prompt
SYSTEM_PROMPT = config.FRED_SYSTEM_PROMPT

def extract_think_content(text):
    """Extract thinking content from <think>...</think> tags."""
    if not text:
        return ""
    matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    return '\n'.join(matches).strip()

# Legacy message preparation functions removed - now handled by CORTEX

def play_audio_locally(audio_file_path):
    """Robust cross-platform audio playback with Windows optimization."""
    if not os.path.exists(audio_file_path):
        return
    
    try:
        system = platform.system().lower()
        
        if system == "windows":
            # Windows: Use multiple fallback methods
            try:
                # Method 1: PowerShell with Windows Media Format SDK
                cmd = [
                    "powershell", "-Command", 
                    f"(New-Object Media.SoundPlayer '{audio_file_path}').PlaySync()"
                ]
                subprocess.run(cmd, check=True, capture_output=True, timeout=None)
                return
            except Exception:
                try:
                    # Method 2: playsound library
                    playsound.playsound(audio_file_path, block=False)
                    return
                except Exception:
                    try:
                        # Method 3: Direct system call to start with associated program
                        subprocess.run(['start', '/wait', audio_file_path], shell=True, check=True, timeout=None)
                        return
                    except Exception:
                        pass
        
        elif system == "linux":
            # Linux: Try multiple audio players
            players = ['aplay', 'paplay', 'mpg123', 'mpv', 'ffplay']
            for player in players:
                try:
                    subprocess.run([player, audio_file_path], check=True, capture_output=True, timeout=None)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            pass
        
        elif system == "darwin":  # macOS
            try:
                subprocess.run(['afplay', audio_file_path], check=True, capture_output=True, timeout=None)
                return
            except Exception:
                pass
        
        else:
            # Fallback: Try playsound for unknown systems
            try:
                playsound.playsound(audio_file_path, block=False)
                return
            except Exception:
                pass
    
    except Exception:
        pass

def fred_speak(text, mute_fred=False, target_device='local'):
    """Deprecated single-shot TTS (kept for compatibility). Use SpeechQueue."""
    if not text:
        return
    # Route through a transient SpeechQueue instance for compatibility
    def _tts_engine_provider():
        return fred_state.get_tts_engine()

    sq = SpeechQueue(
        playback_controller=local_playback_controller,
        tts_engine_provider=_tts_engine_provider,
        send_audio_to_pi_fn=lambda path, t: send_audio_to_pi(path, t),
        prebuffer_sentences=1,
        language=config.FRED_LANGUAGE,
        speaker_wav_path=config.FRED_SPEAKER_WAV_PATH if os.path.exists(config.FRED_SPEAKER_WAV_PATH) else None,
        default_speaker="Ana Florence",
    )
    sq.configure(target_device=target_device, mute=mute_fred)
    sq.enqueue(text)

def send_audio_to_pi(audio_file_path, text):
    """Send audio file to connected Pi clients."""
    try:
        import base64
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            return
        
        # Read audio file and encode as base64
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Send via SocketIO to WebRTC server
        payload = {
            'audio_data': audio_b64,
            'text': text,
            'format': 'wav'
        }
        
        socketio.emit('fred_audio', payload)
        
    except Exception as e:
        pass

def segment_into_sentences(text: str) -> list:
    """Lightweight sentence segmentation for English. Returns list of sentences."""
    if not text:
        return []
    # Simple rule-based segmentation on ., !, ? followed by space or EOL
    import re
    pieces = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in pieces if p.strip()]

def cleanup_wav_files():
    """Clean up old WAV files."""
    pattern = os.path.join(APP_ROOT, "fred_speech_output_*.wav")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except Exception:
            pass

# Initialize databases
try:
    L3.init_db()
except Exception as e:
    pass

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
        with L3.duckdb.connect(L3.DB_FILE) as con:
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
        pass
    
    return jsonify({"nodes": nodes, "edges": edges})

@app.route('/api/interrupt', methods=['POST'])
def api_interrupt():
    """HTTP endpoint for UI interrupt buttons.

    action: 'stop' | 'continue'
    text: optional user text for continuation
    """
    try:
        data = request.json or {}
        action = data.get('action', '').strip().lower()
        if action == 'stop':
            interaction_orchestrator.request_stop()
            return jsonify({"ok": True})
        if action == 'continue':
            user_text = (data.get('text') or '').strip()
            interaction_orchestrator.request_barge_in(user_text)
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "invalid action"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Main chat endpoint with enhanced thinking preservation."""
    global fred_state
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400
        
        user_message = data.get('message')
        # Use FRED_OLLAMA_MODEL for personality responses, but allow override from client
        model_name = data.get('model', config.FRED_OLLAMA_MODEL)
        ollama_base_url = data.get('ollama_url', config.OLLAMA_BASE_URL).strip()
        max_tool_iterations = data.get('max_tool_iterations', config.MAX_TOOL_ITERATIONS)
        mute_fred = data.get('mute_fred', False)
        from_pi_glasses = data.get('from_pi_glasses', False)
        
        if not user_message or not ollama_base_url:
            return jsonify({'error': 'Missing required fields'}), 400
        
        def event_stream():
            # Add user message to history
            fred_state.add_conversation_turn('user', user_message)
            
            print("\n" + "="*80)
            print("🤖 [F.R.E.D.] MAIN AGENT INPUT")
            print("="*80)
            print(f"📝 USER MESSAGE: {user_message}")
            print(f"🔧 MODEL: {model_name}")
            if from_pi_glasses:
                print("👓 SOURCE: Pi Glasses")
            else:
                print("🖥️ SOURCE: Web Interface")
            
            conversation_history = fred_state.get_conversation_history()
            if conversation_history:
                print(f"\n📚 FULL CONVERSATION HISTORY ({len(conversation_history)} turns):")
                for i, turn in enumerate(conversation_history):
                    print(f"  Turn {i+1} [{turn['role']}]: {turn['content']}")
            else:
                print("\n📚 CONVERSATION HISTORY: Empty")
            print("="*80 + "\n")
            
            # CENTRALIZED CONNECTION: All calls use ollama_manager.chat_concurrent_safe() directly
            # No need to store client reference

            # Get visual context if from Pi glasses
            visual_context = ""
            if from_pi_glasses:
                from vision_service import vision_service
                visual_context = vision_service.get_current_visual_context()
                if visual_context:
                    print(f"👁️ VISUAL CONTEXT: {len(visual_context)} characters")

            # Parallel Agent Execution: G.A.T.E. + M.A.D.
            from memory import gate
            from agents.mad import mad_agent
            
            def run_parallel_agents():
                """Run G.A.T.E. and prepare for M.A.D. analysis."""
                # G.A.T.E. processes current user message for context retrieval
                gate_result = gate.run_gate_analysis(
                    user_message, 
                    fred_state.get_conversation_history(),
                    agent_dispatcher,
                    visual_context
                )
                
                # M.A.D. analyzes previous conversation turn for memory creation
                # Only run if we have at least one complete turn (user + assistant)
                conversation_history = fred_state.get_conversation_history()
                if len(conversation_history) >= 2:
                    # Get the last completed turn
                    last_assistant_idx = None
                    last_user_idx = None
                    
                    for i in range(len(conversation_history) - 1, -1, -1):
                        if conversation_history[i]['role'] == 'assistant' and last_assistant_idx is None:
                            last_assistant_idx = i
                        elif conversation_history[i]['role'] == 'user' and last_user_idx is None and last_assistant_idx is not None:
                            last_user_idx = i
                            break
                    
                    if last_user_idx is not None and last_assistant_idx is not None:
                        previous_user_msg = conversation_history[last_user_idx]['content']
                        previous_fred_response = conversation_history[last_assistant_idx]['content']
                        
                        try:
                            mad_result = mad_agent.analyze_turn(
                                previous_user_msg,
                                previous_fred_response,
                                conversation_history[:last_user_idx]  # Context before the analyzed turn
                            )
                            if mad_result.get('success', False):
                                created_count = len(mad_result.get('created_memories', []))
                                if created_count > 0:
                                    pass
                        except Exception as e:
                            pass
                
                return gate_result
            
            # Run agents in parallel - M.A.D. analyzes previous turn while G.A.T.E. processes current
            fred_database = run_parallel_agents()

            # Prepare messages for F.R.E.D. using the context from G.A.T.E./C.R.A.P.
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Add conversation history without thinking content
            for turn in fred_state.get_conversation_history():
                messages.append({"role": turn['role'], "content": turn['content']})
            

            # Check for F.R.E.D. research summaries (direct injection bypassing agents)
            subconscious_results = ""
            try:
                import memory.agenda_system as agenda
                pending_summaries = agenda.get_pending_fred_summaries()
                if pending_summaries:
                    summary_lines = []
                    summary_ids = []
                    for summary in pending_summaries:
                        summary_lines.append(summary['content'])
                        summary_ids.append(summary['summary_id'])
                    
                    subconscious_results = f"""
SUBCONSCIOUS PROCESSING RESULTS:
{chr(10).join(summary_lines)}
"""
                    # Mark as delivered
                    agenda.mark_fred_summaries_delivered(summary_ids)
            except Exception as e:
                pass

            # Format final user message with the retrieved database content
            formatted_input = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)
{subconscious_results}
{fred_database}"""
            messages.append({"role": "user", "content": formatted_input})
            # --- Debug: Show full F.R.E.D. LLM prompt (mirrors G.A.T.E. style) ---
            try:
                print("\n" + "="*80)
                print("🎯 [F.R.E.D.] LLM PROMPT")
                print("="*80)
                print(f"🧠 SYSTEM PROMPT:\n{SYSTEM_PROMPT}")
                print("-"*80)
                print("📝 FULL USER PROMPT:")
                print(messages[-1].get('content', ''))
                print("="*80 + "\n")
            except Exception:
                pass
            
            assistant_response = ""
            raw_thinking = ""
            tool_outputs = []  # Ensure defined even if no tools are called
            # Begin orchestrated response lifecycle
            response_id = interaction_orchestrator.begin_response()
            
            # Tool iteration loop
            sleep_cycle_triggered = False  # Initialize variable
            for iteration in range(max_tool_iterations):
                try:
                    response = ollama_manager.chat_concurrent_safe(
                        host=ollama_base_url,
                        model=model_name,
                        messages=messages,
                        stream=False,
                        options=config.Instruct_Generation_Options
                    )
                except Exception as e:
                    print(f"[SSE] Iteration {iteration+1} chat error: {e}")
                    print(traceback.format_exc())
                    yield json.dumps({"type": "error", "content": str(e)}) + '\n'
                    return
                
                response_message = response.get('message', {})
                raw_content = response_message.get('content', '')

                # Extract and store thinking
                current_thinking = extract_think_content(raw_content)
                raw_thinking += current_thinking + "\n"
                
                clean_content = strip_think_tags(raw_content)
                tool_calls = response_message.get('tool_calls')
                
                # Ensure assistant role and preserve thinking for next iteration
                if 'role' not in response_message:
                    response_message['role'] = 'assistant'
                
                # If there are tool calls, preserve the thinking content for the next iteration
                if tool_calls and current_thinking:
                    # Keep raw content with thinking tags for model's context
                    response_message['content'] = raw_content
                
                messages.append(response_message)
                
                if tool_calls:
                    # Check for sleep cycle trigger (special blocking behavior)
                    sleep_cycle_triggered = any(tc.get('function', {}).get('name') == 'triggerSleepCycle' for tc in tool_calls)
                    
                    if sleep_cycle_triggered:
                        # Immediate sleep cycle message
                        yield json.dumps({
                            "type": "sleep_cycle_start",
                            "content": config.SLEEP_CYCLE_MESSAGE
                        }) + '\n'
                        yield json.dumps({'response': config.SLEEP_CYCLE_MESSAGE}) + '\n'
                    
                    # Execute tools
                    for tc in tool_calls:
                        tool_name = tc.get('function', {}).get('name', 'unknown')
                        if not sleep_cycle_triggered:  # Don't show activity for sleep cycle (already shown)
                            yield json.dumps({
                                "type": "tool_activity",
                                "content": f"Executing {tool_name}..."
                            }) + '\n'
                    
                    try:
                        tool_outputs = handle_tool_calls(tool_calls)
                    except Exception as e:
                        tool_outputs = []
                    
                    # Process tool results without printing
                    if tool_outputs:
                    
                        
                        # Update user message with tool results (skip for sleep cycle - already handled)
                        if not sleep_cycle_triggered:
                            tool_results = []
                            system_notifications = [] # Generic notifications

                            for output in tool_outputs:
                                try:
                                    content_json = json.loads(output.get('content', '{}'))
                                    if content_json.get('status') == 'enrollment_complete':
                                        person_name = content_json.get('person_name', 'unknown person')
                                        system_notifications.append(f"[Facial Recognition Engaged] I've committed \"{person_name}\" to memory. I'll recognize them from now on.")
                                    else:
                                        tool_results.append(f"Tool result: {output.get('content', '{}')}")
                                except json.JSONDecodeError:
                                    tool_results.append(f"Tool result: {output.get('content', '{}')}")

                            fred_database_content = chr(10).join(tool_results)
                            if system_notifications:
                                fred_database_content += f"\n\nSystem Notifications:\n- {chr(10).join(system_notifications)}"

                            enhanced_message = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(NEURAL PROCESSING CORE)
{fred_database_content}
The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END NEURAL PROCESSING CORE)"""
                            
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
            
            # Sleep cycle complete - skip normal response generation
            if sleep_cycle_triggered and tool_outputs:
                assistant_response = config.SLEEP_CYCLE_MESSAGE
                # Store the sleep cycle activation in history
                fred_state.add_conversation_turn('assistant', assistant_response)
                yield json.dumps({'type': 'done'}) + '\n'
                return
            
            # Generate final response if needed
            if not assistant_response:
                try:
                    # Use ollama_manager for consistent connection management
                    final_stream = ollama_manager.chat_concurrent_safe(
                        host=ollama_base_url,
                        model=model_name,
                        messages=messages,
                        stream=True,
                        options=config.Instruct_Generation_Options
                    )

                    # Set up sentence-level speech queue
                    def tts_engine_provider():
                        return fred_state.get_tts_engine()

                    def send_stop_to_pi():
                        try:
                            socketio.emit('fred_audio_stop', {'message': 'stop'})
                        except Exception:
                            pass

                    speech_queue = SpeechQueue(
                        playback_controller=local_playback_controller,
                        tts_engine_provider=tts_engine_provider,
                        send_audio_to_pi_fn=lambda path, t: send_audio_to_pi(path, t),
                        send_stop_to_pi_fn=send_stop_to_pi,
                        prebuffer_sentences=1,
                        language=config.FRED_LANGUAGE,
                        speaker_wav_path=config.FRED_SPEAKER_WAV_PATH if os.path.exists(config.FRED_SPEAKER_WAV_PATH) else None,
                        default_speaker="Ana Florence",
                    )
                    target_device = 'pi' if from_pi_glasses else 'local'
                    speech_queue.configure(target_device=target_device, mute=mute_fred)
                    interaction_orchestrator.bind_speech_queue(speech_queue)

                    # STT coordination during playback (local)
                    speech_queue.on_playback_start = lambda: stt_service.set_speaking_state(True)
                    speech_queue.on_playback_end = lambda: stt_service.set_speaking_state(False)
                    speech_queue.on_sentence_spoken = interaction_orchestrator.record_spoken_sentence

                    sentence_buffer = ""

                    def flush_sentences(text_chunk: str):
                        nonlocal sentence_buffer
                        sentence_buffer += text_chunk
                        sentences = segment_into_sentences(sentence_buffer)
                        if sentences:
                            # If last char is not a terminal, keep the tail as partial
                            if sentence_buffer and sentence_buffer[-1] not in ['.', '!', '?']:
                                tail = sentences.pop() if len(sentences) > 0 else ""
                            else:
                                tail = ""
                            for s in sentences:
                                speech_queue.enqueue(s)
                            sentence_buffer = tail

                    # Primary stream loop
                    for chunk in final_stream:
                        content_chunk = chunk.get('message', {}).get('content')
                        if content_chunk:
                            chunk_thinking = extract_think_content(content_chunk)
                            if chunk_thinking:
                                raw_thinking += chunk_thinking + "\n"
                            clean_chunk = strip_think_tags(content_chunk)
                            if clean_chunk:
                                assistant_response += clean_chunk
                                yield json.dumps({'response': clean_chunk}) + '\n'
                                flush_sentences(clean_chunk)

                        # Check for abort request (stop or barge-in)
                        if interaction_orchestrator.should_abort_generation():
                            break

                        if chunk.get('done', False):
                            break

                    # If abort requested, determine whether to continue with user context
                    if interaction_orchestrator.should_abort_generation():
                        user_barge_text = interaction_orchestrator.consume_barge_in_text()
                        # Flush trailing partial as a sentence if present
                        if sentence_buffer.strip():
                            speech_queue.enqueue(sentence_buffer.strip())
                            sentence_buffer = ""

                        if user_barge_text and user_barge_text.strip():
                            # Build continuation prompt
                            last_spoken = interaction_orchestrator.get_last_spoken(1)
                            spoken_hint = last_spoken[0] if last_spoken else ""
                            continuation_msg = (
                                "(INTERRUPTION) You were interrupted mid-response. "
                                f"User said: '{user_barge_text.strip()}'. "
                                f"Avoid repeating already spoken sentence: '{spoken_hint}'. "
                                "Continue seamlessly from where you left off."
                            )
                            messages.append({"role": "assistant", "content": assistant_response})
                            messages.append({"role": "user", "content": continuation_msg})

                            # Clear abort flag before resuming
                            interaction_orchestrator.clear_abort_flags()

                            # Resume streaming continuation
                            cont_stream = ollama_manager.chat_concurrent_safe(
                                host=ollama_base_url,
                                model=model_name,
                                messages=messages,
                                stream=True,
                                options=config.Instruct_Generation_Options
                            )
                            for chunk in cont_stream:
                                content_chunk = chunk.get('message', {}).get('content')
                                if content_chunk:
                                    chunk_thinking = extract_think_content(content_chunk)
                                    if chunk_thinking:
                                        raw_thinking += chunk_thinking + "\n"
                                    clean_chunk = strip_think_tags(content_chunk)
                                    if clean_chunk:
                                        assistant_response += clean_chunk
                                        yield json.dumps({'response': clean_chunk}) + '\n'
                                        flush_sentences(clean_chunk)
                                if interaction_orchestrator.should_abort_generation():
                                    break
                                if chunk.get('done', False):
                                    break

                    # After stream completes, enqueue any trailing partial sentence
                    if sentence_buffer.strip():
                        speech_queue.enqueue(sentence_buffer.strip())
                        sentence_buffer = ""

                except Exception as e:
                    print(f"[SSE] Final streaming error: {e}")
                    print(traceback.format_exc())
                    yield json.dumps({"type": "error", "content": str(e)}) + '\n'
                    return
            else:
                # Direct response
                yield json.dumps({'response': assistant_response}) + '\n'
            
            # Store assistant response in history (thinking omitted)
            fred_state.add_conversation_turn('assistant', assistant_response)
            

            # Streaming handled sentence-level above; finalize
            yield json.dumps({'type': 'done'}) + '\n'
            interaction_orchestrator.end_response(response_id)
        
        return Response(event_stream(), headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        })
    
    except Exception as e:
        if fred_state.get_conversation_history() and fred_state.get_conversation_history()[-1].get('role') == 'user':
            fred_state.clear_conversation_history()
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory_visualization_data')
def get_memory_viz_data():
    """Get memory visualization data."""
    try:
        return jsonify(L3.get_all_active_nodes_for_viz())
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
    lower = text.lower()
    
    if text.startswith("_acknowledge_"):
        # Silent acknowledgment; do not speak or print
        return
    
    # Interrupt semantics from STT or UI
    if lower == 'stop' or lower == '[interrupt_stop]':
        interaction_orchestrator.request_stop()
        return
    if lower.startswith('fred'):
        # Everything after 'fred' is barge-in continuation context
        remainder = text[4:].strip() if len(text) >= 4 else ''
        interaction_orchestrator.request_barge_in(remainder)
        return
    
    socketio.emit('transcription_result', {'text': text})
    
    def process_voice():
        try:
            # Prepare request data
            request_data = {
                'message': text,
                'model': config.FRED_OLLAMA_MODEL,  # Use FRED's personality model for voice responses
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
    
    # Initialize TTS if not already initialized
    if fred_state.get_tts_engine() is None:
        initialize_tts()
    
    # Initialize STT if not already initialized  
    if not getattr(stt_service, 'is_initialized', False):
        stt_service.initialize()
    
    try:
        socketio.run(app, host=config.HOST, port=config.PORT, debug=False, use_reloader=False, log_output=False)
    except Exception as e:
        pass

if __name__ == '__main__':
    cleanup_wav_files()
    
    # Check Ollama connection
    try:
        requests.get(config.OLLAMA_BASE_URL, timeout=None)
    except:
        pass
    
    # Initialize STT
    stt_service.initialize()
    
    # Start server
    run_app()
