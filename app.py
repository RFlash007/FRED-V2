import os
import ollama
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import json
import requests
import memory.librarian as lib
import logging
import re
import playsound
import uuid
import glob
from Tools import AVAILABLE_TOOLS, handle_tool_calls
from datetime import datetime
import torch
from TTS.api import TTS
import traceback
from stt_service import stt_service
import threading
import time
from config import config

# State Management Class
class FREDState:
    """Centralized state management for F.R.E.D."""
    
    def __init__(self):
        self.conversation_history = []
        self.tts_engine = None
        self.last_played_wav = None
        self.stt_enabled = True
        self._lock = threading.Lock()
    
    def add_conversation_turn(self, role, content, thinking=None):
        """Thread-safe conversation history management."""
        with self._lock:
            turn = {'role': role, 'content': content}
            if thinking:
                turn['thinking'] = thinking
            self.conversation_history.append(turn)
    
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

# Global state instance
fred_state = FREDState()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = config.get_db_path(APP_ROOT)
lib.DB_FILE = DB_PATH
FRED_CORE_NODE_ID = "FRED_CORE"

def initialize_tts():
    """Initialize TTS engine once during startup."""
    if os.path.exists(config.FRED_SPEAKER_WAV_PATH):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_engine = TTS(config.XTTS_MODEL_NAME).to(device)
            fred_state.set_tts_engine(tts_engine)
            logging.info(f"TTS engine initialized on {device}")
        except Exception as e:
            logging.error(f"Failed to initialize TTS: {e}")
            fred_state.set_tts_engine(None)

# Load System Prompt
SYSTEM_PROMPT_FILE = os.path.join(APP_ROOT, 'system_prompt.txt')
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()
except Exception as e:
    logging.error(f"Error loading system prompt: {e}")
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
        prompt = f"""Condense the following reasoning into exactly 2-3 sentences. Focus only on key insights and decisions. Do NOT use thinking tags or meta-commentary:

{thinking_content}

Condensed reasoning (2-3 sentences only):"""

        print(f"\n[THINKING SUMMARY] Summarizing {len(thinking_content)} chars of thinking...")
        response = ollama_client.chat(
            model="hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.5, "max_tokens": 150}  # Lower temp + token limit
        )
        raw_summary_response = response.get('message', {}).get('content', '').strip()
        
        # Extract only the summary content, not the 4B model's thinking
        summary = strip_think_tags(raw_summary_response).strip()
        
        print(f"[THINKING SUMMARY] Raw 4B response ({len(raw_summary_response)} chars):")
        print(f"[THINKING SUMMARY] {raw_summary_response[:200]}{'...' if len(raw_summary_response) > 200 else ''}")
        print(f"[THINKING SUMMARY] Extracted summary ({len(summary)} chars): {summary}")
        return summary
    except Exception as e:
        print(f"[THINKING SUMMARY] Failed to summarize: {e}")
        logging.warning(f"Failed to summarize thinking: {e}")
        return thinking_content[:200] + "..." if len(thinking_content) > 200 else thinking_content

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
                print(f"[THINKING] Including full thinking for recent message (age {age})")
                full_content = f"<think>\n{thinking}\n</think>\n{content}"
                print(f"[THINKING CONTEXT] Message with full thinking being sent to model:")
                print(f"{'='*50}")
                print(full_content)
                print(f"{'='*50}\n")
                messages.append({"role": "assistant", "content": full_content})
            elif age <= 6 and thinking:
                # Older messages: include summarized thinking
                print(f"[THINKING] Including summarized thinking for older message (age {age})")
                if not hasattr(prepare_messages_with_thinking, '_thinking_cache'):
                    prepare_messages_with_thinking._thinking_cache = {}
                
                cache_key = hash(thinking)
                if cache_key not in prepare_messages_with_thinking._thinking_cache:
                    prepare_messages_with_thinking._thinking_cache[cache_key] = summarize_thinking(thinking, ollama_client)
                
                summarized = prepare_messages_with_thinking._thinking_cache[cache_key]
                if summarized:
                    full_content = f"<think>\n{summarized}\n</think>\n{content}"
                    print(f"[THINKING CONTEXT] Message with summarized thinking being sent to model:")
                    print(f"{'='*50}")
                    print(full_content)
                    print(f"{'='*50}\n")
                    messages.append({"role": "assistant", "content": full_content})
                else:
                    messages.append({"role": "assistant", "content": content})
            else:
                # Oldest messages: no thinking context
                if thinking:
                    print(f"[THINKING] Excluding thinking for old message (age {age})")
                messages.append({"role": "assistant", "content": content})
    
    # Add current user message
    pending_tasks_alert = get_pending_tasks_alert()
    formatted_input = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
{pending_tasks_alert}The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END OF FRED DATABASE)"""
    
    messages.append({"role": "user", "content": formatted_input})
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

def fred_speak(text, mute_fred=False):
    """Generate and play speech using TTS."""
    if mute_fred or not text.strip() or not os.path.exists(config.FRED_SPEAKER_WAV_PATH):
        return
    
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
        tts_engine = fred_state.get_tts_engine()
        if tts_engine is None:
            logging.warning("TTS engine not initialized, skipping speech generation")
            return
        
        tts_engine.tts_to_file(
            text=text,
            speaker_wav=config.FRED_SPEAKER_WAV_PATH,
            language=config.FRED_LANGUAGE,
            file_path=output_path
        )
        
        playsound.playsound(output_path, block=False)
        
        # Schedule cleanup after a delay
        def delayed_cleanup():
            time.sleep(config.TTS_CLEANUP_DELAY)  # Wait for playback to complete
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
        
        threading.Thread(target=delayed_cleanup, daemon=True).start()
        fred_state.last_played_wav = output_path
        
    except Exception as e:
        logging.error(f"TTS error: {e}")

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
    print(f"[INFO] Memory DB initialized at: {DB_PATH}")
except Exception as e:
    logging.error(f"Database initialization failed: {e}")

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
        logging.error(f"Graph generation error: {e}")
    
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
        
        if not user_message or not ollama_base_url:
            return jsonify({'error': 'Missing required fields'}), 400
        
        def event_stream():
            # Add user message to history
            fred_state.add_conversation_turn('user', user_message)
            
            client = ollama.Client(host=ollama_base_url)
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
                    print(f"\n{'='*60}")
                    print(f"[MODEL RESPONSE] Full response from iteration {iteration + 1}:")
                    print(f"{'='*60}")
                    print(raw_content)
                    print(f"{'='*60}\n")
                
                # Extract and store thinking
                current_thinking = extract_think_content(raw_content)
                if current_thinking:
                    print(f"[THINKING] Extracted {len(current_thinking)} chars of thinking from iteration {iteration + 1}")
                    raw_thinking += current_thinking + "\n"
                
                clean_content = strip_think_tags(raw_content)
                tool_calls = response_message.get('tool_calls')
                
                # Ensure assistant role
                if 'role' not in response_message:
                    response_message['role'] = 'assistant'
                messages.append(response_message)
                
                if tool_calls:
                    # Execute tools
                    print(f"\n[TOOL CALLS] Model requested {len(tool_calls)} tool(s):")
                    for tc in tool_calls:
                        tool_name = tc.get('function', {}).get('name', 'unknown')
                        tool_args = tc.get('function', {}).get('arguments', {})
                        print(f"[TOOL CALLS] - {tool_name} with args: {tool_args}")
                        yield json.dumps({
                            "type": "tool_activity",
                            "content": f"Executing {tool_name}..."
                        }) + '\n'
                    
                    tool_outputs = handle_tool_calls(response_message)
                    print(f"[TOOL CALLS] Executed {len(tool_outputs) if tool_outputs else 0} tools successfully")
                    
                    # Print tool results
                    if tool_outputs:
                        print(f"\n[TOOL RESULTS] Tool execution results:")
                        print(f"{'='*50}")
                        for i, output in enumerate(tool_outputs):
                            tool_call_id = output.get('tool_call_id', 'unknown')
                            content = output.get('content', '{}')
                            print(f"[TOOL RESULTS] Result {i+1} (ID: {tool_call_id}):")
                            try:
                                # Try to format JSON nicely
                                import json as json_module
                                parsed = json_module.loads(content)
                                formatted = json_module.dumps(parsed, indent=2)
                                print(formatted)
                            except:
                                # If not JSON, print as-is
                                print(content)
                            print(f"{'-'*30}")
                        print(f"{'='*50}\n")
                    
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
                            print(f"\n{'='*60}")
                            print(f"[MODEL RESPONSE] Full streaming response:")
                            print(f"{'='*60}")
                            print(streaming_response)
                            print(f"{'='*60}\n")
                        break
            else:
                # Direct response
                yield json.dumps({'response': assistant_response}) + '\n'
            
            # Store response with thinking in history
            fred_state.add_conversation_turn('assistant', assistant_response, raw_thinking.strip())
            
            # TTS
            fred_speak(assistant_response, mute_fred)
            yield json.dumps({'type': 'done'}) + '\n'
        
        return Response(event_stream(), headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        })
    
    except Exception as e:
        logging.error(f"Chat error: {e}")
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

# STT Functions
def process_transcription(text):
    """Process transcribed text."""
    if not text or not text.strip():
        return
    
    text = text.strip()
    
    if text.startswith("_acknowledge_"):
        acknowledgment = text.replace("_acknowledge_", "")
        socketio.emit('fred_acknowledgment', {'text': acknowledgment})
        fred_speak(acknowledgment)
        return
    
    socketio.emit('transcription_result', {'text': text})
    
    def process_voice():
        try:
            url = f"http://localhost:{os.environ.get('PORT', 5000)}/chat"
            response = requests.post(url, json={
                'message': text,
                'model': 'hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M',
                'mute_fred': False
            }, stream=True)
            
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
            logging.error(f"Voice processing error: {e}")
            socketio.emit('error', {'message': f'Error: {str(e)}'})
    
    threading.Thread(target=process_voice, daemon=True).start()

# SocketIO Handlers
@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to F.R.E.D.', 'stt_enabled': fred_state.stt_enabled})

@socketio.on('disconnect')
def handle_disconnect():
    stt_service.stop_processing()

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

if __name__ == '__main__':
    cleanup_wav_files()
    
    # Check Ollama connection
    try:
        requests.get(config.OLLAMA_BASE_URL, timeout=2)
        print(f"[INFO] Ollama connected at {config.OLLAMA_BASE_URL}")
    except:
        logging.warning(f"Could not connect to Ollama")
    
    # Initialize TTS
    initialize_tts()
    
    # Initialize STT
    stt_service.initialize()
    
    # Start server
    print(f"[INFO] Starting F.R.E.D. server on http://{config.HOST}:{config.PORT}")
    socketio.run(app, debug=config.DEBUG, port=config.PORT, host=config.HOST)