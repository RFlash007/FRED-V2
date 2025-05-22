import os
import ollama
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
import json
import requests
from dotenv import load_dotenv # Added for .env file support
import memory.librarian as lib
import logging # Added logging
import re # Added for stripping think tags
import playsound # Added for background audio playback
import uuid # Added for generating unique filenames
import glob # Added for file pattern matching (cleanup)
# Import tool-related items
from Tools import AVAILABLE_TOOLS, handle_tool_calls # Removed execute_tool import, handled by handle_tool_calls
from datetime import datetime
import sys
import torch # Added for TTS
from TTS.api import TTS # Added for TTS
import subprocess
import traceback # Ensure this is not commented out

load_dotenv() # Load environment variables from .env file

app = Flask(__name__, static_folder='static', template_folder='templates')
# Change logging level to ERROR to reduce console clutter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize memory database
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_ROOT, 'memory', 'memory.db')
lib.DB_FILE = DB_PATH

FRED_CORE_NODE_ID = "FRED_CORE" # Define F.R.E.D. Core Node ID

# --- TTS Configuration ---
# IMPORTANT: You MUST provide a valid .wav file path for FRED_SPEAKER_WAV_PATH for voice cloning to work.
# This sample should be a few seconds of clear audio of the voice you want F.R.E.D. to use.
FRED_SPEAKER_WAV_PATH = "new_voice_sample.wav"  # Updated to use the new voice sample
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2" # Confirmed from clone_and_speak.py
FRED_LANGUAGE = "en"
# FRED_OUTPUT_WAV = os.path.join(APP_ROOT, "fred_speech_output.wav") # Output in app root for simplicity - REPLACED BY DYNAMIC

tts_engine = None
last_played_wav = None # Global variable to keep track of the last played WAV file
# --- End TTS Configuration ---

# --- Ollama Model Options ---
thinking_mode_options = {
    "temperature": 0.6,
    "min_p": 0.0,
    "top_p": 0.95,
    "top_k": 20
}

non_thinking_mode_options = {
    "temperature": 0.7,
    "min_p": 0.0,
    "top_p": 0.8,
    "top_k": 20
}
# --- End Ollama Model Options ---

# --- Load System Prompt from File ---
SYSTEM_PROMPT_FILE = os.path.join(APP_ROOT, 'system_prompt.txt')
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    logging.error(f"CRITICAL: System prompt file not found at {SYSTEM_PROMPT_FILE}")
    SYSTEM_PROMPT = "You are F.R.E.D., a helpful AI assistant. Your primary goal is to assist the user based on their input and your available tools. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
except Exception as e:
    logging.error(f"CRITICAL: Error loading system prompt: {e}")
    SYSTEM_PROMPT = "You are F.R.E.D., a helpful AI assistant. Your primary goal is to assist the user based on their input and your available tools. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Print database path once at startup (keep this as it's helpful)
print(f"[INFO] Setting memory database path to: {DB_PATH}")

# Check if database file exists before initialization
if os.path.exists(DB_PATH):
    print(f"[INFO] Found existing database file at: {DB_PATH} ({os.path.getsize(DB_PATH)} bytes)")
else:
    print(f"[INFO] No existing database found at: {DB_PATH}, will create a new one")

try:
    lib.init_db()
    print(f"[INFO] Memory DB initialized at: {DB_PATH}")
except Exception as e:
    logging.error(f"Failed to initialize database: {e}")

# Check for the DB file in an unexpected location that might cause confusion
ROOT_DB_PATH = os.path.join(APP_ROOT, 'memory.db')
if os.path.exists(ROOT_DB_PATH):
    print(f"[WARNING] Found another database file at: {ROOT_DB_PATH} ({os.path.getsize(ROOT_DB_PATH)} bytes)")
    print(f"[WARNING] This file is NOT being used. The active database is at: {DB_PATH}")

# Create static folder if it doesn't exist
STATIC_FOLDER = app.static_folder
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

conversation_history = []

def strip_think_tags(text_with_tags):
    """
    Removes <think>...</think> blocks from a string.
    """
    if not text_with_tags:
        return ""
    # The re.DOTALL flag makes '.' match newlines as well,
    # in case the <think> block spans multiple lines.
    # The '?' makes the '*' non-greedy, so it matches the shortest possible block.
    cleaned_text = re.sub(r'<think>.*?</think>', '', text_with_tags, flags=re.DOTALL)
    return cleaned_text.strip() # .strip() to remove leading/trailing whitespace

# --- TTS Helper Function ---
def fred_speak(text_to_speak, mute_fred=False):
    global tts_engine
    global APP_ROOT # Ensure APP_ROOT is accessible
    global last_played_wav # Added to manage cleanup
    if mute_fred:
        logging.info("F.R.E.D. is muted. Skipping speech output.")
        return

    if not text_to_speak.strip():
        logging.info("No text provided to speak.")
        return

    if not os.path.exists(FRED_SPEAKER_WAV_PATH):
        logging.error(f"FRED's speaker WAV file not found at '{FRED_SPEAKER_WAV_PATH}'. Cannot generate speech.")
        logging.error("Please update FRED_SPEAKER_WAV_PATH in app.py.")
        return

    # Generate a unique filename for this speech output
    unique_id = uuid.uuid4()
    current_output_wav = os.path.join(APP_ROOT, f"fred_speech_output_{unique_id}.wav")

    try:
        # Attempt to delete the last played WAV file before generating a new one
        if last_played_wav and os.path.exists(last_played_wav):
            try:
                os.remove(last_played_wav)
                logging.info(f"Successfully deleted previous WAV file: {last_played_wav}")
            except Exception as e:
                logging.warning(f"Could not delete previous WAV file {last_played_wav}: {e}")
        
        if tts_engine is None:
            logging.info(f"Initializing TTS engine with model: {XTTS_MODEL_NAME}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"TTS using device: {device}")
            tts_engine = TTS(XTTS_MODEL_NAME).to(device)
            logging.info("TTS engine initialized.")
        
        logging.info(f"F.R.E.D. speaking: \"{text_to_speak[:50]}...\" to {current_output_wav}")
        tts_engine.tts_to_file(
            text=text_to_speak,
            speaker_wav=FRED_SPEAKER_WAV_PATH,
            language=FRED_LANGUAGE,
            file_path=current_output_wav # Use the unique filename
        )
        logging.info(f"F.R.E.D. speech saved to {current_output_wav}")

        # Play the audio using playsound for background playback
        try:
            playsound.playsound(current_output_wav, block=False) # Play the unique file
            logging.info(f"F.R.E.D. speech playback initiated with playsound: {current_output_wav}")
            last_played_wav = current_output_wav # Update the last played WAV path
        except Exception as e:
            logging.error(f"Error playing sound with playsound: {e}", exc_info=True)
            # Fallback or alternative playback method could be considered here if playsound fails critically

    except Exception as e:
        logging.error(f"Error in fred_speak: {e}", exc_info=True)
        # Specific error for xtts_v2 license prompt if it occurs here via console interaction
        if "You must confirm the following" in str(e):
            logging.error("The TTS model requires license confirmation. Please run the script in an interactive terminal once to accept the terms if you haven't already.")

# --- Helper Functions ---

def format_rag_context(memories):
    """Format memory search results in a readable format with NodeIDs."""
    if not memories:
        return "No relevant memories found in your knowledge base for this query."
    
    context_str = ""
    for mem in memories:
        context_str += f"- [NodeID: {mem.get('nodeid', 'N/A')}] (Type: {mem.get('type', 'N/A')}) Label: {mem.get('label', 'N/A')} | Content: {mem.get('text', 'N/A')}\n"
    
    return context_str.strip() if context_str else "No relevant memories found."

def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable types.
    
    Handles custom objects with __dict__, lists, and dictionaries.
    """
    if hasattr(obj, '__dict__'):
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    else:
        # Basic types like str, int, bool are already serializable
        return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/graph')
def get_graph():
    fred_core_node_data = {
        "id": FRED_CORE_NODE_ID,
        "label": "F.R.E.D.",
        "type": "SystemCore", 
        "text": "The Central Processing Core of F.R.E.D.",
        "isFredCore": True,
        "size": 30 
    }

    nodes = [] # Initialize nodes list
    edges = []
    max_orbiting_nodes = 10

    try:
        # Start with F.R.E.D. Core in the nodes list for the try block
        nodes.append(fred_core_node_data) 

        with lib.duckdb.connect(lib.DB_FILE) as con:
            # Fetch most connected non-superseded memories to orbit F.R.E.D.
            satellite_memories_query = f"""
                SELECT 
                    n.nodeid, n.label, n.text, n.type, n.created_at, n.last_access,
                    (SELECT COUNT(*) FROM edges e WHERE e.sourceid = n.nodeid OR e.targetid = n.nodeid) AS degree
                FROM nodes n
                WHERE n.superseded_at IS NULL
                ORDER BY degree DESC
                LIMIT {max_orbiting_nodes}
            """
            satellite_memories = con.execute(satellite_memories_query).fetchall()
            
            current_node_ids_in_graph = {FRED_CORE_NODE_ID} # Keep track of IDs we've added

            for mem_row in satellite_memories:
                memory_node_id = str(mem_row[0])
                if memory_node_id == FRED_CORE_NODE_ID: # Should not happen with DB IDs
                    continue 
                
                if memory_node_id not in current_node_ids_in_graph:
                    nodes.append({
                        "id": memory_node_id,
                        "label": mem_row[1],
                        "text": mem_row[2],
                        "type": mem_row[3],
                        "created_at": mem_row[4].isoformat() if mem_row[4] else None,
                        "last_access": mem_row[5].isoformat() if mem_row[5] else None
                    })
                    current_node_ids_in_graph.add(memory_node_id)
                    
                    edges.append({
                        "source": memory_node_id,
                        "target": FRED_CORE_NODE_ID,
                        "rel_type": "connectsTo"
                    })

            secondary_nodes_to_add = []
            secondary_edges_to_add = []

            # Iterate over copies of satellite nodes that were actually added
            # These are nodes that are not FRED_CORE_NODE_ID and are in current_node_ids_in_graph
            initial_satellites_for_secondary_search = [n for n in nodes if n['id'] != FRED_CORE_NODE_ID and n['id'] in current_node_ids_in_graph]

            for memory_node in initial_satellites_for_secondary_search:
                try:
                    satellite_id_int = int(memory_node['id']) 
                    neighbor_data = lib.get_graph_data(center_nodeid=satellite_id_int, depth=1)

                    for neighbor_node_from_lib in neighbor_data.get('nodes', []):
                        neighbor_id_str = str(neighbor_node_from_lib['id'])
                        if neighbor_id_str not in current_node_ids_in_graph:
                            neighbor_node_from_lib['id'] = neighbor_id_str
                            secondary_nodes_to_add.append(neighbor_node_from_lib)
                            current_node_ids_in_graph.add(neighbor_id_str)
                    
                    for neighbor_edge_from_lib in neighbor_data.get('edges', []):
                        src = str(neighbor_edge_from_lib['source'])
                        tgt = str(neighbor_edge_from_lib['target'])
                        rel = str(neighbor_edge_from_lib.get('rel_type', 'relatedTo'))
                        edge_representation = tuple(sorted((src, tgt)) + [rel])
                        current_edge_reps = {tuple(sorted((str(e['source']), str(e['target']))) + [str(e.get('rel_type', 'relatedTo'))]) for e in edges + secondary_edges_to_add}
                        if edge_representation not in current_edge_reps:
                            secondary_edges_to_add.append({"source": src, "target": tgt, "rel_type": rel})
                except ValueError: # Handles if a memory_node['id'] is somehow not an int
                    logging.warning(f"Skipping secondary search for non-integer node ID: {memory_node['id']}")
                except Exception as e_neighbor: # Catch errors during individual neighbor fetch
                    logging.warning(f"Error fetching neighbors for satellite {memory_node['id']}: {e_neighbor}")
            
            nodes.extend(secondary_nodes_to_add)
            edges.extend(secondary_edges_to_add)

    except Exception as e_main:
        logging.error(f"Main error in get_graph: {e_main}. Returning F.R.E.D. Core only.")
        # Ensure nodes list contains ONLY fred_core_node_data on error
        nodes = [fred_core_node_data] 
        edges = [] # Clear edges

    # Deduplicate nodes and ensure string IDs
    final_nodes = []
    node_ids_seen = set()
    for node in nodes:
        node_id_str = str(node['id'])
        if node_id_str not in node_ids_seen:
            node['id'] = node_id_str
            final_nodes.append(node)
            node_ids_seen.add(node_id_str)

    final_edges = []
    edge_tuples_seen = set()
    for edge in edges:
        source_str = str(edge['source'])
        target_str = str(edge['target'])
        rel_type_str = str(edge.get('rel_type', 'relatedTo'))
        # Check for duplicates considering direction and type
        # For undirected visual representation, (A,B) is same as (B,A) if rel_type is general.
        # For directed, they are different. Assuming directed for now.
        edge_tuple = tuple(sorted((source_str, target_str)) + [rel_type_str]) # Sort source/target for undirected check
        # edge_tuple = (source_str, target_str, rel_type_str) # For directed check
        if edge_tuple not in edge_tuples_seen:
            final_edges.append({"source": source_str, "target": target_str, "rel_type": rel_type_str})
            edge_tuples_seen.add(edge_tuple)
        
    return jsonify({"nodes": final_nodes, "edges": final_edges})

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global conversation_history
    try:
        logging.info("Entered event_stream try block.")
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        user_message = data.get('message')
        model_name = data.get('model', 'huihui_ai/qwen3-abliterated:8b') 
        ollama_base_url = data.get('ollama_url', 'http://localhost:11434').strip()
        max_tool_iterations = data.get('max_tool_iterations', 5)
        mute_fred_flag = data.get('mute_fred', False) # Get mute flag from request

        # Log the resolved model_name for this specific request
        logging.info(f"--- Resolved Ollama Model for this request: {model_name} ---")

        if not user_message or not ollama_base_url:
            missing_field = 'message' if not user_message else 'Ollama URL'
            return jsonify({'error': f'Missing required field: {missing_field}'}), 400

        logging.info(f"Chat request: Model='{model_name}', User Message: '{user_message[:50]}...' Muted: {mute_fred_flag}")

        def event_stream():
            logging.info("EVENT_STREAM: Entered function.")
            current_user_turn = {'role': 'user', 'content': user_message}
            conversation_history.append(current_user_turn) # Add to persistent history immediately

            formatted_system_prompt = SYSTEM_PROMPT
            
            system_message = {"role": "system", "content": formatted_system_prompt}
            
            messages = [system_message]
            # Create a temporary history for this request to include previous turns for the LLM
            # but not the current user turn which is formatted specially below.
            messages.extend(conversation_history[:-1]) 
            
            # --- Inject pending tasks alert --- #
            pending_tasks_alert = ""
            try:
                with lib.duckdb.connect(lib.DB_FILE) as con:
                    pending_count_res = con.execute("SELECT COUNT(*) FROM pending_edge_creation_tasks WHERE status = 'pending';").fetchone()
                    if pending_count_res and pending_count_res[0] > 20:
                        pending_tasks_alert = f"Alert: There are {pending_count_res[0]} memory nodes awaiting full connection processing. Consider using the 'update_knowledge_graph_edges' tool.\n"
            except Exception as e_alert:
                logging.warning(f"Could not query pending edge tasks: {e_alert}")
            # --- End inject pending tasks alert --- #

            formatted_user_input_for_llm = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
{pending_tasks_alert}The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END OF FRED DATABASE)
"""
            messages.append({"role": "user", "content": formatted_user_input_for_llm})
            
            # Terminal Output - Kept as requested
            print("\n=========================================")
            print(f">>> User input formatted for LLM:\n{formatted_user_input_for_llm}")
            print("-----------------------------------------")

            logging.info(f"EVENT_STREAM: Preparing to initialize Ollama client with URL: '{ollama_base_url}'")
            client = ollama.Client(host=ollama_base_url)
            logging.info("EVENT_STREAM: Ollama client object created.")
            
            assistant_response_final_content = ""

            logging.info(f"EVENT_STREAM: Starting tool iteration loop (max_tool_iterations: {max_tool_iterations})...")
            for iteration in range(max_tool_iterations):
                logging.info(f"EVENT_STREAM: Model call iteration {iteration + 1}. Preparing for client.chat() with stream=False and thinking_mode_options.")
                
                response = client.chat(
                    model=model_name,
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    stream=False,
                    options=thinking_mode_options # Added thinking mode options
                )
                logging.info(f"EVENT_STREAM: client.chat() with stream=False completed. Response message: {response.get('message', {}).get('content', '')[:50]}...")
                
                response_message = response.get('message', {})
                assistant_interim_content = response_message.get('content', '')
                # Print raw model output before stripping think tags
                if assistant_interim_content:
                    print("\n-----------------------------------------")
                    print(f">>> Raw Model Output (Non-Streaming):\n{assistant_interim_content}")
                    print("-----------------------------------------")
                # Strip think tags from non-streaming direct response
                assistant_interim_content = strip_think_tags(assistant_interim_content) if assistant_interim_content else ''
                tool_calls = response_message.get('tool_calls')

                # Append assistant's response (potentially with tool calls) to messages for next iteration
                if 'role' not in response_message:
                    response_message['role'] = 'assistant' # Ensure role is set
                messages.append(response_message)

                if tool_calls:
                    logging.info(f"LLM requested {len(tool_calls)} tool call(s).")
                    # Yield messages about tool calls being attempted
                    for tc in tool_calls:
                        tool_name = tc.get('function', {}).get('name', 'unknown_tool')
                        tool_args_str = json.dumps(tc.get('function', {}).get('arguments', {}))
                        activity_message = {
                            "type": "tool_activity", 
                            "content": f"Attempting tool: {tool_name} with args: {tool_args_str}"
                        }
                        yield json.dumps(activity_message) + '\n'

                    tool_outputs = handle_tool_calls(response_message) # Pass the assistant message that contains tool_calls

                    if tool_outputs:
                        logging.info(f"Executed {len(tool_outputs)} tool(s).")
                        # Yield messages about tool results
                        for out_idx, out_msg in enumerate(tool_outputs):
                            messages.append(out_msg) # Add tool output to messages for LLM
                            original_tc = tool_calls[out_idx] if out_idx < len(tool_calls) else {}
                            tool_name = original_tc.get('function', {}).get('name', 'unknown_tool')
                            # Summarize content for brevity in activity log if it's complex JSON
                            tool_content_str = out_msg.get('content', '{}')
                            try:
                                tool_content_parsed = json.loads(tool_content_str)
                                if isinstance(tool_content_parsed, dict) and 'results' in tool_content_parsed and isinstance(tool_content_parsed['results'], list):
                                    result_summary = f"{len(tool_content_parsed['results'])} results found."
                                elif isinstance(tool_content_parsed, dict) and 'success' in tool_content_parsed:
                                    result_summary = f"Success: {tool_content_parsed.get('success')}, Msg: {tool_content_parsed.get('message', tool_content_parsed.get('error', 'N/A'))}"
                                else:
                                    result_summary = tool_content_str[:100] + ('...' if len(tool_content_str) > 100 else '') # Truncate if too long
                            except json.JSONDecodeError:
                                result_summary = tool_content_str[:100] + ('...' if len(tool_content_str) > 100 else '')

                            activity_message = {
                                "type": "tool_activity", 
                                "content": f"Tool {tool_name} completed. Result: {result_summary}"
                            }
                            yield json.dumps(activity_message) + '\n'
                        
                        # Enhanced user message logic (internal, no print)
                        tool_results_formatted_for_llm = []
                        for tc_output in tool_outputs:
                            # Assume tc_output is like {'role': 'tool', 'content': '...', 'tool_call_id': '...'}
                            # Find original tool name for better context
                            tool_name_for_llm = "unknown_tool"
                            for tc_original in tool_calls:
                                if tc_original.get('id') == tc_output.get('tool_call_id'):
                                    tool_name_for_llm = tc_original.get('function', {}).get('name', 'unknown_tool')
                                    break
                            tool_content_for_llm = tc_output.get('content', '{}')
                            # Specific formatting for search_memory, if needed, can go here based on content
                            tool_results_formatted_for_llm.append(f"Tool '{tool_name_for_llm}' result: {tool_content_for_llm}")
                        
                        if tool_results_formatted_for_llm:
                            joined_results_for_llm = "\n\n".join(tool_results_formatted_for_llm)
                            enhanced_user_message_content = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
{joined_results_for_llm}
The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END OF FRED DATABASE)
"""
                            # Update the last user message in 'messages' list for the LLM
                            for i in range(len(messages) -1, -1, -1):
                                if messages[i].get('role') == 'user':
                                    messages[i]['content'] = enhanced_user_message_content
                                    logging.info("Updated user message with tool results for next LLM call.")
                                    break
                    else:
                        logging.warning("handle_tool_calls returned no outputs or an error occurred.")
                        error_payload = {"error": "Tool execution failed or produced no output."}
                        activity_message = {"type": "tool_activity", "content": f"Tool execution failed."}
                        yield json.dumps(activity_message) + '\n'
                        yield json.dumps({"type": "error", "content": json.dumps(error_payload)}) + '\n'
                        # Add a generic tool error message to messages for the LLM
                        tool_call_id_fb = tool_calls[0].get('id', f'fallback_id_iter{iteration}') if tool_calls else f'fallback_id_iter{iteration}'
                        messages.append({"role": "tool", "content": json.dumps(error_payload), "tool_call_id": tool_call_id_fb})
                        break # Break from tool loop if tool execution fails critically
                else: # No tool calls from LLM
                    logging.info("No tool calls detected by LLM. Proceeding to final response.")
                    if assistant_interim_content: # LLM provided content directly
                        assistant_response_final_content = assistant_interim_content
                    break # Exit tool loop

            # --- Final Response Generation --- 
            if assistant_response_final_content: # Already have final content from non-tool response
                logging.info(f"EVENT_STREAM: Preparing to speak direct response. Mute: {mute_fred_flag}. Content: '{assistant_response_final_content[:100]}...'")
                yield json.dumps({'response': assistant_response_final_content}) + '\n'
                conversation_history.append({'role': 'assistant', 'content': assistant_response_final_content})
                fred_speak(assistant_response_final_content, mute_fred_flag) # Speak the direct response
            else:
                logging.info("EVENT_STREAM: Preparing for final streaming response from LLM with non_thinking_mode_options.")
                # The messages list now contains the full context including any tool exchanges
                final_response_stream = client.chat(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    options=thinking_mode_options # Added non-thinking mode options
                )
                aggregated_final_content = ""
                for chunk in final_response_stream:
                    message_chunk = chunk.get('message', {})
                    content_chunk = message_chunk.get('content')
                    # Print raw model output chunk before stripping think tags
                    if content_chunk:
                        print("\n-----------------------------------------")
                        print(f">>> Raw Model Output Chunk (Streaming):\n{content_chunk}")
                        print("-----------------------------------------")
                    # Strip think tags from streaming response chunk
                    content_chunk = strip_think_tags(content_chunk) if content_chunk else None
                    is_done_chunk = chunk.get('done', False)

                    if content_chunk:
                        aggregated_final_content += content_chunk
                        yield json.dumps({'response': content_chunk}) + '\n'
                    
                    if is_done_chunk:
                        logging.info("EVENT_STREAM: Final streaming from LLM finished.")
                        if aggregated_final_content:
                            logging.info(f"EVENT_STREAM: Preparing to speak aggregated streamed response. Mute: {mute_fred_flag}. Content: '{aggregated_final_content[:100]}...'")
                            conversation_history.append({'role': 'assistant', 'content': aggregated_final_content})
                            fred_speak(aggregated_final_content, mute_fred_flag) # Speak the aggregated streamed response
                        else:
                            logging.info("EVENT_STREAM: Aggregated final content is empty. Nothing to speak.")
                        break 
            
            yield json.dumps({'type': 'done'}) + '\n'

        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Good practice for SSE with reverse proxies
        }
        return Response(event_stream(), headers=headers)

    except ollama.ResponseError as e:
        # ... (Error handling remains the same) ...
        error_msg = f'Ollama API Error: {e.error}'
        status_code = e.status_code
        if hasattr(e, 'error') and isinstance(e.error, str) and "connection refused" in e.error.lower():
             error_msg = f'Connection Error: Could not connect to Ollama at {ollama_base_url}'
             status_code = 503
        logging.error(f"Ollama Error: {error_msg} (Status: {status_code})", exc_info=True)
        # Clean up history (remove last user message if error occurred before assistant response)
        if conversation_history and conversation_history[-1].get('role') == 'user':
            conversation_history.pop()
        return jsonify({'error': error_msg, 'status_code': status_code}), status_code

    except Exception as e:
        # ... (Error handling remains the same) ...
        logging.error(f"[ERROR] {e}\n{traceback.format_exc()}")
        if conversation_history and conversation_history[-1].get('role') == 'user':
            conversation_history.pop()
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory_visualization_data')
def get_memory_viz_data():
    """API endpoint to fetch data for the mind map visualization."""
    try:
        # Initialize DB if needed (might already be handled elsewhere)
        # librarian.init_db()
        
        data = lib.get_all_active_nodes_for_viz()
        return jsonify(data)
    except ImportError:
        return jsonify({"error": "Librarian module or DuckDB not available"}), 500
    except Exception as e:
        app.logger.error(f"Error fetching visualization data: {e}") # Use Flask logger
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

# --- WAV File Cleanup Function ---
def cleanup_startup_wav_files():
    global APP_ROOT
    logging.info("Performing startup cleanup of old speech output WAV files...")
    pattern = os.path.join(APP_ROOT, "fred_speech_output_*.wav")
    count = 0
    for f_path in glob.glob(pattern):
        try:
            os.remove(f_path)
            logging.info(f"Deleted old WAV file: {f_path}")
            count += 1
        except Exception as e:
            logging.warning(f"Could not delete old WAV file {f_path}: {e}")
    if count > 0:
        logging.info(f"Startup cleanup: Deleted {count} old WAV file(s).")
    else:
        logging.info("Startup cleanup: No old speech output WAV files found to delete.")
# --- End WAV File Cleanup ---

if __name__ == '__main__':
    # --- Perform startup cleanup ---
    cleanup_startup_wav_files() # Added call to cleanup WAV files
    # --- End startup cleanup ---

    # Check Ollama connection
    try:
        ollama_check_url = (os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434') + '/').rstrip('/')
        requests.get(ollama_check_url, timeout=2)
        print(f"[INFO] Ollama connection successful at {ollama_check_url}")
    except requests.exceptions.RequestException:
        logging.warning(f"Could not connect to Ollama at {ollama_check_url}")
        logging.warning("Ensure Ollama is running")

    # Start server
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask server on http://0.0.0.0:{port}")
    # Use debug=False to reduce console output
    app.run(debug=False, port=port, host='0.0.0.0', use_reloader=True)