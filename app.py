import os
import ollama
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
import json
import requests
import memory.librarian as lib
import logging # Added logging
# Import tool-related items
from Tools import AVAILABLE_TOOLS, handle_tool_calls # Removed execute_tool import, handled by handle_tool_calls
from datetime import datetime
import sys

# Adjust sys.path BEFORE importing librarian
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory import librarian # Import your librarian module

app = Flask(__name__, static_folder='static', template_folder='templates')
# Change logging level to ERROR to reduce console clutter
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize memory database
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_ROOT, 'memory', 'memory.db')
lib.DB_FILE = DB_PATH

FRED_CORE_NODE_ID = "FRED_CORE" # Define F.R.E.D. Core Node ID

# --- Load System Prompt from File ---
SYSTEM_PROMPT_FILE = os.path.join(APP_ROOT, 'system_prompt.txt')
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()
    print(f"[INFO] Loaded system prompt from: {SYSTEM_PROMPT_FILE}")
except FileNotFoundError:
    print(f"[ERROR] System prompt file not found at: {SYSTEM_PROMPT_FILE}")
    SYSTEM_PROMPT = "Error: Could not load system prompt." # Fallback
except Exception as e:
    print(f"[ERROR] Failed to read system prompt file: {e}")
    SYSTEM_PROMPT = "Error: Could not load system prompt." # Fallback

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
    print(f"[ERROR] Failed to initialize database: {e}")

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

def generate_final_response_chunks(response):
    """Streams the final response after tool calls (if any) are handled."""
    full_response_content = ""
    try:
        for chunk in response:
            message = chunk.get('message', {})
            content = message.get('content')
            is_done = chunk.get('done', False)

            if content:
                full_response_content += content
                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}" + "\n\n"

            # Check for done status correctly
            if is_done:
                logging.info("Streaming finished.")
                yield f"data: {json.dumps({'type': 'done'})}" + "\n\n"
                # Add final assistant message to history AFTER streaming is complete
                if full_response_content:
                    conversation_history.append({'role': 'assistant', 'content': full_response_content})
                break # Exit loop once done

    except Exception as e:
        error_message = f"Error streaming final response: {e}"
        logging.error(f"[ERROR] {error_message}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': error_message})}" + "\n\n"

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
                    print(f"[WARN] Skipping secondary search for non-integer node ID: {memory_node['id']}")
                except Exception as e_neighbor: # Catch errors during individual neighbor fetch
                    print(f"[WARN] Error fetching neighbors for satellite {memory_node['id']}: {e_neighbor}")
            
            nodes.extend(secondary_nodes_to_add)
            edges.extend(secondary_edges_to_add)

    except Exception as e_main:
        print(f"[ERROR] Main error in get_graph: {e_main}. Returning F.R.E.D. Core only.")
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
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        user_message = data.get('message')
        # 1. Use the requested model
        model_name = data.get('model', 'qwen3:30b-a3b') # UPDATED Default Model
        ollama_base_url = data.get('ollama_url', 'http://localhost:11434').strip()
        max_tool_iterations = data.get('max_tool_iterations', 5)

        if not user_message or not ollama_base_url:
            missing_field = 'message' if not user_message else 'Ollama URL'
            return jsonify({'error': f'Missing required field: {missing_field}'}), 400

        logging.info(f"Chat request: Model='{model_name}'")

        # Add user message to persistent history FIRST
        current_user_turn = {'role': 'user', 'content': user_message}
        conversation_history.append(current_user_turn)

        # --- Format the System Prompt with Tools --- 
        try:
            # Convert tool schemas to a JSON string representation for the prompt
            tool_schemas_string = json.dumps(AVAILABLE_TOOLS, indent=2)
            # Format the system prompt read from the file
            formatted_system_prompt = SYSTEM_PROMPT.format(tool_schemas=tool_schemas_string)
        except KeyError: # Handle case where placeholder isn't in the file
            logging.warning("'{tool_schemas}' placeholder not found in system_prompt.txt. Using prompt as-is.")
            formatted_system_prompt = SYSTEM_PROMPT
        except Exception as e:
            logging.error(f"Error formatting system prompt with tools: {e}")
            formatted_system_prompt = "Error: Could not format system prompt with tools." # Fallback
        
        system_message = {"role": "system", "content": formatted_system_prompt}
        
        # --- Create message history for this model call ---
        messages = [system_message]  # Start with system prompt
        messages.extend(conversation_history[:-1])  # Add previous conversation history
        
        # Format the user's message
        formatted_user_message = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END OF FRED DATABASE)
"""

        # Add formatted user message to history
        messages.append({"role": "user", "content": formatted_user_message})
        
        # Terminal Output - Keep this as it's important for debugging
        print("\n=========================================")
        print(f">>> User input formatted for LLM:\\n{formatted_user_message}")
        print("-----------------------------------------")

        client = ollama.Client(host=ollama_base_url, timeout=180)

        # --- Tool Calling Loop ---
        assistant_response_content = None
        for iteration in range(max_tool_iterations):
            logging.info(f"Model call iteration {iteration + 1}")
            
            # Call the model with current message history
            response = client.chat(
                model=model_name,
                messages=messages,
                tools=AVAILABLE_TOOLS,
                stream=False
            )
            
            response_message = response.get('message', {})
            assistant_response_content = response_message.get('content', '')
            tool_calls = response_message.get('tool_calls')

            # Add assistant response to messages
            if response_message and 'role' not in response_message:
                response_message['role'] = 'assistant'
            messages.append(response_message)

            # Break if no tool calls
            if not tool_calls:
                logging.info("No tool calls detected, breaking loop.")
                break

            # Log tool calls
            print(f"\n>>> LLM Requested Tool Calls (Iteration {iteration + 1}):")
            try:
                # Convert tool_calls to a serializable format
                serializable_tool_calls = make_json_serializable(tool_calls)
                print(json.dumps(serializable_tool_calls, indent=2))
            except Exception as e:
                print(f"Could not serialize tool calls: {e}")
                print(f"Tool calls: {tool_calls}")
            print("-----------------------------------------")

            # Execute tools
            tool_outputs = handle_tool_calls(response_message)

            if tool_outputs:
                # Log tool results
                print(f"\n>>> Tool Execution Results (Iteration {iteration + 1}):")
                try:
                    parsed_outputs = make_json_serializable(tool_outputs)
                    print(json.dumps(parsed_outputs, indent=2))
                except Exception as e:
                    print(f"Could not serialize tool outputs: {e}")
                    print(f"Tool outputs: {tool_outputs}")
                print("-----------------------------------------")

                # Add tool outputs to messages
                messages.extend(tool_outputs)
                
                # Format tool results for readability
                tool_results_formatted = []
                for output in tool_outputs:
                    tool_name = output.get('name', 'unknown')
                    content = output.get('content', '{}')
                    
                    try:
                        content_json = json.loads(content)
                        # Extract tool name from the tool_call_id if not directly available
                        if tool_name == 'unknown' and 'tool_call_id' in output:
                            # Get function name from original tool calls
                            for tc in tool_calls:
                                if tc.get('id') == output.get('tool_call_id'):
                                    tool_name = tc.get('function', {}).get('name', 'unknown')
                                    break
                        
                        # Special formatting for search_memory results
                        if tool_name == "search_memory" and content_json.get('success') and 'results' in content_json:
                            results = content_json['results']
                            if results:
                                memory_str = format_rag_context(results)
                                tool_results_formatted.append(f"Tool 'search_memory' found these memories:\n{memory_str}")
                            else:
                                tool_results_formatted.append(f"Tool 'search_memory' result: No relevant memories found.")
                        else:
                            tool_results_formatted.append(f"Tool '{tool_name}' result: {content}")
                    except json.JSONDecodeError:
                        tool_results_formatted.append(f"Tool '{tool_name}' result: {content}")
                
                # Join formatted results
                tool_results_formatted = "\n\n".join(tool_results_formatted)
                
                # Create enhanced user message with tool results
                if tool_results_formatted:
                    enhanced_user_message = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
{tool_results_formatted}
The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END OF FRED DATABASE)
"""
                    # Print a shorter version of the enhanced message
                    print(f"\n>>> Enhanced user message with tool results")
                    print(f">>> Original input: '{user_message}'")
                    print(f">>> Added {len(tool_results_formatted.splitlines())} lines of tool results")
                    print("-----------------------------------------")
                    
                    # Replace the most recent user message with enhanced version
                    for i in range(len(messages)-1, -1, -1):
                        if messages[i].get('role') == 'user':
                            messages[i]['content'] = enhanced_user_message
                            break
            else:
                # Log tool error
                logging.warning("handle_tool_calls returned no outputs.")
                error_content = json.dumps({"error": "Tool execution failed or produced no output."})
                tool_call_id_fallback = tool_calls[0].get('id', 'fallback_id') if tool_calls else 'fallback_id'
                messages.append({ 
                    "role": "tool", 
                    "content": error_content, 
                    "tool_call_id": tool_call_id_fallback 
                })
                print(f"\n>>> Tool Execution Error (Iteration {iteration + 1}):")
                print(error_content)
                print("-----------------------------------------")
                break

        # --- Final Response Generation ---
        if assistant_response_content and not tool_calls:
            logging.info("Sending last text response directly.")
            # Add the final assistant text response to persistent history
            conversation_history.append({'role': 'assistant', 'content': assistant_response_content})
            # Simulate stream for client
            def direct_response_stream(content):
                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return Response(direct_response_stream(assistant_response_content), mimetype='text/event-stream')
        else:
            logging.info("Generating final streaming response after tool loop.")
            
            # Generate final streaming response
            final_response_stream = client.chat(
                model=model_name,
                messages=messages,
                stream=True
            )
            headers = {'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
            return Response(generate_final_response_chunks(final_response_stream), headers=headers)

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
        import traceback
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
        
        data = librarian.get_all_active_nodes_for_viz()
        return jsonify(data)
    except ImportError:
        return jsonify({"error": "Librarian module or DuckDB not available"}), 500
    except Exception as e:
        app.logger.error(f"Error fetching visualization data: {e}") # Use Flask logger
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    # Check Ollama connection
    try:
        ollama_check_url = (os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434') + '/').rstrip('/')
        requests.get(ollama_check_url, timeout=2)
        print(f"[INFO] Ollama connection successful at {ollama_check_url}")
    except requests.exceptions.RequestException:
        print(f"*** WARNING: Could not connect to Ollama at {ollama_check_url}")
        print("*** Ensure Ollama is running")

    # Start server
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask server on http://0.0.0.0:{port}")
    # Use debug=False to reduce console output
    app.run(debug=False, port=port, host='0.0.0.0', use_reloader=True)