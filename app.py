import os
import ollama
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
import json
import requests
# Removed old memory imports
# from memory.hybrid_memory import HybridMemory
# from memory.mindmap_bridge import MindMapBridge

# Import the new librarian
import memory.librarian as lib

app = Flask(__name__, static_folder='static', template_folder='templates') # Define folders

# --- New Memory System Initialization ---
# Get the absolute path to the directory containing this app.py file
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# Define the path for the memory database within the memory folder
DB_PATH = os.path.join(APP_ROOT, 'memory', 'memory.db')
# Set the database file path in the librarian module
lib.DB_FILE = DB_PATH
# Initialize the database (creates file and tables if they don't exist)
lib.init_db()
print(f"[INFO] Memory system initialized. Database at: {DB_PATH}")
# ---------------------------------------

# Ensure the static folder exists
STATIC_FOLDER = app.static_folder
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Removed old memory/bridge initialization and data file check

# Global variable to store conversation history (Consider alternatives for production)
conversation_history = []

def generate_chunks(response):
    """Yields response chunks as server-sent events."""
    full_response_content = "" # Accumulate full response
    user_msg_content = conversation_history[-1]['content'] if conversation_history and conversation_history[-1]['role'] == 'user' else "Unknown user message"
    assistant_node_id = None # To store the ID for the assistant's response memory

    try:
        # Add user message to memory immediately
        print(f"[INFO] Adding user message to memory: {user_msg_content[:50]}...")
        user_label = f"User: {user_msg_content[:30]}..."
        user_node_id = lib.add_memory(label=user_label, text=user_msg_content, memory_type="Episodic")
        if user_node_id:
            print(f"[INFO] User message added to memory with Node ID: {user_node_id}")
        else:
            print("[WARNING] Failed to add user message to memory (Ollama issue?).")

        # Process Ollama stream
        for chunk in response:
            content = getattr(getattr(chunk, 'message', None), 'content', None)
            if content:
                full_response_content += content
                # Format as Server-Sent Event (SSE) - Moved \n\n outside f-string
                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}" + "\n\n"

            if getattr(chunk, 'done', False):
                if full_response_content:
                    conversation_history.append({'role': 'assistant', 'content': full_response_content})

                    # Add assistant response to memory
                    print(f"[INFO] Adding assistant response to memory: {full_response_content[:50]}...")
                    assistant_label = f"Assistant: {full_response_content[:30]}..."
                    assistant_node_id = lib.add_memory(label=assistant_label, text=full_response_content, memory_type="Episodic")
                    if assistant_node_id:
                        print(f"[INFO] Assistant response added to memory with Node ID: {assistant_node_id}")
                        # Link user message and assistant response
                        if user_node_id:
                            print(f"[INFO] Linking user node {user_node_id} and assistant node {assistant_node_id}")
                            lib.add_edge(user_node_id, assistant_node_id, 'relatedTo')
                            # Optional: Add edge back? Or keep unidirectional?
                            # lib.add_edge(assistant_node_id, user_node_id, 'relatedTo')
                    else:
                        print("[WARNING] Failed to add assistant response to memory (Ollama issue?).")
                else:
                     print("[WARNING] Assistant response was empty, not adding to memory.")

                # Moved \n\n outside f-string
                yield f"data: {json.dumps({'type': 'done'})}" + "\n\n"
                break
    except Exception as e:
        error_message = f"Error streaming response or processing memory: {e}"
        print(f"[SERVER ERROR] {error_message}")
        try:
            # Moved \n\n outside f-string
            yield f"data: {json.dumps({'type': 'error', 'content': error_message})}" + "\n\n"
        except Exception as send_e:
            print(f"[SERVER ERROR] Failed to send error to client: {send_e}")

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# Route to serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# --- NEW Memory API Endpoints --- #

@app.route('/graph')
def get_graph():
    """Endpoint to get graph data (nodes and edges) for the mind map."""
    center_nodeid_str = request.args.get('center')
    depth_str = request.args.get('depth', '1')

    if not center_nodeid_str:
        # If no center node, find the most recently created node
        try:
            with lib.duckdb.connect(lib.DB_FILE) as con:
                res = con.execute("SELECT nodeid FROM nodes WHERE superseded_at IS NULL ORDER BY created_at DESC LIMIT 1").fetchone()
                if res:
                    center_nodeid = res[0]
                    print(f"[INFO] No center specified, using most recent node: {center_nodeid}")
                else:
                    print("[INFO] No nodes found in DB for graph center.")
                    return jsonify({"nodes": [], "edges": []})
        except Exception as e:
            print(f"[ERROR] Error getting default center node: {e}")
            return jsonify({"error": "Could not determine center node"}), 500
    else:
        try:
            center_nodeid = int(center_nodeid_str)
        except ValueError:
            return jsonify({"error": "Invalid center node ID"}), 400

    try:
        depth = int(depth_str)
    except ValueError:
        return jsonify({"error": "Invalid depth value"}), 400

    try:
        print(f"[INFO] Fetching graph data: center={center_nodeid}, depth={depth}")
        graph_data = lib.get_graph_data(center_nodeid, depth)
        # Ensure IDs are strings for JSON/JS if they are large integers
        for node in graph_data.get('nodes', []):
            node['id'] = str(node['id'])
        for edge in graph_data.get('edges', []):
             edge['source'] = str(edge['source'])
             edge['target'] = str(edge['target'])
        print(f"[INFO] Returning {len(graph_data.get('nodes',[]))} nodes, {len(graph_data.get('edges',[]))} edges.")
        return jsonify(graph_data)
    except Exception as e:
        print(f"[ERROR] Error fetching graph data: {e}")
        return jsonify({"error": "Failed to retrieve graph data"}), 500

@app.route('/search')
def search():
    """Endpoint to search memories."""
    query = request.args.get('q')
    memory_type = request.args.get('type') # Optional: Semantic or Episodic

    if not query:
        return jsonify({"error": "Missing search query 'q'"}), 400

    if memory_type and memory_type not in ['Semantic', 'Episodic']:
        return jsonify({"error": "Invalid type parameter. Use 'Semantic' or 'Episodic'"}), 400

    try:
        print(f"[INFO] Searching memory: query='{query}', type={memory_type}")
        results = lib.search_memory(query, memory_type)
        # Convert nodeid to string for JS compatibility if needed
        for res in results:
             res['nodeid'] = str(res['nodeid'])
        print(f"[INFO] Search returned {len(results)} results.")
        return jsonify(results)
    except Exception as e:
        print(f"[ERROR] Error during search: {e}")
        return jsonify({"error": "Search operation failed"}), 500

# --- Chat Endpoint --- #

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handles chat requests and streams responses from Ollama."""
    try:
        data = request.json
        if not data:
             return jsonify({'error': 'Invalid JSON payload'}), 400

        user_message = data.get('message')
        model = data.get('model', 'phi4-mini:latest') # Default model
        ollama_base_url = data.get('ollama_url', 'http://localhost:11434').strip()

        if not user_message or not ollama_base_url:
            missing_field = 'message' if not user_message else 'Ollama URL'
            return jsonify({'error': f'Missing required field: {missing_field}'}), 400

        print(f"[INFO] Received chat request: Model='{model}', URL='{ollama_base_url}'")

        # Add user message to conversation history (for Ollama context)
        conversation_history.append({'role': 'user', 'content': user_message})

        client = ollama.Client(host=ollama_base_url, timeout=60)

        response = client.chat(
            model=model,
            messages=conversation_history,
            stream=True
        )

        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
        # generate_chunks now handles adding user/assistant messages to memory
        return Response(generate_chunks(response), headers=headers)

    except ollama.ResponseError as e:
        error_msg = f'Ollama API Error: {e.error}'
        status_code = e.status_code
        if "connection refused" in str(e.error).lower() or "Failed to establish a new connection" in str(e.error):
            error_msg = f'Connection Error: Could not connect to Ollama at {ollama_base_url}. Is Ollama running?'
            status_code = 503
        print(f"[OLLAMA ERROR] {error_msg} (Status: {status_code})")
        if conversation_history and conversation_history[-1]['role'] == 'user':
             conversation_history.pop() # Remove last user message on error
        return jsonify({'error': error_msg, 'status_code': status_code}), status_code
    except Exception as e:
        error_msg = f'An unexpected error occurred: {e}'
        print(f"[SERVER ERROR] {error_msg}")
        if conversation_history and conversation_history[-1]['role'] == 'user':
             conversation_history.pop() # Remove last user message on error
        return jsonify({'error': error_msg}), 500

# --- Removed Old Memory Management API Routes --- #
# The concepts of idle mode, threshold setting, and manual map refresh
# from the previous system are not directly applicable here.

if __name__ == '__main__':
    # Check Ollama connection (optional but helpful)
    try:
        ollama_check_url = (os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434') + '/').rstrip('/') # Get base URL
        requests.get(ollama_check_url, timeout=2)
        print(f"[INFO] Ollama connection successful at {ollama_check_url}.")
    except requests.exceptions.RequestException:
        print(f"\n*** WARNING: Could not connect to Ollama at {ollama_check_url}")
        print("*** Ensure Ollama is running.")
        print("*** Memory embedding and chat will likely fail.\n")

    port = int(os.environ.get('PORT', 5000)) # Changed default port back to 5000
    print(f"[INFO] Starting Flask server on http://0.0.0.0:{port}")
    # use_reloader=False can be helpful for debugging initialization
    app.run(debug=True, port=port, host='0.0.0.0', use_reloader=True) # Allow reloader for dev 