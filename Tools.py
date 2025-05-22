import time
import logging
import json
import datetime
from decimal import Decimal # Added import for Decimal
from duckduckgo_search import DDGS # Uncommented DuckDuckGo import
# Import the librarian module
import memory.librarian as lib

# Change logging level to ERROR to reduce console clutter
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tool Definitions ---
AVAILABLE_TOOLS = [
    {
        "name": "add_memory",
        "description": "Adds a new memory node to the knowledge graph. Use for new information, facts, events, or procedures.",
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
        "description": "Replaces a specific old memory node with new, corrected information. Use ONLY after the user explicitly confirms an update to resolve a conflict you identified.",
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
        "description": "Searches the knowledge graph for memories relevant to a query text using semantic similarity.",
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
                    "description": "Optional. The maximum number of search results to return. Defaults to 10.",
                    "default": 10
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
                }
            },
            "required": ["query_text"]
        }
    },
    {
        "name": "search_web_information",
        "description": "Searches the web for information using DuckDuckGo. This tool retrieves current information from the internet. It combines results from general web search and news search.",
        "parameters": {"query_text": {"type": "string", "description": "The text to search for."}}
    },
    {
        "name": "get_node_by_id",
        "description": "Retrieves a specific memory node by its ID, along with its connections to other nodes.",
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
        "description": "Retrieves a subgraph centered around a specific node, showing its connections up to a certain depth.",
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

# --- Tool Implementations ---
# Wrapper for adding memory
def tool_add_memory(label: str, text: str, memory_type: str, parent_id: int | None = None, target_date: str | None = None):
    """Wrapper to call librarian.add_memory."""
    logging.info(f"Tool call: add_memory(label='{label}', type='{memory_type}', target_date='{target_date}')")
    try:
        # Ensure memory_type is valid before calling lib function
        if memory_type not in ('Semantic', "Episodic", "Procedural"):
             raise ValueError(f"Invalid memory_type '{memory_type}' provided to tool.")
        
        # Parse the ISO format date string to datetime object if provided
        parsed_target_date = None
        if target_date:
            try:
                # Try full datetime format first (YYYY-MM-DDTHH:MM:SS)
                parsed_target_date = datetime.datetime.fromisoformat(target_date)
            except ValueError:
                try:
                    # Try date-only format (YYYY-MM-DD)
                    parsed_target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Invalid target_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        
        node_id = lib.add_memory(label=label, text=text, memory_type=memory_type, parent_id=parent_id, target_date=parsed_target_date)
        return {"success": True, "nodeid": node_id, "message": f"Memory added with ID {node_id}."}
    except Exception as e:
        logging.error(f"Error in tool_add_memory: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# Wrapper for superseding memory
def tool_supersede_memory(old_nodeid: int, new_label: str, new_text: str, new_memory_type: str, target_date: str | None = None):
    """Wrapper to call librarian.supersede_memory."""
    logging.info(f"Tool call: supersede_memory(old_nodeid={old_nodeid}, new_label='{new_label}', new_type='{new_memory_type}', target_date='{target_date}')")
    try:
         # Validate old_nodeid is an integer
         if not isinstance(old_nodeid, int):
            error_message = f"Invalid old_nodeid: '{old_nodeid}'. Must be an integer."
            logging.error(error_message)
            return {"success": False, "error": error_message}

         # Ensure memory_type is valid before calling lib function
         if new_memory_type not in ('Semantic', "Episodic", "Procedural"):
             raise ValueError(f"Invalid new_memory_type '{new_memory_type}' provided to tool.")
         
         # Parse the ISO format date string to datetime object if provided
         parsed_target_date = None
         if target_date:
             try:
                 # Try full datetime format first (YYYY-MM-DDTHH:MM:SS)
                 parsed_target_date = datetime.datetime.fromisoformat(target_date)
             except ValueError:
                 try:
                     # Try date-only format (YYYY-MM-DD)
                     parsed_target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d")
                 except ValueError:
                     raise ValueError(f"Invalid target_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
         
         new_nodeid = lib.supersede_memory(old_nodeid=old_nodeid, new_label=new_label, new_text=new_text, new_memory_type=new_memory_type, target_date=parsed_target_date)
         return {"success": True, "new_nodeid": new_nodeid, "message": f"Memory {old_nodeid} superseded by {new_nodeid}."}
    except Exception as e:
        logging.error(f"Error in tool_supersede_memory: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# Wrapper for searching memory
def tool_search_memory(query_text: str, memory_type: str | None = None, limit: int = 10, future_events_only: bool = False, use_keyword_search: bool = False):
    """Wrapper to call librarian.search_memory."""
    logging.info(f"Tool call: search_memory(query='{query_text}', type='{memory_type}', limit={limit}, future_events_only={future_events_only}, use_keyword_search={use_keyword_search})")
    try:
        # Validate memory_type if provided
        if memory_type is not None and memory_type not in ('Semantic', "Episodic", "Procedural"):
            raise ValueError(f"Invalid memory_type filter '{memory_type}' provided to tool.")
        
        # Pass include_connections=True to get edge data in a single efficient query
        results = lib.search_memory(
            query_text=query_text, 
            memory_type=memory_type, 
            limit=limit, 
            future_events_only=future_events_only, 
            use_keyword_search=use_keyword_search,
            include_connections=True  # Always include connections in the tool interface
        )
        
        # Convert datetime and Decimal objects to JSON-compatible types
        for result in results:
            for key, value in result.items():
                if isinstance(value, datetime.datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, Decimal):
                    result[key] = float(value) # Convert Decimal to float
            
        return {"success": True, "results": results}
    except Exception as e:
        logging.error(f"Error in tool_search_memory: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def search_web_information(query_text: str) -> str:
    """
    Searches the web for information using DuckDuckGo.
    This tool retrieves current information from the internet.
    It combines results from general web search and news search.

    Args:
        query_text (str): The text to search for.
        
    Returns:
        str: A JSON string containing the search results or an error message.
    """
    logging.info(f"Tool call: search_web_information(query_text='{query_text}')")
    try:
        ddgs = DDGS(timeout=20) # Initialize with a timeout
        
        search_keywords = query_text # Use the provided query_text directly
        
        text_results = []
        news_results = []
        
        # Perform text search
        try:
            text_search_results = ddgs.text(
                keywords=search_keywords,
                region="us-en",
                safesearch="off",
                max_results=3  # Limiting results for brevity
            )
            if text_search_results:
                text_results = text_search_results
        except Exception as e_text:
            logging.warning(f"DuckDuckGo text search failed for '{search_keywords}': {e_text}")
            # Optionally, include this warning in the output if desired

        # Perform news search
        try:
            news_search_results = ddgs.news(
                keywords=search_keywords,
                region="us-en",
                safesearch="off",
                max_results=2 # Limiting news results
            )
            if news_search_results:
                news_results = news_search_results
        except Exception as e_news:
            logging.warning(f"DuckDuckGo news search failed for '{search_keywords}': {e_news}")

        if not text_results and not news_results:
            return json.dumps({"success": True, "results": "No information found from web search."})
        
        combined_prompt = f"Summarize the following web search results: {text_results} {news_results}"

        combined_results = {
            "text_search_summary": text_results,
            "news_search_summary": news_results
        }

        
        try:
            # Single summarization call
            response = ddgs.chat(
                keywords=combined_prompt,
                model="gpt-4o-mini",
                timeout=60
            )
            if response:
                combined_results["llm_summary"] = response
        except Exception as e_summary:
            logging.warning(f"DuckDuckGo LLM summary failed for '{combined_prompt}': {e_summary}")
            combined_results["llm_summary"] = "Could not generate summary"
        
        return json.dumps({"success": True, "results": combined_results})

    except Exception as e:
        logging.error(f"Error in search_web_information: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)})

# New tool to get a node by ID with its connections
def tool_get_node_by_id(nodeid: int):
    """Retrieves a specific node by ID with its connections."""
    logging.info(f"Tool call: get_node_by_id(nodeid={nodeid})")
    try:
        # Use get_graph_data with depth=1 to get the node and its immediate connections
        graph_data = lib.get_graph_data(nodeid, depth=1)
        
        # Find the target node in the nodes list
        target_node = next((n for n in graph_data.get('nodes', []) if n['id'] == nodeid), None)
        
        if not target_node:
            return {"success": False, "error": f"Node with ID {nodeid} not found or is superseded."}
        
        # Process datetime objects for JSON compatibility
        for key, value in target_node.items():
            if isinstance(value, datetime.datetime):
                target_node[key] = value.isoformat()
        
        # Extract connections specific to this node
        connections = []
        for edge in graph_data.get('edges', []):
            if edge['source'] == nodeid:
                # Outgoing connection
                target_node_info = next((n for n in graph_data.get('nodes', []) if n['id'] == edge['target']), None)
                if target_node_info:
                    connections.append({
                        "direction": "outgoing",
                        "rel_type": edge['rel_type'],
                        "target_nodeid": edge['target'],
                        "target_label": target_node_info.get('label', ''),
                        "target_type": target_node_info.get('type', '')
                    })
            elif edge['target'] == nodeid:
                # Incoming connection
                source_node_info = next((n for n in graph_data.get('nodes', []) if n['id'] == edge['source']), None)
                if source_node_info:
                    connections.append({
                        "direction": "incoming",
                        "rel_type": edge['rel_type'],
                        "source_nodeid": edge['source'],
                        "source_label": source_node_info.get('label', ''),
                        "source_type": source_node_info.get('type', '')
                    })
        
        # Add connections to the result
        result = {
            "node": target_node,
            "connections": connections
        }
        
        return {"success": True, "result": result}
    except Exception as e:
        logging.error(f"Error in tool_get_node_by_id: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# New tool to get graph data
def tool_get_graph_data(center_nodeid: int, depth: int = 1):
    """Retrieves nodes and edges around a center node up to specified depth."""
    logging.info(f"Tool call: get_graph_data(center_nodeid={center_nodeid}, depth={depth})")
    try:
        graph_data = lib.get_graph_data(center_nodeid, depth=depth)
        
        # Process datetime objects for JSON compatibility
        for node in graph_data.get('nodes', []):
            for key, value in node.items():
                if isinstance(value, datetime.datetime):
                    node[key] = value.isoformat()
        
        return {"success": True, "result": graph_data}
    except Exception as e:
        logging.error(f"Error in tool_get_graph_data: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def tool_update_knowledge_graph_edges(limit_per_run: int = 5):
    """Wrapper to call librarian.process_pending_edges."""
    logging.info(f"Tool call: update_knowledge_graph_edges(limit_per_run={limit_per_run})")
    try:
        if not isinstance(limit_per_run, int) or limit_per_run <= 0:
            raise ValueError("limit_per_run must be a positive integer.")
        
        summary = lib.process_pending_edges(limit_per_run=limit_per_run)
        # Ensure all parts of the summary are JSON serializable, though they should be.
        # datetime objects are handled by librarian if any were returned directly.
        return {"success": True, "result": summary}
    except Exception as e:
        logging.error(f"Error in tool_update_knowledge_graph_edges: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# --- Tool Registry ---
TOOL_FUNCTIONS = {
    "add_memory": tool_add_memory,
    "supersede_memory": tool_supersede_memory,
    "search_memory": tool_search_memory,
    "search_web_information": search_web_information,
    "get_node_by_id": tool_get_node_by_id,
    "get_graph_data": tool_get_graph_data,
    "update_knowledge_graph_edges": tool_update_knowledge_graph_edges
}

# --- Tool Execution Logic ---
# Simplified execution logic, assuming args are already parsed dicts by handle_tool_calls
def handle_tool_calls(response_message):
    """
    Handles tool calls found within an LLM response message.

    Args:
        response_message: The 'message' dictionary from the Ollama response.

    Returns:
        A list of dictionaries, where each dictionary represents the result
        of a tool call formatted for the Ollama API (role='tool', content='...', tool_call_id='...').
        Returns an empty list if no tool calls are present or errors occur.
    """
    tool_calls = response_message.get('tool_calls')
    if not tool_calls:
        return [] # No tool calls to handle

    tool_outputs = []
    for tool_call in tool_calls:
        function_name = tool_call.get('function', {}).get('name')
        tool_args = tool_call.get('function', {}).get('arguments', {}) # Arguments should be dict
        tool_call_id = tool_call.get('id') # Ollama now includes an ID

        if not function_name:
            logging.error("Tool call missing function name.")
            # Optionally add an error output for the LLM
            continue # Skip this invalid tool call

        if not tool_call_id:
             logging.warning(f"Tool call for '{function_name}' missing 'id'. Generating one.")
             # Generate a fallback ID if needed, although Ollama should provide it
             tool_call_id = f"call_{time.time_ns()}"

        logging.info(f"Handling tool call '{function_name}' with ID '{tool_call_id}' and args: {tool_args}")

        # Argument parsing now happens directly before calling the function from TOOL_FUNCTIONS
        try:
            # Ensure arguments are a dictionary for ** unpacking
            if not isinstance(tool_args, dict):
                try:
                    # Ollama now seems to pass args as a dict directly sometimes,
                    # but might still be stringified JSON in other cases? Be robust.
                    tool_args = json.loads(str(tool_args)) 
                except json.JSONDecodeError:
                     logging.error(f"Tool arguments for '{function_name}' are not valid JSON or a dictionary: {tool_args}")
                     output_content = json.dumps({"success": False, "error": f"Invalid arguments format for {function_name}."})
                     tool_outputs.append({"tool_call_id": tool_call_id, "role": "tool", "content": output_content})
                     continue # Skip this tool call

            # Execute the tool (using the function from TOOL_FUNCTIONS directly)
            logging.info(f"Executing tool '{function_name}' via handle_tool_calls with args: {tool_args}")
            tool_function = TOOL_FUNCTIONS.get(function_name)
            if tool_function:
                try:
                    # Call the actual tool function (e.g., tool_add_memory)
                    result_dict = tool_function(**tool_args) 
                    output_content = json.dumps(result_dict) # Result should already be a serializable dict
                except TypeError as e:
                    # Handles cases where LLM might provide wrong/missing args
                    logging.error(f"TypeError executing {function_name} via handle_tool_calls: {e}. Args: {tool_args}")
                    output_content = json.dumps({"success": False, "error": f"Argument mismatch for tool '{function_name}': {e}"})
                except Exception as e:
                    logging.error(f"Unexpected error executing {function_name} via handle_tool_calls: {e}", exc_info=True)
                    output_content = json.dumps({"success": False, "error": f"Error during execution of {function_name}: {str(e)}"})
            else:
                logging.error(f"Tool '{function_name}' not found in TOOL_FUNCTIONS.")
                output_content = json.dumps({"success": False, "error": f"Tool '{function_name}' not found"})
            
            # Append the result for this tool call
            tool_outputs.append({"tool_call_id": tool_call_id, "role": "tool", "content": output_content})

        except Exception as e: # Catch broader errors during the loop iteration
            logging.error(f"General error processing tool call {tool_call_id} for {function_name}: {e}", exc_info=True)
            # Provide a generic error response for this specific tool call
            error_content = json.dumps({"success": False, "error": f"Internal error processing tool call '{function_name}'."})
            tool_outputs.append({"tool_call_id": tool_call_id, "role": "tool", "content": error_content})

    return tool_outputs
