import time
import json
import datetime
from decimal import Decimal # Added import for Decimal
from duckduckgo_search import DDGS # Uncommented DuckDuckGo import
# Import the librarian module
import memory.librarian as lib
from config import config
from ollie_print import olliePrint

# Change logging level to ERROR to reduce console clutter (handled by olliePrint)

# --- Utility Functions ---
def parse_target_date(target_date: str | None) -> datetime.datetime | None:
    """Parse ISO format date string to datetime object."""
    if not target_date:
        return None
    
    try:
        # Try full datetime format first (YYYY-MM-DDTHH:MM:SS)
        return datetime.datetime.fromisoformat(target_date)
    except ValueError:
        try:
            # Try date-only format (YYYY-MM-DD)
            return datetime.datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid target_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

def validate_memory_type(memory_type: str, field_name: str = "memory_type") -> None:
    """Validate memory type is one of the allowed values."""
    if memory_type not in ('Semantic', "Episodic", "Procedural"):
        raise ValueError(f"Invalid {field_name} '{memory_type}' provided to tool.")

def convert_datetime_for_json(obj: dict) -> dict:
    """Convert datetime and Decimal objects to JSON-compatible types."""
    for key, value in obj.items():
        if isinstance(value, datetime.datetime):
            obj[key] = value.isoformat()
        elif isinstance(value, Decimal):
            obj[key] = float(value)
    return obj

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
    olliePrint(f"Tool call: add_memory(label='{label}', type='{memory_type}', target_date='{target_date}')")
    try:
        validate_memory_type(memory_type)
        parsed_target_date = parse_target_date(target_date)
        
        node_id = lib.add_memory(label=label, text=text, memory_type=memory_type, parent_id=parent_id, target_date=parsed_target_date)
        return {"success": True, "nodeid": node_id, "message": f"Memory added with ID {node_id}."}
    except Exception as e:
        olliePrint(f"Error in tool_add_memory: {e}", level='error')
        return {"success": False, "error": str(e)}

# Wrapper for superseding memory
def tool_supersede_memory(old_nodeid: int, new_label: str, new_text: str, new_memory_type: str, target_date: str | None = None):
    """Wrapper to call librarian.supersede_memory."""
    olliePrint(f"Tool call: supersede_memory(old_nodeid={old_nodeid}, new_label='{new_label}', new_type='{new_memory_type}', target_date='{target_date}')")
    try:
         # Validate old_nodeid is an integer
         if not isinstance(old_nodeid, int):
            error_message = f"Invalid old_nodeid: '{old_nodeid}'. Must be an integer."
            olliePrint(error_message, level='error')
            return {"success": False, "error": error_message}

         validate_memory_type(new_memory_type, "new_memory_type")
         parsed_target_date = parse_target_date(target_date)
         
         new_nodeid = lib.supersede_memory(old_nodeid=old_nodeid, new_label=new_label, new_text=new_text, new_memory_type=new_memory_type, target_date=parsed_target_date)
         return {"success": True, "new_nodeid": new_nodeid, "message": f"Memory {old_nodeid} superseded by {new_nodeid}."}
    except Exception as e:
        olliePrint(f"Error in tool_supersede_memory: {e}", level='error')
        return {"success": False, "error": str(e)}

# Wrapper for searching memory
def tool_search_memory(query_text: str, memory_type: str | None = None, limit: int = None, future_events_only: bool = False, use_keyword_search: bool = False):
    """Wrapper to call librarian.search_memory."""
    olliePrint(f"Tool call: search_memory(query='{query_text}', type='{memory_type}', limit={limit}, future_events_only={future_events_only}, use_keyword_search={use_keyword_search})")
    try:
        # Validate memory_type if provided
        if memory_type is not None:
            validate_memory_type(memory_type, "memory_type filter")
        
        # Use config default if no limit specified
        search_limit = limit if limit is not None else config.MEMORY_SEARCH_LIMIT
        
        # Pass include_connections=True to get edge data in a single efficient query
        results = lib.search_memory(
            query_text=query_text, 
            memory_type=memory_type, 
            limit=search_limit, 
            future_events_only=future_events_only, 
            use_keyword_search=use_keyword_search,
            include_connections=True  # Always include connections in the tool interface
        )
        
        # Convert datetime and Decimal objects to JSON-compatible types
        for result in results:
            convert_datetime_for_json(result)
            
        return {"success": True, "results": results}
    except Exception as e:
        olliePrint(f"Error in tool_search_memory: {e}", level='error')
        return {"success": False, "error": str(e)}

def search_web_information(query_text: str) -> dict:
    """
    Searches the web for information using DuckDuckGo.
    This tool retrieves current information from the internet.
    It combines results from general web search and news search.

    Args:
        query_text (str): The text to search for.
        
    Returns:
        dict: A dictionary containing the search results or an error message.
    """
    olliePrint(f"Tool call: search_web_information(query_text='{query_text}')")
    try:
        ddgs = DDGS(timeout=config.WEB_SEARCH_TIMEOUT)
        
        text_results = []
        news_results = []
        
        # Perform text search
        try:
            text_search_results = ddgs.text(
                keywords=query_text,
                region="us-en",
                safesearch="off",
                max_results=config.WEB_SEARCH_MAX_RESULTS
            )
            if text_search_results:
                text_results = text_search_results
        except Exception as e_text:
            olliePrint(f"DuckDuckGo text search failed for '{query_text}': {e_text}", level='warning')

        # Perform news search
        try:
            news_search_results = ddgs.news(
                keywords=query_text,
                region="us-en",
                safesearch="off",
                max_results=config.WEB_SEARCH_NEWS_MAX_RESULTS
            )
            if news_search_results:
                news_results = news_search_results
        except Exception as e_news:
            olliePrint(f"DuckDuckGo news search failed for '{query_text}': {e_news}", level='warning')

        if not text_results and not news_results:
            return {"success": True, "results": "No information found from web search."}
        
        # Deduplicate results by URL
        seen_urls = set()
        unique_text_results = []
        unique_news_results = []
        
        for result in text_results:
            url = result.get('href', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_text_results.append(result)
                
        for result in news_results:
            url = result.get('href', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_news_results.append(result)

        combined_results = {
            "text_search_results": unique_text_results,
            "news_search_results": unique_news_results
        }

        # Create concise summary using local Ollama
        try:
            # Extract key info for prompt (titles and snippets only)
            text_summaries = [f"• {r.get('title', '')}: {r.get('body', '')[:100]}..." 
                            for r in unique_text_results[:3]]  # Limit to top 3
            news_summaries = [f"• {r.get('title', '')}: {r.get('body', '')[:100]}..." 
                            for r in unique_news_results[:2]]  # Limit to top 2
            
            prompt = f"""Query: {query_text}

Web Results:
{chr(10).join(text_summaries)}

News Results:
{chr(10).join(news_summaries)}

Provide a 2-3 sentence summary of the key findings relevant to the query."""

            summary_response = lib.call_ollama_generate(prompt, model="hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M")
            if summary_response and isinstance(summary_response, dict):
                combined_results["llm_summary"] = summary_response.get("summary", "Summary generation failed")
            else:
                combined_results["llm_summary"] = "Summary generation failed"
                
        except Exception as e_summary:
            olliePrint(f"Local LLM summarization failed for '{query_text}': {e_summary}", level='warning')
            # Return raw results if LLM fails
            combined_results["llm_summary"] = "Summarization failed - raw results provided"
        
        return {"success": True, "results": combined_results}

    except Exception as e:
        olliePrint(f"Error in search_web_information for query '{query_text}': {e}", level='error')
        return {"success": False, "error": str(e)}

# New tool to get a node by ID with its connections
def tool_get_node_by_id(nodeid: int):
    """Retrieves a specific node by ID with its connections."""
    olliePrint(f"Tool call: get_node_by_id(nodeid={nodeid})")
    try:
        # Use get_graph_data with depth=1 to get the node and its immediate connections
        graph_data = lib.get_graph_data(nodeid, depth=1)
        
        # Find the target node in the nodes list
        target_node = next((n for n in graph_data.get('nodes', []) if n['id'] == nodeid), None)
        
        if not target_node:
            return {"success": False, "error": f"Node with ID {nodeid} not found or is superseded."}
        
        # Process datetime objects for JSON compatibility
        convert_datetime_for_json(target_node)
        
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
        olliePrint(f"Error in tool_get_node_by_id: {e}", level='error')
        return {"success": False, "error": str(e)}

# New tool to get graph data
def tool_get_graph_data(center_nodeid: int, depth: int = 1):
    """Retrieves nodes and edges around a center node up to specified depth."""
    olliePrint(f"Tool call: get_graph_data(center_nodeid={center_nodeid}, depth={depth})")
    try:
        graph_data = lib.get_graph_data(center_nodeid, depth=depth)
        
        # Process datetime objects for JSON compatibility
        for node in graph_data.get('nodes', []):
            convert_datetime_for_json(node)
        
        return {"success": True, "result": graph_data}
    except Exception as e:
        olliePrint(f"Error in tool_get_graph_data: {e}", level='error')
        return {"success": False, "error": str(e)}

def tool_update_knowledge_graph_edges(limit_per_run: int = 5):
    """Wrapper to call librarian.process_pending_edges."""
    olliePrint(f"Tool call: update_knowledge_graph_edges(limit_per_run={limit_per_run})")
    try:
        if not isinstance(limit_per_run, int) or limit_per_run <= 0:
            raise ValueError("limit_per_run must be a positive integer.")
        
        summary = lib.process_pending_edges(limit_per_run=limit_per_run)
        # Ensure all parts of the summary are JSON serializable, though they should be.
        # datetime objects are handled by librarian if any were returned directly.
        return {"success": True, "result": summary}
    except Exception as e:
        olliePrint(f"Error in tool_update_knowledge_graph_edges: {e}", level='error')
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
            olliePrint("Tool call missing function name.", level='error')
            # Optionally add an error output for the LLM
            continue # Skip this invalid tool call

        if not tool_call_id:
             olliePrint(f"Tool call for '{function_name}' missing 'id'. Generating one.", level='warning')
             # Generate a fallback ID if needed, although Ollama should provide it
             tool_call_id = f"call_{time.time_ns()}"

        olliePrint(f"Handling tool call '{function_name}' with ID '{tool_call_id}' and args: {tool_args}")

        # Argument parsing now happens directly before calling the function from TOOL_FUNCTIONS
        try:
            # Ensure arguments are a dictionary for ** unpacking
            if not isinstance(tool_args, dict):
                try:
                    # Ollama now seems to pass args as a dict directly sometimes,
                    # but might still be stringified JSON in other cases? Be robust.
                    tool_args = json.loads(str(tool_args)) 
                except json.JSONDecodeError:
                     olliePrint(f"Tool arguments for '{function_name}' are not valid JSON or a dictionary: {tool_args}", level='error')
                     output_content = json.dumps({"success": False, "error": f"Invalid arguments format for {function_name}."})
                     tool_outputs.append({"tool_call_id": tool_call_id, "role": "tool", "content": output_content})
                     continue # Skip this tool call

            # Execute the tool (using the function from TOOL_FUNCTIONS directly)
            olliePrint(f"Executing tool '{function_name}' via handle_tool_calls with args: {tool_args}")
            tool_function = TOOL_FUNCTIONS.get(function_name)
            if tool_function:
                try:
                    # Call the actual tool function (e.g., tool_add_memory)
                    result_dict = tool_function(**tool_args) 
                    output_content = json.dumps(result_dict) # Result should already be a serializable dict
                except TypeError as e:
                    # Handles cases where LLM might provide wrong/missing args
                    olliePrint(f"TypeError executing {function_name} via handle_tool_calls: {e}. Args: {tool_args}", level='error')
                    output_content = json.dumps({"success": False, "error": f"Argument mismatch for tool '{function_name}': {e}"})
                except Exception as e:
                    olliePrint(f"Unexpected error executing {function_name} via handle_tool_calls: {e}", level='error')
                    output_content = json.dumps({"success": False, "error": f"Error during execution of {function_name}: {str(e)}"})
            else:
                olliePrint(f"Tool '{function_name}' not found in TOOL_FUNCTIONS.", level='error')
                output_content = json.dumps({"success": False, "error": f"Tool '{function_name}' not found"})
            
            # Append the result for this tool call
            tool_outputs.append({"tool_call_id": tool_call_id, "role": "tool", "content": output_content})

        except Exception as e: # Catch broader errors during the loop iteration
            olliePrint(f"General error processing tool call {tool_call_id} for {function_name}: {e}", level='error')
            # Provide a generic error response for this specific tool call
            error_content = json.dumps({"success": False, "error": f"Internal error processing tool call '{function_name}'."})
            tool_outputs.append({"tool_call_id": tool_call_id, "role": "tool", "content": error_content})

    return tool_outputs
