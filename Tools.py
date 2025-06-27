import time
import json
import datetime
from decimal import Decimal # Added import for Decimal
from duckduckgo_search import DDGS # Uncommented DuckDuckGo import
# Import the librarian module
import memory.librarian as lib
from config import config
from ollie_print import olliePrint
import uuid

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

# --- Tool Implementations ---
# Wrapper for adding memory
def tool_add_memory(label: str, text: str, memory_type: str, parent_id: int | None = None, target_date: str | None = None):
    """Adds a new memory node to the knowledge graph."""
    try:
        olliePrint(f"Adding memory: '{label}' ({memory_type})")
        validate_memory_type(memory_type)
        parsed_target_date = parse_target_date(target_date)
        
        node_id = lib.add_memory(label=label, text=text, memory_type=memory_type, parent_id=parent_id, target_date=parsed_target_date)
        return f"Memory '{label}' added with ID {node_id}"
    except Exception as e:
        olliePrint(f"Error adding memory: {e}", level='error')
        return f"Failed to add memory: {e}"

# Wrapper for superseding memory
def tool_supersede_memory(old_nodeid: int, new_label: str, new_text: str, new_memory_type: str, target_date: str | None = None):
    """Supersedes an existing memory with a new one."""
    try:
        olliePrint(f"Superseding memory {old_nodeid} with '{new_label}' ({new_memory_type})")
        validate_memory_type(new_memory_type, "new_memory_type")
        parsed_target_date = parse_target_date(target_date)
        
        new_id = lib.supersede_memory(old_nodeid=old_nodeid, new_label=new_label, new_text=new_text, new_memory_type=new_memory_type, target_date=parsed_target_date)
        return f"Memory superseded. New ID: {new_id}"
    except ValueError as e:
        error_message = f"Cannot supersede memory {old_nodeid}: {e}"
        olliePrint(error_message, level='error')
        return error_message
    except Exception as e:
        olliePrint(f"Error superseding memory: {e}", level='error')
        return f"Failed to supersede memory: {e}"

# Wrapper for searching memory
def tool_search_memory(query_text: str, memory_type: str | None = None, limit: int = None, future_events_only: bool = False, use_keyword_search: bool = False):
    """Searches the knowledge graph for relevant memories."""
    try:
        olliePrint(f"Searching memories: '{query_text}' (type: {memory_type}, limit: {limit})")
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
        olliePrint(f"Error searching memory: {e}", level='error')
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
    try:
        olliePrint(f"Searching web: '{query_text}'")
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
        olliePrint(f"Error in web search: {e}", level='error')
        return {"success": False, "error": str(e)}

# New tool to get a node by ID with its connections
def tool_get_node_by_id(nodeid: int):
    """Gets a specific memory node by its ID."""
    try:
        olliePrint(f"Getting memory node: {nodeid}")
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
        olliePrint(f"Error getting node: {e}", level='error')
        return {"success": False, "error": str(e)}

# New tool to get graph data
def tool_get_graph_data(center_nodeid: int, depth: int = 1):
    """Gets graph data for visualization."""
    try:
        olliePrint(f"Getting graph data (center: {center_nodeid}, depth: {depth})")
        graph_data = lib.get_graph_data(center_nodeid, depth=depth)
        
        # Process datetime objects for JSON compatibility
        for node in graph_data.get('nodes', []):
            convert_datetime_for_json(node)
        
        return {"success": True, "result": graph_data}
    except Exception as e:
        olliePrint(f"Error getting graph data: {e}", level='error')
        return {"nodes": [], "links": []}

def tool_enroll_person(name: str):
    """Enrolls a new person by capturing their face from the live camera feed."""
    try:
        olliePrint(f"Received request to enroll new person: {name}")
        # Local imports to prevent circular dependencies and access services
        import asyncio
        from vision_service import vision_service
        from persona_service import persona_service
        # This local import is critical to avoid a circular dependency loop at startup
        from webrtc_server import request_frame_from_client

        if not vision_service.pi_connected:
            return "Enrollment failed: No camera client is connected."

        async def enroll_async_wrapper():
            # Request fresh capture from Pi for enrollment
            from webrtc_server import send_capture_request_to_pi
            
            olliePrint(f"Requesting fresh image capture for enrollment of '{name}'")
            
            # For enrollment, we need to handle this differently since we need immediate response
            # This is a simplified approach - in production you might want a more sophisticated callback system
            success = await send_capture_request_to_pi()
            
            if success:
                # Wait for image to be processed by vision service and use current frame
                await asyncio.sleep(3)  # Give time for capture and processing
                
                # Get the latest processed image data from vision service
                if hasattr(vision_service, 'last_processed_image_array'):
                    frame_array = vision_service.last_processed_image_array
                    result = persona_service.enroll_person(frame_array, name)
                    return result
                else:
                    return "Enrollment failed: Could not get processed image from vision service."
            else:
                return "Enrollment failed: Could not request image capture from Pi."
        
        # Run the async enrollment function in a new event loop
        return asyncio.run(enroll_async_wrapper())

    except Exception as e:
        olliePrint(f"Error during enrollment tool execution: {e}", level='error')
        return f"An unexpected error occurred during enrollment: {str(e)}"

def tool_update_knowledge_graph_edges(limit_per_run: int = 3):
    """Updates edges in the knowledge graph."""
    try:
        olliePrint(f"Updating graph edges (limit: {limit_per_run})")
        if not isinstance(limit_per_run, int) or limit_per_run <= 0:
            raise ValueError("limit_per_run must be a positive integer.")
        
        summary = lib.process_pending_edges(limit_per_run=limit_per_run)
        # Ensure all parts of the summary are JSON serializable, though they should be.
        # datetime objects are handled by librarian if any were returned directly.
        return {"success": True, "result": summary}
    except Exception as e:
        olliePrint(f"Error updating edges: {e}", level='error')
        return "Failed to update edges"

# --- Tool Registry ---
TOOL_FUNCTIONS = {
    "add_memory": tool_add_memory,
    "supersede_memory": tool_supersede_memory,
    "search_memory": tool_search_memory,
    "search_web_information": search_web_information,
    "get_node_by_id": tool_get_node_by_id,
    "get_graph_data": tool_get_graph_data,
    "enroll_person": tool_enroll_person,
    "update_knowledge_graph_edges": tool_update_knowledge_graph_edges
}

# --- Tool Execution Logic ---
# Simplified execution logic, assuming args are already parsed dicts by handle_tool_calls
def handle_tool_calls(tool_calls):
    """Handles multiple tool calls and returns results."""
    if not tool_calls:
        return []
    
    results = []
    for tool_call in tool_calls:
        tool_call_id = tool_call.get('id', '')
        function_name = tool_call.get('function', {}).get('name', '')
        tool_args = tool_call.get('function', {}).get('arguments', {})
        
        if not function_name:
            olliePrint("Tool call missing function name", level='error')
            continue
            
        if not tool_call_id:
            olliePrint(f"Tool '{function_name}' missing ID - generating one", level='warning')
            tool_call_id = str(uuid.uuid4())
        
        olliePrint(f"Executing tool: {function_name}")
        
        try:
            # Convert arguments if they're a string
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    olliePrint(f"Invalid JSON arguments for '{function_name}': {tool_args}", level='error')
                    continue
            
            # Execute the tool function
            if function_name in TOOL_FUNCTIONS:
                function = TOOL_FUNCTIONS[function_name]
                content = function(**tool_args)
                results.append({
                    "tool_call_id": tool_call_id,
                    "content": content
                })
            else:
                olliePrint(f"Tool '{function_name}' not found", level='error')
                
        except TypeError as e:
            olliePrint(f"TypeError executing {function_name}: {e}", level='error')
        except Exception as e:
            olliePrint(f"Error executing {function_name}: {e}", level='error')
    
    return results
