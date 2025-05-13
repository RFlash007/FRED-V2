import os
import time
import logging
import json
import datetime
# from duckduckgo_search import DDGS  # Commented out DuckDuckGo import
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
        "description": "Searches the web for information relevant to a query text.",
        "parameters": {"query_text": {"type": "string", "description": "The text to search for relevant information."}}
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
        results = lib.search_memory(query_text=query_text, memory_type=memory_type, limit=limit, future_events_only=future_events_only, use_keyword_search=use_keyword_search)
        # Convert datetime objects to strings for JSON compatibility
        for result in results:
            if isinstance(result.get('created_at'), datetime.datetime):
                result['created_at'] = result['created_at'].isoformat()
            if isinstance(result.get('last_access'), datetime.datetime):
                result['last_access'] = result['last_access'].isoformat()
            if isinstance(result.get('target_date'), datetime.datetime):
                result['target_date'] = result['target_date'].isoformat()
        return {"success": True, "results": results}
    except Exception as e:
        logging.error(f"Error in tool_search_memory: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def search_web_information(topics: str) -> str:
    """
    Searches for EXTERNAL information from the web using DuckDuckGo.
    This tool retrieves current information from the internet - NOT from F.R.E.D.'s memory.
    
    1. Perform text & news searches on each topic (comma-separated).
    2. Summarize the combined results in the specified style.
    3. If summarization fails, return raw search results.
    
    Args:
        topics (str): The user's desired topics, comma-separated.
        mode (str): The summarization mode - "educational" or "news".
        
    Returns:
        str: A summary or raw results if summarization fails.
    """
    ddgs = DDGS()
    
    # Split topics by commas
    topics_list = [t.strip() for t in topics.split(",") if t.strip()]
    
    region = "us-en"
    safesearch = "off"
    
    # Gather results
    text_results = []
    news_results = []
    
    # Configure search parameters based on mode
    text_max_results = 2
    news_max_results = 1 if mode == "educational" else 2
    
    # For each topic, do text + news
    for topic in topics_list:
        text_results.extend(ddgs.text(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=text_max_results
        ))
        news_results.extend(ddgs.news(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=news_max_results
        ))
    
    # Configure prompt based on mode
    if mode == "educational":
        combined_prompt = (
            "You are an educational summarizer. Summarize both the search results and news "
            "in a concise but thorough manner, focusing on learning and clarity. "
            "If not enough info is provided, do your best to fill in context. "
            "Structure your response with 'Text Summary:' and 'News Summary:' sections. "
            "Return ONLY the summary.\n\n"
            f"Search Results:\n{text_results}\n\n"
            f"News Results:\n{news_results}"
        )
    else:  # news mode
        combined_prompt = (
            "You are a news summarizer. Summarize both the text results and news "
            "in a journalistic style, highlighting recent or important events. "
            "Structure your response with 'Text Summary:' and 'News Summary:' sections. "
            "Return ONLY the summary:\n\n"
            f"Text Results:\n{text_results}\n\n"
            f"News Results:\n{news_results}"
        )
    
    # Fallback if summarization fails
    bare_info = (
        f"[Text Results]\n{text_results}\n\n"
        f"[News Results]\n{news_results}"
    )
    
    try:
        # Single summarization call
        response = ddgs.chat(
            keywords=combined_prompt,
            model="gpt-4o-mini",
            timeout=60
        )
        return response
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return bare_info

# --- Tool Registry ---
TOOL_FUNCTIONS = {
    "add_memory": tool_add_memory,
    "supersede_memory": tool_supersede_memory,
    "search_memory": tool_search_memory,
    "search_web_information": search_web_information    
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

# --- Test Code ---
if __name__ == '__main__':
    print("Available Tools:")
    print(json.dumps(AVAILABLE_TOOLS, indent=2))

    print("\n--- Testing Tools ---")

    # Ensure DB is initialized for testing
    try:
        lib.init_db()
        print("Memory DB initialized for testing.")
    except Exception as e:
        print(f"DB init failed for testing: {e}")

    print("\nTesting add_memory:")
    add_args = {"label": "Test Memory", "text": "This is a test memory added via tool.", "memory_type": "Episodic"}
    add_result = handle_tool_calls({"tool_calls": [{"function": {"name": "add_memory", "arguments": add_args}, "id": "call_12345"}]})
    print(f"Result: {add_result}")
    added_node_id = None
    try:
        added_node_id = json.loads(add_result[0]['content']).get("nodeid")
        print(f"Added Node ID: {added_node_id}")
    except: pass

    print("\nTesting search_memory:")
    search_args = {"query_text": "test memory"}
    search_result = handle_tool_calls({"tool_calls": [{"function": {"name": "search_memory", "arguments": search_args}, "id": "call_12346"}]})
    print(f"Result: {search_result}")

    if added_node_id:
        print("\nTesting supersede_memory:")
        supersede_args = {
            "old_nodeid": added_node_id,
            "new_label": "Updated Test Memory",
            "new_text": "This memory has been superseded.",
            "new_memory_type": "Episodic"
        }
        supersede_result = handle_tool_calls({"tool_calls": [{"function": {"name": "supersede_memory", "arguments": supersede_args}, "id": "call_12347"}]})
        print(f"Result: {supersede_result}")

    print("\n--- Testing Finished ---")
