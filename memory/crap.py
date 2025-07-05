import os
import ollama
import re
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional
from ollie_print import olliePrint_simple
from config import config
import memory.L2_memory as L2

# Import memory tools directly
from Tools import TOOL_FUNCTIONS, handle_tool_calls

# C.R.A.P.-specific tool schema (memory tools only)
CRAP_TOOLS = [
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
        "description": "Replaces a specific old memory node with new, corrected information. Use ONLY when you have a specific NodeID to replace.",
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
        "name": "search_general",
        "description": "General web search for broad topics, documentation, or official sources.",
        "parameters": {
            "type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]
        }
    },
    {
        "name": "search_news",
        "description": "Search for recent news articles and current events.",
        "parameters": {
            "type": "object", "properties": {"query": {"type": "string", "description": "Search query for news"}}, "required": ["query"]
        }
    },
    {
        "name": "search_academic",
        "description": "Search for academic papers and research articles.",
        "parameters": {
            "type": "object", "properties": {"query": {"type": "string", "description": "Search query for academic content"}}, "required": ["query"]
        }
    },
    {
        "name": "search_forums",
        "description": "Search forums and community discussion platforms like Reddit and Stack Overflow.",
        "parameters": {
            "type": "object", "properties": {"query": {"type": "string", "description": "Search query for forum discussions"}}, "required": ["query"]
        }
    },
    {
        "name": "read_webpage",
        "description": "Extract content from a specific URL. Use after a search to read promising sources.",
        "parameters": {
            "type": "object", "properties": {"url": {"type": "string", "description": "The complete URL of the webpage to read."}}, "required": ["url"]
        }
    }
]

# Use C.R.A.P. system prompt from config  
CRAP_SYSTEM_PROMPT = config.CRAP_SYSTEM_PROMPT
CRAP_USER_PROMPT = config.CRAP_USER_PROMPT

class CrapState:
    """State management for C.R.A.P. with aggressive truncation."""
    
    def __init__(self):
        self.conversation_history = []
        self._lock = threading.Lock()
        # Align with F.R.E.D.'s conversation message limit from config
        self.MAX_CONVERSATION_MESSAGES = config.CRAP_MAX_CONVERSATION_MESSAGES
        # Number of messages (user + assistant) for which full thinking is retained (3 full turns)
        self.MAX_MESSAGES_WITH_FULL_THINKING = 6
    
    def extract_think_content(self, text):
        """Extract thinking content from <think>...</think> tags."""
        if not text:
            return ""
        matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
        return '\n'.join(matches).strip()
    
    def strip_think_tags(self, text):
        """Remove <think>...</think> blocks from text."""
        if not text:
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    def prepare_crap_messages(self, user_message, conversation_history, ollama_client):
        """Prepare messages for C.R.A.P. with aggressive truncation but NO F.R.E.D. thinking context."""
        messages = [{"role": "system", "content": CRAP_SYSTEM_PROMPT}]
        
        # Get L2 context and inject into C.R.A.P. context
        l2_context = L2.query_l2_context(user_message)
        if l2_context:
            crap_database_section = f"""(C.R.A.P. MEMORY DATABASE)
{l2_context}
(END C.R.A.P. MEMORY DATABASE)"""
        else:
            crap_database_section = ""
        
        # Apply aggressive truncation to conversation history
        # C.R.A.P. only keeps a limited number of turns, and does NOT include F.R.E.D.'s thinking
        if len(conversation_history) > self.MAX_CONVERSATION_MESSAGES:
            olliePrint_simple(f"[C.R.A.P.] Context truncated: {len(conversation_history)} → {self.MAX_CONVERSATION_MESSAGES}")
            conversation_history = conversation_history[-self.MAX_CONVERSATION_MESSAGES:]
        
        # Add conversation history WITHOUT any thinking context from F.R.E.D.
        for turn in conversation_history:
            if turn['role'] == 'user':
                messages.append({"role": "user", "content": turn['content']})
            elif turn['role'] == 'assistant':
                # Only include F.R.E.D.'s final response content, NO thinking
                content = turn['content']
                messages.append({"role": "assistant", "content": content})
        
        # Check for pending notifications and include them
        notifications_text = ""
        try:
            import memory.agenda_system as agenda
            pending_notifications = agenda.get_pending_notifications()
            if pending_notifications:
                notification_lines = []
                notification_ids = []
                for notif in pending_notifications:
                    notification_lines.append(f"• {notif['message']}")
                    notification_ids.append(notif['notification_id'])
                
                notifications_text = f"""(PENDING NOTIFICATIONS)
{chr(10).join(notification_lines)}
(END PENDING NOTIFICATIONS)

"""
                # Mark as delivered
                agenda.mark_notifications_delivered(notification_ids)
        except Exception as e:
            olliePrint_simple(f"Failed to get pending notifications: {e}", level='error')
        
        # Add current user message with C.R.A.P. database context
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        formatted_input = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

{notifications_text}{crap_database_section}

Current analysis time: {current_time}"""
        
        messages.append({"role": "user", "content": formatted_input})
        return messages

# Global C.R.A.P. state
crap_state = CrapState()

def get_pending_tasks_alert():
    """Get pending tasks alert for system status."""
    try:
        import memory.L3_memory as L3
        import memory.agenda_system as agenda
        import duckdb
        
        alerts = []
        
        # Check L3 edge creation backlog
        with duckdb.connect(L3.DB_FILE) as con:
            count = con.execute("SELECT COUNT(*) FROM pending_edge_creation_tasks WHERE status = 'pending';").fetchone()
            if count and count[0] > 5:
                # Try background processing if backlog is building
                if count[0] > 15:
                    olliePrint_simple(f"[BACKGROUND] High edge backlog ({count[0]}), starting background processing...")
                    try:
                        import threading
                        threading.Thread(
                            target=L3.process_pending_edges,
                            args=(3,),  # Process 3 at a time in background
                            daemon=True
                        ).start()
                    except Exception as e:
                        olliePrint_simple(f"[BACKGROUND] Edge processing failed: {e}", level='error')
                
                alerts.append(f"{count[0]} memory nodes awaiting connection processing")
        
        # Check agenda system status
        agenda_summary = agenda.get_agenda_summary()
        pending_tasks = agenda_summary.get('tasks', {}).get('pending', 0)
        pending_notifications = agenda_summary.get('pending_notifications', 0)
        
        if pending_tasks > 0:
            alerts.append(f"{pending_tasks} research tasks in agenda")
        if pending_notifications > 0:
            alerts.append(f"{pending_notifications} notifications ready")
        
        return "Alert: " + ", ".join(alerts) + "." if alerts else ""
    except Exception:
        pass
    return ""

def run_crap_analysis(user_message, conversation_history, ollama_client):
    """Run C.R.A.P. memory analysis and return structured database content."""
    try:
        olliePrint_simple(f"[C.R.A.P.] Analyzing conversation ({len(conversation_history)} turns)...")
        
        # Prepare messages for C.R.A.P.
        messages = crap_state.prepare_crap_messages(user_message, conversation_history, ollama_client)
        
        # Add CRAP_USER_PROMPT as the first user message for continuous instruction
        messages.insert(1, {"role": "user", "content": CRAP_USER_PROMPT})
        
        # Run C.R.A.P. with tool access
        max_tool_iterations = config.CRAP_MAX_TOOL_ITERATIONS
        assistant_response = ""
        raw_thinking = ""
        total_tool_calls_made = 0
        
        for iteration in range(max_tool_iterations):
            response = ollama_client.chat(
                model="hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M",
                messages=messages,
                tools=CRAP_TOOLS,
                stream=False,
                options=config.THINKING_MODE_OPTIONS
            )
            
            response_message = response.get('message', {})
            raw_content = response_message.get('content', '')
            
            # Extract and store C.R.A.P.'s thinking
            current_thinking = crap_state.extract_think_content(raw_content)
            if current_thinking:
                # Display CRAP's thinking in terminal
                olliePrint_simple(f"\n{'='*60}")
                olliePrint_simple(f"[C.R.A.P. THINKING] Iteration {iteration + 1}:")
                olliePrint_simple(f"{'='*60}")
                olliePrint_simple(current_thinking)
                olliePrint_simple(f"{'='*60}\n")
                raw_thinking += current_thinking + "\n"
            
            clean_content = crap_state.strip_think_tags(raw_content)
            tool_calls = response_message.get('tool_calls')
            
            # Ensure assistant role and preserve thinking for next iteration
            if 'role' not in response_message:
                response_message['role'] = 'assistant'
            
            # If there are tool calls, preserve the thinking content for the next iteration
            if tool_calls and current_thinking:
                # Keep raw content with thinking tags for C.R.A.P.'s context continuity
                response_message['content'] = raw_content
            
            messages.append(response_message)
            
            if tool_calls:
                # Log tool calls for debugging
                olliePrint_simple(f"\n[C.R.A.P. TOOLS] Executing {len(tool_calls)} operations:")
                for tc in tool_calls:
                    tool_name = tc.get('function', {}).get('name', 'unknown')
                    tool_args = tc.get('function', {}).get('arguments', {})
                    olliePrint_simple(f"  → {tool_name}({tool_args})")
                
                # Execute tools
                tool_outputs = handle_tool_calls(tool_calls)
                total_tool_calls_made += len(tool_calls)
                
                # Log tool results
                if tool_outputs:
                    olliePrint_simple(f"[C.R.A.P. RESULTS] {len(tool_outputs)} tool results received\n")
                
                if tool_outputs:
                    messages.extend(tool_outputs)
                    
                    # Update user message with tool results
                    tool_results = []
                    for output in tool_outputs:
                        tool_results.append(f"Tool result: {output.get('content', '{}')}")
                    
                    enhanced_message = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(TOOL RESULTS)
{chr(10).join(tool_results)}
(END TOOL RESULTS)

Current analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
                    
                    # Update last user message
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get('role') == 'user':
                            messages[i]['content'] = enhanced_message
                            break
                else:
                    break
            else:
                # No tools, we have final response
                if clean_content:
                    assistant_response = clean_content
                break
        
        # Get final response if needed
        if not assistant_response:
            final_response = ollama_client.chat(
                model="hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M",
                messages=messages,
                stream=False,
                options=config.THINKING_MODE_OPTIONS
            )
            
            final_content = final_response.get('message', {}).get('content', '')
            final_thinking = crap_state.extract_think_content(final_content)
            if final_thinking:
                # Display CRAP's final thinking in terminal
                olliePrint_simple(f"\n{'='*60}")
                olliePrint_simple(f"[C.R.A.P. THINKING] Final response:")
                olliePrint_simple(f"{'='*60}")
                olliePrint_simple(final_thinking)
                olliePrint_simple(f"{'='*60}\n")
                raw_thinking += final_thinking + "\n"
            
            assistant_response = crap_state.strip_think_tags(final_content)
        
        # Check if C.R.A.P. used tools (anti-hallucination check)
        if total_tool_calls_made == 0:
            olliePrint_simple("⚠️  [C.R.A.P. ALERT] No tools used! Possible hallucination detected.")
            
            
        olliePrint_simple(f"[C.R.A.P.] Analysis complete: {total_tool_calls_made} tools used, {len(assistant_response)} chars output")
        
        # Log C.R.A.P.'s full output for debugging
        if assistant_response:
            olliePrint_simple(f"\n{'='*60}")
            olliePrint_simple(f"[C.R.A.P. OUTPUT] Full response:")
            olliePrint_simple(f"{'='*60}")
            olliePrint_simple(assistant_response)
            olliePrint_simple(f"{'='*60}\n")
        
        # Extract the (FRED DATABASE) section from C.R.A.P. response
        if "(FRED DATABASE)" in assistant_response and "(END FRED DATABASE)" in assistant_response:
            start = assistant_response.find("(FRED DATABASE)")
            end = assistant_response.find("(END FRED DATABASE)") + len("(END FRED DATABASE)")
            return assistant_response[start:end]
        else:
            # C.R.A.P. didn't format properly, create minimal fallback
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pending_alert = get_pending_tasks_alert()
            
            fallback = f"""(FRED DATABASE)
RELEVANT MEMORIES:
{assistant_response if assistant_response else "No specific memory context identified."}

RECENT CONTEXT:
{L2.query_l2_context(user_message) if L2.query_l2_context(user_message) else "No relevant recent context."}

SYSTEM STATUS:
{pending_alert if pending_alert else ""}The current time is: {current_time}
(END FRED DATABASE)"""
            
            return fallback
            
    except Exception as e:
        olliePrint_simple(f"[C.R.A.P.] Memory analysis failed: {e}", level='error')
        
        # Minimal fallback on complete failure
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""(FRED DATABASE)
RELEVANT MEMORIES:
Memory system temporarily unavailable.

RECENT CONTEXT:
Analysis system offline.

SYSTEM STATUS:
The current time is: {current_time}
(END FRED DATABASE)"""

def prepare_messages_with_memory_context(system_prompt, user_message, ollama_client, from_pi_glasses=False):
    """Prepare messages with C.R.A.P.-managed memory context for F.R.E.D."""
    try:
        # Import here to avoid circular dependency
        import app
        conversation_history = app.fred_state.get_conversation_history()
        
        # Run C.R.A.P. analysis to get database content
        database_content = run_crap_analysis(user_message, conversation_history, ollama_client)
        
        # Apply F.R.E.D.'s normal thinking context logic to conversation
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history with F.R.E.D.'s thinking context (normal F.R.E.D. limits)
        for i, turn in enumerate(conversation_history):
            age = len(conversation_history) - i
            
            if turn['role'] == 'user':
                messages.append({"role": "user", "content": turn['content']})
            elif turn['role'] == 'assistant':
                content = turn['content']
                thinking = turn.get('thinking', '')
                
                if age <= 3 and thinking:
                    # Recent messages: include full thinking
                    full_content = f"<think>\n{thinking}\n</think>\n{content}"
                    messages.append({"role": "assistant", "content": full_content})
                else:
                    # Oldest messages: no thinking context
                    messages.append({"role": "assistant", "content": content})
        
        # Add visual context if from Pi glasses
        if from_pi_glasses:
            # Import here to avoid circular dependency
            from vision_service import vision_service
            visual_context = vision_service.get_current_visual_context()
            enhanced_database = database_content.replace(
                "(FRED DATABASE)",
                f"(FRED DATABASE)\nCurrent Visual Context (Pi Glasses): {visual_context}\n"
            )
            database_content = enhanced_database
        
        # Format final user message with C.R.A.P.-provided database content
        formatted_input = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

{database_content}"""
        
        messages.append({"role": "user", "content": formatted_input})
        return messages
        
    except Exception as e:
        olliePrint_simple(f"[C.R.A.P.] Failed to prepare memory context: {e}", level='error')
        
        # Fallback to minimal context
        messages = [{"role": "system", "content": system_prompt}]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        fallback_input = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

(FRED DATABASE)
RELEVANT MEMORIES:
Memory system unavailable.

RECENT CONTEXT:
Context analysis unavailable.

SYSTEM STATUS:
The current time is: {current_time}
(END FRED DATABASE)"""
        
        messages.append({"role": "user", "content": fallback_input})
        return messages 