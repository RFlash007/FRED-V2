import os
import re
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional
from ollie_print import olliePrint_simple
from config import config, ollama_manager
import memory.L2_memory as L2

# Import memory tools directly
from Tools import TOOL_FUNCTIONS, handle_tool_calls

# C.R.A.P. tool schemas are now consolidated in config.py
# Import and use config.CRAP_TOOLS for all C.R.A.P. operations

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
    
    def prepare_crap_messages(self, user_message, conversation_history):
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
        
        # Notifications are now handled directly by F.R.E.D., not C.R.A.P.
        
        # Add current user message with C.R.A.P. database context
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        formatted_input = f"""(USER INPUT)
{user_message}
(END OF USER INPUT)

{crap_database_section}

Current analysis time: {current_time}"""
        
        messages.append({"role": "user", "content": formatted_input})
        return messages

# Global C.R.A.P. state
crap_state = CrapState()

# Alert handling removed - notifications now handled directly by F.R.E.D.

def run_crap_analysis(user_message, conversation_history):
    """Run C.R.A.P. memory analysis and return structured database content."""
    try:
        olliePrint_simple(f"[C.R.A.P.] Analyzing conversation ({len(conversation_history)} turns)...")
        
        # Prepare messages for C.R.A.P.
        # This no longer needs the ollama_client passed in
        messages = crap_state.prepare_crap_messages(user_message, conversation_history)
        
        # Add CRAP_USER_PROMPT as the first user message for continuous instruction
        messages.insert(1, {"role": "user", "content": CRAP_USER_PROMPT})
        
        # Run C.R.A.P. with tool access
        max_tool_iterations = config.CRAP_MAX_TOOL_ITERATIONS
        assistant_response = ""
        raw_thinking = ""
        total_tool_calls_made = 0
        
        for iteration in range(max_tool_iterations):
            # Use safe attribute access to prevent AttributeError during import timing issues
            # Define minimal memory tools if CRAP_TOOLS is not available
            try:
                crap_tools = config.CRAP_TOOLS
            except AttributeError:
                # Fallback: Define basic memory tools directly
                crap_tools = [
                    {
                        "name": "search_memory",
                        "description": "Search the knowledge graph for relevant memories using semantic similarity.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query_text": {"type": "string", "description": "The text to search for relevant memories."},
                                "memory_type": {"type": ["string", "null"], "description": "Optional filter by memory type."},
                                "limit": {"type": "integer", "description": "Maximum results to return.", "default": 10}
                            },
                            "required": ["query_text"]
                        }
                    },
                    {
                        "name": "add_memory",
                        "description": "Add new memory node to knowledge graph.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "A concise label for the memory."},
                                "text": {"type": "string", "description": "The full memory content."},
                                "memory_type": {"type": "string", "description": "Memory classification.", "enum": ["Semantic", "Episodic", "Procedural"]}
                            },
                            "required": ["label", "text", "memory_type"]
                        }
                    },
                    {
                        "name": "supersede_memory",
                        "description": "Replace existing memory with corrected information.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "old_nodeid": {"type": "integer", "description": "NodeID to replace."},
                                "new_label": {"type": "string", "description": "New memory label."},
                                "new_text": {"type": "string", "description": "New memory content."},
                                "new_memory_type": {"type": "string", "enum": ["Semantic", "Episodic", "Procedural"]}
                            },
                            "required": ["old_nodeid", "new_label", "new_text", "new_memory_type"]
                        }
                    }
                ]
                # Cache the fallback tools for future use
                config.CRAP_TOOLS = crap_tools
            
            # DEBUG: Show CRAP tools available
            olliePrint_simple(f"[DEBUG] CRAP Tools Available: {len(crap_tools)} tools")
            for tool in crap_tools:
                olliePrint_simple(f"  - {tool.get('name', 'Unknown')}")
            
            # DEBUG: Show messages being sent to CRAP
            olliePrint_simple(f"[DEBUG] Sending {len(messages)} messages to CRAP model: {config.CRAP_MODEL}")
            
            response = ollama_manager.chat_concurrent_safe(
                model=config.CRAP_MODEL,
                messages=messages,
                tools=crap_tools,
                stream=False,
                options=config.THINKING_MODE_OPTIONS
            )
            
            response_message = response.get('message', {})
            raw_content = response_message.get('content', '')
            tool_calls = response_message.get('tool_calls', [])
            
            # DEBUG: Comprehensive tool_calls debugging
            olliePrint_simple(f"[DEBUG] CRAP Response Length: {len(raw_content)} chars")
            olliePrint_simple(f"[DEBUG] tool_calls type: {type(tool_calls)}")
            olliePrint_simple(f"[DEBUG] tool_calls value: {tool_calls}")
            olliePrint_simple(f"[DEBUG] tool_calls is None: {tool_calls is None}")
            
            # Safe len() call with additional protection
            if tool_calls is not None:
                olliePrint_simple(f"[DEBUG] CRAP Tool Calls: {len(tool_calls)}")
            else:
                olliePrint_simple(f"[DEBUG] CRAP Tool Calls: 0 (tool_calls was None despite fallback)")
                tool_calls = []  # Force fallback
            if tool_calls:
                for i, call in enumerate(tool_calls):
                    func_name = call.get('function', {}).get('name', 'Unknown')
                    olliePrint_simple(f"  Tool Call {i+1}: {func_name}")
            else:
                olliePrint_simple("[DEBUG] No tool calls made by CRAP")
            
            # DEBUG: Show first 200 chars of response
            olliePrint_simple(f"[DEBUG] CRAP Response Preview: {raw_content[:200]}...")
            
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
            # tool_calls already safely assigned above with fallback
            
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
            final_response = ollama_manager.chat_concurrent_safe(
                model=config.CRAP_MODEL,
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
        
        # Extract the (MEMORY CONTEXT) section from C.R.A.P. response
        if "(MEMORY CONTEXT)" in assistant_response and "(END MEMORY CONTEXT)" in assistant_response:
            start = assistant_response.find("(MEMORY CONTEXT)")
            end = assistant_response.find("(END MEMORY CONTEXT)") + len("(END MEMORY CONTEXT)")
            return assistant_response[start:end]
        else:
            # C.R.A.P. didn't format properly, create minimal fallback
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            fallback = f"""(MEMORY CONTEXT)
RELEVANT MEMORIES:
{assistant_response if assistant_response else "No specific memory context identified."}

RECENT CONTEXT:
{L2.query_l2_context(user_message) if L2.query_l2_context(user_message) else "No relevant recent context."}

SYSTEM STATUS:
The current time is: {current_time}
(END MEMORY CONTEXT)"""
            
            return fallback
            
    except Exception as e:
        olliePrint_simple(f"[C.R.A.P.] Memory analysis failed: {e}", level='error')
        
        # Minimal fallback on complete failure
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""(MEMORY CONTEXT)
RELEVANT MEMORIES:
Memory system temporarily unavailable.

RECENT CONTEXT:
Analysis system offline.

SYSTEM STATUS:
The current time is: {current_time}
(END MEMORY CONTEXT)"""  