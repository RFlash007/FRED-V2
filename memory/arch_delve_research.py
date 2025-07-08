"""
A.R.C.H./D.E.L.V.E. Iterative Research System
Advanced research director/analyst conversation system for F.R.E.D.'s agenda processing
"""

import os
import json
import uuid
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import re

from config import config, ollama_manager
from ollie_print import olliePrint_simple

# Import memory tools for complete_research functionality
from Tools import tool_add_memory, tool_read_webpage, TOOL_FUNCTIONS

# Use centralized Ollama connection manager for efficient concurrent calls
def get_ollama_client():
    """Get Ollama client from centralized connection manager."""
    return ollama_manager.get_client()

def _log_synthesis_event(conversation_path: str, event_type: str, data: dict):
    """Helper to log synthesis events to the correct directory."""
    if not conversation_path:
        return
    try:
        events_log_path = Path(conversation_path) / "research_events.jsonl"
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        with open(events_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception as e:
        olliePrint_simple(f"Failed to log synthesis event: {e}", level='error')

class ArchDelveState:
    """State management for A.R.C.H./D.E.L.V.E. research conversations with thinking removal."""
    
    def __init__(self, task_id: str, original_task: str):
        self.task_id = task_id
        self.original_task = original_task
        self.conversation_history = []
        self.arch_context = []  # A.R.C.H.'s thinking context (last 5 messages)
        self.delve_context = []  # D.E.L.V.E.'s thinking context (last 5 messages)
        self.research_complete = False
        self.final_findings = ""
        self._lock = threading.Lock()
        
        # Tool call deduplication tracking
        self.executed_tool_calls: Set[str] = set()  # Track executed tool calls to prevent duplicates
        self.delve_iteration_count = 0  # Track D.E.L.V.E. iterations to prevent infinite loops
        
        # Create conversation storage directory
        self.conversation_dir = Path(config.ARCH_DELVE_CONVERSATION_STORAGE_PATH) / task_id
        self.conversation_dir.mkdir(parents=True, exist_ok=True)
        self.events_log_path = self.conversation_dir / "research_events.jsonl"
    
    def log_event(self, event_type: str, data: dict):
        """Logs a structured event to a JSONL file for the research task."""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "data": data
            }
            with open(self.events_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            olliePrint_simple(f"Failed to log research event: {e}", level='error')
    
    def get_tool_call_signature(self, tool_name: str, arguments: dict) -> str:
        """Generate a unique signature for a tool call to detect duplicates."""
        # Create a deterministic signature based on tool name and arguments
        import hashlib
        sig_data = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        return hashlib.md5(sig_data.encode()).hexdigest()
    
    def is_tool_call_duplicate(self, tool_name: str, arguments: dict) -> bool:
        """Check if this tool call has already been executed in this session."""
        signature = self.get_tool_call_signature(tool_name, arguments)
        return signature in self.executed_tool_calls
    
    def mark_tool_call_executed(self, tool_name: str, arguments: dict):
        """Mark a tool call as executed to prevent future duplicates."""
        signature = self.get_tool_call_signature(tool_name, arguments)
        self.executed_tool_calls.add(signature)

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
    
    def add_conversation_turn(self, role: str, content: str, model_type: str, thinking: str = ""):
        """Add a turn to conversation history with thinking context management."""
        with self._lock:
            # Create full turn record
            turn = {
                'role': role,
                'content': content,
                'model_type': model_type,  # 'arch' or 'delve'
                'thinking': thinking,
                'timestamp': datetime.now().isoformat()
            }
            
            self.conversation_history.append(turn)
            
            # Manage thinking context per model with limits to prevent context overflow
            if model_type == 'arch':
                self.arch_context.append(turn)
                # Keep only last 15 messages for A.R.C.H. to prevent context overflow
                if len(self.arch_context) > 15:
                    self.arch_context = self.arch_context[-15:]
            elif model_type == 'delve':
                self.delve_context.append(turn)
                # Keep only last 10 messages for D.E.L.V.E. to prevent context overflow
                if len(self.delve_context) > 10:
                    self.delve_context = self.delve_context[-10:]
    
    def get_context_for_model(self, model_type: str) -> List[Dict]:
        """Get conversation context for specific model with thinking limited to past 3 messages."""
        with self._lock:
            if model_type == 'arch':
                context = self.arch_context.copy()
            elif model_type == 'delve':
                context = self.delve_context.copy()
            else:
                return []
            
            # Prepare messages with thinking only from past 3 messages
            messages = []
            total_turns = len(context)
            
            for i, turn in enumerate(context):
                if turn['role'] == 'user':
                    messages.append({"role": "user", "content": turn['content']})
                elif turn['role'] == 'assistant':
                    # Only include thinking for the LAST 3 assistant messages
                    turns_from_end = total_turns - i
                    assistant_turns_from_end = sum(1 for t in context[i:] if t['role'] == 'assistant')
                    
                    if assistant_turns_from_end <= 3 and turn.get('thinking'):
                        # Recent message: include thinking
                        full_content = f"<think>{turn['thinking']}</think>\n{turn['content']}"
                        messages.append({"role": "assistant", "content": full_content})
                    else:
                        # Older message: only final content, no thinking
                        messages.append({"role": "assistant", "content": turn['content']})
            
            return messages
    
    def save_conversation_state(self):
        """Save current conversation state to storage."""
        try:
            # Save full conversation log
            conversation_file = self.conversation_dir / "full_conversation.json"
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'task_id': self.task_id,
                    'original_task': self.original_task,
                    'conversation_history': self.conversation_history,
                    'research_complete': self.research_complete,
                    'final_findings': self.final_findings,
                    'executed_tool_calls': list(self.executed_tool_calls),  # Save for debugging
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            # Save model contexts separately
            arch_context_file = self.conversation_dir / "arch_context.json"
            with open(arch_context_file, 'w', encoding='utf-8') as f:
                json.dump(self.arch_context, f, indent=2, ensure_ascii=False)
            
            delve_context_file = self.conversation_dir / "delve_context.json"
            with open(delve_context_file, 'w', encoding='utf-8') as f:
                json.dump(self.delve_context, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            olliePrint_simple(f"Failed to save A.R.C.H./D.E.L.V.E. conversation state: {e}", level='error')

def log_tool_result(tool_name: str, arguments: dict, result: dict):
    """Logs the output of a tool call in a readable format with content size limits."""
    olliePrint_simple("\n" + "="*25 + f" 📖 TOOL RESULT: {tool_name} " + "="*25)
    
    query = arguments.get('query', arguments.get('url', 'N/A'))
    olliePrint_simple(f"  ➡️  Input: {query}")

    if not isinstance(result, dict) or 'success' not in result:
        olliePrint_simple(f"  ❌ Status: Failed (Invalid result format)", level='warning')
        olliePrint_simple(f"  Raw Result: {result}")
        olliePrint_simple("="*70)
        return

    success = result.get('success', False)
    status = "✅ Success" if success else "❌ Failed"
    olliePrint_simple(f"  Status: {status}")

    if not success:
        olliePrint_simple(f"  Error: {result.get('error', 'Unknown error')}", level='warning')
    else:
        # Successful tool call
        if "results" in result and isinstance(result["results"], list):
            search_results = result["results"]
            olliePrint_simple(f"\n  📄 Found {len(search_results)} results:")
            olliePrint_simple("-" * 40)
            if not search_results:
                olliePrint_simple("   (No results returned)")
            else:
                for i, res in enumerate(search_results[:5]): # Limit print to first 5
                    olliePrint_simple(f"  {i+1}. {res.get('title', 'No Title')}")
                    olliePrint_simple(f"     URL: {res.get('url', 'N/A')}")
                    olliePrint_simple(f"     Snippet: {res.get('snippet', 'N/A')[:200]}...")
            olliePrint_simple("-" * 40)

        # Handle read_webpage - FIXED: Limit content output to prevent terminal spam
        elif tool_name == "read_webpage":
            content = result.get("content", "")
            content_length = len(content)
            olliePrint_simple(f"\n  📄 Page Content ({content_length} chars):")
            olliePrint_simple("-" * 40)
            
            if content.strip():
                # Only show first 500 characters to prevent terminal spam
                if content_length > 500:
                    truncated_content = content[:500] + f"\n... [TRUNCATED - {content_length - 500} more chars] ..."
                    olliePrint_simple(truncated_content)
                else:
                    olliePrint_simple(content)
            else:
                olliePrint_simple("   (No content extracted)")
            
            olliePrint_simple("-" * 40)
            olliePrint_simple(f"  🔗 Links Found on Page: {result.get('links_found', 0)}")

    olliePrint_simple("="*70)

# Global storage for active research sessions
active_research_sessions: Dict[str, ArchDelveState] = {}

def create_research_session(task_id: str, original_task: str) -> ArchDelveState:
    """Create a new A.R.C.H./D.E.L.V.E. research session."""
    session = ArchDelveState(task_id, original_task)
    active_research_sessions[task_id] = session
    
    olliePrint_simple(f"[A.R.C.H./D.E.L.V.E.] Research session created: {task_id}")
    olliePrint_simple(f"   Task: {original_task[:100]}...")
    
    return session

def prepare_arch_messages(session: ArchDelveState) -> List[Dict]:
    """Prepare messages for A.R.C.H. with system prompt and task injection."""
    current_time_iso = datetime.now().isoformat()
    current_date_readable = datetime.now().strftime("%B %d, %Y")
    olliePrint_simple(f"[SYSTEM TIME] {current_time_iso}")
    
    messages = [
        {
            "role": "system", 
            "content": config.ARCH_SYSTEM_PROMPT.format(
                original_task=session.original_task,
                current_date_time=current_time_iso,
                current_date=current_date_readable
            )
        }
    ]
    
    # Add conversation context
    context_messages = session.get_context_for_model('arch')
    messages.extend(context_messages)
    
    # Add task reinforcement if starting new session
    if len(context_messages) == 0:
        messages.append({
            "role": "user",
            "content": config.ARCH_TASK_PROMPT.format(original_task=session.original_task)
        })
    
    return messages

def prepare_delve_messages(session: ArchDelveState, arch_instruction: str) -> List[Dict]:
    """Prepare messages for D.E.L.V.E. with system prompt and director instruction."""
    current_time_iso = datetime.now().isoformat()
    current_date_readable = datetime.now().strftime("%B %d, %Y")
    messages = [
        {"role": "system", "content": config.DELVE_SYSTEM_PROMPT.format(
            current_date_time=current_time_iso,
            current_date=current_date_readable
        )}
    ]
    
    # Add conversation context (D.E.L.V.E.'s own context only)
    context_messages = session.get_context_for_model('delve')
    messages.extend(context_messages)
    
    # Add current instruction from A.R.C.H.
    messages.append({
        "role": "user",
        "content": arch_instruction
    })
    
    # Simplified debug output
    olliePrint_simple(f"\n[D.E.L.V.E. CONTEXT] {len(context_messages)} previous messages | Instruction: '{arch_instruction}'")
    
    return messages

def prepare_vet_messages(delve_output: str, arch_instruction: str) -> List[Dict]:
    """Prepare messages for V.E.T. with system prompt and D.E.L.V.E.'s raw data."""
    messages = [
        {"role": "system", "content": config.VET_SYSTEM_PROMPT},
        {"role": "user", "content": f"DIRECTIVE: {arch_instruction}\n\nRAW DATA:\n{delve_output}"}
    ]
    return messages

def run_arch_iteration(session: ArchDelveState, ollama_client=None) -> Tuple[str, bool]:
    """Run A.R.C.H. iteration and return (response, is_complete)."""
    try:
        messages = prepare_arch_messages(session)
        
        # A.R.C.H. tools (only complete_research)
        arch_tools = [
            {
                "name": "complete_research",
                "description": "Signal that the research is 100% complete and all objectives have been met.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        
        # Use centralized connection manager for concurrent-safe calls
        response = ollama_manager.chat_concurrent_safe(
            model=config.ARCH_MODEL,
            messages=messages,
            tools=arch_tools,
            stream=False,
            options=config.THINKING_MODE_OPTIONS
        )
        
        response_message = response.get('message', {})
        raw_content = response_message.get('content', '')
        tool_calls = response_message.get('tool_calls')
        
        # Extract thinking and get clean content for D.E.L.V.E.
        thinking = session.extract_think_content(raw_content)
        clean_content = session.strip_think_tags(raw_content)
        
        # Show A.R.C.H. thinking and instruction
        olliePrint_simple(f"\n[A.R.C.H. THINKING]:\n{thinking}")
        olliePrint_simple(f"\n[A.R.C.H. → D.E.L.V.E.]:\n{clean_content}")
        olliePrint_simple("-" * 70)
        
        # Check for completion tool call
        research_complete = False
        if tool_calls:
            for tool_call in tool_calls:
                olliePrint_simple(f"\n[A.R.C.H. TOOL CALL DEBUG]:")
                olliePrint_simple(f"  Full tool_call: {tool_call}")
                
                # Safely extract tool name
                tool_name = None
                try:
                    tool_name = tool_call.get('function', {}).get('name')
                    if not tool_name:
                        olliePrint_simple(f"  ERROR: No tool name found in tool_call")
                        continue
                except Exception as e:
                    olliePrint_simple(f"  ERROR: Failed to extract tool name: {e}")
                    continue
                
                if tool_name == 'complete_research':
                    research_complete = True
                    session.research_complete = True
                    olliePrint_simple(f"[A.R.C.H.] Research completion signaled!")
                    break
                else:
                    olliePrint_simple(f"  ERROR: Unknown tool name: {tool_name}")
                    olliePrint_simple(f"  Available tools: complete_research")
                    olliePrint_simple(f"  A.R.C.H. should only use complete_research tool!")
        
        # Handle empty A.R.C.H. responses
        if not clean_content.strip() and not research_complete:
            olliePrint_simple(f"[ERROR] A.R.C.H. provided no instruction and did not complete the task!")
            return "ERROR: No instruction provided", False
        
        # Store A.R.C.H.'s response
        session.add_conversation_turn('assistant', clean_content, 'arch', thinking)
        session.save_conversation_state()
        
        return clean_content, research_complete
        
    except Exception as e:
        olliePrint_simple(f"A.R.C.H. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        return f"A.R.C.H. system error: {str(e)}", False

def run_delve_iteration(session: ArchDelveState, arch_instruction: str, ollama_client=None) -> str:
    """Run D.E.L.V.E. iteration and return raw JSON data as a string."""
    try:
        messages = prepare_delve_messages(session, arch_instruction)
        
        # D.E.L.V.E. tools (specialized web search)
        delve_tools = [
            {
                "name": "search_general",
                "description": "General web search for broad topics, documentation, or official sources.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}
            },
            {
                "name": "search_news",
                "description": "Search for recent news articles and current events.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query for news"}}, "required": ["query"]}
            },
            {
                "name": "search_academic",
                "description": "Search for academic papers and research articles.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query for academic content"}}, "required": ["query"]}
            },
            {
                "name": "search_forums",
                "description": "Search forums and community discussion platforms like Reddit and Stack Overflow.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query for forum discussions"}}, "required": ["query"]}
            },
            {
                "name": "read_webpage",
                "description": "Extract full content from a specific webpage URL. Use after a search to read promising sources.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The complete URL of the webpage to read."}}, "required": ["url"]}
            }
        ]
        
        # Store A.R.C.H.'s instruction first
        session.add_conversation_turn('user', arch_instruction, 'delve')
        
        # DELVE can use tool iterations for thorough research
        raw_thinking = ""
        max_delve_iterations = 10  # Increased from 8 - allow DELVE more thinking iterations
        session.delve_iteration_count = 0
        
        # Continue until DELVE stops using tools and provides JSON output
        while session.delve_iteration_count < max_delve_iterations:
            session.delve_iteration_count += 1
            olliePrint_simple(f"\n[D.E.L.V.E. ITERATION {session.delve_iteration_count}/{max_delve_iterations}]")
            
            response = ollama_manager.chat_concurrent_safe(
                model=config.DELVE_MODEL,
                messages=messages,
                tools=delve_tools,
                stream=False,
                format="json",
                options=config.THINKING_MODE_OPTIONS
            )
            
            response_message = response.get('message', {})
            raw_content = response_message.get('content', '')
            tool_calls = response_message.get('tool_calls')
            
            # Extract thinking
            current_thinking = session.extract_think_content(raw_content)
            if current_thinking:
                raw_thinking = current_thinking
            
            clean_content = session.strip_think_tags(raw_content)
            
            # Show D.E.L.V.E.'s thinking and response
            if current_thinking:
                olliePrint_simple(f"\n[D.E.L.V.E. THINKING]:\n{current_thinking}")
            if clean_content and not tool_calls:
                olliePrint_simple(f"\n[D.E.L.V.E. RAW DATA OUTPUT]:\n{clean_content}")
            if tool_calls:
                olliePrint_simple(f"\n[D.E.L.V.E. TOOL CALLS] {len(tool_calls)} calls")
            
            if 'role' not in response_message:
                response_message['role'] = 'assistant'
            
            # Preserve thinking for next iteration if there are tools
            if tool_calls and current_thinking:
                response_message['content'] = raw_content
            
            messages.append(response_message)
            
            if tool_calls:
                # Execute web search tools with deduplication
                tool_results_for_next_iteration = []
                executed_this_round = 0

                for i, tool_call in enumerate(tool_calls):
                    olliePrint_simple(f"\n[TOOL CALL {i+1}]")
                    
                    tool_name = tool_call.get('function', {}).get('name')
                    if not tool_name:
                        olliePrint_simple(f"  ERROR: No tool name found")
                        tool_results_for_next_iteration.append({
                            "role": "tool", "tool_call_id": tool_call.get('id'),
                            "content": json.dumps({"success": False, "error": "Tool name missing."})
                        })
                        continue

                    # Parse arguments safely
                    try:
                        raw_arguments = tool_call.get('function', {}).get('arguments', {})
                        if isinstance(raw_arguments, str):
                            arguments = json.loads(raw_arguments)
                        elif isinstance(raw_arguments, dict):
                            arguments = raw_arguments
                        else:
                            raise ValueError(f"Unexpected arguments format: {type(raw_arguments)}")
                    except Exception as e:
                        olliePrint_simple(f"  ERROR: Failed to parse arguments for {tool_name}: {e}")
                        tool_results_for_next_iteration.append({
                            "role": "tool", "tool_call_id": tool_call.get('id'),
                            "content": json.dumps({"success": False, "error": f"Argument parsing failed: {e}"})
                        })
                        continue

                    # CRITICAL FIX: Check for duplicate tool calls
                    if session.is_tool_call_duplicate(tool_name, arguments):
                        olliePrint_simple(f"  SKIPPED: Duplicate tool call - {tool_name} with same args already executed")
                        # Return cached "already executed" response instead of re-executing
                        tool_results_for_next_iteration.append({
                            "role": "tool",
                            "content": json.dumps({"success": True, "note": "Duplicate call skipped - data already available in context"}),
                            "tool_call_id": tool_call.get('id', 'unknown')
                        })
                        continue

                    # Execute the tool call using the TOOL_FUNCTIONS registry
                    if tool_name in TOOL_FUNCTIONS:
                        try:
                            tool_function = TOOL_FUNCTIONS[tool_name]
                            olliePrint_simple(f"  Executing {tool_name} with args: {arguments}")
                            
                            result = tool_function(**arguments)
                            log_tool_result(tool_name, arguments, result)
                            
                            # Mark this tool call as executed to prevent future duplicates
                            session.mark_tool_call_executed(tool_name, arguments)
                            executed_this_round += 1
                            
                            # Log the tool call event
                            session.log_event('tool_call', {
                                'tool_name': tool_name,
                                'arguments': arguments,
                                'result': result
                            })
                            
                            tool_output_content = json.dumps(result)
                            
                        except Exception as e:
                            olliePrint_simple(f"  ERROR: Tool {tool_name} execution failed: {e}")
                            import traceback
                            traceback.print_exc()
                            tool_output_content = json.dumps({"success": False, "error": f"Tool execution failed **MOVE ON TO A DIFFERENT LINK**: {str(e)}"})
                    else:
                        olliePrint_simple(f"  ERROR: Unknown tool name: {tool_name}")
                        tool_output_content = json.dumps({"success": False, "error": f"Tool '{tool_name}' not found."})

                    tool_results_for_next_iteration.append({
                        "role": "tool",
                        "content": tool_output_content,
                        "tool_call_id": tool_call.get('id', 'unknown')
                    })

                # Only add tool results if we executed new tools
                if executed_this_round > 0:
                    messages.extend(tool_results_for_next_iteration)
                    olliePrint_simple(f"[D.E.L.V.E.] Executed {executed_this_round} new tools, skipped {len(tool_calls) - executed_this_round} duplicates")
                else:
                    olliePrint_simple(f"[D.E.L.V.E.] All {len(tool_calls)} tool calls were duplicates - forcing final response")
                    # Force D.E.L.V.E. to provide final output by not adding any tool results
                    break
            else:
                # No more tools, D.E.L.V.E. should provide its JSON output
                # The clean_content should be the raw JSON data
                if not clean_content:
                    olliePrint_simple("[D.E.L.V.E.] ERROR: Finished tool use but provided no final data output.", level='error')
                    return json.dumps([{"error": "D.E.L.V.E. provided no data."}])
                
                # We no longer strictly parse here. V.E.T. is responsible for handling malformed data.
                # This makes the pipeline more resilient to model failures.
                
                # Store D.E.L.V.E.'s raw output and thinking
                session.add_conversation_turn('assistant', clean_content, 'delve', raw_thinking.strip())
                session.save_conversation_state()
                
                return clean_content
        
        # If we exit the loop without a final response, D.E.L.V.E. hit max iterations
        olliePrint_simple(f"[D.E.L.V.E.] WARNING: Reached max iterations ({max_delve_iterations}) without completion", level='warning')
        
        # Force a final response attempt
        final_attempt_messages = [
            {"role": "system", "content": config.DELVE_SYSTEM_PROMPT.format(
                current_date_time=datetime.now().isoformat(),
                current_date=datetime.now().strftime("%B %d, %Y")
            )},
            {"role": "user", "content": f"URGENT: Provide your final JSON data summary based on all the research conducted so far. You have reached the iteration limit. Summarize what you found: {arch_instruction}"}
        ]
        
        try:
            final_response = ollama_manager.chat_concurrent_safe(
                model=config.DELVE_MODEL,
                messages=final_attempt_messages,
                stream=False,
                format="json",
                options=config.THINKING_MODE_OPTIONS
            )
            
            final_content = final_response.get('message', {}).get('content', '')
            final_clean = session.strip_think_tags(final_content)
            
            if final_clean:
                olliePrint_simple(f"[D.E.L.V.E.] Emergency final response generated")
                session.add_conversation_turn('assistant', final_clean, 'delve', "")
                session.save_conversation_state()
                return final_clean
            else:
                olliePrint_simple(f"[D.E.L.V.E.] Emergency response failed - returning error")
                return json.dumps([{"error": "D.E.L.V.E. exceeded iteration limit without providing data"}])
        except Exception as final_e:
            olliePrint_simple(f"[D.E.L.V.E.] Emergency response failed: {final_e}", level='error')
            return json.dumps([{"error": f"D.E.L.V.E. iteration limit exceeded, emergency response failed: {str(final_e)}"}])
        
    except Exception as e:
        olliePrint_simple(f"D.E.L.V.E. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        return json.dumps([{"error": f"D.E.L.V.E. system error: {str(e)}"}])

def run_vet_iteration(session: ArchDelveState, delve_output_str: str, arch_instruction: str, ollama_client=None) -> str:
    """Run V.E.T. iteration to analyze D.E.L.V.E.'s output and return a verified report."""
    try:
        olliePrint_simple("\n[V.E.T.] Analyzing raw data...")
        messages = prepare_vet_messages(delve_output_str, arch_instruction)

        response = ollama_manager.chat_concurrent_safe(
            model=config.SAGE_MODEL, # Using SAGE model for high-quality analysis
            messages=messages,
            stream=False,
            options=config.THINKING_MODE_OPTIONS
        )

        raw_verified_report = response.get('message', {}).get('content', '')
        
        # Strip any thinking tags to get the clean report for A.R.C.H.
        verified_report = session.strip_think_tags(raw_verified_report)
        
        if not verified_report.strip().startswith("VERIFIED REPORT"):
            olliePrint_simple("[V.E.T.] ERROR: Output did not follow the required report format.", level='error')
            # Fallback report
            verified_report = f"VERIFIED REPORT: Analysis Failed\nDIRECTIVE: {arch_instruction}\n\nFINDINGS:\nV.E.T. failed to produce a valid report.\n\nASSESSMENT:\nConfidence: Low"

        olliePrint_simple(f"\n[V.E.T. → A.R.C.H.]:\n{verified_report}")
        olliePrint_simple("-" * 70)

        # Log the verified report for record-keeping
        session.log_event('vet_report', {
            'instruction': arch_instruction,
            'delve_data': json.loads(delve_output_str),
            'verified_report': verified_report
        })
        
        return verified_report

    except Exception as e:
        olliePrint_simple(f"V.E.T. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        return f"VERIFIED REPORT: System Error\nDIRECTIVE: {arch_instruction}\n\nFINDINGS:\nV.E.T. system encountered an error: {e}\n\nASSESSMENT:\nConfidence: Low"

def conduct_iterative_research(task_id: str, original_task: str) -> Dict:
    """Main function to conduct iterative A.R.C.H./D.E.L.V.E./V.E.T. research."""
    session = None
    try:
        olliePrint_simple(f"[RESEARCH] Starting A.R.C.H./D.E.L.V.E./V.E.T. investigation...")
        olliePrint_simple(f"   Task ID: {task_id}")
        olliePrint_simple(f"   Objective: {original_task[:100]}...")
        
        # Create research session
        session = create_research_session(task_id, original_task)
        verified_reports_for_sage = []
        
        # Iterative research loop
        iteration_count = 0
        max_iterations = config.ARCH_DELVE_MAX_RESEARCH_ITERATIONS
        
        while iteration_count < max_iterations:
            iteration_count += 1
            olliePrint_simple(f"\n{'='*30} [RESEARCH ITERATION {iteration_count}/{max_iterations}] {'='*30}")
            
            # 1. A.R.C.H. provides direction
            arch_instruction, is_complete = run_arch_iteration(session)
            
            if is_complete:
                olliePrint_simple(f"[RESEARCH] A.R.C.H. declared research complete!")
                break
            
            # Manually append the reminder to A.R.C.H.'s instruction for D.E.L.V.E.
            arch_instruction_with_reminder = (
                f"{arch_instruction}\n\n"
                "REMINDER: Do not just search; **you must READ** the most promising sources too "
                "gather deep, comprehensive information. No stone unturned."
            )

            # 2. D.E.L.V.E. executes research and returns raw data
            delve_raw_data_str = run_delve_iteration(session, arch_instruction_with_reminder)

            # 3. V.E.T. analyzes the raw data and creates a verified report
            verified_report = run_vet_iteration(session, delve_raw_data_str, arch_instruction)
            verified_reports_for_sage.append(verified_report) # Collect for final synthesis

            # 4. Pass V.E.T.'s report to A.R.C.H.'s context for the next iteration
            session.add_conversation_turn('user', verified_report, 'arch')
        
        # After loop completion, finalize results
        final_result = {}
        if session.research_complete:
            olliePrint_simple(f"[SUCCESS] Research process completed in {iteration_count} iterations.")
            
            # 5. S.A.G.E. creates the final user-facing report
            final_report_str = synthesize_final_report(
                original_task=original_task,
                verified_reports=verified_reports_for_sage,
                conversation_path=str(session.conversation_dir)
            )
            session.final_findings = final_report_str
            
            final_result = {
                'success': True,
                'findings': session.final_findings,
                'reason': 'completed'
            }
        else:
            olliePrint_simple(f"[WARNING] Research reached max iterations without completion", level='warning')
            final_findings = f"Research inconclusive after {iteration_count} iterations. Task: {original_task}. Review logs for details."
            session.final_findings = final_findings
            final_result = {
                'success': False,
                'findings': final_findings,
                'reason': 'max_iterations_reached'
            }

        # Common result fields
        final_result.update({
            'task_id': task_id,
            'original_task': original_task,
            'conversation_path': str(session.conversation_dir),
            'iterations': iteration_count
        })
        
        # Save final report and findings text file
        try:
            # Save final summary
            summary_file = session.conversation_dir / "research_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            # Save readable findings
            findings_file = session.conversation_dir / "research_findings.txt"
            with open(findings_file, 'w', encoding='utf-8') as f:
                f.write(f"Research Task: {original_task}\n")
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Success: {final_result.get('success', False)}\n")
                f.write(f"Iterations: {iteration_count}\n")
                f.write("=" * 80 + "\n")
                f.write("FINAL RESEARCH FINDINGS:\n")
                f.write("=" * 80 + "\n")
                f.write(final_result.get('findings', 'No findings available'))
        except Exception as e:
            olliePrint_simple(f"Failed to save final research reports: {e}", level='error')
            
        return final_result

    except Exception as e:
        olliePrint_simple(f"Research system error: {e}", level='error')
        import traceback
        traceback.print_exc()
        # Return a consistent error structure
        error_result = {
            'success': False,
            'task_id': task_id,
            'findings': f"Research system encountered an error: {str(e)}",
            'conversation_path': str(session.conversation_dir) if session else None,
            'iterations': 0,
            'reason': 'system_error'
        }
        if session:
            try:
                summary_file = session.conversation_dir / "research_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)
            except Exception as save_e:
                olliePrint_simple(f"Additionally failed to save error summary: {save_e}", level='error')
        return error_result
    
    finally:
        # Clean up session
        if session and task_id in active_research_sessions:
            del active_research_sessions[task_id]

def synthesize_final_report(original_task: str, verified_reports: List[str], conversation_path: str) -> str:
    """Use S.A.G.E. to synthesize a final, user-facing report from all verified reports."""
    try:
        olliePrint_simple("\n[S.A.G.E.] Synthesizing final user-facing report...")
        
        # Combine all verified reports into a single string for the prompt
        all_reports_text = "\n\n---\n\n".join(verified_reports)
        
        report_prompt = config.SAGE_FINAL_REPORT_USER_PROMPT.format(
            original_task=original_task,
            verified_reports=all_reports_text
        )
        
        messages = [
            {"role": "system", "content": config.SAGE_FINAL_REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": report_prompt}
        ]
        
        response = ollama_manager.chat_concurrent_safe(
            model=config.SAGE_MODEL,
            messages=messages,
            stream=False
        )
        
        final_report = response.get('message', {}).get('content', '').strip()
        if not final_report:
            olliePrint_simple("[S.A.G.E.] No final report generated from model", level='error')
            return "Error: S.A.G.E. failed to generate the final report."
            
        olliePrint_simple("[S.A.G.E.] Final report synthesis complete.")
        _log_synthesis_event(conversation_path, 'sage_final_report', {'report': final_report})
        
        return final_report
        
    except Exception as e:
        olliePrint_simple(f"[S.A.G.E.] Final report synthesis failed: {e}", level='error')
        _log_synthesis_event(conversation_path, 'sage_final_report_error', {'error': str(e)})
        return f"Error during final report synthesis: {e}"

def synthesize_research_to_memory(research_result: Dict, original_task: str) -> str:
    """Convert final research report to L3 memory node using S.A.G.E. synthesis."""
    conversation_path = research_result.get('conversation_path')
    try:
        olliePrint_simple("[S.A.G.E.] Synthesizing research findings for L3 memory...")
        
        # Prepare S.A.G.E. synthesis prompt
        synthesis_prompt = config.SAGE_L3_MEMORY_USER_PROMPT.format(
            original_task=original_task,
            research_findings=research_result['findings']
        )
        
        # Call S.A.G.E. for synthesis using the modern chat client
        messages = [
            {"role": "system", "content": config.SAGE_L3_MEMORY_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        response = ollama_manager.chat_concurrent_safe(
            model=config.SAGE_MODEL,
            messages=messages,
            stream=False,
            format="json"
        )
        
        response_text = response.get('message', {}).get('content', '').strip()
        if not response_text:
            olliePrint_simple("[S.A.G.E.] No response from synthesis model", level='error')
            return ""
        
        try:
            synthesis_result = json.loads(response_text)
            _log_synthesis_event(conversation_path, 'synthesis_output', {
                'model': config.SAGE_MODEL,
                'output': synthesis_result
            })
        except json.JSONDecodeError as e:
            olliePrint_simple(f"[S.A.G.E.] Failed to parse synthesis JSON: {e}", level='error')
            _log_synthesis_event(conversation_path, 'synthesis_error', {
                'error': 'JSONDecodeError',
                'message': str(e),
                'raw_response': response_text
            })
            return ""
        
        # Extract synthesized components
        memory_type = synthesis_result.get('memory_type', 'Semantic')
        memory_label = synthesis_result.get('label', f"Research: {original_task[:80]}...")
        memory_text = synthesis_result.get('text', research_result['findings'])
        
        olliePrint_simple(f"[S.A.G.E.] Synthesis complete - Type: {memory_type}")
        
        # Create optimized memory node
        result = tool_add_memory(
            label=memory_label,
            text=memory_text,
            memory_type=memory_type
        )
        
        _log_synthesis_event(conversation_path, 'l3_memory_add_attempt', {
            'label': memory_label,
            'text_length': len(memory_text),
            'memory_type': memory_type,
            'result': result
        })
        
        # tool_add_memory returns a string, not a dict
        if result and "added with ID" in result:
            # Extract node ID from success message: "Memory 'label' added with ID 12345"
            try:
                node_id = result.split("ID ")[-1]
                olliePrint_simple(f"[S.A.G.E.] Optimized memory created: {node_id}")
                return node_id
            except:
                olliePrint_simple(f"[S.A.G.E.] Memory created but couldn't extract ID: {result}")
                return "created"
        else:
            olliePrint_simple(f"[S.A.G.E.] Failed to create memory node: {result}", level='error')
            return ""
            
    except Exception as e:
        olliePrint_simple(f"[S.A.G.E.] Synthesis system error: {e}", level='error')
        # Fallback to simple storage if S.A.G.E. fails
        olliePrint_simple("[S.A.G.E.] Falling back to direct storage...", level='warning')
        try:
            label = f"Research: {original_task[:100]}..."
            text = research_result['findings']
            result = tool_add_memory(
                label=label,
                text=text,
                memory_type="Semantic"
            )
            
            _log_synthesis_event(conversation_path, 'l3_memory_add_attempt_fallback', {
                'label': label,
                'text_length': len(text),
                'memory_type': "Semantic",
                'result': result
            })

            if result and "added with ID" in result:
                try:
                    return result.split("ID ")[-1]
                except:
                    return "created"
        except Exception as fallback_error:
            olliePrint_simple(f"[S.A.G.E.] Fallback also failed: {fallback_error}", level='error')
        return ""