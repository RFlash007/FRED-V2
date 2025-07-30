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
import time

from tool_schemas import ARCH_TOOLS, PIPELINE_CONTROL_TOOLS, DELVE_TOOLS, RESEARCH_TOOLS
from config import config, ollama_manager
from ollie_print import olliePrint_simple

# MEMORY OPTIMIZATION: Use single model for all agents with context switching
# All agents (ARCH, DELVE, SAGE, VET) use the same model to prevent multiple model loads
# Model personality is defined by system prompts, not different model instances
# Import unified model from centralized config
CONSOLIDATED_MODEL = config.ARCH_OLLAMA_MODEL

# Use same response handling pattern as app.py (which works)

# Import memory tools for complete_research functionality
from Tools import tool_add_memory, tool_read_webpage, TOOL_FUNCTIONS

# CENTRALIZED CONNECTION MANAGEMENT:
# All Ollama calls use ollama_manager.chat_concurrent_safe() directly
# This ensures proper connection pooling and prevents connection exhaustion
# No local client instances should be created

# STANDARDIZED ERROR HANDLING:
# All functions should use consistent error response patterns to improve reliability

class ResearchResult:
    """Standardized result structure for research operations."""
    
    def __init__(self, success: bool, data=None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error
        }
    
    @classmethod
    def success_result(cls, data):
        """Create a successful result."""
        return cls(success=True, data=data)
    
    @classmethod
    def error_result(cls, error: str):
        """Create an error result."""
        return cls(success=False, error=error)

def create_standard_error_response(operation: str, error: str) -> dict:
    """Create a standardized error response for backward compatibility."""
    return {
        "success": False,
        "error": f"{operation}: {error}",
        "operation": operation
    }

# UNIFIED RESOURCE MANAGEMENT:
# Consolidates all cleanup mechanisms into a single, simple system

class ResourceManager:
    """Unified resource management system for the research pipeline."""
    
    def __init__(self):
        self.cleanup_callbacks = []
        self.active_resources = {
            'sessions': {},
            'connections': [],
            'temp_files': []
        }
    
    def register_cleanup(self, callback, priority=0):
        """Register a cleanup callback with priority (higher = runs first)."""
        self.cleanup_callbacks.append((priority, callback))
        self.cleanup_callbacks.sort(key=lambda x: x[0], reverse=True)
    
    def register_session(self, session_id: str, session_obj):
        """Register a research session for tracking."""
        self.active_resources['sessions'][session_id] = session_obj
    
    def unregister_session(self, session_id: str):
        """Unregister a research session."""
        self.active_resources['sessions'].pop(session_id, None)
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a specific session."""
        try:
            session = self.active_resources['sessions'].get(session_id)
            if session:
                # Safe cleanup of session data
                for attr in ['conversation_history', 'arch_context', 'delve_context', 'executed_tool_calls']:
                    try:
                        getattr(session, attr, []).clear()
                    except:
                        pass
                
                # Remove from tracking
                self.unregister_session(session_id)
                olliePrint_simple(f"[RESOURCE MANAGER] Session {session_id} cleaned up")
                return True
        except Exception as e:
            olliePrint_simple(f"[RESOURCE MANAGER] Error cleaning session {session_id}: {e}", level='error')
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old sessions and return count cleaned."""
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        sessions_to_clean = []
        for session_id in list(self.active_resources['sessions'].keys()):
            try:
                # Parse timestamp from session_id
                timestamp_str = session_id.split('_')[1]
                session_time = int(timestamp_str)
                if current_time - session_time > max_age_seconds:
                    sessions_to_clean.append(session_id)
            except:
                # If we can't parse timestamp, consider it old
                sessions_to_clean.append(session_id)
        
        for session_id in sessions_to_clean:
            if self.cleanup_session(session_id):
                cleaned_count += 1
        
        if cleaned_count > 0:
            olliePrint_simple(f"[RESOURCE MANAGER] Cleaned up {cleaned_count} old sessions")
        
        return cleaned_count
    
    def cleanup_connections(self):
        """Clean up database connections and other resources."""
        try:
            # Enhanced DuckDB cleanup
            import gc
            
            try:
                import duckdb
                # Clear various DuckDB state
                for attr in ['_thread_connections', '_connections']:
                    try:
                        if hasattr(duckdb, attr):
                            delattr(duckdb, attr)
                    except:
                        pass
                
                # Close default connection
                try:
                    conn = duckdb.connect()
                    conn.close()
                except:
                    pass
            except ImportError:
                pass
            
            # Efficient garbage collection
            collected = 0
            for _ in range(2):  # Reduced from 3 to 2 rounds
                round_collected = gc.collect()
                if round_collected == 0:
                    break
                collected += round_collected
            
            if collected > 0:
                olliePrint_simple(f"[RESOURCE MANAGER] GC freed {collected} objects")
                
        except Exception as e:
            olliePrint_simple(f"[RESOURCE MANAGER] Connection cleanup error: {e}", level='warning')
    
    def force_cleanup_all(self):
        """Force cleanup of all resources."""
        try:
            # Clean all sessions
            session_count = len(self.active_resources['sessions'])
            for session_id in list(self.active_resources['sessions'].keys()):
                self.cleanup_session(session_id)
            
            # Clean connections
            self.cleanup_connections()
            
            # Run all registered cleanup callbacks
            for priority, callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    olliePrint_simple(f"[RESOURCE MANAGER] Cleanup callback error: {e}", level='warning')
            
            olliePrint_simple(f"[RESOURCE MANAGER] Force cleanup complete - {session_count} sessions cleaned")
            
        except Exception as e:
            olliePrint_simple(f"[RESOURCE MANAGER] Force cleanup error: {e}", level='error')
    
    def get_resource_status(self) -> dict:
        """Get current resource usage status."""
        return {
            'active_sessions': len(self.active_resources['sessions']),
            'cleanup_callbacks': len(self.cleanup_callbacks),
            'session_ids': list(self.active_resources['sessions'].keys())
        }

# Global resource manager instance
resource_manager = ResourceManager()

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
        """Add a turn to conversation history with thinking context management - thread safe."""
        with self._lock:
            # Create full turn record
            turn = {
                'role': role,
                'content': content,
                'model_type': model_type,  # 'arch' or 'delve'
                'thinking': thinking,
                'timestamp': datetime.now().isoformat()
            }
            
            # Safely add to conversation history
            try:
                self.conversation_history.append(turn)
            except Exception as e:
                olliePrint_simple(f"[STATE MANAGEMENT] Error adding turn to conversation history: {e}", level='error')
                return
            
            # Manage thinking context per model with atomic operations to prevent race conditions
            if model_type == 'arch':
                try:
                    # Atomic append operation
                    self.arch_context.append(turn)
                    
                    # Safe trimming - create new list atomically instead of slice reassignment
                    if len(self.arch_context) > 15:
                        # Create trimmed copy atomically
                        trimmed_context = self.arch_context[-15:]
                        # Atomic reassignment
                        self.arch_context = trimmed_context
                        
                except Exception as e:
                    olliePrint_simple(f"[STATE MANAGEMENT] Error managing ARCH context: {e}", level='error')
                    
            elif model_type == 'delve':
                try:
                    # Atomic append operation
                    self.delve_context.append(turn)
                    
                    # Safe trimming - create new list atomically instead of slice reassignment
                    if len(self.delve_context) > 10:
                        # Create trimmed copy atomically
                        trimmed_context = self.delve_context[-10:]
                        # Atomic reassignment
                        self.delve_context = trimmed_context
                        
                except Exception as e:
                    olliePrint_simple(f"[STATE MANAGEMENT] Error managing DELVE context: {e}", level='error')
    
    def get_context_for_model(self, model_type: str) -> List[Dict]:
        """Get conversation context for specific model with thinking limited to past 3 messages - thread safe."""
        with self._lock:
            try:
                # Get context safely with error handling
                if model_type == 'arch':
                    # Create defensive copy to prevent concurrent modification issues
                    try:
                        context = list(self.arch_context)  # More defensive than .copy()
                    except Exception as e:
                        olliePrint_simple(f"[STATE MANAGEMENT] Error copying ARCH context: {e}", level='error')
                        return []
                elif model_type == 'delve':
                    # Create defensive copy to prevent concurrent modification issues
                    try:
                        context = list(self.delve_context)  # More defensive than .copy()
                    except Exception as e:
                        olliePrint_simple(f"[STATE MANAGEMENT] Error copying DELVE context: {e}", level='error')
                        return []
                else:
                    return []
                
                # Validate context before processing
                if not context or not isinstance(context, list):
                    return []
                
                # Prepare messages with thinking only from past 3 messages
                messages = []
                total_turns = len(context)
                
                for i, turn in enumerate(context):
                    try:
                        # Validate turn structure
                        if not isinstance(turn, dict) or 'role' not in turn or 'content' not in turn:
                            olliePrint_simple(f"[STATE MANAGEMENT] Invalid turn structure at index {i}, skipping", level='warning')
                            continue
                            
                        if turn['role'] == 'user':
                            messages.append({"role": "user", "content": turn['content']})
                        elif turn['role'] == 'assistant':
                            # Only include thinking for the LAST 3 assistant messages
                            turns_from_end = total_turns - i
                            assistant_turns_from_end = sum(1 for t in context[i:] if isinstance(t, dict) and t.get('role') == 'assistant')
                            
                            if assistant_turns_from_end <= 3 and turn.get('thinking'):
                                # Recent message: include thinking
                                thinking_content = turn.get('thinking', '')
                                main_content = turn.get('content', '')
                                full_content = f"<think>{thinking_content}</think>\n{main_content}"
                                messages.append({"role": "assistant", "content": full_content})
                            else:
                                # Older message: only final content, no thinking
                                messages.append({"role": "assistant", "content": turn.get('content', '')})
                    except Exception as e:
                        olliePrint_simple(f"[STATE MANAGEMENT] Error processing turn {i}: {e}", level='error')
                        continue
                
                return messages
                
            except Exception as e:
                olliePrint_simple(f"[STATE MANAGEMENT] Error in get_context_for_model: {e}", level='error')
                return []
    
    def save_conversation_state(self):
        """Save current conversation state to storage - thread safe."""
        with self._lock:
            try:
                # Create defensive copies of all state data to prevent race conditions during save
                try:
                    conversation_history_copy = list(self.conversation_history)
                    arch_context_copy = list(self.arch_context)
                    delve_context_copy = list(self.delve_context)
                    executed_tool_calls_copy = set(self.executed_tool_calls).copy()
                except Exception as e:
                    olliePrint_simple(f"[STATE MANAGEMENT] Error creating defensive copies for save: {e}", level='error')
                    return
                
                # Save full conversation log with defensive copy
                conversation_file = self.conversation_dir / "full_conversation.json"
                try:
                    with open(conversation_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'task_id': self.task_id,
                            'original_task': self.original_task,
                            'conversation_history': conversation_history_copy,
                            'research_complete': self.research_complete,
                            'final_findings': self.final_findings,
                            'executed_tool_calls': list(executed_tool_calls_copy),  # Save for debugging
                            'last_updated': datetime.now().isoformat()
                        }, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    olliePrint_simple(f"[STATE MANAGEMENT] Error saving full conversation: {e}", level='error')
                
                # Save model contexts separately with defensive copies
                try:
                    arch_context_file = self.conversation_dir / "arch_context.json"
                    with open(arch_context_file, 'w', encoding='utf-8') as f:
                        json.dump(arch_context_copy, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    olliePrint_simple(f"[STATE MANAGEMENT] Error saving ARCH context: {e}", level='error')
                
                try:
                    delve_context_file = self.conversation_dir / "delve_context.json"
                    with open(delve_context_file, 'w', encoding='utf-8') as f:
                        json.dump(delve_context_copy, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    olliePrint_simple(f"[STATE MANAGEMENT] Error saving DELVE context: {e}", level='error')
                
            except Exception as e:
                olliePrint_simple(f"[STATE MANAGEMENT] Failed to save A.R.C.H./D.E.L.V.E. conversation state: {e}", level='error')

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

# Global storage for active research sessions - DEPRECATED, use resource_manager instead
# Kept for backward compatibility during transition
active_research_sessions: Dict[str, ArchDelveState] = {}
_session_lock = threading.Lock()  # Thread safety for session management
_max_active_sessions = 10  # Prevent unlimited session accumulation

def cleanup_research_session(task_id: str) -> None:
    """Clean up a completed research session - uses unified ResourceManager."""
    resource_manager.cleanup_session(task_id)
    # Also remove from compatibility storage
    with _session_lock:
        if task_id in active_research_sessions:
            del active_research_sessions[task_id]

def cleanup_old_research_sessions(max_age_hours: int = 24) -> None:
    """Clean up old research sessions - uses unified ResourceManager."""
    resource_manager.cleanup_old_sessions(max_age_hours)
    # Also remove from compatibility storage
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    with _session_lock:
        sessions_to_remove = []
        for task_id in list(active_research_sessions.keys()):
            try:
                timestamp_str = task_id.split('_')[1]
                session_time = int(timestamp_str)
                if current_time - session_time > max_age_seconds:
                    sessions_to_remove.append(task_id)
            except:
                sessions_to_remove.append(task_id)
        
        for task_id in sessions_to_remove:
            active_research_sessions.pop(task_id, None)

def cleanup_duckdb_connections():
    """DuckDB connection cleanup - uses unified ResourceManager."""
    resource_manager.cleanup_connections()

def create_research_session(task_id: str, original_task: str) -> ArchDelveState:
    """Create a new A.R.C.H./D.E.L.V.E. research session using unified ResourceManager."""
    # Check if session already exists in ResourceManager
    existing_session = resource_manager.active_resources['sessions'].get(task_id)
    if existing_session:
        olliePrint_simple(f"[SESSION MANAGEMENT] Research session {task_id} already exists! Using existing session.")
        return existing_session
    
    # Clean up old sessions before creating new ones
    resource_manager.cleanup_old_sessions()
    
    # Enforce session limits using ResourceManager
    if len(resource_manager.active_resources['sessions']) >= _max_active_sessions:
        olliePrint_simple(f"[SESSION MANAGEMENT] At session limit ({_max_active_sessions}), cleaning up oldest sessions", level='warning')
        # Clean up sessions until under limit
        sessions_to_clean = len(resource_manager.active_resources['sessions']) - _max_active_sessions + 1
        session_items = list(resource_manager.active_resources['sessions'].items())
        
        # Sort by timestamp and remove oldest
        sorted_sessions = []
        for sid, session in session_items:
            try:
                timestamp_str = sid.split('_')[1]
                session_time = int(timestamp_str)
                sorted_sessions.append((session_time, sid))
            except:
                sorted_sessions.append((0, sid))  # Consider unparseable as very old
        
        sorted_sessions.sort()
        for i in range(min(sessions_to_clean, len(sorted_sessions))):
            old_session_id = sorted_sessions[i][1]
            resource_manager.cleanup_session(old_session_id)
    
    # Create new session
    olliePrint_simple(f"[SESSION MANAGEMENT] Creating new research session: {task_id}")
    try:
        session = ArchDelveState(task_id, original_task)
        
        # Register with ResourceManager
        resource_manager.register_session(task_id, session)
        
        # Also store in compatibility storage for older clients
        with _session_lock:
            active_research_sessions[task_id] = session
        
        olliePrint_simple(f"[SESSION MANAGEMENT] Research session created: {task_id}")
        olliePrint_simple(f"   Task: {original_task[:100]}...")
        olliePrint_simple(f"   Active sessions: {len(resource_manager.active_resources['sessions'])}/{_max_active_sessions}")
        
        return session
    except Exception as e:
        olliePrint_simple(f"[SESSION MANAGEMENT] Failed to create session {task_id}: {e}", level='error')
        raise

def force_cleanup_all_sessions() -> None:
    """Force cleanup of all active research sessions using unified ResourceManager."""
    resource_manager.force_cleanup_all()
    # Also clear compatibility storage
    with _session_lock:
        active_research_sessions.clear()

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
    
    return messages

def prepare_vet_messages(delve_output: str, arch_instruction: str) -> List[Dict]:
    """Prepare messages for V.E.T. with system prompt and D.E.L.V.E.'s raw data."""
    messages = [
        {"role": "system", "content": config.VET_SYSTEM_PROMPT},
        {"role": "user", "content": f"DIRECTIVE: {arch_instruction}\n\nRAW DATA:\n{delve_output}"}
    ]
    return messages

# DEPRECATED FUNCTION REMOVED - Use run_enhanced_arch_iteration() instead

# DEPRECATED FUNCTION REMOVED - Use run_fresh_delve_iteration() instead

# DEPRECATED FUNCTION REMOVED - Use run_fresh_vet_iteration() instead

# DEPRECATED FUNCTION REMOVED - Use conduct_enhanced_iterative_research() instead

# DEPRECATED FUNCTION REMOVED - Use synthesize_final_report_with_rag() instead

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
        
        # MEMORY OPTIMIZATION: Use consolidated model for all agents
        response = ollama_manager.chat_concurrent_safe(
            host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
            model=CONSOLIDATED_MODEL,
            messages=messages,
            stream=False,
            format="json"
        )
        
        # Use same response handling pattern as app.py
        response_message = response.get('message', {})
        response_text = response_message.get('content', '').strip()
        if not response_text:
            olliePrint_simple("[S.A.G.E.] No response from synthesis model", level='error')
            return ""
        
        try:
            synthesis_result = json.loads(response_text)
            _log_synthesis_event(conversation_path, 'synthesis_output', {
                'model': CONSOLIDATED_MODEL,  # Use consolidated model
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

# ========================= PHASE 4: QUALITY FRAMEWORK =========================

class QualityAssessmentFramework:
    """
    Phase 4: Multi-layer quality assessment framework for enhanced research pipeline.
    Provides comprehensive quality metrics across DELVE, VET, and SAGE layers.
    """
    
    def __init__(self):
        self.config = config.QUALITY_ASSESSMENT_CONFIG
        self.credibility_weights = self.config['credibility_weights']
        self.source_diversity_target = self.config['source_diversity_target']
        self.data_balance_target = self.config['data_balance_target']
        self.contradiction_indicators = self.config['contradiction_indicators']
    
    def assess_delve_quality(self, delve_data: dict) -> Dict:
        """DELVE layer quality assessment: source diversity, data balance, credibility distribution."""
        try:
            sources = delve_data if isinstance(delve_data, list) else [delve_data]
            
            # Check for standardized error format
            if (isinstance(delve_data, dict) and 
                not delve_data.get("success", True)):
                return self._create_quality_report('DELVE', 'LOW', f'DELVE error: {delve_data.get("error", "Unknown error")}')
            
            if not sources or (isinstance(sources[0], dict) and not sources[0].get("success", True)):
                return self._create_quality_report('DELVE', 'LOW', 'System error or no data')
            
            # Source diversity analysis
            source_types = [s.get('source_type', 'other') for s in sources if isinstance(s, dict)]
            unique_types = len(set(source_types))
            diversity_score = min(unique_types / 5.0, 1.0)  # Max 5 types: academic, news, gov, forum, blog
            
            # Credibility distribution 
            credibility_dist = {'high': 0, 'medium': 0, 'low': 0}
            for source in sources:
                if isinstance(source, dict):
                    cred = source.get('credibility', 'low')
                    credibility_dist[cred] = credibility_dist.get(cred, 0) + 1
            
            total_sources = sum(credibility_dist.values())
            if total_sources == 0:
                return self._create_quality_report('DELVE', 'LOW', 'No valid sources found')
            
            # Data type balance (quantitative vs qualitative)
            quant_count = 0
            qual_count = 0
            for source in sources:
                if isinstance(source, dict):
                    data_types = source.get('data_types', [])
                    if 'quantitative' in data_types:
                        quant_count += 1
                    if 'qualitative' in data_types:
                        qual_count += 1
            
            data_balance_score = 1.0 - abs((quant_count - qual_count) / max(quant_count + qual_count, 1))
            
            # Overall DELVE quality score
            high_cred_ratio = credibility_dist['high'] / total_sources
            overall_score = (diversity_score * 0.3 + high_cred_ratio * 0.4 + data_balance_score * 0.3)
            
            if overall_score >= 0.7:
                quality_level = 'HIGH'
            elif overall_score >= 0.4:
                quality_level = 'MEDIUM'
            else:
                quality_level = 'LOW'
            
            return {
                'layer': 'DELVE',
                'quality_level': quality_level,
                'overall_score': overall_score,
                'source_diversity_score': diversity_score,
                'credibility_distribution': credibility_dist,
                'data_balance_score': data_balance_score,
                'total_sources': total_sources,
                'metrics': {
                    'unique_source_types': unique_types,
                    'quantitative_sources': quant_count,
                    'qualitative_sources': qual_count
                }
            }
            
        except Exception as e:
            olliePrint_simple(f"DELVE quality assessment failed: {e}", level='error')
            return self._create_quality_report('DELVE', 'LOW', f'Assessment error: {e}')
    
    def assess_vet_quality(self, vet_report: str, delve_quality: Dict) -> Dict:
        """VET layer quality assessment: information preservation, organization clarity, quality flagging."""
        try:
            if not vet_report or 'VERIFIED REPORT' not in vet_report:
                return self._create_quality_report('VET', 'LOW', 'Invalid or missing VET report')
            
            # Information preservation check
            has_findings = 'VERIFIED FINDINGS:' in vet_report
            has_assessment = 'ASSESSMENT:' in vet_report
            has_sources = 'SOURCES:' in vet_report
            
            preservation_score = sum([has_findings, has_assessment, has_sources]) / 3.0
            
            # Organization clarity check
            quantitative_mentioned = any(term in vet_report.lower() for term in ['number', 'statistic', 'percent', 'data', 'metric'])
            qualitative_mentioned = any(term in vet_report.lower() for term in ['opinion', 'expert', 'analysis', 'context', 'trend'])
            
            organization_score = sum([quantitative_mentioned, qualitative_mentioned]) / 2.0
            
            # Quality flagging accuracy
            contradiction_flagged = any(indicator in vet_report.lower() for indicator in self.contradiction_indicators)
            confidence_stated = any(level in vet_report for level in ['High', 'Medium', 'Low'])
            
            flagging_score = sum([contradiction_flagged, confidence_stated]) / 2.0
            
            # Integration with DELVE quality
            delve_integration_bonus = 0.2 if delve_quality.get('quality_level') == 'HIGH' else 0.0
            
            overall_score = (preservation_score * 0.4 + organization_score * 0.3 + flagging_score * 0.3) + delve_integration_bonus
            overall_score = min(overall_score, 1.0)  # Cap at 1.0
            
            if overall_score >= 0.7:
                quality_level = 'HIGH'
            elif overall_score >= 0.4:
                quality_level = 'MEDIUM' 
            else:
                quality_level = 'LOW'
            
            return {
                'layer': 'VET',
                'quality_level': quality_level,
                'overall_score': overall_score,
                'preservation_score': preservation_score,
                'organization_score': organization_score,
                'flagging_score': flagging_score,
                'delve_integration_bonus': delve_integration_bonus,
                'metrics': {
                    'has_findings': has_findings,
                    'has_assessment': has_assessment,
                    'has_sources': has_sources,
                    'quantitative_mentioned': quantitative_mentioned,
                    'qualitative_mentioned': qualitative_mentioned,
                    'contradiction_flagged': contradiction_flagged,
                    'confidence_stated': confidence_stated
                }
            }
            
        except Exception as e:
            olliePrint_simple(f"VET quality assessment failed: {e}", level='error')
            return self._create_quality_report('VET', 'LOW', f'Assessment error: {e}')
    
    def assess_sage_quality(self, truth_analysis: Dict, vet_reports: List[Dict]) -> Dict:
        """SAGE layer quality assessment: consensus reliability, contradiction resolution, source traceability."""
        try:
            if not truth_analysis or not vet_reports:
                return self._create_quality_report('SAGE', 'LOW', 'Missing truth analysis or VET reports')
            
            # Consensus reliability
            consensus_score = truth_analysis.get('consensus_score', 0)
            high_confidence_count = truth_analysis.get('high_confidence_count', 0)
            total_reports = len(vet_reports)
            
            reliability_score = consensus_score
            if total_reports > 0:
                high_confidence_ratio = high_confidence_count / total_reports
                reliability_score = (consensus_score * 0.7 + high_confidence_ratio * 0.3)
            
            # Contradiction resolution
            contradiction_count = truth_analysis.get('contradiction_count', 0)
            max_contradictions = self.config.get('contradiction_tolerance', 2)
            resolution_score = max(0, (max_contradictions - contradiction_count) / max_contradictions)
            
            # Source traceability (check if analysis references diverse sources)
            total_reports_analyzed = truth_analysis.get('total_reports_analyzed', 0)
            traceability_score = min(total_reports_analyzed / max(total_reports, 1), 1.0)
            
            # Overall SAGE quality score
            overall_score = (reliability_score * 0.5 + resolution_score * 0.3 + traceability_score * 0.2)
            
            if overall_score >= 0.7:
                quality_level = 'HIGH'
            elif overall_score >= 0.4:
                quality_level = 'MEDIUM'
            else:
                quality_level = 'LOW'
            
            return {
                'layer': 'SAGE',
                'quality_level': quality_level,
                'overall_score': overall_score,
                'reliability_score': reliability_score,
                'resolution_score': resolution_score,
                'traceability_score': traceability_score,
                'metrics': {
                    'consensus_score': consensus_score,
                    'high_confidence_count': high_confidence_count,
                    'contradiction_count': contradiction_count,
                    'total_reports_analyzed': total_reports_analyzed,
                    'max_contradictions_tolerated': max_contradictions
                }
            }
            
        except Exception as e:
            olliePrint_simple(f"SAGE quality assessment failed: {e}", level='error')
            return self._create_quality_report('SAGE', 'LOW', f'Assessment error: {e}')
    
    def calculate_overall_confidence(self, delve_quality: Dict, vet_quality: Dict, sage_quality: Dict) -> Dict:
        """
        Calculate overall research confidence based on multi-layer quality assessment.
        
        Confidence Levels:
        - HIGH (70-100%): Multiple layers show high quality, minimal contradictions
        - MEDIUM (40-69%): Mixed quality with some issues but generally reliable
        - LOW (0-39%): Significant quality issues across multiple layers
        """
        try:
            # Weight each layer's contribution
            layer_weights = {'DELVE': 0.3, 'VET': 0.3, 'SAGE': 0.4}  # SAGE weighted more heavily
            
            # Calculate weighted average
            weighted_score = (
                delve_quality.get('overall_score', 0) * layer_weights['DELVE'] +
                vet_quality.get('overall_score', 0) * layer_weights['VET'] +
                sage_quality.get('overall_score', 0) * layer_weights['SAGE']
            )
            
            # Penalty for low-quality layers
            low_quality_layers = sum(1 for q in [delve_quality, vet_quality, sage_quality] 
                                   if q.get('quality_level') == 'LOW')
            
            if low_quality_layers >= 2:
                weighted_score *= 0.7  # 30% penalty for multiple low-quality layers
            elif low_quality_layers == 1:
                weighted_score *= 0.85  # 15% penalty for one low-quality layer
            
            # Determine final confidence level
            if weighted_score >= 0.7:
                confidence_level = 'HIGH'
                confidence_percentage = int(weighted_score * 100)
            elif weighted_score >= 0.4:
                confidence_level = 'MEDIUM'
                confidence_percentage = int(weighted_score * 100)
            else:
                confidence_level = 'LOW'
                confidence_percentage = int(weighted_score * 100)
            
            return {
                'overall_confidence': confidence_level,
                'confidence_percentage': confidence_percentage,
                'weighted_score': weighted_score,
                'layer_quality_summary': {
                    'delve': delve_quality.get('quality_level', 'UNKNOWN'),
                    'vet': vet_quality.get('quality_level', 'UNKNOWN'),
                    'sage': sage_quality.get('quality_level', 'UNKNOWN')
                },
                'quality_penalties': {
                    'low_quality_layers': low_quality_layers,
                    'penalty_applied': 0.7 if low_quality_layers >= 2 else (0.85 if low_quality_layers == 1 else 1.0)
                }
            }
            
        except Exception as e:
            olliePrint_simple(f"Overall confidence calculation failed: {e}", level='error')
            return {
                'overall_confidence': 'LOW',
                'confidence_percentage': 0,
                'weighted_score': 0,
                'error': str(e)
            }
    
    def _create_quality_report(self, layer: str, quality_level: str, reason: str) -> Dict:
        """Create a standard quality report for error cases."""
        return {
            'layer': layer,
            'quality_level': quality_level,
            'overall_score': 0.0,
            'error': reason,
            'metrics': {}
        }

def enhanced_conduct_iterative_research_with_quality(task_id: str, original_task: str, enable_streaming: bool = False) -> Dict:
    """
    Enhanced research with comprehensive Phase 4 quality assessment framework.
    This is the complete implementation with all phases integrated.
    """
    session = None
    quality_framework = QualityAssessmentFramework()
    
    try:
        olliePrint_simple(f"[ENHANCED RESEARCH WITH QUALITY] Starting comprehensive research pipeline...")
        olliePrint_simple(f"   Task ID: {task_id}")
        olliePrint_simple(f"   Objective: {original_task[:100]}...")
        
        # MEMORY OPTIMIZATION: Preload model to prevent unloading during research
        ollama_manager.preload_model(CONSOLIDATED_MODEL)
        
        # Create research session - only ARCH maintains context
        session = create_research_session(task_id, original_task)
        vet_reports_for_rag = []  # Store VET reports for SAGE RAG database
        quality_assessments = []  # Store quality assessments for each iteration
        
        # Global citation tracking across all DELVE sessions
        global_citation_db = {}
        citation_counter = 1
        
        # Enhanced research loop with quality assessment
        iteration_count = 0
        max_iterations = config.ENHANCED_RESEARCH_CONFIG['max_iterations']
        
        while iteration_count < max_iterations:
            iteration_count += 1
            olliePrint_simple(f"\n{'='*50} [ENHANCED RESEARCH WITH QUALITY - ITERATION {iteration_count}/{max_iterations}] {'='*50}")
            
            # 1. A.R.C.H. provides strategic direction (maintains context of VET summaries)
            arch_instruction, is_complete = run_enhanced_arch_iteration(session, enable_streaming)
            
            # Check for ARCH errors
            if isinstance(arch_instruction, dict) and not arch_instruction.get("success", True):
                olliePrint_simple(f"[ENHANCED RESEARCH] A.R.C.H. iteration {iteration_count} failed: {arch_instruction.get('error', 'Unknown error')}", level='error')
                break
            
            if is_complete:
                olliePrint_simple(f"[ENHANCED RESEARCH] A.R.C.H. declared research complete!")
                break
            
            # 2. Fresh DELVE instance executes research with enhanced data gathering
            delve_enhanced_data = run_fresh_delve_iteration(
                arch_instruction, 
                task_id, 
                iteration_count,
                global_citation_db,
                citation_counter,
                enable_streaming
            )
            
            # CRITICAL: Validate DELVE results before proceeding
            if isinstance(delve_enhanced_data, dict) and not delve_enhanced_data.get("success", True):
                olliePrint_simple(f"[ENHANCED RESEARCH] DELVE iteration {iteration_count} failed: {delve_enhanced_data.get('error', 'Unknown error')}", level='error')
                olliePrint_simple(f"[ENHANCED RESEARCH] Skipping iteration and continuing...", level='warning')
                continue
            
            # Phase 4: DELVE Quality Assessment
            delve_quality = quality_framework.assess_delve_quality(delve_enhanced_data)
            olliePrint_simple(f"[QUALITY ASSESSMENT] DELVE Layer: {delve_quality['quality_level']} ({delve_quality['overall_score']:.2f})")
            
            # Update global citations from DELVE results
            citation_counter = update_global_citations(delve_enhanced_data, global_citation_db, citation_counter)
            
            # 3. Fresh VET instance formats data into strategic summary  
            vet_summary = run_fresh_vet_iteration(
                delve_enhanced_data, 
                arch_instruction,
                task_id,
                iteration_count,
                global_citation_db,
                enable_streaming
            )
            
            # Phase 4: VET Quality Assessment
            vet_quality = quality_framework.assess_vet_quality(vet_summary, delve_quality)
            olliePrint_simple(f"[QUALITY ASSESSMENT] VET Layer: {vet_quality['quality_level']} ({vet_quality['overall_score']:.2f})")
            
            # Store VET report for SAGE RAG database with quality metrics
            vet_reports_for_rag.append({
                'iteration': iteration_count,
                'instruction': arch_instruction,
                'summary': vet_summary,
                'timestamp': datetime.now().isoformat(),
                'delve_quality': delve_quality,
                'vet_quality': vet_quality
            })
            
            # Store quality assessment for this iteration
            quality_assessments.append({
                'iteration': iteration_count,
                'delve_quality': delve_quality,
                'vet_quality': vet_quality
            })
            
            # 4. Pass VET summary to A.R.C.H.'s context (only ARCH maintains context)
            session.add_conversation_turn('user', vet_summary, 'arch')
            
            # 5. Save state and cleanup instances (hardware efficiency)
            session.save_conversation_state()
            olliePrint_simple(f"[ENHANCED RESEARCH] Iteration {iteration_count} complete with quality assessment.")
        
        # After loop completion, use SAGE with RAG database and quality assessment
        final_result = {}
        if session.research_complete:
            olliePrint_simple(f"[SUCCESS] Enhanced research with quality assessment completed in {iteration_count} iterations.")
            
            # MEMORY OPTIMIZATION: Use RAG database context manager for proper cleanup
            with RAGDatabase(vet_reports_for_rag, task_id) as rag_database:
                # Apply enhanced truth determination with quality metrics
                all_vet_reports = [{'summary': r['summary'], 'iteration': r['iteration']} for r in vet_reports_for_rag]
                query_tools = SAGEQueryTools(rag_database)
                truth_analysis = determine_truth_consensus_advanced(all_vet_reports, query_tools)
                
                # Phase 4: SAGE Quality Assessment
                sage_quality = quality_framework.assess_sage_quality(truth_analysis, vet_reports_for_rag)
                olliePrint_simple(f"[QUALITY ASSESSMENT] SAGE Layer: {sage_quality['quality_level']} ({sage_quality['overall_score']:.2f})")
                
                # Calculate overall confidence across all layers
                avg_delve_quality = {
                    'overall_score': sum(qa['delve_quality']['overall_score'] for qa in quality_assessments) / len(quality_assessments),
                    'quality_level': 'HIGH' if sum(qa['delve_quality']['overall_score'] for qa in quality_assessments) / len(quality_assessments) >= 0.7 else 'MEDIUM'
                }
                avg_vet_quality = {
                    'overall_score': sum(qa['vet_quality']['overall_score'] for qa in quality_assessments) / len(quality_assessments),
                    'quality_level': 'HIGH' if sum(qa['vet_quality']['overall_score'] for qa in quality_assessments) / len(quality_assessments) >= 0.7 else 'MEDIUM'
                }
                
                overall_confidence = quality_framework.calculate_overall_confidence(avg_delve_quality, avg_vet_quality, sage_quality)
                olliePrint_simple(f"[OVERALL CONFIDENCE] {overall_confidence['overall_confidence']} ({overall_confidence['confidence_percentage']}%)")
                
                # SAGE creates final report using RAG queries with quality assessment
                final_report_str = synthesize_final_report_with_rag(
                    original_task=original_task,
                    rag_database=rag_database,
                    global_citations=global_citation_db,
                    conversation_path=str(session.conversation_dir)
                )
                session.final_findings = final_report_str
            # RAG database automatically closed here by context manager
            
            final_result = {
                'success': True,
                'findings': session.final_findings,
                'reason': 'completed',
                'enhancement': 'sequential_fresh_context_with_quality_assessment',
                'quality_assessment': {
                    'overall_confidence': overall_confidence,
                    'sage_quality': sage_quality,
                    'average_delve_quality': avg_delve_quality,
                    'average_vet_quality': avg_vet_quality,
                    'iteration_quality_assessments': quality_assessments
                }
            }
        else:
            olliePrint_simple(f"[WARNING] Enhanced research reached max iterations", level='warning')
            final_findings = f"Enhanced research inconclusive after {iteration_count} iterations. Task: {original_task}."
            session.final_findings = final_findings
            final_result = {
                'success': False,
                'findings': final_findings,
                'reason': 'max_iterations_reached',
                'enhancement': 'sequential_fresh_context_with_quality_assessment'
            }

        # Store final results with enhanced metadata and quality assessment
        final_result.update({
            'task_id': task_id,
            'original_task': original_task,
            'conversation_path': str(session.conversation_dir),
            'iterations': iteration_count,
            'vet_reports_count': len(vet_reports_for_rag),
            'global_citations_count': len(global_citation_db)
        })
        
        return final_result

    except Exception as e:
        olliePrint_simple(f"Enhanced research with quality assessment error: {e}", level='error')
        import traceback
        traceback.print_exc()
        
        error_result = {
            'success': False,
            'task_id': task_id,
            'findings': f"Enhanced research system error: {str(e)}",
            'conversation_path': str(session.conversation_dir) if session else None,
            'iterations': 0,
            'reason': 'system_error',
            'enhancement': 'sequential_fresh_context_with_quality_assessment'
        }
        return error_result
    
    finally:
        # UNIFIED RESOURCE MANAGEMENT: Single cleanup call handles everything
        if session and task_id:
            resource_manager.cleanup_session(task_id)
        
        # Additional global cleanup for thoroughness
        resource_manager.cleanup_connections()

def run_enhanced_arch_iteration(session: ArchDelveState, enable_streaming: bool = False) -> Tuple[str, bool]:
    """Enhanced ARCH iteration using updated system prompts and VET format analysis."""
    try:
        # A.R.C.H. tools - imported from consolidated config with safe access
        try:
            arch_tools = config.ARCH_TOOLS
        except AttributeError:
            # Fallback: Initialize ARCH_TOOLS if not available
            config.ARCH_TOOLS = config.PIPELINE_CONTROL_TOOLS.copy()
            arch_tools = config.ARCH_TOOLS
        
        messages = prepare_arch_messages(session)
        olliePrint_simple(f"\n[A.R.C.H. ITERATION] Analyzing strategic direction...")
        
        start_time = time.time()
        
        # MEMORY OPTIMIZATION: Use consolidated model for all agents (A.R.C.H. personality via system prompt)
        if enable_streaming:
            olliePrint_simple(f"\n🔴 [A.R.C.H. LIVE STREAM] Starting strategic analysis...")
            
            response_stream = ollama_manager.chat_concurrent_safe(
                host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
                model=CONSOLIDATED_MODEL,
                messages=messages,
                tools=arch_tools,
                stream=True,
                options=config.LLM_GENERATION_OPTIONS
            )
            
            raw_content = ""
            thinking_buffer = ""
            clean_buffer = ""
            tool_calls = None
            in_thinking = False
            
            olliePrint_simple("🧠 [A.R.C.H. THINKING]:")
            
            for chunk in response_stream:
                chunk_message = chunk.get('message', {})
                chunk_content = chunk_message.get('content', '')
                if chunk_content:
                    raw_content += chunk_content
                    
                    # Buffer the content and process thinking tags properly
                    thinking_buffer += chunk_content
                    
                    # Check for complete thinking blocks with safety limits
                    thinking_loop_count = 0
                    max_thinking_loops = 100  # Prevent infinite loops
                    max_buffer_size = 50000   # Prevent memory bloat
                    
                    while ('<think>' in thinking_buffer and '</think>' in thinking_buffer and 
                           thinking_loop_count < max_thinking_loops and 
                           len(thinking_buffer) < max_buffer_size):
                        
                        start_idx = thinking_buffer.find('<think>')
                        end_idx = thinking_buffer.find('</think>')
                        
                        # Handle malformed tags gracefully
                        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                            olliePrint_simple("[WARNING] Malformed thinking tags detected, clearing buffer", level='warning')
                            thinking_buffer = ""
                            break
                            
                        thinking_block = thinking_buffer[start_idx+7:end_idx]
                        
                        # Print thinking content cleanly
                        if thinking_block.strip():
                            print(f"   {thinking_block.strip()}")
                        
                        thinking_buffer = thinking_buffer[end_idx+8:]
                        thinking_loop_count += 1
                    
                    # Log if we hit safety limits
                    if thinking_loop_count >= max_thinking_loops:
                        olliePrint_simple("[WARNING] Thinking tag processing hit loop limit", level='warning')
                    if len(thinking_buffer) >= max_buffer_size:
                        olliePrint_simple("[WARNING] Thinking buffer hit size limit, truncating", level='warning')
                        thinking_buffer = thinking_buffer[:1000]  # Keep first 1000 chars
                    
                    # Extract clean content for non-thinking parts
                    clean_chunk = session.strip_think_tags(chunk_content)
                    if clean_chunk.strip():
                        clean_buffer += clean_chunk
                
                # Handle tool calls in streaming chunks
                chunk_tool_calls = chunk_message.get('tool_calls')
                if chunk_tool_calls:
                    tool_calls = chunk_tool_calls
                            
                # Handle done flag
                if chunk.get('done', False):
                    if clean_buffer.strip():
                        olliePrint_simple(f"\n✅ [A.R.C.H. STRATEGIC DIRECTION]:")
                        olliePrint_simple(f"   {clean_buffer.strip()}")
                    break
                    
            # Create response object for compatibility
            response = {
                'message': {
                    'content': raw_content,
                    'tool_calls': tool_calls
                }
            }
        else:
            response = ollama_manager.chat_concurrent_safe(
                host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
                model=CONSOLIDATED_MODEL,
                messages=messages,
                tools=arch_tools,
                stream=False,
                options=config.LLM_GENERATION_OPTIONS
            )
        
        call_duration = time.time() - start_time
        
        # Extract content and tool calls using app.py pattern
        response_message = response.get('message', {})
        raw_content = response_message.get('content', '')
        tool_calls = response_message.get('tool_calls')
        
        # Extract thinking and get clean content for DELVE
        thinking = session.extract_think_content(raw_content)
        clean_content = session.strip_think_tags(raw_content)
        
        # Show A.R.C.H. enhanced thinking and strategic instruction
        olliePrint_simple(f"\n[ENHANCED A.R.C.H. STRATEGIC THINKING]:\n{thinking}")
        olliePrint_simple(f"\n[ENHANCED A.R.C.H. → FRESH D.E.L.V.E.]:\n{clean_content}")
        olliePrint_simple("-" * 70)
        
        # Check for completion tool call
        research_complete = False
        if tool_calls:
            for tool_call in tool_calls:
                # Safely extract tool name
                tool_name = None
                try:
                    if isinstance(tool_call, dict):
                        # Standard format
                        tool_name = tool_call.get('function', {}).get('name')
                        if not tool_name:
                            # Alternative format
                            tool_name = tool_call.get('name')
                    else:
                        continue
                        
                    if not tool_name:
                        continue
                        
                except Exception as e:
                    continue
                
                if tool_name == 'complete_research':
                    research_complete = True
                    session.research_complete = True
                    olliePrint_simple(f"[A.R.C.H.] Research completion signaled!")
                    break
        
        # Handle empty A.R.C.H. responses
        if not clean_content.strip() and not research_complete:
            olliePrint_simple(f"[ERROR] A.R.C.H. provided no instruction and did not complete the task!", level='error')
            # Return standardized error instead of string tuple
            error_response = create_standard_error_response("ARCH", "No instruction provided and task not completed")
            return error_response, False
        
        # Store A.R.C.H.'s response (only ARCH maintains context in enhanced system)
        session.add_conversation_turn('assistant', clean_content, 'arch', thinking)
        session.save_conversation_state()
        
        # Memory cleanup - clear large variables
        del response, messages, raw_content, thinking
        
        return clean_content, research_complete
        
    except Exception as e:
        olliePrint_simple(f"Enhanced A.R.C.H. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        # Return standardized error instead of string tuple
        error_response = create_standard_error_response("ARCH", f"System error: {str(e)}")
        return error_response, False

def run_fresh_delve_iteration(arch_instruction: str, task_id: str, iteration_count: int, 
                             global_citation_db: dict, citation_counter: int, enable_streaming: bool = False) -> dict:
    """Fresh DELVE instance with no conversation history - enhanced data gathering with memory management."""
    try:
        olliePrint_simple(f"\n[FRESH D.E.L.V.E. ITERATION {iteration_count}] Starting with clean context...")
        
        # Create completely fresh messages - no conversation history
        current_time_iso = datetime.now().isoformat()
        current_date_readable = datetime.now().strftime("%B %d, %Y")
        
        messages = [
            {"role": "system", "content": config.DELVE_SYSTEM_PROMPT.format(
                current_date_time=current_time_iso,
                current_date=current_date_readable
            )},
            {"role": "user", "content": arch_instruction}
        ]
        
        # D.E.L.V.E. tools - imported from consolidated config
        # Use safe attribute access to prevent AttributeError during import timing issues
        try:
            delve_tools = config.DELVE_TOOLS
        except AttributeError:
            # Fallback: Initialize DELVE_TOOLS if not available
            config.DELVE_TOOLS = config.RESEARCH_TOOLS.copy()
            delve_tools = config.DELVE_TOOLS
        
        # Track tool calls for this fresh instance (memory efficient)
        fresh_delve_tool_calls = set()
        max_fresh_delve_iterations = 8
        fresh_iteration = 0
        
        # Continue until DELVE provides enhanced JSON output
        while fresh_iteration < max_fresh_delve_iterations:
            fresh_iteration += 1
            olliePrint_simple(f"[FRESH D.E.L.V.E. SUB-ITERATION {fresh_iteration}/{max_fresh_delve_iterations}]")
            
            # MEMORY OPTIMIZATION: Use consolidated model for all agents (D.E.L.V.E. personality via system prompt)
            if enable_streaming:
                olliePrint_simple(f"\n🔴 [D.E.L.V.E. LIVE STREAM] Sub-iteration {fresh_iteration} - Data analysis...")
                
                response_stream = ollama_manager.chat_concurrent_safe(
                    host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
                    model=CONSOLIDATED_MODEL,
                    messages=messages,
                    tools=delve_tools,
                    stream=True,
                    format="json",
                    options=config.LLM_GENERATION_OPTIONS
                )
                
                raw_content = ""
                tool_calls = None
                thinking_buffer = ""
                clean_buffer = ""
                
                olliePrint_simple("🔍 [D.E.L.V.E. THINKING]:")
                
                for chunk in response_stream:
                    # Extract content from streaming chunk like app.py
                    chunk_message = chunk.get('message', {})
                    chunk_content = chunk_message.get('content', '')
                    chunk_tool_calls = chunk_message.get('tool_calls')
                    
                    if chunk_content:
                        raw_content += chunk_content
                        
                        # Buffer and process thinking tags properly
                        thinking_buffer += chunk_content
                        
                        # Check for complete thinking blocks with safety limits
                        thinking_loop_count = 0
                        max_thinking_loops = 100  # Prevent infinite loops
                        max_buffer_size = 50000   # Prevent memory bloat
                        
                        while ('<think>' in thinking_buffer and '</think>' in thinking_buffer and 
                               thinking_loop_count < max_thinking_loops and 
                               len(thinking_buffer) < max_buffer_size):
                            
                            start_idx = thinking_buffer.find('<think>')
                            end_idx = thinking_buffer.find('</think>')
                            
                            # Handle malformed tags gracefully
                            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                                olliePrint_simple("[WARNING] Malformed thinking tags detected, clearing buffer", level='warning')
                                thinking_buffer = ""
                                break
                                
                            thinking_block = thinking_buffer[start_idx+7:end_idx]
                            
                            # Print thinking content cleanly
                            if thinking_block.strip():
                                olliePrint_simple(f"   {thinking_block.strip()}")
                            
                            thinking_buffer = thinking_buffer[end_idx+8:]
                            thinking_loop_count += 1
                        
                        # Log if we hit safety limits
                        if thinking_loop_count >= max_thinking_loops:
                            olliePrint_simple("[WARNING] Thinking tag processing hit loop limit", level='warning')
                        if len(thinking_buffer) >= max_buffer_size:
                            olliePrint_simple("[WARNING] Thinking buffer hit size limit, truncating", level='warning')
                            thinking_buffer = thinking_buffer[:1000]  # Keep first 1000 chars
                        
                        # Extract clean content
                        clean_chunk = strip_think_tags_helper(chunk_content)
                        if clean_chunk.strip() and not chunk_tool_calls:
                            clean_buffer += clean_chunk
                            
                    # Handle tool calls in streaming chunks
                    if chunk_tool_calls:
                        tool_calls = chunk_tool_calls
                        olliePrint_simple(f"\n🔧 [D.E.L.V.E. TOOLS]: {len(tool_calls)} tool calls")
                        
                    if chunk.get('done', False):
                        if clean_buffer.strip():
                            olliePrint_simple(f"\n📊 [D.E.L.V.E. DATA OUTPUT]:")
                            olliePrint_simple(f"   {clean_buffer.strip()[:200]}...")
                        olliePrint_simple(f"\n✅ [D.E.L.V.E. SUB-ITERATION {fresh_iteration} COMPLETE]")
                        break
                        
                # Create response object for compatibility
                response = {
                    'message': {
                        'content': raw_content,
                        'tool_calls': tool_calls
                    }
                }
            else:
                            response = ollama_manager.chat_concurrent_safe(
                host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
                model=CONSOLIDATED_MODEL,
                messages=messages,
                tools=delve_tools,
                stream=False,
                format="json",
                options=config.LLM_GENERATION_OPTIONS
            )
            
            # Extract content and tool calls using app.py pattern
            response_message = response.get('message', {})
            raw_content = response_message.get('content', '')
            tool_calls = response_message.get('tool_calls')
            
            # Extract thinking
            current_thinking = extract_think_content_helper(raw_content)
            clean_content = strip_think_tags_helper(raw_content)
            
            # Show fresh DELVE thinking and response
            if current_thinking:
                olliePrint_simple(f"\n[FRESH D.E.L.V.E. THINKING]:\n{current_thinking}")
            if clean_content and not tool_calls:
                olliePrint_simple(f"\n[FRESH D.E.L.V.E. ENHANCED DATA OUTPUT]:\n{clean_content}")
            if tool_calls:
                olliePrint_simple(f"\n[FRESH D.E.L.V.E. TOOL CALLS] {len(tool_calls)} calls")
            
            # Ensure we have a proper message format for context
            if hasattr(response_message, 'content'):
                # ChatResponse object - convert to dict for messages
                message_dict = {
                    'role': 'assistant',
                    'content': response_message.content or '',
                    'tool_calls': getattr(response_message, 'tool_calls', None)
                }
                messages.append(message_dict)
            else:
                # Already a dict
                if 'role' not in response_message:
                    response_message['role'] = 'assistant'
                messages.append(response_message)
            
            # Memory management: Limit message history to prevent RAM buildup
            if len(messages) > 10:  # Keep only recent messages
                messages = messages[:2] + messages[-8:]  # Keep system + user + last 8
                olliePrint_simple(f"[MEMORY] Trimmed DELVE message history to prevent RAM buildup")
            
            if tool_calls:
                # Execute tools with robust state management and error boundaries
                try:
                    # Prepare for tool execution with clean state
                    tool_results = []
                    executed_this_round = 0
                    failed_tools = []
                    
                    # Process each tool call with individual error handling
                    for tool_index, tool_call in enumerate(tool_calls):
                        try:
                            # Extract tool name safely
                            tool_name = tool_call.get('function', {}).get('name')
                            if not tool_name:
                                olliePrint_simple(f"  WARNING: Tool call {tool_index} has no name, skipping")
                                continue

                            # Parse arguments with robust error handling
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
                                failed_tools.append(f"{tool_name}: argument parsing failed")
                                continue

                            # Check for duplicates within this fresh instance
                            tool_signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                            if tool_signature in fresh_delve_tool_calls:
                                olliePrint_simple(f"  SKIPPED: Duplicate tool call - {tool_name}")
                                continue

                            # Execute the tool call with isolated error handling
                            tool_execution_result = None
                            try:
                                if tool_name in TOOL_FUNCTIONS:
                                    tool_function = TOOL_FUNCTIONS[tool_name]
                                    olliePrint_simple(f"  Fresh DELVE executing {tool_name} with args: {arguments}")
                                    
                                    tool_execution_result = tool_function(**arguments)
                                    log_tool_result(tool_name, arguments, tool_execution_result)
                                    
                                    # Only mark as executed if successful
                                    fresh_delve_tool_calls.add(tool_signature)
                                    executed_this_round += 1
                                    
                                    tool_output_content = json.dumps(tool_execution_result)
                                    
                                else:
                                    olliePrint_simple(f"  ERROR: Unknown tool name: {tool_name}")
                                    tool_output_content = json.dumps({"success": False, "error": f"Tool '{tool_name}' not found."})
                                    failed_tools.append(f"{tool_name}: unknown tool")
                                    
                            except Exception as e:
                                olliePrint_simple(f"  ERROR: Tool {tool_name} execution failed: {e}")
                                tool_output_content = json.dumps({"success": False, "error": f"Tool execution failed: {str(e)}"})
                                failed_tools.append(f"{tool_name}: execution failed - {str(e)}")

                            # Add tool result (success or failure) to results
                            tool_results.append({
                                "role": "tool",
                                "content": tool_output_content,
                                "tool_call_id": tool_call.get('id', f'unknown_{tool_index}')
                            })
                            
                        except Exception as e:
                            olliePrint_simple(f"  CRITICAL: Tool processing error for tool {tool_index}: {e}", level='error')
                            failed_tools.append(f"tool_{tool_index}: critical processing error")
                            continue
                    
                    # Update message history only if we have results to add
                    if tool_results:
                        try:
                            messages.extend(tool_results)
                            olliePrint_simple(f"[FRESH D.E.L.V.E.] Executed {executed_this_round} tools successfully")
                            if failed_tools:
                                olliePrint_simple(f"[FRESH D.E.L.V.E.] Failed tools: {failed_tools}", level='warning')
                        except Exception as e:
                            olliePrint_simple(f"[FRESH D.E.L.V.E.] Error updating message history: {e}", level='error')
                            # Continue without adding tool results to prevent corruption
                            
                    else:
                        olliePrint_simple(f"[FRESH D.E.L.V.E.] No tool results to add - all failed or were duplicates")
                        if len(failed_tools) == len(tool_calls):
                            olliePrint_simple(f"[FRESH D.E.L.V.E.] All tool calls failed - forcing final response")
                            break
                            
                except Exception as e:
                    olliePrint_simple(f"[FRESH D.E.L.V.E.] CRITICAL: Tool execution system error: {e}", level='error')
                    # Continue with next iteration instead of crashing
                    continue
            else:
                # No more tools, return enhanced JSON data
                if not clean_content:
                    olliePrint_simple("[FRESH D.E.L.V.E.] ERROR: No enhanced data output provided.", level='error')
                    return create_standard_error_response("FRESH_DELVE", "No enhanced data output provided")
                
                # Parse enhanced JSON data
                try:
                    enhanced_data = json.loads(clean_content)
                    olliePrint_simple(f"[FRESH D.E.L.V.E.] Enhanced data collection complete - {len(enhanced_data) if isinstance(enhanced_data, list) else 1} sources")
                    
                    # Memory cleanup before returning
                    del response, messages, raw_content, current_thinking, clean_content
                    olliePrint_simple(f"[MEMORY] DELVE iteration cleaned up large variables")
                    
                    return enhanced_data
                except json.JSONDecodeError as e:
                    olliePrint_simple(f"[FRESH D.E.L.V.E.] JSON parse error: {e}", level='error')
                    return create_standard_error_response("FRESH_DELVE", f"JSON parsing failed: {str(e)}")
        
        # Max iterations reached
        olliePrint_simple(f"[FRESH D.E.L.V.E.] WARNING: Reached max iterations without completion", level='warning')
        return create_standard_error_response("FRESH_DELVE", "Exceeded iteration limit without completion")
        
    except Exception as e:
        olliePrint_simple(f"Fresh D.E.L.V.E. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        return create_standard_error_response("FRESH_DELVE", f"System error: {str(e)}")

def run_fresh_vet_iteration(delve_data: dict, arch_instruction: str, task_id: str,
                           iteration_count: int, global_citation_db: dict, enable_streaming: bool = False) -> str:
    """Fresh VET instance to format DELVE data into strategic summary with memory management."""
    try:
        olliePrint_simple(f"\n[FRESH V.E.T. ITERATION {iteration_count}] Analyzing enhanced data with clean context...")
        
        # Create completely fresh messages - no conversation history
        delve_data_str = json.dumps(delve_data, indent=2)
        
        messages = [
            {"role": "system", "content": config.VET_SYSTEM_PROMPT},
            {"role": "user", "content": f"DIRECTIVE: {arch_instruction}\n\nENHANCED RAW DATA:\n{delve_data_str}"}
        ]

        # MEMORY OPTIMIZATION: Use consolidated model for all agents (V.E.T. personality via system prompt)
        if enable_streaming:
            olliePrint_simple(f"\n🔴 [V.E.T. LIVE STREAM] Verification and evidence triangulation...")
            
            response_stream = ollama_manager.chat_concurrent_safe(
                host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
                model=CONSOLIDATED_MODEL,
                messages=messages,
                stream=True,
                options=config.LLM_GENERATION_OPTIONS
            )
            
            raw_verified_report = ""
            thinking_buffer = ""
            clean_buffer = ""
            
            olliePrint_simple("🔬 [V.E.T. THINKING]:")
            
            for chunk in response_stream:
                chunk_message = chunk.get('message', {})
                chunk_content = chunk_message.get('content', '')
                if chunk_content:
                    raw_verified_report += chunk_content
                    
                    # Buffer and process thinking tags properly
                    thinking_buffer += chunk_content
                    
                    # Check for complete thinking blocks with safety limits
                    thinking_loop_count = 0
                    max_thinking_loops = 100  # Prevent infinite loops
                    max_buffer_size = 50000   # Prevent memory bloat
                    
                    while ('<think>' in thinking_buffer and '</think>' in thinking_buffer and 
                           thinking_loop_count < max_thinking_loops and 
                           len(thinking_buffer) < max_buffer_size):
                        
                        start_idx = thinking_buffer.find('<think>')
                        end_idx = thinking_buffer.find('</think>')
                        
                        # Handle malformed tags gracefully
                        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                            olliePrint_simple("[WARNING] Malformed thinking tags detected, clearing buffer", level='warning')
                            thinking_buffer = ""
                            break
                            
                        thinking_block = thinking_buffer[start_idx+7:end_idx]
                        
                        # Print thinking content cleanly
                        if thinking_block.strip():
                            olliePrint_simple(f"   {thinking_block.strip()}")
                        
                        thinking_buffer = thinking_buffer[end_idx+8:]
                        thinking_loop_count += 1
                    
                    # Log if we hit safety limits
                    if thinking_loop_count >= max_thinking_loops:
                        olliePrint_simple("[WARNING] Thinking tag processing hit loop limit", level='warning')
                    if len(thinking_buffer) >= max_buffer_size:
                        olliePrint_simple("[WARNING] Thinking buffer hit size limit, truncating", level='warning')
                        thinking_buffer = thinking_buffer[:1000]  # Keep first 1000 chars
                    
                    # Extract clean content
                    clean_chunk = strip_think_tags_helper(chunk_content)
                    if clean_chunk.strip():
                        clean_buffer += clean_chunk
                        if "VERIFIED REPORT" in clean_chunk:
                            olliePrint_simple(f"\n📋 [V.E.T. REPORT]: Generating verified findings...")
                        
                if chunk.get('done', False):
                    if clean_buffer.strip():
                        olliePrint_simple(f"\n📋 [V.E.T. VERIFIED REPORT]:")
                        olliePrint_simple(f"   Report generated ({len(clean_buffer)} chars)")
                    olliePrint_simple(f"\n✅ [V.E.T. VERIFICATION COMPLETE]")
                    break
                    
            # Create response object for compatibility
            response = {
                'message': {
                    'content': raw_verified_report
                }
            }
        else:
                    response = ollama_manager.chat_concurrent_safe(
            host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
            model=CONSOLIDATED_MODEL,
            messages=messages,
            stream=False,
            options=config.LLM_GENERATION_OPTIONS
        )

        # Extract content using app.py pattern
        response_message = response.get('message', {})
        raw_verified_report = response_message.get('content', '')
        
        # Strip any thinking tags to get the clean report
        verified_report = strip_think_tags_helper(raw_verified_report)
        
        if not verified_report.strip().startswith("VERIFIED REPORT"):
            olliePrint_simple("[FRESH V.E.T.] ERROR: Output did not follow the required enhanced report format.", level='error')
            # Fallback enhanced report
            verified_report = f"VERIFIED REPORT: Enhanced Analysis Failed\nDIRECTIVE: {arch_instruction}\n\nVERIFIED FINDINGS:\n• Fresh V.E.T. failed to produce a valid enhanced report.\n\nASSESSMENT:\n• Overall Confidence: Low\n• Key Contradictions: Analysis system error\n• Notable Gaps: Complete analysis failure\n\nSOURCES:\n• No sources processed due to system error"
        
        olliePrint_simple(f"\n[FRESH V.E.T. → A.R.C.H.]:\n{verified_report}")
        olliePrint_simple("-" * 70)
        
        # Memory cleanup before returning
        del response, messages, delve_data_str, raw_verified_report
        olliePrint_simple(f"[MEMORY] VET iteration cleaned up large variables")
        
        return verified_report

    except Exception as e:
        olliePrint_simple(f"Fresh V.E.T. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        return f"VERIFIED REPORT: Fresh V.E.T. System Error\nDIRECTIVE: {arch_instruction}\n\nVERIFIED FINDINGS:\n• Fresh V.E.T. system encountered an error: {e}\n\nASSESSMENT:\n• Overall Confidence: Low\n• Key Contradictions: System malfunction\n• Notable Gaps: Complete processing failure\n\nSOURCES:\n• No sources processed due to error"

def update_global_citations(delve_data: dict, citation_db: dict, counter: int) -> int:
    """Update global citation database with new sources from DELVE."""
    try:
        # Handle both single dict and list of dicts from DELVE
        sources = delve_data if isinstance(delve_data, list) else [delve_data]
        
        for source in sources:
            if isinstance(source, dict) and 'url' in source:
                url = source['url']
                
                # Check if URL already exists in citation database
                url_exists = False
                for cit_id, data in citation_db.items():
                    if data.get('url') == url:
                        url_exists = True
                        break
                
                # Add new URL to global citation database
                if not url_exists:
                    citation_db[counter] = {
                        'url': url,
                        'title': source.get('title', f"Source {counter}"),
                        'credibility': source.get('credibility', 'unknown'),
                        'source_type': source.get('source_type', 'other'),
                        'first_seen': datetime.now().isoformat()
                    }
                    olliePrint_simple(f"[GLOBAL CITATIONS] Added [{counter}]: {url}")
                    counter += 1
        
        olliePrint_simple(f"[GLOBAL CITATIONS] Database now contains {len(citation_db)} unique sources")
        return counter
        
    except Exception as e:
        olliePrint_simple(f"Global citation update failed: {e}", level='error')
        return counter

class RAGDatabase:
    """Context manager for DuckDB RAG database to ensure proper connection cleanup."""
    
    def __init__(self, vet_reports: list, task_id: str):
        self.vet_reports = vet_reports
        self.task_id = task_id
        self.conn = None
        self.is_duckdb = False
        self._cleanup_called = False  # Track if cleanup has been called
    
    def __del__(self):
        """Destructor to ensure connection cleanup even if context manager isn't used properly."""
        if not self._cleanup_called:
            try:
                self._perform_cleanup()
            except:
                pass  # Ignore errors in destructor
    
    def _perform_cleanup(self):
        """Internal method to perform connection cleanup."""
        if self._cleanup_called:
            return  # Avoid double cleanup
            
        self._cleanup_called = True
        
        try:
            if self.is_duckdb and self.conn:
                try:
                    self.conn.close()
                except:
                    pass
                self.conn = None
                self.is_duckdb = False
            elif isinstance(self.conn, dict):
                self.conn.clear()
                self.conn = None
        except:
            pass
    
    def execute(self, query: str, params=None):
        """Execute query on the database."""
        if self.is_duckdb:
            return self.conn.execute(query, params)
        else:
            # Fallback behavior for non-DuckDB
            raise RuntimeError("Database fallback mode - query not supported")
    
    def get_reports(self):
        """Get all reports (fallback compatibility)."""
        if self.is_duckdb:
            results = self.conn.execute("SELECT * FROM vet_reports ORDER BY iteration_number").fetchall()
            return [
                {
                    'iteration': row[2],
                    'instruction': row[3],
                    'findings': row[4],
                    'confidence': row[5],
                    'summary': row[8]
                }
                for row in results
            ]
        else:
            return self.conn.get('reports', [])
    
    def __enter__(self):
        try:
            import duckdb
            olliePrint_simple(f"[RAG DATABASE] Creating DuckDB database from {len(self.vet_reports)} VET reports...")
            
            # Create in-memory DuckDB connection
            self.conn = duckdb.connect(':memory:')
            self.is_duckdb = True
            
            # Create schema for VET reports
            self.conn.execute("""
                CREATE TABLE vet_reports (
                    id INTEGER PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    iteration_number INTEGER NOT NULL,
                    instruction_text TEXT NOT NULL,
                    verified_findings TEXT NOT NULL,
                    assessment_confidence TEXT,
                    contradictions TEXT,
                    sources_list TEXT,
                    raw_summary TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create global citations table
            self.conn.execute("""
                CREATE TABLE global_citations (
                    citation_id INTEGER PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    credibility_score TEXT CHECK(credibility_score IN ('high', 'medium', 'low')),
                    source_type TEXT,
                    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert VET reports into database
            for report in self.vet_reports:
                # Parse VET report to extract structured data
                raw_summary = report.get('summary', '')
                iteration = report.get('iteration', 0)
                instruction = report.get('instruction', '')
                
                # Extract structured data from VET report
                verified_findings, assessment_confidence, contradictions, sources_list = parse_vet_report(raw_summary)
                
                self.conn.execute("""
                    INSERT INTO vet_reports 
                    (task_id, iteration_number, instruction_text, verified_findings, 
                     assessment_confidence, contradictions, sources_list, raw_summary, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.task_id,
                    iteration,
                    instruction,
                    verified_findings,
                    assessment_confidence,
                    contradictions,
                    sources_list,
                    raw_summary,
                    report.get('timestamp', datetime.now().isoformat())
                ))
            
            olliePrint_simple(f"[RAG DATABASE] DuckDB database created with {len(self.vet_reports)} reports")
            return self
            
        except Exception as e:
            olliePrint_simple(f"DuckDB RAG database creation failed: {e}, using fallback", level='error')
            # Fallback to in-memory structure
            self.is_duckdb = False
            self.conn = {
                'task_id': self.task_id,
                'reports': self.vet_reports,
                'created_at': datetime.now().isoformat(),
                'total_reports': len(self.vet_reports),
                'type': 'fallback'
            }
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """MEMORY OPTIMIZATION: Enhanced DuckDB connection cleanup to prevent memory leaks."""
        try:
            if self.is_duckdb and self.conn:
                # Enhanced connection cleanup
                connection_was_open = True
                try:
                    # Check if connection is still open by attempting a simple query
                    self.conn.execute("SELECT 1")
                except:
                    connection_was_open = False
                
                # Close connection if it's still open
                if connection_was_open:
                    try:
                        # Explicitly close all cursors first
                        try:
                            self.conn.execute("PRAGMA database_list").close()
                        except:
                            pass
                        
                        # Close the main connection
                        self.conn.close()
                        olliePrint_simple("[RAG DATABASE] DuckDB connection closed successfully")
                    except Exception as e:
                        olliePrint_simple(f"[RAG DATABASE] Error during connection close: {e}", level='warning')
                else:
                    olliePrint_simple("[RAG DATABASE] DuckDB connection was already closed")
                
                # Clear connection reference to prevent lingering references
                self.conn = None
                self.is_duckdb = False
                
                # Force garbage collection to clean up connection objects
                import gc
                gc.collect()
                
            elif not self.is_duckdb:
                # Fallback mode - clear the data structure
                if isinstance(self.conn, dict):
                    self.conn.clear()
                self.conn = None
            
            # Mark cleanup as completed
            self._cleanup_called = True
                
        except Exception as e:
            olliePrint_simple(f"[RAG DATABASE] Critical error during connection cleanup: {e}", level='error')
            # Force cleanup even if there were errors
            try:
                self.conn = None
                self.is_duckdb = False
                self._cleanup_called = True
            except:
                pass

def parse_vet_report(vet_report: str) -> tuple:
    """Parse VET report text into structured components."""
    try:
        verified_findings = ""
        assessment_confidence = "Unknown"
        contradictions = ""
        sources_list = ""
        
        # Extract VERIFIED FINDINGS section
        if "VERIFIED FINDINGS:" in vet_report:
            findings_start = vet_report.find("VERIFIED FINDINGS:") + len("VERIFIED FINDINGS:")
            findings_end = vet_report.find("ASSESSMENT:", findings_start)
            if findings_end == -1:
                findings_end = len(vet_report)
            verified_findings = vet_report[findings_start:findings_end].strip()
        
        # Extract ASSESSMENT section
        if "ASSESSMENT:" in vet_report:
            assessment_start = vet_report.find("ASSESSMENT:") + len("ASSESSMENT:")
            assessment_end = vet_report.find("SOURCES:", assessment_start)
            if assessment_end == -1:
                assessment_end = len(vet_report)
            assessment_text = vet_report[assessment_start:assessment_end].strip()
            
            # Extract confidence level
            if "Overall Confidence:" in assessment_text:
                conf_start = assessment_text.find("Overall Confidence:") + len("Overall Confidence:")
                conf_line = assessment_text[conf_start:].split('\n')[0].strip()
                assessment_confidence = conf_line
            
            # Extract contradictions
            if "Key Contradictions:" in assessment_text:
                contra_start = assessment_text.find("Key Contradictions:") + len("Key Contradictions:")
                contra_end = assessment_text.find("Notable Gaps:", contra_start)
                if contra_end == -1:
                    contra_end = len(assessment_text)
                contradictions = assessment_text[contra_start:contra_end].strip()
        
        # Extract SOURCES section
        if "SOURCES:" in vet_report:
            sources_start = vet_report.find("SOURCES:") + len("SOURCES:")
            sources_list = vet_report[sources_start:].strip()
        
        return verified_findings, assessment_confidence, contradictions, sources_list
        
    except Exception as e:
        olliePrint_simple(f"VET report parsing failed: {e}", level='error')
        return vet_report, "Unknown", "", ""

class SAGEQueryTools:
    """Advanced query tools for SAGE truth determination using DuckDB with proper memory management."""
    
    def __init__(self, rag_database):
        self.db = rag_database
        self.is_duckdb = rag_database.is_duckdb if hasattr(rag_database, 'is_duckdb') else hasattr(rag_database, 'execute')
    
    def query_findings_by_topic(self, topic: str) -> List[Dict]:
        """Search VET reports for specific topics/keywords."""
        if not self.is_duckdb:
            # Fallback for non-DuckDB
            return self._fallback_topic_search(topic)
        
        try:
            results = self.db.execute("""
                SELECT iteration_number, instruction_text, verified_findings, assessment_confidence
                FROM vet_reports 
                WHERE verified_findings LIKE ? OR instruction_text LIKE ?
                ORDER BY iteration_number
            """, (f'%{topic}%', f'%{topic}%')).fetchall()
            
            return [
                {
                    'iteration': row[0],
                    'instruction': row[1],
                    'findings': row[2],
                    'confidence': row[3]
                }
                for row in results
            ]
        except Exception as e:
            olliePrint_simple(f"Topic query failed: {e}", level='error')
            return []
    
    def query_consensus_level(self, claim: str) -> Dict:
        """Find agreement/disagreement on specific claims across reports."""
        if not self.is_duckdb:
            return self._fallback_consensus_search(claim)
        
        try:
            results = self.db.execute("""
                SELECT verified_findings, assessment_confidence, contradictions
                FROM vet_reports 
                WHERE verified_findings LIKE ?
                ORDER BY iteration_number
            """, (f'%{claim}%',)).fetchall()
            
            high_confidence = sum(1 for r in results if 'High' in r[1])
            medium_confidence = sum(1 for r in results if 'Medium' in r[1])
            low_confidence = sum(1 for r in results if 'Low' in r[1])
            total = len(results)
            
            return {
                'total_mentions': total,
                'high_confidence_count': high_confidence,
                'medium_confidence_count': medium_confidence,
                'low_confidence_count': low_confidence,
                'consensus_score': (high_confidence * 3 + medium_confidence * 2 + low_confidence) / (total * 3) if total > 0 else 0,
                'findings': [r[0] for r in results]
            }
        except Exception as e:
            olliePrint_simple(f"Consensus query failed: {e}", level='error')
            return {'total_mentions': 0, 'consensus_score': 0, 'findings': []}
    
    def query_source_credibility(self, credibility_level: str) -> List[Dict]:
        """Filter findings by source reliability level."""
        if not self.is_duckdb:
            return []
        
        try:
            results = self.db.execute("""
                SELECT iteration_number, verified_findings, sources_list
                FROM vet_reports 
                WHERE sources_list LIKE ?
                ORDER BY iteration_number
            """, (f'%{credibility_level}%',)).fetchall()
            
            return [
                {
                    'iteration': row[0],
                    'findings': row[1],
                    'sources': row[2]
                }
                for row in results
            ]
        except Exception as e:
            olliePrint_simple(f"Credibility query failed: {e}", level='error')
            return []
    
    def _fallback_topic_search(self, topic: str) -> List[Dict]:
        """Fallback topic search for non-DuckDB databases."""
        if not isinstance(self.db, dict) or 'reports' not in self.db:
            return []
        
        results = []
        for report in self.db['reports']:
            summary = report.get('summary', '')
            if topic.lower() in summary.lower():
                results.append({
                    'iteration': report.get('iteration', 0),
                    'instruction': report.get('instruction', ''),
                    'findings': summary,
                    'confidence': 'Unknown'
                })
        return results
    
    def _fallback_consensus_search(self, claim: str) -> Dict:
        """Fallback consensus search for non-DuckDB databases."""
        if not isinstance(self.db, dict) or 'reports' not in self.db:
            return {'total_mentions': 0, 'consensus_score': 0, 'findings': []}
        
        findings = []
        for report in self.db['reports']:
            summary = report.get('summary', '')
            if claim.lower() in summary.lower():
                findings.append(summary)
        
        return {
            'total_mentions': len(findings),
            'consensus_score': 0.5 if findings else 0,
            'findings': findings
        }

def synthesize_final_report_with_rag(original_task: str, rag_database: object,
                                    global_citations: dict, conversation_path: str) -> str:
    """SAGE synthesis using RAG database and global citations with truth determination."""
    try:
        olliePrint_simple("\n[S.A.G.E. RAG] Synthesizing final report with advanced truth determination...")
        
        # Initialize SAGE query tools
        query_tools = SAGEQueryTools(rag_database)
        
        # Extract VET reports from RAG database (using new RAGDatabase context manager)
        if rag_database.is_duckdb:
            # DuckDB implementation
            vet_reports = rag_database.get_reports()
        else:
            # Fallback implementation
            vet_reports = rag_database.conn.get('reports', [])
        
        # Apply SAGE truth determination algorithm
        truth_analysis = determine_truth_consensus_advanced(vet_reports, query_tools)
        
        # Combine all VET summaries with truth analysis
        all_summaries = []
        for report in vet_reports:
            summary = report.get('summary', '')
            if summary:
                confidence = truth_analysis.get('iteration_confidence', {}).get(report.get('iteration', 0), 'Unknown')
                all_summaries.append(f"Iteration {report.get('iteration', '?')} (Confidence: {confidence}): {summary}")
        
        combined_summaries = "\n\n---\n\n".join(all_summaries)
        
        # Create global citation reference list
        citation_list = []
        for cit_id, data in global_citations.items():
            citation_list.append(f"[{cit_id}] {data.get('url', 'Unknown URL')} ({data.get('credibility', 'unknown')} credibility)")
        
        citations_text = "\n".join(citation_list)
        
        # Create truth determination summary
        truth_summary = f"""
TRUTH DETERMINATION ANALYSIS:
- Overall Confidence: {truth_analysis.get('overall_confidence', 'Unknown')}
- High Confidence Findings: {truth_analysis.get('high_confidence_count', 0)}
- Contradictions Identified: {truth_analysis.get('contradiction_count', 0)}
- Consensus Score: {truth_analysis.get('consensus_score', 0):.2f}
"""
        
        # Use SAGE for final synthesis with truth determination
        synthesis_prompt = f"""# FINAL RESEARCH SYNTHESIS WITH TRUTH DETERMINATION

## Original Research Mission
{original_task}

## Truth Determination Analysis
{truth_summary}

## All Verified Research Reports
{combined_summaries}

## Global Citation Database
{citations_text}

## Your Mission
Synthesize all research findings into a comprehensive, user-facing final report. Use global citation numbers [1], [2], [3] etc. when referencing sources. Apply the truth determination analysis to highlight high-confidence conclusions and note any contradictions. Provide confidence levels for major conclusions.

Create a final report that answers the original research mission completely with confidence assessment."""
        
        messages = [
            {"role": "system", "content": config.SAGE_FINAL_REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        # MEMORY OPTIMIZATION: Use consolidated model for all agents (S.A.G.E. personality via system prompt)
        response = ollama_manager.chat_concurrent_safe(
            host=config.OLLAMA_BASE_URL,  # Add host parameter like app.py
            model=CONSOLIDATED_MODEL,
            messages=messages,
            stream=False
        )
        
        # Extract content using app.py pattern
        response_message = response.get('message', {})
        final_report = response_message.get('content', '').strip()
        if not final_report:
            olliePrint_simple("[S.A.G.E. RAG] No final report generated from model", level='error')
            return "Error: S.A.G.E. RAG synthesis failed to generate the final report."
        
        olliePrint_simple("[S.A.G.E. RAG] Advanced final report synthesis complete with truth determination.")
        _log_synthesis_event(conversation_path, 'sage_advanced_rag_final_report', {
            'report': final_report,
            'vet_reports_used': len(vet_reports),
            'global_citations_used': len(global_citations),
            'truth_analysis': truth_analysis
        })
        
        return final_report
        
    except Exception as e:
        olliePrint_simple(f"[S.A.G.E. RAG] Advanced final report synthesis failed: {e}", level='error')
        return f"Error during S.A.G.E. advanced RAG synthesis: {e}"

def determine_truth_consensus_advanced(vet_reports: List[Dict], query_tools: SAGEQueryTools) -> Dict:
    """
    Advanced SAGE truth determination algorithm with confidence scoring.
    
    Steps:
    1. Evidence Aggregation: Group findings by confidence level
    2. Source Credibility Analysis: Weight findings by source reliability
    3. Consensus Detection: Find agreement patterns
    4. Contradiction Analysis: Identify conflicts
    5. Confidence Scoring: Calculate reliability scores
    6. Truth Synthesis: Build final conclusions
    """
    try:
        # Step 1: Evidence Aggregation
        high_confidence_findings = []
        medium_confidence_findings = []
        low_confidence_findings = []
        
        iteration_confidence = {}
        
        for report in vet_reports:
            confidence = report.get('confidence', 'Unknown')
            findings = report.get('findings', report.get('summary', ''))
            iteration = report.get('iteration', 0)
            
            iteration_confidence[iteration] = confidence
            
            if 'High' in confidence:
                high_confidence_findings.append(findings)
            elif 'Medium' in confidence:
                medium_confidence_findings.append(findings)
            else:
                low_confidence_findings.append(findings)
        
        # Step 2 & 3: Source Credibility Analysis and Consensus Detection
        credibility_weights = {'high': 3, 'medium': 2, 'low': 1}
        total_weighted_score = (
            len(high_confidence_findings) * credibility_weights['high'] +
            len(medium_confidence_findings) * credibility_weights['medium'] +
            len(low_confidence_findings) * credibility_weights['low']
        )
        max_possible_score = len(vet_reports) * credibility_weights['high']
        
        consensus_score = total_weighted_score / max_possible_score if max_possible_score > 0 else 0
        
        # Step 4: Contradiction Analysis
        contradiction_indicators = [
            'contradicts', 'disagrees', 'conflicting', 'however', 'but', 'although', 
            'different from', 'opposes', 'disputes'
        ]
        
        contradiction_count = 0
        for report in vet_reports:
            findings = report.get('findings', report.get('summary', '')).lower()
            if any(indicator in findings for indicator in contradiction_indicators):
                contradiction_count += 1
        
        # Step 5: Confidence Scoring
        if consensus_score >= 0.7 and contradiction_count <= 1:
            overall_confidence = "High"
        elif consensus_score >= 0.4 and contradiction_count <= 2:
            overall_confidence = "Medium"
        else:
            overall_confidence = "Low"
        
        # Step 6: Truth Synthesis Results
        return {
            'overall_confidence': overall_confidence,
            'consensus_score': consensus_score,
            'high_confidence_count': len(high_confidence_findings),
            'medium_confidence_count': len(medium_confidence_findings),
            'low_confidence_count': len(low_confidence_findings),
            'contradiction_count': contradiction_count,
            'iteration_confidence': iteration_confidence,
            'total_reports_analyzed': len(vet_reports)
        }
            
    except Exception as e:
        olliePrint_simple(f"Truth determination analysis failed: {e}", level='error')
        return {
            'overall_confidence': 'Unknown',
            'consensus_score': 0,
            'contradiction_count': 0,
            'high_confidence_count': 0,
            'medium_confidence_count': 0,
            'low_confidence_count': 0,
            'iteration_confidence': {},
            'total_reports_analyzed': 0
        }

# Helper functions for the fresh instances
def extract_think_content_helper(text):
    """Extract thinking content from <think>...</think> tags."""
    if not text:
        return ""
    matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    return '\n'.join(matches).strip()

def strip_think_tags_helper(text):
    """Remove <think>...</think> blocks from text."""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def conduct_enhanced_iterative_research(task_id: str, original_task: str, enable_streaming: bool = False) -> Dict:
    """
    Enhanced A.R.C.H./D.E.L.V.E./V.E.T./S.A.G.E. research with sequential processing and fresh context.
    
    This function provides the basic enhanced research pipeline without quality assessment.
    For full quality assessment, use enhanced_conduct_iterative_research_with_quality().
    
    Key Features:
    - Each DELVE instance gets fresh context (no conversation history)
    - Each VET instance gets fresh context  
    - Only ARCH maintains context of VET summaries for strategic planning
    - Sequential processing: ARCH → DELVE (fresh) → VET (fresh) → ARCH
    - Instance cleanup after each cycle for hardware efficiency
    - Global citation tracking
    - RAG database preparation for SAGE
    """
    session = None
    try:
        olliePrint_simple(f"[ENHANCED RESEARCH] Starting A.R.C.H./D.E.L.V.E./V.E.T./S.A.G.E. investigation...")
        olliePrint_simple(f"   Task ID: {task_id}")
        olliePrint_simple(f"   Objective: {original_task[:100]}...")
        
        # MEMORY OPTIMIZATION: Preload model to prevent unloading during research
        ollama_manager.preload_model(CONSOLIDATED_MODEL)
        
        # Create research session - only ARCH maintains context
        session = create_research_session(task_id, original_task)
        vet_reports_for_rag = []  # Store VET reports for SAGE RAG database
        
        # Global citation tracking across all DELVE sessions
        global_citation_db = {}
        citation_counter = 1
        
        # Enhanced research loop with fresh context
        iteration_count = 0
        max_iterations = config.ENHANCED_RESEARCH_CONFIG.get('max_iterations', config.ARCH_DELVE_MAX_RESEARCH_ITERATIONS)
        
        while iteration_count < max_iterations:
            iteration_count += 1
            olliePrint_simple(f"\n{'='*40} [ENHANCED RESEARCH ITERATION {iteration_count}/{max_iterations}] {'='*40}")
            
            # 1. A.R.C.H. provides strategic direction (maintains context of VET summaries)
            arch_instruction, is_complete = run_enhanced_arch_iteration(session, enable_streaming)
            
            # Check for ARCH errors
            if isinstance(arch_instruction, dict) and not arch_instruction.get("success", True):
                olliePrint_simple(f"[ENHANCED RESEARCH] A.R.C.H. iteration {iteration_count} failed: {arch_instruction.get('error', 'Unknown error')}", level='error')
                break
            
            if is_complete:
                olliePrint_simple(f"[ENHANCED RESEARCH] A.R.C.H. declared research complete!")
                break
            
            # 2. Fresh DELVE instance executes research with enhanced data gathering
            delve_enhanced_data = run_fresh_delve_iteration(
                arch_instruction, 
                task_id, 
                iteration_count,
                global_citation_db,
                citation_counter,
                enable_streaming
            )
            
            # CRITICAL: Validate DELVE results before proceeding
            if isinstance(delve_enhanced_data, dict) and "error" in delve_enhanced_data:
                olliePrint_simple(f"[ENHANCED RESEARCH] DELVE iteration {iteration_count} failed: {delve_enhanced_data['error']}", level='error')
                olliePrint_simple(f"[ENHANCED RESEARCH] Skipping iteration and continuing...", level='warning')
                continue
            
            # Update global citations from DELVE results
            citation_counter = update_global_citations(delve_enhanced_data, global_citation_db, citation_counter)
            
            # 3. Fresh VET instance formats data into strategic summary  
            vet_summary = run_fresh_vet_iteration(
                delve_enhanced_data, 
                arch_instruction,
                task_id,
                iteration_count,
                global_citation_db,
                enable_streaming
            )
            
            # Store VET report for SAGE RAG database
            vet_reports_for_rag.append({
                'iteration': iteration_count,
                'instruction': arch_instruction,
                'summary': vet_summary,
                'timestamp': datetime.now().isoformat()
            })
            
            # 4. Pass VET summary to A.R.C.H.'s context (only ARCH maintains context)
            session.add_conversation_turn('user', vet_summary, 'arch')
            
            # 5. Save state and cleanup instances (hardware efficiency)
            session.save_conversation_state()
            olliePrint_simple(f"[ENHANCED RESEARCH] Iteration {iteration_count} complete. Instances cleaned up.")
        
        # After loop completion, use SAGE with RAG database
        final_result = {}
        if session.research_complete:
            olliePrint_simple(f"[SUCCESS] Enhanced research completed in {iteration_count} iterations.")
            
            # MEMORY OPTIMIZATION: Use RAG database context manager for proper cleanup
            with RAGDatabase(vet_reports_for_rag, task_id) as rag_database:
                # SAGE creates final report using RAG queries
                final_report_str = synthesize_final_report_with_rag(
                    original_task=original_task,
                    rag_database=rag_database,
                    global_citations=global_citation_db,
                    conversation_path=str(session.conversation_dir)
                )
                session.final_findings = final_report_str
            # RAG database automatically closed here by context manager
            
            final_result = {
                'success': True,
                'findings': session.final_findings,
                'reason': 'completed',
                'enhancement': 'sequential_fresh_context'
            }
        else:
            olliePrint_simple(f"[WARNING] Enhanced research reached max iterations", level='warning')
            final_findings = f"Enhanced research inconclusive after {iteration_count} iterations. Task: {original_task}."
            session.final_findings = final_findings
            final_result = {
                'success': False,
                'findings': final_findings,
                'reason': 'max_iterations_reached',
                'enhancement': 'sequential_fresh_context'
            }

        # Store final results with enhanced metadata
        final_result.update({
            'task_id': task_id,
            'original_task': original_task,
            'conversation_path': str(session.conversation_dir),
            'iterations': iteration_count,
            'vet_reports_count': len(vet_reports_for_rag),
            'global_citations_count': len(global_citation_db)
        })
        
        # Save enhanced research summary
        try:
            summary_file = session.conversation_dir / "enhanced_research_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            findings_file = session.conversation_dir / "enhanced_research_findings.txt"
            with open(findings_file, 'w', encoding='utf-8') as f:
                f.write(f"Enhanced Research Task: {original_task}\n")
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Success: {final_result.get('success', False)}\n")
                f.write(f"Iterations: {iteration_count}\n")
                f.write(f"Enhancement: Sequential Fresh Context Processing\n")
                f.write("=" * 80 + "\n")
                f.write("FINAL ENHANCED RESEARCH FINDINGS:\n")
                f.write("=" * 80 + "\n")
                f.write(final_result.get('findings', 'No findings available'))
        except Exception as e:
            olliePrint_simple(f"Failed to save enhanced research reports: {e}", level='error')
            
        return final_result

    except Exception as e:
        olliePrint_simple(f"Enhanced research system error: {e}", level='error')
        import traceback
        traceback.print_exc()
        
        error_result = {
            'success': False,
            'task_id': task_id,
            'findings': f"Enhanced research system error: {str(e)}",
            'conversation_path': str(session.conversation_dir) if session else None,
            'iterations': 0,
            'reason': 'system_error',
            'enhancement': 'sequential_fresh_context'
        }
        return error_result
    
    finally:
        # UNIFIED RESOURCE MANAGEMENT: Single cleanup call handles everything
        if session and task_id:
            resource_manager.cleanup_session(task_id)
        
        # Additional global cleanup for thoroughness
        resource_manager.cleanup_connections()

def _log_synthesis_event(conversation_path: str, event_type: str, data: dict):
    """Log structured synthesis events for debugging and monitoring."""
    try:
        events_log = Path(conversation_path) / "research_events.jsonl"
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        with open(events_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception as e:
        olliePrint_simple(f"Failed to log synthesis event: {e}", level='error')