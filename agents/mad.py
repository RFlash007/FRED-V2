"""
M.A.D. (Memory Addition Daemon) - Microagent for Memory Management
================================================================

Single-responsibility agent for detecting and storing new learnable information
from conversation turns. Runs parallel to G.A.T.E. on every user message.

Design Principles:
- Single responsibility: Only memory creation, no context retrieval
- Lightweight context: Own analysis history + minimal F.R.E.D. context
- Independent operation: No dependency on G.A.T.E. or other agents
- Graceful degradation: Failures don't block main conversation flow
"""

import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from ollie_print import olliePrint, olliePrint_simple
from config import config, ollama_manager
from Tools import tool_add_memory, tool_add_memory_with_observations
from prompts import MAD_SYSTEM_PROMPT
from tool_schemas import MAD_TOOLS


class MADAgent:
    """Memory Addition Daemon - Specialized agent for memory creation."""
    
    def __init__(self):
        self.analysis_history = []  # M.A.D.'s own analysis context
        self.max_analysis_history = 6  # Last 3 analysis turns (user + mad response)
        self._lock = threading.Lock()
        
        # Use centralized configurations
        self.mad_tools = MAD_TOOLS
        self.system_prompt = MAD_SYSTEM_PROMPT

    def _prepare_mad_context(self, user_message: str, fred_response: str, fred_context: List[Dict]) -> List[Dict]:
        """Prepare optimized context for M.A.D. analysis."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add M.A.D.'s own analysis history (with thinking truncation)
        for turn in self.analysis_history:
            if turn['role'] == 'user':
                messages.append({"role": "user", "content": turn['content']})
            elif turn['role'] == 'assistant':
                content = turn['content']
                thinking = turn.get('thinking', '')
                
                # Apply thinking truncation like main pipeline
                if thinking and len(self.analysis_history) > 6:  # Keep thinking for recent messages only
                    full_content = f"<think>\n{thinking}\n</think>\n{content}"
                    messages.append({"role": "assistant", "content": full_content})
                else:
                    messages.append({"role": "assistant", "content": content})
        
        # Add minimal F.R.E.D. context (last 2-3 messages)
        recent_fred_context = fred_context[-4:] if len(fred_context) > 4 else fred_context
        if recent_fred_context:
            context_summary = "Recent F.R.E.D. conversation context:\n"
            for turn in recent_fred_context:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')[:200] + "..." if len(turn.get('content', '')) > 200 else turn.get('content', '')
                context_summary += f"{role}: {content}\n"
            messages.append({"role": "user", "content": context_summary})
        
        # Add current turn for analysis
        current_turn = f"""Current conversation turn to analyze for new memories:

USER: {user_message}

FRED: {fred_response}

Analyze this turn: What new information about Ian or his interests should be stored in memory? What general knowledge is learned about the world that you do not inherently know? Add that to memory."""
        
        messages.append({"role": "user", "content": current_turn})
        
        return messages

    def _handle_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute M.A.D. tool calls (add_memory and add_memory_with_observations)."""
        results = []
        
        print(f"\n--- M.A.D. Tool Calls ({len(tool_calls)}) ---")
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            function_args_str = tool_call.get("function", {}).get("arguments", "{}")
            
            print(f"[M.A.D.] Calling tool: {function_name}")
            print(f"[M.A.D.] Arguments: {function_args_str}")

            try:
                # The arguments are a JSON string, so we need to parse them
                function_args = json.loads(function_args_str)

                if function_name == "add_memory":
                    result = tool_add_memory(**function_args)
                elif function_name == "add_memory_with_observations":
                    result = tool_add_memory_with_observations(**function_args)
                else:
                    result = {"success": False, "error": f"Unknown tool: {function_name}"}
                
                print(f"[M.A.D.] Tool Result: {result}")

                results.append({
                    "tool": function_name,
                    "args": function_args,
                    "result": result
                })
            except Exception as e:
                olliePrint_simple(f"[M.A.D.] Tool call failed: {e}", level='error')
                results.append({
                    "tool": function_name,
                    "args": function_args_str, # Log the raw string on failure
                    "result": {"success": False, "error": str(e)}
                })
        print("---------------------------\n")
        
        return results

    def _extract_think_content(self, text: str) -> str:
        """Extract thinking content from <think>...</think> tags."""
        if not text:
            return ""
        import re
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, text, re.DOTALL)
        return '\n'.join(matches).strip()

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from text."""
        if not text:
            return ""
        import re
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _truncate_analysis_history(self):
        """Truncate M.A.D.'s analysis history to maintain context window."""
        with self._lock:
            if len(self.analysis_history) > self.max_analysis_history:
                messages_to_remove = len(self.analysis_history) - self.max_analysis_history
                self.analysis_history = self.analysis_history[messages_to_remove:]

    def analyze_turn(self, user_message: str, fred_response: str, fred_context: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a conversation turn for new memory creation opportunities.
        
        Args:
            user_message: The user's message
            fred_response: F.R.E.D.'s response
            fred_context: Recent F.R.E.D. conversation context
            
        Returns:
            Dict containing analysis results and any created memories
        """
        try:
            start_time = time.time()
            
            # Prepare context for M.A.D.
            messages = self._prepare_mad_context(user_message, fred_response, fred_context)
            
            # Get M.A.D. analysis
            response = ollama_manager.chat_concurrent_safe(
                host=config.OLLAMA_BASE_URL,
                model=config.MAD_OLLAMA_MODEL,
                messages=messages,
                tools=self.mad_tools,
                stream=False,
                options=config.LLM_GENERATION_OPTIONS
            )
            
            response_message = response.get('message', {})
            raw_content = response_message.get('content', '')
            tool_calls = response_message.get('tool_calls', [])

            print("\n--- M.A.D. Raw Response ---")
            print(raw_content)
            print("---------------------------\n")
            
            # Extract thinking and clean content
            thinking = self._extract_think_content(raw_content)
            clean_content = self._strip_think_tags(raw_content)

            if thinking:
                print("\n--- M.A.D. Thinking ---")
                print(thinking)
                print("------------------------\n")
            
            # Add to M.A.D.'s analysis history
            analysis_turn = f"USER: {user_message}\nFRED: {fred_response}"
            with self._lock:
                self.analysis_history.append({
                    'role': 'user', 
                    'content': analysis_turn
                })
                self.analysis_history.append({
                    'role': 'assistant', 
                    'content': clean_content,
                    'thinking': thinking
                })
            
            # Truncate history if needed
            self._truncate_analysis_history()
            
            # Execute tool calls
            tool_results = []
            created_memories = []
            if tool_calls:
                tool_results = self._handle_tool_calls(tool_calls)
                created_memories = [r for r in tool_results if r.get('result', {}).get('success', False)]
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log results
            if created_memories:
                memory_count = len(created_memories)
                olliePrint_simple(f"[M.A.D.] Created {memory_count} memories in {execution_time:.2f}s")
            else:
                olliePrint_simple(f"[M.A.D.] No new memories needed (analysis: {execution_time:.2f}s)")
            
            return {
                'success': True,
                'analysis': clean_content,
                'thinking': thinking,
                'created_memories': created_memories,
                'tool_results': tool_results,
                'execution_time': execution_time
            }
            
        except Exception as e:
            olliePrint_simple(f"[M.A.D.] Analysis failed: {e}", level='error')
            return {
                'success': False,
                'error': str(e),
                'created_memories': [],
                'analysis': '',
                'thinking': '',
                'execution_time': 0
            }

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get M.A.D. analysis statistics for debugging."""
        with self._lock:
            return {
                'analysis_history_length': len(self.analysis_history),
                'max_analysis_history': self.max_analysis_history,
                'last_analysis_time': getattr(self, 'last_analysis_time', None)
            }


# Global M.A.D. instance
mad_agent = MADAgent()
