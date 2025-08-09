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
        
        # Add M.A.D.'s own analysis history without thinking content
        for turn in self.analysis_history:
            messages.append({"role": turn['role'], "content": turn['content']})
        
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
        
        olliePrint_simple(f"--- M.A.D. Tool Calls ({len(tool_calls)}) ---", level='debug')
        print("\n" + "-"*80)
        print(f"🔧 [M.A.D.] Executing Tool Calls: {len(tool_calls)}")
        print("-"*80)
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            function_args_str = tool_call.get("function", {}).get("arguments", "{}")
            
            olliePrint_simple(f"[M.A.D.] Calling tool: {function_name}", level='debug')
            olliePrint_simple(f"[M.A.D.] Arguments: {function_args_str}", level='debug')
            try:
                preview = (function_args_str[:500] + "...") if len(function_args_str) > 500 else function_args_str
            except Exception:
                preview = str(function_args_str)
            print(f"➡️  Tool: {function_name}")
            print(f"   Args: {preview}")

            try:
                # The arguments are a JSON string, so we need to parse them
                function_args = json.loads(function_args_str)

                if function_name == "add_memory":
                    result = tool_add_memory(**function_args)
                elif function_name == "add_memory_with_observations":
                    result = tool_add_memory_with_observations(**function_args)
                else:
                    result = {"success": False, "error": f"Unknown tool: {function_name}"}
                
                olliePrint_simple(f"[M.A.D.] Tool Result: {result}", level='debug')
                status = "✅ success" if result.get("success") else "❌ failure"
                print(f"   Result: {status}")
                if not result.get("success") and result.get("error"):
                    err_preview = (str(result.get("error"))[:500] + "...") if len(str(result.get("error"))) > 500 else str(result.get("error"))
                    print(f"   Error: {err_preview}")

                results.append({
                    "tool": function_name,
                    "args": function_args,
                    "result": result
                })
            except Exception as e:
                olliePrint_simple(f"[M.A.D.] Tool call failed: {e}", level='error')
                print(f"   Result: ❌ failure")
                print(f"   Error: {str(e)}")
                results.append({
                    "tool": function_name,
                    "args": function_args_str, # Log the raw string on failure
                    "result": {"success": False, "error": str(e)}
                })
        olliePrint_simple("---------------------------", level='debug')
        print("-"*80 + "\n")
        
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
            print("\n" + "="*80)
            print("🧠 [M.A.D.] ANALYSIS START")
            print("="*80)
            print(f"⏱️  {datetime.now().isoformat()}")
            
            def truncate_for_log(text: str, limit: int = 800) -> str:
                try:
                    return (text[:limit] + "...") if len(text) > limit else text
                except Exception:
                    return str(text)
            
            print("— Current Turn to Analyze —")
            print(f"USER: {truncate_for_log(user_message)}")
            print(f"FRED: {truncate_for_log(fred_response)}")
            
            # Prepare context for M.A.D.
            messages = self._prepare_mad_context(user_message, fred_response, fred_context)
            
            # Get M.A.D. analysis
            print("\n🔮 Invoking model for M.A.D. analysis...")
            print(f"Model: {config.MAD_OLLAMA_MODEL}")
            print(f"Host: {config.OLLAMA_BASE_URL}")
            print(f"Messages: {len(messages)} | Tools: {len(self.mad_tools) if self.mad_tools else 0}")
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
            raw_preview = truncate_for_log(raw_content)
            print("\n📨 [M.A.D.] Model Response Received")
            if raw_preview:
                print(f"Content (preview): {raw_preview}")
            print(f"Tool Calls: {len(tool_calls)}")

            olliePrint_simple("--- M.A.D. Raw Response ---", level='debug')
            olliePrint_simple(raw_content, level='debug')
            olliePrint_simple("---------------------------", level='debug')
            
            # Extract thinking and clean content
            thinking = self._extract_think_content(raw_content)
            clean_content = self._strip_think_tags(raw_content)

            if thinking:
                olliePrint_simple("--- M.A.D. Thinking ---", level='debug')
                olliePrint_simple(thinking, level='debug')
                olliePrint_simple("------------------------", level='debug')
            
            # Add to M.A.D.'s analysis history
            analysis_turn = f"USER: {user_message}\nFRED: {fred_response}"
            with self._lock:
                self.analysis_history.append({
                    'role': 'user',
                    'content': analysis_turn
                })
                self.analysis_history.append({
                    'role': 'assistant',
                    'content': clean_content
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
                print(f"✅ [M.A.D.] Created {memory_count} memories | {execution_time:.2f}s")
            else:
                olliePrint_simple(f"[M.A.D.] No new memories needed (analysis: {execution_time:.2f}s)")
                print(f"ℹ️  [M.A.D.] No new memories needed | {execution_time:.2f}s")
            print("="*80 + "\n")
            
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
            print(f"❌ [M.A.D.] Analysis failed: {str(e)}")
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
