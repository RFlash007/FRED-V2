"""
Agent Dispatch System
Manages parallel execution of FRED agents according to G.A.T.E. routing decisions
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Callable
from ollie_print import olliePrint_simple
from config import config

from .scout import ScoutAgent
from .remind import RemindAgent
from .pivot import PivotAgent
from .synapse import SynapseAgent
try:
    import memory.crap as crap
except ImportError:
    crap = None

class AgentDispatcher:
    """Manages parallel execution of FRED agents."""
    
    def __init__(self):
        self.max_concurrent = getattr(config, 'MAX_CONCURRENT_AGENTS', 1)
        self.timeout_seconds = 30  # Agent timeout
        
        self.agents = {
            'scout': ScoutAgent(),
            'remind': RemindAgent(),
            'pivot': PivotAgent(),
            'synapse': SynapseAgent()
        }
        
        olliePrint_simple(f"[DISPATCHER] Initialized with {len(self.agents)} agents, max concurrent: {self.max_concurrent}")
    
    def dispatch_agents(self, routing_flags: Dict, user_message: str, conversation_history: List[Dict], visual_context: str = "") -> str:
        """
        Dispatch agents based on G.A.T.E. routing flags and return synthesized FRED DATABASE.
        
        Args:
            routing_flags: Dict with needs_memory, needs_web_search, needs_deep_research, needs_pi_tools, needs_reminders
            user_message: Current user message
            conversation_history: Full conversation history
            visual_context: Visual context from Pi glasses if available
        
        Returns:
            Formatted FRED DATABASE content
        """
        try:
            olliePrint_simple(f"[DISPATCHER] Dispatching agents for flags: {routing_flags}")
            
            agent_tasks = self._plan_agent_execution(routing_flags, user_message, conversation_history, visual_context)
            
            if not agent_tasks:
                olliePrint_simple("[DISPATCHER] No agents needed, using fallback")
                return self._generate_fallback_database(user_message, visual_context)
            
            agent_outputs = self._execute_agents_parallel(agent_tasks)
            
            l2_summaries = self._get_l2_context(user_message)
            
            synapse_result = self.agents['synapse'].synthesize_context(
                agent_outputs=agent_outputs,
                l2_summaries=l2_summaries,
                user_query=user_message,
                visual_context=visual_context
            )
            
            return synapse_result
            
        except Exception as e:
            olliePrint_simple(f"[DISPATCHER] Critical error: {e}", level='error')
            return self._generate_fallback_database(user_message, visual_context)
    
    def _plan_agent_execution(self, routing_flags: Dict, user_message: str, conversation_history: List[Dict], visual_context: str) -> List[Tuple[str, Callable]]:
        """Plan which agents to execute based on routing flags."""
        tasks = []
        
        if routing_flags.get('needs_memory', False):
            def run_crap():
                return self._run_crap_agent(user_message, conversation_history)
            tasks.append(('crap', run_crap))
        
        if routing_flags.get('needs_web_search', False):
            def run_scout():
                return self.agents['scout'].search_and_assess(user_message)
            tasks.append(('scout', run_scout))
        
        if routing_flags.get('needs_reminders', True):  # Default to True
            def run_remind():
                return self.agents['remind'].process_conversation_turn(user_message)
            tasks.append(('remind', run_remind))
        
        if routing_flags.get('needs_pi_tools', False):
            pi_command, pi_params = self._extract_pi_command(user_message)
            if pi_command:
                def run_pivot():
                    return self.agents['pivot'].process_pi_command(pi_command, pi_params)
                tasks.append(('pivot', run_pivot))
        
        if routing_flags.get('needs_deep_research', False):
            olliePrint_simple("[DISPATCHER] Deep research flagged - will be added to agenda")
        
        return tasks
    
    def _execute_agents_parallel(self, agent_tasks: List[Tuple[str, Callable]]) -> Dict:
        """Execute agent tasks in parallel with timeout handling."""
        agent_outputs = {}
        
        if self.max_concurrent == 1:
            for agent_name, task_func in agent_tasks:
                try:
                    olliePrint_simple(f"[DISPATCHER] Running {agent_name}...")
                    start_time = time.time()
                    result = task_func()
                    duration = time.time() - start_time
                    
                    agent_outputs[agent_name] = result
                    olliePrint_simple(f"[DISPATCHER] {agent_name} completed in {duration:.2f}s")
                    
                except Exception as e:
                    olliePrint_simple(f"[DISPATCHER] {agent_name} failed: {e}", level='error')
                    agent_outputs[agent_name] = {"error": str(e)}
        else:
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                future_to_agent = {
                    executor.submit(task_func): agent_name 
                    for agent_name, task_func in agent_tasks
                }
                
                for future in as_completed(future_to_agent, timeout=self.timeout_seconds):
                    agent_name = future_to_agent[future]
                    try:
                        result = future.result()
                        agent_outputs[agent_name] = result
                        olliePrint_simple(f"[DISPATCHER] {agent_name} completed")
                    except Exception as e:
                        olliePrint_simple(f"[DISPATCHER] {agent_name} failed: {e}", level='error')
                        agent_outputs[agent_name] = {
                            "error": config.AGENT_ERRORS.get(f"{agent_name}_failure", f"{agent_name} failed")
                        }
        
        return agent_outputs
    
    def _run_crap_agent(self, user_message: str, conversation_history: List[Dict]) -> Dict:
        """Run C.R.A.P. agent and return structured output."""
        try:
            if crap is None:
                return {"error": "C.R.A.P. module not available"}
            
            crap_result = crap.run_crap_analysis(user_message, conversation_history)
            
            if isinstance(crap_result, str) and "(FRED DATABASE)" in crap_result:
                start = crap_result.find("(FRED DATABASE)")
                end = crap_result.find("(END FRED DATABASE)")
                if start != -1 and end != -1:
                    database_content = crap_result[start:end + len("(END FRED DATABASE)")]
                    return {"memories": database_content, "raw_output": crap_result}
            
            return {"memories": crap_result, "raw_output": crap_result}
            
        except Exception as e:
            olliePrint_simple(f"[DISPATCHER] C.R.A.P. execution error: {e}", level='error')
            return {"error": config.AGENT_ERRORS.get("memory_failure", "Memory system failed")}
    
    def _extract_pi_command(self, user_message: str) -> Tuple[Optional[str], Dict]:
        """Extract Pi command from user message (simplified implementation)."""
        message_lower = user_message.lower()
        
        if any(phrase in message_lower for phrase in ["this is", "my name is", "meet", "introduce"]):
            import re
            name_patterns = [
                r"this is (\w+)",
                r"my name is (\w+)",
                r"meet (\w+)",
                r"i'm (\w+)"
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    name = match.group(1).title()
                    return "enroll_person", {"name": name}
        
        return None, {}
    
    def _get_l2_context(self, user_message: str) -> List[Dict]:
        """Get L2 memory context for synthesis."""
        try:
            import memory.L2_memory as L2
            l2_context = L2.query_l2_context(user_message)
            
            if l2_context:
                return [{"summary": l2_context}]
            
            return []
            
        except Exception as e:
            olliePrint_simple(f"[DISPATCHER] L2 context error: {e}", level='error')
            return []
    
    def _generate_fallback_database(self, user_message: str, visual_context: str = "") -> str:
        """Generate fallback FRED DATABASE when agent dispatch fails."""
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""(FRED DATABASE)
• Processing your query: {user_message[:100]}...
• My agent systems are working to gather information
• Putting it together... ready to help with what I know

SYSTEM STATUS:
The current time is: {current_time}
(END FRED DATABASE)"""
