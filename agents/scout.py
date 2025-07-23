"""
S.C.O.U.T. (Search & Confidence Optimization Utility Tool)
Performs quick web searches with confidence scoring and auto-escalation
"""

import json
from typing import Dict, Optional, Tuple
from ollie_print import olliePrint_simple
from config import config, ollama_manager
from Tools import handle_tool_calls, tool_add_task_to_agenda

class ScoutAgent:
    """S.C.O.U.T. agent for web search with confidence assessment."""
    
    def __init__(self):
        self.name = "S.C.O.U.T."
        self.confidence_threshold = getattr(config, 'SCOUT_CONFIDENCE_THRESHOLD', 70)
    
    def search_and_assess(self, query: str, context: str = "") -> Dict:
        """
        Perform web search and assess confidence in the answer.
        Returns dict with search_result, confidence_score, and escalation_needed.
        """
        try:
            olliePrint_simple(f"[{self.name}] Searching: {query[:100]}...")
            
            search_tool_call = [{
                "function": {
                    "name": "search_general",
                    "arguments": {"query": query}
                }
            }]
            
            search_results = handle_tool_calls(search_tool_call)
            
            if not search_results or not search_results[0].get('content'):
                return {
                    "search_result": "Search failed",
                    "confidence_score": 0,
                    "escalation_needed": True,
                    "error": config.AGENT_ERRORS.get("search_failure", "Search failed")
                }
            
            search_content = search_results[0]['content']
            
            confidence_score = self._assess_confidence(query, search_content, context)
            
            escalation_needed = confidence_score < self.confidence_threshold
            
            if escalation_needed:
                olliePrint_simple(f"[{self.name}] Low confidence ({confidence_score}%), escalating to deep research")
                
                # Directly call the centralized tool_add_task_to_agenda function
                tool_add_task_to_agenda(
                    task_description=f"Deep research required: {query}",
                    priority=1
                )
            
            return {
                "search_result": search_content,
                "confidence_score": confidence_score,
                "escalation_needed": escalation_needed,
                "query": query
            }
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Error: {e}", level='error')
            return {
                "search_result": "",
                "confidence_score": 0,
                "escalation_needed": True,
                "error": config.AGENT_ERRORS.get("search_failure", "Search failed")
            }
    
    def _assess_confidence(self, query: str, search_content: str, context: str) -> int:
        """Assess confidence in search results using LLM."""
        try:
            assessment_prompt = config.SCOUT_CONFIDENCE_PROMPT.format(
                query=query,
                context=context,
                search_content=search_content
            )
            
            response = ollama_manager.chat_concurrent_safe(
                model=config.LLM_DECISION_MODEL,
                messages=[{"role": "user", "content": assessment_prompt}],
                stream=False
            )
            
            confidence_text = response.get('message', {}).get('content', '0').strip()
            
            try:
                confidence_score = int(''.join(filter(str.isdigit, confidence_text)))
                return max(0, min(100, confidence_score))
            except:
                return 50  # Default moderate confidence
                
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Confidence assessment error: {e}", level='error')
            return 50
