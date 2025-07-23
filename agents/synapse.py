"""
S.Y.N.A.P.S.E. (Synthesis & Yielding Neural Analysis for Prompt Structure Enhancement)
Generates the "Fleeting Thoughts" NEURAL PROCESSING CORE section from agent outputs
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from ollie_print import olliePrint_simple
from config import config, ollama_manager

class SynapseAgent:
    """S.Y.N.A.P.S.E. agent for context synthesis and NEURAL PROCESSING CORE generation."""
    
    def __init__(self):
        self.name = "S.Y.N.A.P.S.E."
        self.max_bullets = getattr(config, 'SYNAPSE_MAX_BULLETS', 8)
    
    def synthesize_context(self, agent_outputs: Dict, l2_summaries: List[Dict], user_query: str, visual_context: str = "") -> str:
        """
        Synthesize all agent outputs into NEURAL PROCESSING CORE format.
        Creates bullet points that read like F.R.E.D.'s fleeting thoughts.
        """
        try:
            olliePrint_simple(f"[{self.name}] Synthesizing context from {len(agent_outputs)} agents...")
            
            synthesis_prompt = self._build_synthesis_prompt(
                agent_outputs, l2_summaries, user_query, visual_context
            )
            
            response = ollama_manager.chat_concurrent_safe(
                model=config.LLM_DECISION_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": synthesis_prompt}
                ],
                stream=False
            )
            
            thoughts_content = response.get('message', {}).get('content', '')
            
            database_content = self._format_neural_core(thoughts_content, visual_context)
            
            olliePrint_simple(f"[{self.name}] Generated {len(thoughts_content.split('•'))-1} thought bullets")
            
            return database_content
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Synthesis error: {e}", level='error')
            return self._generate_fallback_database(user_query, visual_context)
    
    def _get_system_prompt(self) -> str:
        """Get S.Y.N.A.P.S.E. system prompt from config."""
        return config.SYNAPSE_SYSTEM_PROMPT.format(max_bullets=self.max_bullets)
    
    def _build_synthesis_prompt(self, agent_outputs: Dict, l2_summaries: List[Dict], user_query: str, visual_context: str) -> str:
        """Build the synthesis prompt with all available context."""
        
        prompt_parts = [f"USER QUERY: {user_query}"]
        
        if l2_summaries:
            memory_context = "\n".join([f"- {summary.get('summary', '')}" for summary in l2_summaries])
            prompt_parts.append(f"RECENT MEMORIES:\n{memory_context}")
        
        if agent_outputs.get('crap'):
            crap_data = agent_outputs['crap']
            if isinstance(crap_data, dict) and crap_data.get('memories'):
                prompt_parts.append(f"RETRIEVED KNOWLEDGE:\n{crap_data['memories']}")
        
        if agent_outputs.get('scout'):
            scout_data = agent_outputs['scout']
            if isinstance(scout_data, dict) and scout_data.get('search_result'):
                confidence = scout_data.get('confidence_score', 0)
                prompt_parts.append(f"WEB SEARCH (confidence: {confidence}%):\n{scout_data['search_result']}")
        
        if agent_outputs.get('remind'):
            remind_data = agent_outputs['remind']
            if isinstance(remind_data, dict):
                if remind_data.get('detected_reminders'):
                    reminders = remind_data['detected_reminders']
                    reminder_text = ", ".join([r.get('content', '') for r in reminders])
                    prompt_parts.append(f"NEW REMINDERS: {reminder_text}")
                
                if remind_data.get('active_reminders'):
                    active = remind_data['active_reminders']
                    active_text = ", ".join([r.get('content', '') for r in active])
                    prompt_parts.append(f"ACTIVE REMINDERS: {active_text}")
        
        if agent_outputs.get('pivot'):
            pivot_data = agent_outputs['pivot']
            if isinstance(pivot_data, dict) and pivot_data.get('result'):
                prompt_parts.append(f"PI OPERATION: {pivot_data['result']}")
        
        if visual_context:
            prompt_parts.append(f"VISUAL CONTEXT: {visual_context}")
        
        return "\n\n".join(prompt_parts)
    
    def _format_neural_core(self, thoughts_content: str, visual_context: str = "") -> str:
        """Format the thoughts into proper NEURAL PROCESSING CORE structure."""
        
        thoughts_lines = []
        for line in thoughts_content.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-')):
                if not line.startswith('•'):
                    line = '•' + line[1:]
                thoughts_lines.append(line)
        
        thoughts_text = '\n'.join(thoughts_lines)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""(NEURAL PROCESSING CORE)
{thoughts_text}

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""
    
    def _generate_fallback_database(self, user_query: str, visual_context: str = "") -> str:
        """Generate a basic NEURAL PROCESSING CORE when synthesis fails."""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        fallback_content = f"""(NEURAL PROCESSING CORE)
Your Neural Processing Core Deemed No Memories Needed.

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""
        
        return fallback_content
