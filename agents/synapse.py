"""
S.Y.N.A.P.S.E. (Synthesis & Yielding Neural Analysis for Prompt Structure Enhancement)
Generates the "Fleeting Thoughts" FRED DATABASE section from agent outputs
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from ollie_print import olliePrint_simple
from config import config, ollama_manager

class SynapseAgent:
    """S.Y.N.A.P.S.E. agent for context synthesis and FRED DATABASE generation."""
    
    def __init__(self):
        self.name = "S.Y.N.A.P.S.E."
        self.max_bullets = getattr(config, 'SYNAPSE_MAX_BULLETS', 8)
    
    def synthesize_context(self, agent_outputs: Dict, l2_summaries: List[Dict], user_query: str, visual_context: str = "") -> str:
        """
        Synthesize all agent outputs into FRED DATABASE format.
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
            
            database_content = self._format_fred_database(thoughts_content, visual_context)
            
            olliePrint_simple(f"[{self.name}] Generated {len(thoughts_content.split('•'))-1} thought bullets")
            
            return database_content
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Synthesis error: {e}", level='error')
            return self._generate_fallback_database(user_query, visual_context)
    
    def _get_system_prompt(self) -> str:
        """Get S.Y.N.A.P.S.E. system prompt."""
        return f"""You are S.Y.N.A.P.S.E., F.R.E.D.'s internal thought synthesis system.

Your job is to create "Fleeting Thoughts" - bullet points that read like F.R.E.D.'s own passing thoughts and observations. These thoughts should feel natural and human-like, as if F.R.E.D. is recalling memories, processing information, and making connections.

GUIDELINES:
- Write in first person as F.R.E.D.
- Keep bullets concise but insightful
- Include recalled memories, web insights, reminders, and observations
- Make connections between different pieces of information
- The final bullet must ALWAYS be "Putting it together..." with a summary insight
- Maximum {self.max_bullets} bullets total
- Sound natural and conversational, not robotic

FORMAT:
• [Thought about memory/context]
• [Insight from web search]
• [Reminder or observation]
• [Connection or pattern]
• Putting it together... [overall insight]

The thoughts should feel like F.R.E.D.'s internal monologue as he processes the user's query."""
    
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
    
    def _format_fred_database(self, thoughts_content: str, visual_context: str = "") -> str:
        """Format the thoughts into proper FRED DATABASE structure."""
        
        thoughts_lines = []
        for line in thoughts_content.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-')):
                if not line.startswith('•'):
                    line = '•' + line[1:]
                thoughts_lines.append(line)
        
        if thoughts_lines and not any("putting it together" in line.lower() for line in thoughts_lines):
            thoughts_lines.append("• Putting it together... processing the available information to provide a helpful response.")
        
        if len(thoughts_lines) > self.max_bullets:
            thoughts_lines = thoughts_lines[:self.max_bullets-1]
            thoughts_lines.append("• Putting it together... synthesizing the key insights from the available information.")
        
        thoughts_text = '\n'.join(thoughts_lines)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""(FRED DATABASE)
{thoughts_text}

SYSTEM STATUS:
The current time is: {current_time}
(END FRED DATABASE)"""
    
    def _generate_fallback_database(self, user_query: str, visual_context: str = "") -> str:
        """Generate a basic FRED DATABASE when synthesis fails."""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        fallback_content = f"""(FRED DATABASE)
• I'm processing your query about: {user_query[:100]}...
• My memory systems are working to find relevant information
• Putting it together... ready to help with what I know

SYSTEM STATUS:
The current time is: {current_time}
(END FRED DATABASE)"""
        
        return fallback_content
