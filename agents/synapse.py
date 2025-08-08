"""
S.Y.N.A.P.S.E. (Synthesis & Yielding Neural Analysis for Prompt Structure Enhancement)
Generates the "Fleeting Thoughts" NEURAL PROCESSING CORE section from agent outputs
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from ollie_print import olliePrint_simple, log_model_io
from config import config, ollama_manager


class SynapseAgent:
    """S.Y.N.A.P.S.E. agent for context synthesis and NEURAL PROCESSING CORE generation."""

    def __init__(self):
        self.name = "S.Y.N.A.P.S.E."
        self.max_bullets = getattr(config, "SYNAPSE_MAX_BULLETS", 8)

    def synthesize_context(
        self,
        agent_outputs: Dict,
        l2_summaries: List[Dict],
        user_query: str,
        visual_context: str = "",
    ) -> str:
        """
        Synthesize all agent outputs into NEURAL PROCESSING CORE format.
        Creates bullet points that read like F.R.E.D.'s fleeting thoughts.
        """
        try:
            print("\n" + "="*80)
            print(f"⚡ [S.Y.N.A.P.S.E.] SYNTHESIS INPUT")
            print("="*80)
            print(f"📝 USER QUERY: {user_query}")
            print(f"🧠 AGENT OUTPUTS ({len(agent_outputs)} agents):")
            for agent_name, output in agent_outputs.items():
                print(f"\n  ┌─ {agent_name.upper()} OUTPUT:")
                if isinstance(output, dict):
                    print(f"  │ {json.dumps(output, indent=4).replace(chr(10), chr(10) + '  │ ')}")
                else:
                    print(f"  │ {str(output)}")
                print(f"  └─ END {agent_name.upper()}")
            
            if l2_summaries:
                print(f"\n💭 L2 SUMMARIES ({len(l2_summaries)} entries):")
                for i, summary in enumerate(l2_summaries):
                    print(f"  [{i+1}] {summary}")
            else:
                print("\n💭 L2 SUMMARIES: None")
            
            if visual_context:
                print(f"\n👁️ VISUAL CONTEXT: {visual_context}")
            
            print("="*80 + "\n")
            olliePrint_simple(
                f"[{self.name}] Synthesizing context from {len(agent_outputs)} agents..."
            )

            synthesis_prompt = self._build_synthesis_prompt(
                agent_outputs, l2_summaries, user_query, visual_context
            )

            # Synthesis prompt debug (keeping internal for now)
            messages_payload = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": synthesis_prompt},
            ]
            response = ollama_manager.chat_concurrent_safe(
                model=config.SYNAPSE_OLLAMA_MODEL,
                messages=messages_payload,
                stream=False,
                options=config.LLM_GENERATION_OPTIONS,
            )

            # Model I/O logging
            try:
                log_model_io(str(config.SYNAPSE_OLLAMA_MODEL), messages_payload, response)
            except Exception:
                pass

            thoughts_content = response.get("message", {}).get("content", "")

            database_content = self._format_neural_core(
                thoughts_content, l2_summaries, visual_context
            )

            print("\n" + "="*80)
            print(f"⚡ [S.Y.N.A.P.S.E.] SYNTHESIS OUTPUT")
            print("="*80)
            thoughts_count = len([line for line in thoughts_content.split('\n') if line.strip().startswith('•')])
            print(f"💡 THOUGHT BULLETS GENERATED: {thoughts_count}")
            print(f"📊 DATABASE LENGTH: {len(database_content)} characters")
            print(f"\n🤖 RAW THOUGHTS CONTENT:\n{thoughts_content}")
            print(f"\n📊 FULL NEURAL PROCESSING CORE:\n{database_content}")
            print("="*80 + "\n")

            return database_content

        except Exception as e:
            olliePrint_simple(f"[{self.name}] Synthesis error: {e}", level="error")
            return self._generate_fallback_database(
                user_query, l2_summaries, visual_context
            )

    def _get_system_prompt(self) -> str:
        """Get S.Y.N.A.P.S.E. system prompt from config."""
        return config.SYNAPSE_SYSTEM_PROMPT.format(max_bullets=self.max_bullets)

    def _build_synthesis_prompt(
        self,
        agent_outputs: Dict,
        l2_summaries: List[Dict],
        user_query: str,
        visual_context: str,
    ) -> str:
        """Build the synthesis prompt with all available context."""

        prompt_parts = [f"USER QUERY: {user_query}"]

        if agent_outputs.get("memory"):
            mem_data = agent_outputs["memory"]
            if isinstance(mem_data, dict) and mem_data.get("memories"):
                prompt_parts.append(f"RETRIEVED KNOWLEDGE:\n{mem_data['memories']}")

        if agent_outputs.get("pivot"):
            pivot_data = agent_outputs["pivot"]
            if isinstance(pivot_data, dict) and pivot_data.get("result"):
                prompt_parts.append(f"PI OPERATION: {pivot_data['result']}")

        if visual_context:
            prompt_parts.append(f"VISUAL CONTEXT: {visual_context}")

        return "\n\n".join(prompt_parts)

    def _format_neural_core(
        self,
        thoughts_content: str,
        l2_summaries: Optional[List[Dict]] = None,
        visual_context: str = "",
    ) -> str:
        """Format the thoughts into proper NEURAL PROCESSING CORE structure."""
        # Collect candidate bullet lines from model output
        candidate_lines: List[str] = []
        for raw in thoughts_content.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("•") or line.startswith("-"):
                # Normalize dash bullets to dot bullets
                if not line.startswith("•"):
                    line = "•" + line[1:]
                candidate_lines.append(line)

        # Pass-through: no deduplication, trimming, or recall enforcement
        thoughts_lines: List[str] = []
        for line in candidate_lines:
            content_only = line[1:].strip()
            if content_only:
                thoughts_lines.append(f"• {content_only}")

        thoughts_text = "\n".join(thoughts_lines)

        # Build fleeting thoughts from L2 summaries
        fleeting_lines: List[str] = []
        if l2_summaries:
            seen_fleeting = set()
            for summary in l2_summaries:
                text = summary.get("summary", "").strip()
                if not text:
                    continue
                # Pass-through: no trimming or recall enforcement
                if not text.startswith("•"):
                    text = f"• {text}"
                fleeting_lines.append(text)
        fleeting_text = "\n".join(fleeting_lines)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""(NEURAL PROCESSING CORE)
{thoughts_text}

<FLEETING THOUGHTS>
{fleeting_text}
</FLEETING THOUGHTS>

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""

    def _generate_fallback_database(
        self, user_query: str, l2_summaries: List[Dict], visual_context: str = ""
    ) -> str:
        """Generate a basic NEURAL PROCESSING CORE when synthesis fails."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build fleeting thoughts from L2 summaries (if any)
        fleeting_lines = []
        if l2_summaries:
            for summary in l2_summaries:
                text = summary.get("summary", "").strip()
                if text:
                    if not text.startswith("•"):
                        text = f"• {text}"
                    fleeting_lines.append(text)
        fleeting_text = "\n".join(fleeting_lines)

        fallback_content = f"""(NEURAL PROCESSING CORE)
Unable to fully synthesize thoughts; providing minimal context.

<FLEETING THOUGHTS>
{fleeting_text}
</FLEETING THOUGHTS>

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""

        return fallback_content
