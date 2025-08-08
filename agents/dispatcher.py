"""
Agent Dispatch System
Manages parallel execution of FRED agents according to G.A.T.E. routing decisions
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Callable
from config import config

from .pivot import PivotAgent
from .synapse import SynapseAgent


# No-op logger to disable all logging output in this module
class _NoOpLogger:
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass

logger = _NoOpLogger()

class AgentDispatcher:
    """Manages parallel execution of FRED agents."""

    def __init__(self):
        self.max_concurrent = getattr(config, "MAX_CONCURRENT_AGENTS", 1)
        self.timeout_seconds = 30  # Agent timeout

        # Remove persistent agent instances - create them per-request instead
        logger.info(
            f"[DISPATCHER] Initialized, max concurrent: {self.max_concurrent}"
        )

    def _create_agent(self, agent_type: str):
        """Create agent instance on-demand for proper memory management."""
        if agent_type == "pivot":
            return PivotAgent()
        elif agent_type == "synapse":
            return SynapseAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def dispatch_agents(
        self,
        routing_flags: Dict,
        user_message: str,
        conversation_history: List[Dict],
        visual_context: str = "",
        memory_context: str = "",
    ) -> str:
        """
        Dispatch agents based on G.A.T.E. routing flags and return synthesized NEURAL PROCESSING CORE.

        Args:
            routing_flags: Dict with needs_memory, web_search_strategy, needs_pi_tools
            user_message: Current user message
            conversation_history: Full conversation history
            visual_context: Visual context from Pi glasses if available

        Returns:
            Formatted NEURAL PROCESSING CORE content
        """
        try:
            logger.info(
                f"[DISPATCHER] Dispatching agents for flags: {routing_flags}"
            )

            agent_tasks = self._plan_agent_execution(
                routing_flags, user_message, conversation_history, visual_context
            )

            agent_outputs = {}

            # Execute agents if any are planned
            if agent_tasks:
                agent_outputs = self._execute_agents_parallel(agent_tasks)
            else:
                logger.info(
                    "[DISPATCHER] No agent tasks scheduled (memory-only request)"
                )

            # Inject memory context prepared by G.A.T.E. directly (new pathway)
            if memory_context:
                agent_outputs["memory"] = {"memories": memory_context}
                logger.debug(
                    "[DISPATCHER] Injected memory_context into agent_outputs"
                )
            elif not agent_tasks:
                # If still no context or agents, fallback
                logger.warning(
                    "[DISPATCHER] Neither agents nor memory context provided. Using fallback"
                )
                return self._generate_fallback_database(user_message, visual_context)

            l2_summaries = self._get_l2_context(user_message)

            # Create SYNAPSE agent for this request
            synapse_agent = self._create_agent("synapse")
            synapse_result = synapse_agent.synthesize_context(
                agent_outputs=agent_outputs,
                l2_summaries=l2_summaries,
                user_query=user_message,
                visual_context=visual_context,
            )

            # Explicitly clean up SYNAPSE agent
            del synapse_agent

            return synapse_result

        except Exception as e:
            logger.error(f"[DISPATCHER] Critical error: {e}")
            return self._generate_fallback_database(user_message, visual_context)

    def _plan_agent_execution(
        self,
        routing_flags: Dict,
        user_message: str,
        conversation_history: List[Dict],
        visual_context: str,
    ) -> List[Tuple[str, Callable]]:
        """Plan which agents to execute based on routing flags."""
        tasks = []

        # Web search now handled by new intelligent search system in Gate

        if routing_flags.get("needs_pi_tools", False):
            pi_command, pi_params = self._extract_pi_command(user_message)
            if pi_command:

                def run_pivot():
                    return self._create_agent("pivot").process_pi_command(
                        pi_command, pi_params
                    )

                tasks.append(("pivot", run_pivot))

        # Deep research flag deprecated; thorough searches are handled via
        # web_search_strategy in G.A.T.E.

        return tasks

    def _execute_agents_parallel(self, agent_tasks: List[Tuple[str, Callable]]) -> Dict:
        """Execute agent tasks in parallel with timeout handling."""
        agent_outputs = {}

        if self.max_concurrent == 1:
            for agent_name, task_func in agent_tasks:
                try:
                    logger.info(f"[DISPATCHER] Running {agent_name}...")
                    start_time = time.time()
                    result = task_func()
                    duration = time.time() - start_time

                    agent_outputs[agent_name] = result
                    logger.info(
                        f"[DISPATCHER] {agent_name} completed in {duration:.2f}s"
                    )

                except Exception as e:
                    logger.error(
                        f"[DISPATCHER] {agent_name} failed: {e}"
                    )
                    agent_outputs[agent_name] = {"error": str(e)}
        else:
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                future_to_agent = {
                    executor.submit(task_func): agent_name
                    for agent_name, task_func in agent_tasks
                }

                for future in as_completed(
                    future_to_agent, timeout=self.timeout_seconds
                ):
                    agent_name = future_to_agent[future]
                    try:
                        result = future.result()
                        agent_outputs[agent_name] = result
                        logger.info(f"[DISPATCHER] {agent_name} completed")
                    except Exception as e:
                        logger.error(
                            f"[DISPATCHER] {agent_name} failed: {e}"
                        )
                        agent_outputs[agent_name] = {
                            "error": config.AGENT_ERRORS.get(
                                f"{agent_name}_failure", f"{agent_name} failed"
                            )
                        }

        return agent_outputs

    def _extract_pi_command(self, user_message: str) -> Tuple[Optional[str], Dict]:
        """Extract Pi command from user message (simplified implementation)."""
        message_lower = user_message.lower()

        if any(
            phrase in message_lower
            for phrase in ["this is", "my name is", "meet", "introduce"]
        ):
            import re

            name_patterns = [
                r"this is (\w+)",
                r"my name is (\w+)",
                r"meet (\w+)",
                r"i'm (\w+)",
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
            logger.error(f"[DISPATCHER] L2 context error: {e}")
            return []

    def _generate_fallback_database(
        self, user_message: str, visual_context: str = ""
    ) -> str:
        """Generate fallback NEURAL PROCESSING CORE when agent dispatch fails."""
        from datetime import datetime

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""(NEURAL PROCESSING CORE)
Your internal system deemed no memories to be retrieved for this query. Answer normally.

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""
