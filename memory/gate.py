import json
from datetime import datetime
from typing import Dict, List
from ollie_print import olliePrint_simple
from config import config, ollama_manager
from utils import strip_think_tags

from agents.dispatcher import AgentDispatcher
import memory.L2_memory as L2  # noqa: F401


def run_gate_analysis(
    user_message: str,
    conversation_history: list,
    agent_dispatcher: AgentDispatcher,
    visual_context: str = "",
) -> str:
    """
    Runs the new G.A.T.E. routing system with multi-agent dispatch.
    1. Analyzes user query to determine routing flags
    2. Dispatches appropriate agents based on flags
    3. Returns synthesized NEURAL PROCESSING CORE from S.Y.N.A.P.S.E.
    """
    olliePrint_simple("[G.A.T.E.] Multi-agent routing initiated...")

    try:
        recent_history_for_prompt: List[Dict] = []
        for turn in conversation_history[-config.GATE_MAX_CONVERSATION_MESSAGES :]:
            clean_content = strip_think_tags(turn["content"])
            if clean_content:
                recent_history_for_prompt.append(
                    {"role": turn["role"], "content": clean_content}
                )

        olliePrint_simple(
            f"[G.A.T.E.] User Message Input: {user_message}", level="debug"
        )
        olliePrint_simple(
            f"[G.A.T.E.] Recent History Input: {recent_history_for_prompt}",
            level="debug",
        )

        routing_flags = _get_routing_flags(user_message, recent_history_for_prompt)

        memory_context = ""
        if routing_flags.get("needs_memory", False):
            search_query = routing_flags.get("memory_search_query") or user_message
            try:
                import memory.L3_memory as L3

                olliePrint_simple(
                    f"[G.A.T.E.] Performing memory search for query: '{search_query}'",
                    level="debug",
                )
                memory_results = L3.search_memory(query_text=search_query, limit=2)
                memory_context = json.dumps(
                    memory_results, ensure_ascii=False, default=str
                )
                olliePrint_simple(
                    f"[G.A.T.E.] Retrieved {len(memory_results)} memory nodes",
                    level="debug",
                )
            except Exception as mem_err:
                olliePrint_simple(
                    f"[G.A.T.E.] Memory search error: {mem_err}", level="warning"
                )
                memory_context = ""

        olliePrint_simple(f"[G.A.T.E.] Routing flags: {routing_flags}")

        web_search_strategy = routing_flags.get("web_search_strategy", {})
        if web_search_strategy.get("needed", False):
            search_priority = web_search_strategy.get("search_priority", "quick")
            search_query = web_search_strategy.get("search_query", user_message)

            if search_priority == "thorough":
                return _handle_thorough_search(
                    search_query, user_message, conversation_history
                )
            else:
                search_results = _handle_quick_search(search_query)
                if search_results and search_results.get("summary"):
                    memory_context += (
                        f"\n\nWeb Search Results:\n{search_results['summary']}"
                    )

        database_content = agent_dispatcher.dispatch_agents(
            routing_flags=routing_flags,
            user_message=user_message,
            conversation_history=conversation_history,
            visual_context=visual_context,
            memory_context=memory_context,
        )

        olliePrint_simple(
            f"[G.A.T.E.] Final Neural Processing Core Content: {database_content}",
            level="debug",
        )
        return database_content

    except Exception as e:
        olliePrint_simple(
            f"[G.A.T.E.] Critical failure during routing: {e}. Using fallback.",
            level="error",
        )
        return _generate_fallback_database(user_message)


def _get_routing_flags(user_message: str, recent_history: list) -> dict:
    """Get routing flags from G.A.T.E. LLM analysis."""
    try:
        formatted_recent_history = "\n".join(
            [f"{msg['role'].title()}: {msg['content']}" for msg in recent_history]
        )

        prompt = config.GATE_USER_PROMPT.format(
            user_query=user_message, recent_history=formatted_recent_history
        )

        messages = [
            {"role": "system", "content": config.GATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = ollama_manager.chat_concurrent_safe(
            model=config.GATE_OLLAMA_MODEL,
            messages=messages,
            options=config.LLM_GENERATION_OPTIONS,
            format="json",
            stream=False,
        )

        if not response:
            olliePrint_simple(
                "[G.A.T.E.] Null or empty LLM response – using default routing flags",
                level="warning",
            )
            return _get_default_routing_flags()

        def _normalize_and_fill_defaults(flags: dict) -> dict:
            required_keys = ["needs_memory", "needs_pi_tools"]
            for key in required_keys:
                if key not in flags:
                    flags[key] = False
            if "web_search_strategy" not in flags or not isinstance(
                flags.get("web_search_strategy"), dict
            ):
                flags["web_search_strategy"] = {
                    "needed": False,
                    "search_priority": "quick",
                    "search_query": "",
                }
            return flags

        if isinstance(response, dict) and any(
            key in response for key in ["needs_memory", "needs_pi_tools", "web_search_strategy"]
        ) and "message" not in response:
            routing_flags = _normalize_and_fill_defaults(response)
            olliePrint_simple("\n┏━━[G.A.T.E.] Raw Routing Response (direct JSON) ━━")
            olliePrint_simple(json.dumps(routing_flags, ensure_ascii=False))
            olliePrint_simple("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            return routing_flags

        if isinstance(response, dict):
            message_dict = response.get("message")
        else:
            message_dict = getattr(response, "message", None)

        response_content = None

        if isinstance(message_dict, dict):
            response_content = (message_dict.get("content") or "").strip()
        elif isinstance(response, str):
            response_content = response.strip()

        if not response_content:
            olliePrint_simple(
                "[G.A.T.E.] LLM response missing usable content – using default routing flags",
                level="warning",
            )
            return _get_default_routing_flags()

        olliePrint_simple("\n┏━━[G.A.T.E.] Raw Routing Response ━━")
        olliePrint_simple(response_content)
        olliePrint_simple("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        try:
            routing_flags = json.loads(response_content)
            routing_flags = _normalize_and_fill_defaults(routing_flags)
            return routing_flags
        except json.JSONDecodeError as e:
            olliePrint_simple(
                f"[G.A.T.E.] JSON parse error: {e}. Using default flags.",
                level="warning",
            )
            return _get_default_routing_flags()

    except Exception as e:
        olliePrint_simple(
            f"[G.A.T.E.] Routing analysis error: {e}. Using default flags.",
            level="error",
        )
        return _get_default_routing_flags()


def _handle_thorough_search(
    search_query: str, user_message: str, conversation_history: list
) -> str:
    """Route thorough search requests to the Agenda System for background research."""
    try:
        import memory.agenda_system as agenda

        research_task = f"Conduct comprehensive web research on: {search_query}"
        task_id = agenda.add_task_to_agenda(research_task, priority=2)

        if task_id:
            olliePrint_simple(
                f"[G.A.T.E.] Thorough search enqueued in Agenda System (task_id={task_id})"
            )
            return (
                f"I've queued a thorough background research task (ID: {task_id}) for '{search_query}'. "
                "I'll share the findings once they are ready."
            )
        else:
            olliePrint_simple(
                "[G.A.T.E.] Failed to enqueue thorough search; falling back to quick search",
                level="warning",
            )
            return _handle_quick_search(search_query)

    except Exception as e:
        olliePrint_simple(
            f"[G.A.T.E.] Error enqueuing thorough search: {e}", level="error"
        )
        return _handle_quick_search(search_query)


def _handle_quick_search(search_query: str) -> dict:
    """Handle quick web searches using the new intelligent_search function."""
    try:
        from web_search_core import intelligent_search

        olliePrint_simple(f"[G.A.T.E.] Executing quick web search: {search_query}")

        search_results = intelligent_search(
            query=search_query, search_priority="quick", mode="auto"
        )

        return search_results

    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Error in quick search: {e}", level="error")
        return {
            "query": search_query,
            "summary": f"Web search error: {str(e)}",
            "links": [],
            "extracted_content": [],
        }


def _get_default_routing_flags() -> dict:
    """Get default routing flags when analysis fails."""
    return {
        "needs_memory": True,
        "needs_pi_tools": False,
        "web_search_strategy": {
            "needed": False,
            "search_priority": "quick",
            "search_query": "",
        },
    }


def _generate_fallback_database(user_message: str) -> str:
    """Generate fallback NEURAL PROCESSING CORE when routing fails."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""(NEURAL PROCESSING CORE)
Your Internal Systems determined no memory was needed to answer your query.

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""
