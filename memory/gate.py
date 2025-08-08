import json
from datetime import datetime
from typing import Dict, List
import logging
import traceback
from config import config, ollama_manager
from utils import strip_think_tags

from agents.dispatcher import AgentDispatcher
import memory.L2_memory as L2  # noqa: F401

# Configure a real logger that prints to console
logger = logging.getLogger("G.A.T.E.")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
try:
    logger.setLevel(getattr(logging, getattr(config, "LOG_LEVEL", "INFO")))
except Exception:
    logger.setLevel(logging.INFO)


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
    logger.info("[G.A.T.E.] Multi-agent routing initiated...")

    try:
        recent_history_for_prompt: List[Dict] = []
        for turn in conversation_history[-config.GATE_MAX_CONVERSATION_MESSAGES :]:
            clean_content = strip_think_tags(turn["content"])
            if clean_content:
                recent_history_for_prompt.append(
                    {"role": turn["role"], "content": clean_content}
                )

        logger.debug(f"[G.A.T.E.] User Message Input: {user_message}")
        logger.debug(
            f"[G.A.T.E.] Recent History Input: {recent_history_for_prompt}"
        )
        
        print("\n" + "="*80)
        print("🎯 [G.A.T.E.] ROUTING INPUT")
        print("="*80)
        print(f"📝 USER MESSAGE: {user_message}")
        if recent_history_for_prompt:
            print(f"📚 CONTEXT HISTORY ({len(recent_history_for_prompt)} turns):")
            for i, turn in enumerate(recent_history_for_prompt):
                print(f"  Turn {i+1} [{turn['role']}]: {turn['content']}")
        else:
            print("📚 CONTEXT HISTORY: None")
        print("="*80 + "\n")

        routing_flags = _get_routing_flags(user_message, recent_history_for_prompt)
        
        print("\n" + "="*80)
        print("🎯 [G.A.T.E.] ROUTING ANALYSIS COMPLETE")
        print("="*80)
        print(f"📊 ROUTING FLAGS: {json.dumps(routing_flags, indent=2)}")
        print("="*80 + "\n")

        memory_context = ""
        if routing_flags.get("needs_memory", False):
            search_query = routing_flags.get("memory_search_query") or user_message
            try:
                import memory.L3_memory as L3

                logger.debug(
                    f"[G.A.T.E.] Performing memory search for query: '{search_query}'"
                )
                memory_results = L3.search_memory(query_text=search_query, limit=2)
                memory_context = json.dumps(
                    memory_results, ensure_ascii=False, default=str
                )
                logger.debug(
                    f"[G.A.T.E.] Retrieved {len(memory_results)} memory nodes"
                )
            except Exception as mem_err:
                logger.warning(
                    f"[G.A.T.E.] Memory search error: {mem_err}"
                )
                memory_context = ""

        logger.info(f"[G.A.T.E.] Routing flags: {routing_flags}")

        web_search_strategy = routing_flags.get("web_search_strategy", {})
        if web_search_strategy.get("needed", False):
            search_priority = web_search_strategy.get("search_priority", "quick")
            search_query = web_search_strategy.get("search_query", user_message)

            if search_priority == "research":
                return _handle_research_search(
                    search_query, user_message, conversation_history
                )
            elif search_priority == "thorough":
                search_results = _handle_thorough_search(search_query)
                if search_results and search_results.get("summary"):
                    memory_context += (
                        f"\n\nWeb Search Results:\n{search_results['summary']}"
                    )
            else:  # quick
                search_results = _handle_quick_search(search_query)
                if search_results and search_results.get("summary"):
                    memory_context += (
                        f"\n\nWeb Search Results:\n{search_results['summary']}"
                    )

        try:
            database_content = agent_dispatcher.dispatch_agents(
                routing_flags=routing_flags,
                user_message=user_message,
                conversation_history=conversation_history,
                visual_context=visual_context,
                memory_context=memory_context,
            )
        except Exception as dispatch_err:
            logger.error(f"[G.A.T.E.] Agent dispatch error: {dispatch_err}")
            print(traceback.format_exc())
            raise

        logger.debug(
            f"[G.A.T.E.] Final Neural Processing Core Content: {database_content}"
        )
        return database_content

    except Exception as e:
        logger.error(
            f"[G.A.T.E.] Critical failure during routing: {e}. Using fallback."
        )
        print(traceback.format_exc())
        return _generate_fallback_database(user_message)


def _get_routing_flags(user_message: str, recent_history: list) -> dict:
    """Get routing flags from G.A.T.E. LLM analysis."""
    try:
        # Local import to avoid hard dependency if logging utility changes
        try:
            from ollie_print import log_model_io
        except Exception:
            log_model_io = None
        formatted_recent_history = "\n".join(
            [f"{msg['role'].title()}: {msg['content']}" for msg in recent_history]
        )


        # Avoid Python str.format interpreting JSON braces in the template.
        # Replace only our known placeholders to prevent KeyErrors (e.g., on "needs_memory").
        prompt_template = config.GATE_USER_PROMPT
        prompt = (
            prompt_template
            .replace("{user_query}", user_message)
            .replace("{recent_history}", formatted_recent_history)
        )

        messages = [
            {"role": "system", "content": config.GATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        print("\n" + "="*80)
        print("🎯 [G.A.T.E.] LLM PROMPT")
        print("="*80)
        print(f"📝 FULL USER PROMPT:\n{prompt}")
        print("="*80 + "\n")

        response = ollama_manager.chat_concurrent_safe(
            model=config.GATE_OLLAMA_MODEL,
            messages=messages,
            options=config.LLM_GENERATION_OPTIONS,
            format="json",
            stream=False,
        )

        # Model I/O logging
        if 'log_model_io' in locals() and callable(log_model_io):
            try:
                log_model_io(str(config.GATE_OLLAMA_MODEL), messages, response)
            except Exception:
                pass

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
            logger.warning(
                "[G.A.T.E.] LLM response missing usable content – using default routing flags"
            )
            return _get_default_routing_flags()

        print("\n" + "="*80)
        print("🎯 [G.A.T.E.] RAW LLM RESPONSE")
        print("="*80)
        print(f"🤖 FULL RESPONSE:\n{response_content}")
        print("="*80 + "\n")

        try:
            routing_flags = json.loads(response_content)
            routing_flags = _normalize_and_fill_defaults(routing_flags)
            return routing_flags
        except json.JSONDecodeError as e:
            logger.warning(
                f"[G.A.T.E.] JSON parse error: {e}. Using default flags."
            )
            return _get_default_routing_flags()

    except Exception as e:
        logger.error(
            f"[G.A.T.E.] Routing analysis error: {e}. Using default flags."
        )
        print(traceback.format_exc())
        return _get_default_routing_flags()


def _handle_research_search(
    search_query: str, user_message: str, conversation_history: list
) -> str:
    """Route research search requests to the Agenda System for background research."""
    try:
        import memory.agenda_system as agenda

        research_task = f"Conduct comprehensive web research on: {search_query}"
        task_id = agenda.add_task_to_agenda(research_task, priority=2)

        if task_id:
            logger.info(
                f"[G.A.T.E.] Research search enqueued in Agenda System (task_id={task_id})"
            )
            return (
                f"I've queued a deep background research task (ID: {task_id}) for '{search_query}'. "
                "I'll share the comprehensive findings once they are ready."
            )
        else:
            logger.warning(
                "[G.A.T.E.] Failed to enqueue research search; falling back to thorough search"
            )
            search_results = _handle_thorough_search(search_query)
            return f"Research queue unavailable. Conducted thorough search instead: {search_results.get('summary', '')}"

    except Exception as e:
        logger.error(
            f"[G.A.T.E.] Error enqueuing research search: {e}"
        )
        search_results = _handle_thorough_search(search_query)
        return f"Research queue error. Conducted thorough search instead: {search_results.get('summary', '')}"


def _handle_thorough_search(search_query: str) -> dict:
    """Handle thorough web searches using intelligent_search with comprehensive processing."""
    try:
        from web_search_core import intelligent_search

        logger.info(f"[G.A.T.E.] Executing thorough web search: {search_query}")

        search_results = intelligent_search(
            query=search_query, search_priority="thorough", mode="auto"
        )

        return search_results

    except Exception as e:
        logger.error(f"[G.A.T.E.] Error in thorough search: {e}")
        return {
            "query": search_query,
            "summary": f"Thorough search error: {str(e)}",
            "links": [],
            "extracted_content": [],
        }


def _handle_quick_search(search_query: str) -> dict:
    """Handle quick web searches using the new intelligent_search function."""
    try:
        from web_search_core import intelligent_search

        logger.info(f"[G.A.T.E.] Executing quick web search: {search_query}")

        search_results = intelligent_search(
            query=search_query, search_priority="quick", mode="auto"
        )

        return search_results

    except Exception as e:
        logger.error(f"[G.A.T.E.] Error in quick search: {e}")
        return {
            "query": search_query,
            "summary": f"Web search error: {str(e)}",
            "links": [],
            "extracted_content": [],
        }


def _get_default_routing_flags() -> dict:
    """Get default routing flags when analysis fails."""
    return {
        "needs_memory": False,
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
