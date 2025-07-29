
import json
from datetime import datetime
from ollie_print import olliePrint_simple
from config import config, ollama_manager
from utils import strip_think_tags

from agents.dispatcher import AgentDispatcher

def run_gate_analysis(user_message: str, conversation_history: list, agent_dispatcher: AgentDispatcher, visual_context: str = "") -> str:
    """
    Runs the new G.A.T.E. routing system with multi-agent dispatch.
    1. Analyzes user query to determine routing flags
    2. Dispatches appropriate agents based on flags
    3. Returns synthesized NEURAL PROCESSING CORE from S.Y.N.A.P.S.E.
    """
    olliePrint_simple("[G.A.T.E.] Multi-agent routing initiated...")

    try:
        # Get G.A.T.E. Specific Context - last 5 messages with thinking content removed
        recent_history_for_prompt = []
        # Ensure we only get the last 5 relevant turns
        for turn in conversation_history[-config.GATE_MAX_CONVERSATION_MESSAGES:]: 
            # Filter out assistant's internal thinking before adding to history for G.A.T.E.
            clean_content = strip_think_tags(turn["content"]) 
            if clean_content:  # Only include if content remains after stripping
                recent_history_for_prompt.append({
                    "role": turn["role"],
                    "content": clean_content
                })
        
        # Log inputs for _get_routing_flags
        olliePrint_simple(f"[G.A.T.E.] User Message Input: {user_message}", level='debug')

        olliePrint_simple(f"[G.A.T.E.] Recent History Input: {recent_history_for_prompt}", level='debug')

        # 2. Get routing flags from G.A.T.E. LLM
        routing_flags = _get_routing_flags(user_message, recent_history_for_prompt)

        # -----------------------------------------------------------------
        # NEW: Retrieve memory context when needs_memory is true
        memory_context = ""
        if routing_flags.get('needs_memory', False):
            search_query = routing_flags.get('memory_search_query') or user_message
            try:
                import memory.L3_memory as L3
                olliePrint_simple(f"[G.A.T.E.] Performing memory search for query: '{search_query}'", level='debug')
                memory_results = L3.search_memory(query_text=search_query, limit=2)
                # Convert to a compact string representation for Synapse
                memory_context = json.dumps(memory_results, ensure_ascii=False, default=str)
                olliePrint_simple(f"[G.A.T.E.] Retrieved {len(memory_results)} memory nodes", level='debug')
            except Exception as mem_err:
                olliePrint_simple(f"[G.A.T.E.] Memory search error: {mem_err}", level='warning')
                memory_context = ""
        # -----------------------------------------------------------------
        
        olliePrint_simple(f"[G.A.T.E.] Routing flags: {routing_flags}")

        database_content = agent_dispatcher.dispatch_agents(
            routing_flags=routing_flags,
            user_message=user_message,
            conversation_history=conversation_history,
            visual_context=visual_context,
            memory_context=memory_context
        )

        olliePrint_simple(f"[G.A.T.E.] Final Neural Processing Core Content: {database_content}", level='debug')
        return database_content

    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Critical failure during routing: {e}. Using fallback.", level='error')
        # Fallback to basic database
        return _generate_fallback_database(user_message)

def _get_routing_flags(user_message: str, recent_history: list) -> dict:
    """Get routing flags from G.A.T.E. LLM analysis."""
    try:
        # Format recent_history for the prompt
        formatted_recent_history = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in recent_history])
        
        prompt = config.GATE_USER_PROMPT.format(
            user_query=user_message,
            recent_history=formatted_recent_history
        )

        messages = [
            {"role": "system", "content": config.GATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = ollama_manager.chat_concurrent_safe(
            model=config.LLM_DECISION_MODEL,
            messages=messages,
            options=config.THINKING_MODE_OPTIONS,
            format="json"  # Ensure JSON output
        )

        response_content = response.get('message', {}).get('content', '').strip()

        print("\n--- G.A.T.E. Raw Routing Response ---")
        print(response_content)
        print("-------------------------------------\n")
        
        try:
            routing_flags = json.loads(response_content)
            
            required_keys = ['needs_memory', 'needs_web_search', 'needs_deep_research', 'needs_pi_tools', 'needs_reminders']
            for key in required_keys:
                if key not in routing_flags:
                    routing_flags[key] = False
            
            return routing_flags
            
        except json.JSONDecodeError as e:
            olliePrint_simple(f"[G.A.T.E.] JSON parse error: {e}. Using default flags.", level='warning')
            return _get_default_routing_flags()

    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Routing analysis error: {e}. Using default flags.", level='error')
        return _get_default_routing_flags()

def _get_default_routing_flags() -> dict:
    """Get default routing flags when analysis fails."""
    return {
        "needs_memory": True,
        "needs_web_search": False,
        "needs_deep_research": False,
        "needs_pi_tools": False,
        "needs_reminders": True
    }

def _generate_fallback_database(user_message: str) -> str:
    """Generate fallback NEURAL PROCESSING CORE when routing fails."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"""(NEURAL PROCESSING CORE)
Your Internal Systems determined no memory was needed to answer your query.

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""   