
import json
from datetime import datetime
from ollie_print import olliePrint_simple
from config import config, ollama_manager
try:
    import memory.L2_memory as L2
except ImportError:
    class MockL2:
        @staticmethod
        def query_l2_context(message):
            return "No recent context available"
    L2 = MockL2()
from agents.dispatcher import AgentDispatcher

agent_dispatcher = AgentDispatcher()

def run_gate_analysis(user_message: str, conversation_history: list, visual_context: str = "") -> str:
    """
    Runs the new G.A.T.E. routing system with multi-agent dispatch.
    1. Analyzes user query to determine routing flags
    2. Dispatches appropriate agents based on flags
    3. Returns synthesized FRED DATABASE from S.Y.N.A.P.S.E.
    """
    olliePrint_simple("[G.A.T.E.] Multi-agent routing initiated...")

    try:
        l2_context_raw = L2.query_l2_context(user_message)
        l2_context_clean = l2_context_raw.replace("(L2 EPISODIC CONTEXT)", "").replace("(END L2 EPISODIC CONTEXT)", "").strip()
        if not l2_context_clean:
            l2_context_clean = "No relevant recent context found."

        # 2. Get routing flags from G.A.T.E. LLM
        routing_flags = _get_routing_flags(user_message, l2_context_clean)
        
        olliePrint_simple(f"[G.A.T.E.] Routing flags: {routing_flags}")

        database_content = agent_dispatcher.dispatch_agents(
            routing_flags=routing_flags,
            user_message=user_message,
            conversation_history=conversation_history,
            visual_context=visual_context
        )

        return database_content

    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Critical failure during routing: {e}. Using fallback.", level='error')
        # Fallback to basic database
        return _generate_fallback_database(user_message)

def _get_routing_flags(user_message: str, l2_context: str) -> dict:
    """Get routing flags from G.A.T.E. LLM analysis."""
    try:
        prompt = config.GATE_ROUTING_USER_PROMPT.format(
            user_query=user_message,
            l2_context=l2_context
        )

        messages = [
            {"role": "system", "content": config.GATE_ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = ollama_manager.chat_concurrent_safe(
            model=config.LLM_DECISION_MODEL,
            messages=messages,
            stream=False
        )

        response_content = response.get('message', {}).get('content', '').strip()
        
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
    """Generate fallback FRED DATABASE when routing fails."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"""(FRED DATABASE)
• Processing your query: {user_message[:100]}...
• My routing systems are working to analyze your request
• Putting it together... ready to help with what I know

SYSTEM STATUS:
The current time is: {current_time}
(END FRED DATABASE)"""   