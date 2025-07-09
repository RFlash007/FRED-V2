
from datetime import datetime
from ollie_print import olliePrint_simple
from config import config, ollama_manager
import memory.L2_memory as L2
import memory.crap as crap

def run_gate_analysis(user_message: str, conversation_history: list) -> str:
    """
    Runs the G.A.T.E. triage agent.
    1. Fetches L2 context.
    2. Asks G.A.T.E. LLM if L2 is sufficient.
    3. If sufficient, returns the formatted L2 context.
    4. If insufficient, escalates to C.R.A.P. and returns its result.
    """
    olliePrint_simple("[G.A.T.E.] Triage analysis initiated...")

    # 1. Fetch L2 Context
    l2_context_raw = L2.query_l2_context(user_message)
    # Clean the context for the prompt, but keep the raw version for the final output
    l2_context_clean = l2_context_raw.replace("(L2 EPISODIC CONTEXT)", "").replace("(END L2 EPISODIC CONTEXT)", "").strip()
    if not l2_context_clean:
        l2_context_clean = "No relevant recent context found."

    # 2. Prepare and call G.A.T.E. LLM
    prompt = config.GATE_USER_PROMPT.format(
        user_query=user_message,
        l2_context=l2_context_clean
    )

    messages = [
        {"role": "system", "content": config.GATE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        response = ollama_manager.chat_concurrent_safe(
            model=config.LLM_DECISION_MODEL, # Using a capable model for routing
            messages=messages,
            tools=config.GATE_TOOLS,
            stream=False
        )

        response_message = response.get('message', {})
        tool_calls = response_message.get('tool_calls')
        final_content = response_message.get('content', '')

        # 3. Decision Point
        if tool_calls:
            # Escalation needed
            tool_call = tool_calls[0] # Assuming one tool call
            if tool_call.get('function', {}).get('name') == 'escalate_to_crap':
                olliePrint_simple("[G.A.T.E.] Decision: L2 insufficient. Escalating to C.R.A.P.")
                # 4. Escalate to C.R.A.P.
                # We pass the original user_message and conversation_history
                return crap.run_crap_analysis(user_message, conversation_history)
            else:
                 olliePrint_simple(f"[G.A.T.E.] Warning: Unknown tool '{tool_call.get('function', {}).get('name')}' called. Defaulting to C.R.A.P.", level='warning')
                 return crap.run_crap_analysis(user_message, conversation_history)
        else:
            # L2 is sufficient. The LLM has synthesized the relevant context text.
            olliePrint_simple("[G.A.T.E.] Decision: L2 context is sufficient. Wrapping synthesized context.")
            # Manually wrap the synthesized content in the required database block.
            synthesized_context = final_content.strip()

            # Construct the final database block.
            # If the LLM returns nothing, the RELEVANT MEMORIES section will be empty.
            database_block = f"""(FRED DATABASE)
RELEVANT MEMORIES:
{synthesized_context}

SYSTEM STATUS:
The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
(END FRED DATABASE)"""
            return database_block

    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Critical failure during analysis: {e}. Defaulting to full C.R.A.P. analysis.", level='error')
        # Fallback to C.R.A.P. on any error
        return crap.run_crap_analysis(user_message, conversation_history) 