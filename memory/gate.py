import json
import uuid
import duckdb
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ollie_print import olliePrint_simple
from config import config, ollama_manager
from utils import strip_think_tags

from agents.dispatcher import AgentDispatcher
import memory.L2_memory as L2

# Reminder database path
REMINDER_DB_PATH = "memory/reminders.db"

def _init_reminder_db():
    """Initialize reminder database with schema."""
    try:
        with duckdb.connect(REMINDER_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id VARCHAR PRIMARY KEY DEFAULT (CONCAT('remind_', CAST(epoch_ms(current_timestamp) AS VARCHAR))),
                    user_id VARCHAR DEFAULT 'default_user',
                    content TEXT NOT NULL,
                    reminder_type VARCHAR DEFAULT 'general' CHECK (reminder_type IN ('general', 'scheduled', 'deadline', 'recurring')),
                    target_datetime TIMESTAMP,
                    recurrence_type VARCHAR CHECK (recurrence_type IN ('daily', 'weekly')),
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    status VARCHAR DEFAULT 'active' CHECK (status IN ('active', 'completed')),
                    completed_at TIMESTAMP
                );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status, target_datetime);")

        olliePrint_simple("[G.A.T.E.] Reminder database initialized")
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Reminder database init error: {e}", level='error')

def _create_reminder(content: str, target_time: str = None, is_recurring: str = None, append_mode: bool = False) -> Optional[str]:
    """Create a new reminder in the database."""
    try:
        _init_reminder_db()  # Ensure DB exists
        
        # Handle append mode - find existing reminder with similar content
        if append_mode:
            with duckdb.connect(REMINDER_DB_PATH) as conn:
                existing = conn.execute("""
                    SELECT reminder_id, content FROM reminders 
                    WHERE status = 'active' AND content LIKE ? 
                    ORDER BY created_at DESC LIMIT 1
                """, (f"%{content[:20]}%",)).fetchone()
                
                if existing:
                    # Append to existing reminder
                    updated_content = f"{existing[1]}. {content}"
                    conn.execute("""
                        UPDATE reminders SET content = ? WHERE reminder_id = ?
                    """, (updated_content, existing[0]))
                    
                    olliePrint_simple(f"[G.A.T.E.] Appended to existing reminder: {existing[0]}")
                    return existing[0]
        
        # Create new reminder using strict time parser
        target_datetime, parsed_recurrence = _parse_target_datetime(target_time)
        
        # Use parsed recurrence if available, otherwise use provided is_recurring
        final_recurrence = parsed_recurrence or is_recurring
        reminder_type = "recurring" if final_recurrence else ("scheduled" if target_datetime else "general")
        
        with duckdb.connect(REMINDER_DB_PATH) as conn:
            result = conn.execute("""
                INSERT INTO reminders (content, reminder_type, target_datetime, recurrence_type)
                VALUES (?, ?, ?, ?)
                RETURNING reminder_id
            """, (content, reminder_type, target_datetime, final_recurrence)).fetchone()
            
            if result:
                reminder_id = result[0]
                olliePrint_simple(f"[G.A.T.E.] Created reminder: {content[:50]}...")
                return reminder_id
        
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Create reminder error: {e}", level='error')
    
    return None

def _complete_reminder(content_hint: str = None) -> List[str]:
    """Mark reminders as completed based on content hint."""
    try:
        _init_reminder_db()
        
        with duckdb.connect(REMINDER_DB_PATH) as conn:
            if content_hint:
                # Complete specific reminder by content match
                results = conn.execute("""
                    UPDATE reminders 
                    SET status = 'completed', completed_at = current_timestamp
                    WHERE status = 'active' AND content LIKE ?
                    RETURNING reminder_id
                """, (f"%{content_hint}%",)).fetchall()
            else:
                # Complete the most recent active reminder
                results = conn.execute("""
                    UPDATE reminders 
                    SET status = 'completed', completed_at = current_timestamp
                    WHERE reminder_id IN (
                        SELECT reminder_id FROM reminders 
                        WHERE status = 'active' 
                        ORDER BY created_at DESC LIMIT 1
                    )
                    RETURNING reminder_id
                """).fetchall()
            
            completed_ids = [row[0] for row in results]
            if completed_ids:
                olliePrint_simple(f"[G.A.T.E.] Completed {len(completed_ids)} reminder(s)")
            
            return completed_ids
        
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Complete reminder error: {e}", level='error')
        return []

def _get_active_reminders() -> List[Dict]:
    """Get all active reminders."""
    try:
        _init_reminder_db()
        
        with duckdb.connect(REMINDER_DB_PATH) as conn:
            results = conn.execute("""
                SELECT reminder_id, content, reminder_type, target_datetime, recurrence_type, created_at
                FROM reminders 
                WHERE status = 'active'
                ORDER BY 
                    CASE WHEN target_datetime IS NOT NULL THEN target_datetime ELSE created_at END ASC
            """).fetchall()
            
            reminders = []
            for row in results:
                reminders.append({
                    "reminder_id": row[0],
                    "content": row[1],
                    "type": row[2],
                    "target_datetime": row[3],
                    "recurrence_type": row[4],
                    "created_at": row[5]
                })
            
            return reminders
            
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Get active reminders error: {e}", level='error')
        return []

def _parse_target_datetime(time_str: str) -> tuple:
    """Parse strict time format into (datetime_obj, recurrence_type).
    
    Returns:
        tuple: (datetime_obj or None, recurrence_type or None)
    """
    if not time_str or time_str == 'null':
        return None, None
    
    now = datetime.now()
    
    try:
        # Handle recurring patterns
        if time_str.startswith('daily@'):
            time_part = time_str[6:]  # Remove 'daily@'
            if time_part == 'morning':
                target_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
            elif time_part == 'afternoon':
                target_time = now.replace(hour=14, minute=0, second=0, microsecond=0)
            elif time_part == 'evening':
                target_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
            else:
                # Parse time like '12:00'
                hour, minute = map(int, time_part.split(':'))
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return target_time, 'daily'
        
        elif time_str.startswith('weekly@'):
            parts = time_str[7:].split('@')  # Remove 'weekly@'
            day_name = parts[0].lower()
            time_part = parts[1] if len(parts) > 1 else '09:00'
            
            # Map day names to weekday numbers (Monday=0)
            day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
                      'friday': 4, 'saturday': 5, 'sunday': 6}
            
            if day_name in day_map:
                target_weekday = day_map[day_name]
                days_ahead = (target_weekday - now.weekday()) % 7
                if days_ahead == 0:  # Today
                    days_ahead = 7  # Next week
                
                if time_part == 'morning':
                    hour, minute = 9, 0
                elif time_part == 'afternoon':
                    hour, minute = 14, 0
                elif time_part == 'evening':
                    hour, minute = 18, 0
                else:
                    hour, minute = map(int, time_part.split(':'))
                
                target_time = now + timedelta(days=days_ahead)
                target_time = target_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time, 'weekly'
        
        # Handle relative times with specific hour
        elif '@' in time_str:
            parts = time_str.split('@')
            base_time = parts[0]
            time_part = parts[1]
            
            if time_part == 'morning':
                hour, minute = 9, 0
            elif time_part == 'afternoon':
                hour, minute = 14, 0
            elif time_part == 'evening':
                hour, minute = 18, 0
            else:
                hour, minute = map(int, time_part.split(':'))
            
            if base_time == 'tomorrow':
                target_time = now + timedelta(days=1)
                target_time = target_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time, None
            elif base_time == 'tonight':
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time, None
        
        # Handle simple relative times
        elif time_str == 'tomorrow':
            target_time = now + timedelta(days=1)
            target_time = target_time.replace(hour=9, minute=0, second=0, microsecond=0)  # Default morning
            return target_time, None
        elif time_str == 'tonight':
            target_time = now.replace(hour=20, minute=0, second=0, microsecond=0)
            return target_time, None
        elif time_str == 'later':
            target_time = now + timedelta(hours=2)
            return target_time, None
        
        # Handle ISO datetime (2024-08-02T15:00)
        elif 'T' in time_str:
            target_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return target_time, None
        
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Time parsing error for '{time_str}': {e}", level='warning')
    
    return None, None

def _handle_reminder_action(reminder_action: Dict) -> str:
    """Process reminder action and return result message for context."""
    action_type = reminder_action.get('type')
    content = reminder_action.get('content', '')
    target_time = reminder_action.get('target_time')
    is_recurring = reminder_action.get('is_recurring')
    append_mode = reminder_action.get('append_mode', False)
    
    try:
        if action_type == 'create':
            reminder_id = _create_reminder(content, target_time, is_recurring, append_mode)
            if reminder_id:
                if append_mode:
                    return f"\n\nREMINDER UPDATED: Appended '{content}' to existing reminder."
                else:
                    recur_text = f" ({is_recurring})" if is_recurring else ""
                    time_text = f" for {target_time}" if target_time else ""
                    return f"\n\nREMINDER CREATED: '{content}'{time_text}{recur_text}"
            else:
                return "\n\nREMINDER ERROR: Failed to create reminder."
        
        elif action_type == 'complete':
            completed_ids = _complete_reminder(content)
            if completed_ids:
                return f"\n\nREMINDER COMPLETED: Marked {len(completed_ids)} reminder(s) as done."
            else:
                return "\n\nREMINDER ERROR: No matching active reminders found to complete."
        
        elif action_type == 'retrieve':
            active_reminders = _get_active_reminders()
            if active_reminders:
                reminder_list = []
                for r in active_reminders:
                    time_info = f" (due: {r['target_datetime'].strftime('%Y-%m-%d %H:%M') if r['target_datetime'] else 'no specific time'})" 
                    recur_info = f" [{r['recurrence_type']}]" if r['recurrence_type'] else ""
                    reminder_list.append(f"• {r['content']}{time_info}{recur_info}")
                
                return f"\n\nACTIVE REMINDERS:\n" + "\n".join(reminder_list)
            else:
                return "\n\nNO ACTIVE REMINDERS: You have no pending reminders."
        
        else:
            return f"\n\nREMINDER ERROR: Unknown action type '{action_type}'."
    
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Reminder action error: {e}", level='error')
        return f"\n\nREMINDER ERROR: {str(e)}"

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
        # NEW: Handle reminder actions directly in GATE
        reminder_context = ""
        if routing_flags.get('needs_reminders', False):
            reminder_action = routing_flags.get('reminder_action')
            
            if reminder_action:
                # Process specific reminder action (create, complete, retrieve)
                reminder_context = _handle_reminder_action(reminder_action)
                olliePrint_simple(f"[G.A.T.E.] Processed reminder action: {reminder_action.get('type')}")
            else:
                # Always show active reminders when reminder system is triggered
                active_reminders = _get_active_reminders()
                if active_reminders:
                    reminder_list = []
                    for r in active_reminders:
                        time_info = f" (due: {r['target_datetime'].strftime('%Y-%m-%d %H:%M') if r['target_datetime'] else 'no specific time'})"
                        recur_info = f" [{r['recurrence_type']}]" if r['recurrence_type'] else ""
                        reminder_list.append(f"• {r['content']}{time_info}{recur_info}")
                    
                    reminder_context = f"\n\nACTIVE REMINDERS:\n" + "\n".join(reminder_list)
                    olliePrint_simple(f"[G.A.T.E.] Retrieved {len(active_reminders)} active reminders")
        
        # Add reminder context to memory context for agent processing
        if reminder_context:
            memory_context += reminder_context
        # -----------------------------------------------------------------
        
        olliePrint_simple(f"[G.A.T.E.] Routing flags: {routing_flags}")

        # Handle web search routing before dispatching agents
        web_search_strategy = routing_flags.get('web_search_strategy', {})
        if web_search_strategy.get('needed', False):
            search_priority = web_search_strategy.get('search_priority', 'quick')
            search_query = web_search_strategy.get('search_query', user_message)
            
            if search_priority == 'thorough':
                # Route thorough searches to arch/delve queue
                return _handle_thorough_search(search_query, user_message, conversation_history)
            else:
                # Handle quick searches with intelligent_search
                search_results = _handle_quick_search(search_query)
                # Add search results to memory context for main agent dispatch
                if search_results and search_results.get('summary'):
                    memory_context += f"\n\nWeb Search Results:\n{search_results['summary']}"
        
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
            model=config.GATE_OLLAMA_MODEL,
            messages=messages,
            options=config.LLM_GENERATION_OPTIONS,
            format="json",  # Ensure JSON output
            stream=False  # Require full JSON dict, not streaming generator
        )

        # Defensive guard against null or malformed responses
        if not response:
            olliePrint_simple("[G.A.T.E.] Null or empty LLM response – using default routing flags", level='warning')
            return _get_default_routing_flags()

        # Ollama client >=0.1.8 returns a dict; older versions return ChatResponse objects
        if isinstance(response, dict):
            message_dict = response.get('message')
        else:
            # Fallback for ChatResponse-like objects
            message_dict = getattr(response, 'message', None)

        if not message_dict or not isinstance(message_dict, dict):
            olliePrint_simple("[G.A.T.E.] LLM response missing 'message' field – using default routing flags", level='warning')
            return _get_default_routing_flags()

        response_content = message_dict.get('content', '').strip()
        
        # Pretty, readable terminal output (colorized via olliePrint_simple)
        olliePrint_simple("\n┏━━[G.A.T.E.] Raw Routing Response ━━")
        olliePrint_simple(response_content)
        olliePrint_simple("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        try:
            routing_flags = json.loads(response_content)
            
            required_keys = ['needs_memory', 'needs_pi_tools', 'needs_reminders']
            for key in required_keys:
                if key not in routing_flags:
                    routing_flags[key] = False
            
            # Ensure web_search_strategy has proper structure
            if 'web_search_strategy' not in routing_flags:
                routing_flags['web_search_strategy'] = {
                    'needed': False,
                    'search_priority': 'quick',
                    'search_query': ''
                }
            
            return routing_flags
            
        except json.JSONDecodeError as e:
            olliePrint_simple(f"[G.A.T.E.] JSON parse error: {e}. Using default flags.", level='warning')
            return _get_default_routing_flags()
            
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Routing analysis error: {e}. Using default flags.", level='error')
        return _get_default_routing_flags()


def _handle_thorough_search(search_query: str, user_message: str, conversation_history: list) -> str:
    """Route thorough search requests to the Agenda System for background research."""
    try:
        import memory.agenda_system as agenda
        
        # Build task description and enqueue via Agenda System
        research_task = f"Conduct comprehensive web research on: {search_query}"
        task_id = agenda.add_task_to_agenda(research_task, priority=2)

        if task_id:
            olliePrint_simple(f"[G.A.T.E.] Thorough search enqueued in Agenda System (task_id={task_id})")
            return (
                f"I've queued a thorough background research task (ID: {task_id}) for '{search_query}'. "
                "I'll share the findings once they are ready."
            )
        else:
            olliePrint_simple("[G.A.T.E.] Failed to enqueue thorough search; falling back to quick search", level='warning')
            return _handle_quick_search(search_query)
            
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Error enqueuing thorough search: {e}", level='error')
        # Fallback to quick search
        return _handle_quick_search(search_query)


def _handle_quick_search(search_query: str) -> dict:
    """Handle quick web searches using the new intelligent_search function."""
    try:
        from web_search_core import intelligent_search
        
        olliePrint_simple(f"[G.A.T.E.] Executing quick web search: {search_query}")
        
        # Use intelligent search in quick mode
        search_results = intelligent_search(
            query=search_query,
            search_priority="quick",
            mode="auto"
        )
        
        return search_results
        
    except Exception as e:
        olliePrint_simple(f"[G.A.T.E.] Error in quick search: {e}", level='error')
        return {
            'query': search_query,
            'summary': f"Web search error: {str(e)}",
            'links': [],
            'extracted_content': []
        }

def _get_default_routing_flags() -> dict:
    """Get default routing flags when analysis fails."""
    return {
        "needs_memory": True,
        "needs_pi_tools": False,
        "needs_reminders": True,
        "web_search_strategy": {
            "needed": False,
            "search_priority": "quick",
            "search_query": ""
        }
    }

def _generate_fallback_database(user_message: str) -> str:
    """Generate fallback NEURAL PROCESSING CORE when routing fails."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"""(NEURAL PROCESSING CORE)
Your Internal Systems determined no memory was needed to answer your query.

SYSTEM STATUS:
The current time is: {current_time}
(END NEURAL PROCESSING CORE)"""   