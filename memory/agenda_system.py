"""
F.R.E.D. Agenda System - Proactive Learning Engine
Implements The Agenda for background research and task processing
As specified in Memory Upgrade Plan Section 4
"""

import duckdb
import json
import uuid
import numpy as np
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional
from config import config, ollama_manager
from ollie_print import olliePrint_simple

# Use L3 database for agenda tables (part of librarian.py database)
AGENDA_DB_PATH = "memory/memory.db"
AGENDA_DUPLICATE_THRESHOLD = 0.95  # Similarity threshold for detecting duplicate tasks

def _cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return 0.0
    v1 = np.array(v1)
    v2 = np.array(v2)
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def generate_fred_research_summary(original_task: str, research_findings: str) -> str:
    """Generate F.R.E.D.-ready research summary using R.E.F.L.E.X."""
    try:
        olliePrint_simple("[R.E.F.L.E.X.] Generating F.R.E.D. research summary...")
        
        summary_prompt = config.REFLEX_USER_PROMPT.format(
            original_task=original_task,
            research_findings=research_findings
        )
        
        messages = [
            {"role": "system", "content": config.REFLEX_SYSTEM_PROMPT},
            {"role": "user", "content": summary_prompt}
        ]
        
        response = ollama_manager.chat_concurrent_safe(
            model=config.REFLEX_MODEL,
            messages=messages,
            stream=False
        )
        
        fred_summary = response.get('message', {}).get('content', '').strip()
        
        if fred_summary:
            olliePrint_simple("[R.E.F.L.E.X.] F.R.E.D. summary generated successfully")
            return fred_summary
        else:
            olliePrint_simple("[R.E.F.L.E.X.] No summary generated, using fallback", level='warning')
            return f"I was thinking about that {original_task[:50]}... question - I worked through it in the back of my mind and have some insights to share."
            
    except Exception as e:
        olliePrint_simple(f"[R.E.F.L.E.X.] Summary generation failed: {e}", level='error')
        return f"I was thinking about that {original_task[:50]}... question - I worked through it in the back of my mind and have some insights to share."

def init_agenda_db():
    """Initialize agenda and notification tables in L3 database."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            # Create agenda tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agenda_tasks (
                    task_id VARCHAR PRIMARY KEY DEFAULT (CONCAT('agenda_', CAST(epoch_ms(current_timestamp) AS VARCHAR))),
                    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
                    priority INTEGER DEFAULT 2 CHECK (priority IN (1, 2)),
                    content TEXT NOT NULL,
                    embedding BLOB,
                    source_conversation_id VARCHAR DEFAULT 'default_conversation',
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    completed_at TIMESTAMP,
                    result_node_id VARCHAR,
                    decomposed_questions JSON,
                    research_results JSON,
                    error_message TEXT
                );
            """)
            
            # Create notification queue table for compatibility
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notification_queue (
                    notification_id VARCHAR PRIMARY KEY DEFAULT (CONCAT('notif_', CAST(epoch_ms(current_timestamp) AS VARCHAR))),
                    user_id VARCHAR DEFAULT 'default_user',
                    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'delivered')),
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    source_task_id VARCHAR,
                    message_summary TEXT NOT NULL,
                    related_node_id VARCHAR
                );
            """)
            
            # Create F.R.E.D. research summaries table (direct injection to F.R.E.D.)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fred_research_summaries (
                    summary_id VARCHAR PRIMARY KEY,
                    source_task_id VARCHAR NOT NULL,
                    summary_content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    delivered BOOLEAN DEFAULT false
                );
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agenda_status ON agenda_tasks(status, priority, created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_status ON notification_queue(status, user_id, created_at);")
        
        olliePrint_simple("Agenda system database initialized")
    except Exception as e:
        olliePrint_simple(f"Failed to initialize agenda database: {e}", level='error')
        raise

def _is_duplicate_task(new_task_embedding: np.ndarray) -> bool:
    """Check if a task with a similar embedding already exists."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            results = conn.execute("""
                SELECT embedding 
                FROM agenda_tasks 
                WHERE status IN ('pending', 'processing') AND embedding IS NOT NULL
            """).fetchall()

            if not results:
                return False

            for (existing_embedding_bytes,) in results:
                existing_embedding = np.frombuffer(existing_embedding_bytes, dtype=np.float32)
                similarity = _cosine_similarity(new_task_embedding, existing_embedding)
                
                if similarity > AGENDA_DUPLICATE_THRESHOLD:
                    olliePrint_simple(f"Duplicate agenda task detected with similarity: {similarity:.2f}", level='warning')
                    return True
        return False
    except Exception as e:
        olliePrint_simple(f"Error checking for duplicate agenda tasks: {e}", level='error')
        return False

def add_task_to_agenda(task_description: str, priority: int = 2) -> Optional[str]:
    """Add a new task to the agenda if it's not a duplicate."""
    try:
        if priority not in [1, 2]:
            priority = 2
        
        olliePrint_simple("Generating embedding for new agenda task...")
        embedding_list = ollama_manager.embeddings(model=config.DEFAULT_EMBEDDING_MODEL, prompt=task_description)
        new_task_embedding = np.array(embedding_list['embedding'], dtype=np.float32)

        if _is_duplicate_task(new_task_embedding):
            olliePrint_simple(f"Rejected duplicate task: {task_description[:100]}...")
            return None
        
        task_id = f"agenda_{uuid.uuid4().hex[:12]}"
        
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO agenda_tasks (task_id, content, priority, embedding)
                VALUES (?, ?, ?, ?)
            """, (task_id, task_description, priority, new_task_embedding.tobytes()))
        
        priority_text = "IMPORTANT" if priority == 1 else "Normal"
        olliePrint_simple(f"Added {priority_text} task to agenda: {task_description[:100]}...")
        return task_id
        
    except Exception as e:
        olliePrint_simple(f"Failed to add task to agenda: {e}", level='error')
        return None

def get_pending_agenda_tasks(limit: int = None) -> List[Dict]:
    """Get pending tasks from the agenda."""
    try:
        if limit is None:
            limit = config.SLEEP_CYCLE_MAX_AGENDA_TASKS
        
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            results = conn.execute("""
                SELECT task_id, content, priority, created_at
                FROM agenda_tasks 
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
                LIMIT ?
            """, (limit,)).fetchall()
            
            tasks = []
            for row in results:
                tasks.append({
                    'task_id': row[0],
                    'content': row[1],
                    'priority': row[2],
                    'created_at': row[3]
                })
            
            return tasks
            
    except Exception as e:
        olliePrint_simple(f"Failed to get pending agenda tasks: {e}", level='error')
        return []

# NOTE: Old linear research pipeline functions removed - now using A.R.C.H./D.E.L.V.E. iterative system

def complete_agenda_task(task_id: str, result_node_id: str, research_results: Dict):
    """Mark agenda task as completed and create F.R.E.D. summary for direct injection."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            # Update task status
            conn.execute("""
                UPDATE agenda_tasks 
                SET status = 'completed', completed_at = current_timestamp, 
                    result_node_id = ?, research_results = ?
                WHERE task_id = ?
            """, (result_node_id, json.dumps(research_results), task_id))
            
            # Store F.R.E.D. summary for direct injection (bypasses all agents)
            fred_summary = research_results.get('fred_summary', 'Research completed.')
            fred_notification_id = f"fred_summary_{uuid.uuid4().hex[:12]}"
            
            conn.execute("""
                INSERT INTO fred_research_summaries 
                (summary_id, source_task_id, summary_content, created_at, delivered)
                VALUES (?, ?, ?, current_timestamp, false)
            """, (fred_notification_id, task_id, fred_summary))
            
            olliePrint_simple(f"Research summary created for F.R.E.D.: {task_id}")
        
    except Exception as e:
        olliePrint_simple(f"Failed to complete agenda task: {e}", level='error')

def get_pending_fred_summaries() -> List[Dict]:
    """Get pending F.R.E.D. research summaries for direct injection."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            result = conn.execute("""
                SELECT summary_id, summary_content, created_at
                FROM fred_research_summaries
                WHERE delivered = false
                ORDER BY created_at ASC
            """).fetchall()
            
            return [
                {
                    'summary_id': row[0],
                    'content': row[1],
                    'created_at': row[2]
                }
                for row in result
            ]
    except Exception as e:
        olliePrint_simple(f"Failed to get pending F.R.E.D. summaries: {e}", level='error')
        return []

def mark_fred_summaries_delivered(summary_ids: List[str]):
    """Mark F.R.E.D. summaries as delivered."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            for summary_id in summary_ids:
                conn.execute("""
                    UPDATE fred_research_summaries
                    SET delivered = true
                    WHERE summary_id = ?
                """, (summary_id,))
        olliePrint_simple(f"Marked {len(summary_ids)} F.R.E.D. summaries as delivered")
    except Exception as e:
        olliePrint_simple(f"Failed to mark F.R.E.D. summaries as delivered: {e}", level='error')

def fail_agenda_task(task_id: str, error_message: str):
    """Mark agenda task as failed."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            conn.execute("""
                UPDATE agenda_tasks 
                SET status = 'failed', error_message = ?
                WHERE task_id = ?
            """, (error_message, task_id))
        
        olliePrint_simple(f"Agenda task failed: {task_id} - {error_message}", level='error')
        
    except Exception as e:
        olliePrint_simple(f"Failed to mark agenda task as failed: {e}", level='error')

def process_agenda_task(task: Dict) -> bool:
    """Process a single agenda task using A.R.C.H./D.E.L.V.E. iterative research system."""
    try:
        task_id = task['task_id']
        task_content = task['content']
        
        olliePrint_simple(f"Processing agenda task with A.R.C.H./D.E.L.V.E.: {task_content[:100]}...")
        
        # Mark as processing
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            conn.execute("""
                UPDATE agenda_tasks SET status = 'processing' WHERE task_id = ?
            """, (task_id,))
        
        # Conduct iterative research using A.R.C.H./D.E.L.V.E. system
        from memory.arch_delve_research import conduct_enhanced_iterative_research, synthesize_research_to_memory
        
        research_result = conduct_enhanced_iterative_research(task_id, task_content)
        
        if not research_result.get('success', False):
            fail_agenda_task(task_id, f"A.R.C.H./D.E.L.V.E. research failed: {research_result.get('reason', 'unknown')}")
            return False
        
        # Synthesize research findings to L3 memory
        result_node_id = synthesize_research_to_memory(research_result, task_content)
        
        if not result_node_id:
            fail_agenda_task(task_id, "Failed to create memory node from research findings")
            return False
        
        # Generate F.R.E.D.-ready summary using R.E.F.L.E.X.
        fred_summary = generate_fred_research_summary(task_content, research_result.get('findings', ''))
        
        # Complete task and create notification
        research_results = {
            'method': 'arch_delve_iterative',
            'iterations': research_result.get('iterations', 0),
            'conversation_path': research_result.get('conversation_path', ''),
            'findings': research_result.get('findings', ''),
            'node_id': result_node_id,
            'fred_summary': fred_summary
        }
        
        complete_agenda_task(task_id, result_node_id, research_results)
        
        olliePrint_simple(f"A.R.C.H./D.E.L.V.E. research completed in {research_result.get('iterations', 0)} iterations")
        return True
        
    except Exception as e:
        olliePrint_simple(f"Failed to process agenda task: {e}", level='error')
        fail_agenda_task(task.get('task_id', 'unknown'), str(e))
        return False

def get_pending_notifications(user_id: str = 'default_user') -> List[Dict]:
    """Get pending notifications for user."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            results = conn.execute("""
                SELECT notification_id, message_summary, created_at, related_node_id
                FROM notification_queue 
                WHERE user_id = ? AND status = 'pending'
                ORDER BY created_at ASC
            """, (user_id,)).fetchall()
            
            notifications = []
            for row in results:
                notifications.append({
                    'notification_id': row[0],
                    'message': row[1],
                    'created_at': row[2],
                    'related_node_id': row[3]
                })
            
            return notifications
            
    except Exception as e:
        olliePrint_simple(f"Failed to get pending notifications: {e}", level='error')
        return []

def mark_notifications_delivered(notification_ids: List[str]):
    """Mark notifications as delivered."""
    try:
        if not notification_ids:
            return
        
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            placeholders = ','.join(['?' for _ in notification_ids])
            conn.execute(f"""
                UPDATE notification_queue 
                SET status = 'delivered' 
                WHERE notification_id IN ({placeholders})
            """, notification_ids)
        
        olliePrint_simple(f"Marked {len(notification_ids)} notifications as delivered")
        
    except Exception as e:
        olliePrint_simple(f"Failed to mark notifications as delivered: {e}", level='error')

def get_agenda_summary() -> Dict:
    """Get summary of agenda system status."""
    try:
        with duckdb.connect(AGENDA_DB_PATH) as conn:
            # Get task counts
            task_counts = conn.execute("""
                SELECT status, COUNT(*) 
                FROM agenda_tasks 
                GROUP BY status
            """).fetchall()
            
            # Get pending notifications count
            pending_notifications = conn.execute("""
                SELECT COUNT(*) 
                FROM notification_queue 
                WHERE status = 'pending'
            """).fetchone()
            
            summary = {
                'tasks': {row[0]: row[1] for row in task_counts},
                'pending_notifications': pending_notifications[0] if pending_notifications else 0
            }
            
            return summary
            
    except Exception as e:
        olliePrint_simple(f"Failed to get agenda summary: {e}", level='error')
        return {'tasks': {}, 'pending_notifications': 0}

# Initialize database on import
init_agenda_db() 