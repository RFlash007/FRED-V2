"""
R.E.M.I.N.D. (Reminder & Event Management Intelligence Daemon)
Monitors conversation for scheduling/reminder language and manages reminder lifecycle
"""

import re
import json
import duckdb
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from ollie_print import olliePrint_simple
from config import config

REMINDER_DB_PATH = "memory/reminders.db"

class RemindAgent:
    """R.E.M.I.N.D. agent for reminder detection and management."""
    
    def __init__(self):
        self.name = "R.E.M.I.N.D."
        self.reminder_keywords = getattr(config, 'REMINDER_KEYWORDS', [
            "remind me", "schedule", "appointment", "meeting", "deadline",
            "tomorrow", "next week", "later", "don't forget", "remember to"
        ])
        self.acknowledgment_phrases = getattr(config, 'REMINDER_ACKNOWLEDGMENT_PHRASES', [
            "thanks", "got it", "okay", "ok", "sure", "alright", "understood",
            "will do", "noted", "roger", "copy that"
        ])
        self._init_reminder_db()
    
    def _init_reminder_db(self):
        """Initialize reminder database."""
        try:
            with duckdb.connect(REMINDER_DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        reminder_id VARCHAR PRIMARY KEY DEFAULT (CONCAT('remind_', CAST(epoch_ms(current_timestamp) AS VARCHAR))),
                        user_id VARCHAR DEFAULT 'default_user',
                        content TEXT NOT NULL,
                        reminder_type VARCHAR DEFAULT 'general' CHECK (reminder_type IN ('general', 'scheduled', 'deadline')),
                        target_datetime TIMESTAMP,
                        created_at TIMESTAMP DEFAULT current_timestamp,
                        status VARCHAR DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'expired')),
                        acknowledgment_text TEXT,
                        acknowledged_at TIMESTAMP
                    );
                """)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status, target_datetime);")
            
            olliePrint_simple(f"[{self.name}] Database initialized")
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Database init error: {e}", level='error')
    
    def process_conversation_turn(self, user_message: str, assistant_response: str = "") -> Dict:
        """
        Process a conversation turn for reminder detection and acknowledgments.
        Returns dict with detected_reminders and acknowledged_reminders.
        """
        try:
            result = {
                "detected_reminders": [],
                "acknowledged_reminders": [],
                "active_reminders": []
            }
            
            acknowledged = self._check_acknowledgments(user_message)
            if acknowledged:
                result["acknowledged_reminders"] = acknowledged
            
            detected = self._detect_reminders(user_message)
            if detected:
                result["detected_reminders"] = detected
            
            active = self._get_active_reminders()
            if active:
                result["active_reminders"] = active
            
            return result
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Processing error: {e}", level='error')
            return {
                "detected_reminders": [],
                "acknowledged_reminders": [],
                "active_reminders": [],
                "error": config.AGENT_ERRORS.get("reminder_failure", "Reminder system failed")
            }
    
    def _detect_reminders(self, text: str) -> List[Dict]:
        """Detect reminder requests in text."""
        detected = []
        
        text_lower = text.lower()
        has_reminder_keyword = any(keyword in text_lower for keyword in self.reminder_keywords)
        
        if not has_reminder_keyword:
            return detected
        
        try:
            reminder_patterns = [
                r"remind me to (.+?)(?:\s+(?:tomorrow|next week|later|at|on|in).*)?",
                r"don't forget to (.+?)(?:\s+(?:tomorrow|next week|later|at|on|in).*)?",
                r"remember to (.+?)(?:\s+(?:tomorrow|next week|later|at|on|in).*)?",
                r"schedule (.+?)(?:\s+(?:for|at|on|in).*)?",
                r"(?:meeting|appointment|deadline).*?(?:for|about|regarding)\s+(.+?)(?:\s+(?:at|on|in).*)?",
            ]
            
            for pattern in reminder_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    reminder_content = match.group(1).strip()
                    if len(reminder_content) > 3:  # Minimum content length
                        
                        target_datetime = self._extract_datetime(text)
                        
                        reminder_id = self._create_reminder(
                            content=reminder_content,
                            target_datetime=target_datetime,
                            original_text=text
                        )
                        
                        if reminder_id:
                            detected.append({
                                "reminder_id": reminder_id,
                                "content": reminder_content,
                                "target_datetime": target_datetime.isoformat() if target_datetime else None,
                                "type": "scheduled" if target_datetime else "general"
                            })
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Detection error: {e}", level='error')
        
        return detected
    
    def _check_acknowledgments(self, text: str) -> List[Dict]:
        """Check for acknowledgment phrases and mark reminders as acknowledged."""
        acknowledged = []
        
        text_lower = text.lower().strip()
        
        has_acknowledgment = any(phrase in text_lower for phrase in self.acknowledgment_phrases)
        
        if not has_acknowledgment:
            return acknowledged
        
        try:
            active_reminders = self._get_active_reminders()
            
            if active_reminders:
                for reminder in active_reminders:
                    self._acknowledge_reminder(reminder['reminder_id'], text)
                    acknowledged.append({
                        "reminder_id": reminder['reminder_id'],
                        "content": reminder['content'],
                        "acknowledgment_text": text
                    })
                
                olliePrint_simple(f"[{self.name}] Acknowledged {len(acknowledged)} reminders")
        
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Acknowledgment error: {e}", level='error')
        
        return acknowledged
    
    def _extract_datetime(self, text: str) -> Optional[datetime]:
        """Extract datetime from text (basic implementation)."""
        try:
            text_lower = text.lower()
            now = datetime.now()
            
            if "tomorrow" in text_lower:
                return now + timedelta(days=1)
            elif "next week" in text_lower:
                return now + timedelta(weeks=1)
            elif "later" in text_lower:
                return now + timedelta(hours=2)
            
            return None
            
        except Exception:
            return None
    
    def _create_reminder(self, content: str, target_datetime: Optional[datetime] = None, original_text: str = "") -> Optional[str]:
        """Create a new reminder in the database."""
        try:
            with duckdb.connect(REMINDER_DB_PATH) as conn:
                reminder_type = "scheduled" if target_datetime else "general"
                
                result = conn.execute("""
                    INSERT INTO reminders (content, reminder_type, target_datetime)
                    VALUES (?, ?, ?)
                    RETURNING reminder_id
                """, (content, reminder_type, target_datetime)).fetchone()
                
                if result:
                    reminder_id = result[0]
                    olliePrint_simple(f"[{self.name}] Created reminder: {content[:50]}...")
                    return reminder_id
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Create reminder error: {e}", level='error')
        
        return None
    
    def _acknowledge_reminder(self, reminder_id: str, acknowledgment_text: str):
        """Mark a reminder as acknowledged."""
        try:
            with duckdb.connect(REMINDER_DB_PATH) as conn:
                conn.execute("""
                    UPDATE reminders 
                    SET status = 'acknowledged', 
                        acknowledgment_text = ?, 
                        acknowledged_at = current_timestamp
                    WHERE reminder_id = ?
                """, (acknowledgment_text, reminder_id))
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Acknowledge error: {e}", level='error')
    
    def _get_active_reminders(self) -> List[Dict]:
        """Get all active reminders."""
        try:
            with duckdb.connect(REMINDER_DB_PATH) as conn:
                results = conn.execute("""
                    SELECT reminder_id, content, reminder_type, target_datetime, created_at
                    FROM reminders 
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                """).fetchall()
                
                reminders = []
                for row in results:
                    reminders.append({
                        "reminder_id": row[0],
                        "content": row[1],
                        "type": row[2],
                        "target_datetime": row[3],
                        "created_at": row[4]
                    })
                
                return reminders
                
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Get active reminders error: {e}", level='error')
            return []
