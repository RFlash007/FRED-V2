"""
F.R.E.D. L2 Episodic Cache - Rolling RAG Database
Implements the "Recent Past" memory layer with semantic similarity triggers
Replaces short_term_memory.py as specified in Memory Upgrade Plan
"""

import duckdb
import json
import numpy as np
import requests
import threading
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from config import config, ollama_manager
from ollie_print import olliePrint_simple

# L2 Database path
L2_DB_PATH = "memory/L2_episodic_cache.db"

# Use centralized Ollama connection manager for all L2 operations

# Thread-safe state for rolling embeddings
class L2State:
    def __init__(self):
        self.rolling_embeddings = []  # Last N user message embeddings for topic change detection
        self.last_topic_start_turn = 0  # Track when current topic started
        self.last_l2_creation_turn = 0  # Track when last L2 was created
        self.processed_ranges = set()  # Track processed turn ranges to prevent duplicates
        self._lock = threading.Lock()
    
    def add_user_embedding(self, embedding: np.ndarray, turn_number: int):
        """Add user message embedding to rolling window."""
        with self._lock:
            self.rolling_embeddings.append((embedding, turn_number))
            # Keep only the last N embeddings for rolling average
            if len(self.rolling_embeddings) > config.L2_ROLLING_AVERAGE_WINDOW:
                self.rolling_embeddings.pop(0)
    
    def get_rolling_average_embedding(self) -> Optional[np.ndarray]:
        """Get rolling average embedding from recent user messages."""
        with self._lock:
            if len(self.rolling_embeddings) < 2:  # Need at least 2 for comparison
                return None
            
            # Calculate average of recent embeddings
            embeddings = [emb for emb, _ in self.rolling_embeddings[:-1]]  # Exclude current message
            if not embeddings:
                return None
            
            return np.mean(embeddings, axis=0)
    
    def should_create_l2_summary(self, current_embedding: np.ndarray, current_turn: int) -> Tuple[bool, str]:
        """Determine if L2 summary should be created based on semantic or fallback triggers."""
        with self._lock:
            # Check minimum gap since last L2 creation
            if current_turn - self.last_l2_creation_turn < config.L2_MIN_CREATION_GAP:
                return False, f"too_soon_gap_{current_turn - self.last_l2_creation_turn}"
            
            # Check if we have enough content to process
            chunk_size = current_turn - self.last_topic_start_turn
            if chunk_size < config.L2_MIN_CHUNK_SIZE:
                return False, f"chunk_too_small_{chunk_size}"
            
            # Fallback trigger: single topic too long
            if current_turn - self.last_topic_start_turn >= config.L2_FALLBACK_TURN_LIMIT:
                self.last_topic_start_turn = current_turn
                self.last_l2_creation_turn = current_turn
                return True, "fallback_turn_limit"
            
            # Semantic trigger: topic change detection
            rolling_avg = self.get_rolling_average_embedding()
            if rolling_avg is not None:
                similarity = cosine_similarity([current_embedding], [rolling_avg])[0][0]
                if similarity < config.L2_SIMILARITY_THRESHOLD:
                    self.last_topic_start_turn = current_turn
                    self.last_l2_creation_turn = current_turn
                    return True, f"semantic_change_similarity_{similarity:.3f}"
            
            return False, "no_trigger"
    
    def mark_range_processed(self, start_turn: int, end_turn: int):
        """Mark a range as processed to prevent duplicates."""
        with self._lock:
            self.processed_ranges.add((start_turn, end_turn))
    
    def is_range_processed(self, start_turn: int, end_turn: int) -> bool:
        """Check if a range has already been processed."""
        with self._lock:
            # Check for exact match or overlap
            for processed_start, processed_end in self.processed_ranges:
                if (start_turn >= processed_start and start_turn < processed_end) or \
                   (end_turn > processed_start and end_turn <= processed_end) or \
                   (start_turn <= processed_start and end_turn >= processed_end):
                    return True
            return False

# Global L2 state
l2_state = L2State()

def init_l2_db():
    """Initialize the L2 episodic cache database."""
    try:
        with duckdb.connect(L2_DB_PATH) as conn:
            # Create L2 episodic summaries table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS l2_episodic_summaries (
                    l2_id VARCHAR PRIMARY KEY DEFAULT (CONCAT('l2_', CAST(epoch_ms(current_timestamp) AS VARCHAR))),
                    conversation_id VARCHAR DEFAULT 'default_conversation',
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    turn_start INTEGER NOT NULL,
                    turn_end INTEGER NOT NULL,
                    embedding FLOAT[{config.EMBEDDING_DIM}],
                    topic VARCHAR,
                    key_outcomes TEXT[],
                    entities_mentioned TEXT[],
                    user_sentiment VARCHAR,
                    raw_text_summary TEXT NOT NULL,
                    source_messages JSON,
                    eligible_for_consolidation BOOLEAN DEFAULT false,
                    trigger_reason VARCHAR
                );
            """)
            
            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_l2_created_at ON l2_episodic_summaries(created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_l2_consolidation ON l2_episodic_summaries(eligible_for_consolidation, created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_l2_conversation ON l2_episodic_summaries(conversation_id, turn_start);")
        
        olliePrint_simple("L2 Episodic Cache database initialized")
    except Exception as e:
        olliePrint_simple(f"Failed to initialize L2 database: {e}", level='error')
        raise

def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for text using Ollama."""
    try:
        client = ollama_manager.get_client()
        response = client.embeddings(
            model=config.EMBED_MODEL,
            prompt=text
        )
        embedding = response.get("embedding", [])
        if len(embedding) != config.EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {config.EMBEDDING_DIM}, got {len(embedding)}")
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        olliePrint_simple(f"Failed to get embedding: {e}", level='error')
        return None

def analyze_conversation_chunk(messages: List[Dict], turn_start: int, turn_end: int, trigger_reason: str) -> Optional[Dict]:
    """Analyze conversation chunk and extract structured L2 summary."""
    try:
        # Format messages for analysis
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = msg['role'].upper()
            content = msg['content']
            
            # Include thinking context if available (for rich analysis)
            if 'thinking' in msg and msg['thinking']:
                thinking = msg['thinking']
                formatted_messages.append(f"{role}: {content}\n[THINKING: {thinking}]")
            else:
                formatted_messages.append(f"{role}: {content}")
        
        messages_text = "\n\n".join(formatted_messages)
        
        prompt = config.L2_ANALYSIS_PROMPT.format(
            turn_start=turn_start,
            turn_end=turn_end,
            trigger_reason=trigger_reason,
            messages_text=messages_text
        )

        chat_messages = [
            {"role": "system", "content": config.L2_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = ollama_manager.chat_concurrent_safe(
            model=config.L2_ANALYSIS_MODEL,
            messages=chat_messages,
            stream=False,
            format="json"
        )
        
        response_text = response.get('message', {}).get('content', '').strip()
        if not response_text:
            return None
        
        try:
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError as e:
            olliePrint_simple(f"Failed to parse L2 analysis JSON: {e}", level='error')
            return None
        
    except Exception as e:
        olliePrint_simple(f"Failed to analyze conversation chunk: {e}", level='error')
        return None

def create_l2_summary(conversation_history: List[Dict], chunk_start: int, chunk_end: int, trigger_reason: str) -> bool:
    """Create and store L2 episodic summary."""
    try:
        # Check for duplicate processing
        if l2_state.is_range_processed(chunk_start + 1, chunk_end):
            olliePrint_simple(f"L2 range {chunk_start + 1}-{chunk_end} already processed, skipping", level='warning')
            return False
        
        # Get the conversation chunk
        chunk_messages = conversation_history[chunk_start:chunk_end]
        if not chunk_messages:
            return False
        
        # Check minimum chunk size
        if len(chunk_messages) < config.L2_MIN_CHUNK_SIZE:
            olliePrint_simple(f"L2 chunk too small ({len(chunk_messages)} < {config.L2_MIN_CHUNK_SIZE}), skipping", level='warning')
            return False
        
        # Analyze the chunk
        analysis = analyze_conversation_chunk(chunk_messages, chunk_start + 1, chunk_end, trigger_reason)
        if not analysis:
            olliePrint_simple(f"L2 analysis failed for turns {chunk_start + 1}-{chunk_end}", level='warning')
            return False
        
        # Get embedding for the summary
        summary_text = f"{analysis.get('topic', '')} {analysis.get('raw_text_summary', '')}"
        embedding = get_embedding(summary_text)
        if embedding is None:
            return False
        
        # Store in L2 database
        l2_id = f"l2_{uuid.uuid4().hex[:12]}"
        conversation_id = "default_conversation"  # Could be enhanced later
        
        with duckdb.connect(L2_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO l2_episodic_summaries 
                (l2_id, conversation_id, turn_start, turn_end, embedding, topic, 
                 key_outcomes, entities_mentioned, user_sentiment, raw_text_summary, 
                 source_messages, trigger_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                l2_id,
                conversation_id,
                chunk_start + 1,  # Convert to 1-based indexing
                chunk_end,
                embedding.tolist(),
                analysis.get('topic', ''),
                analysis.get('key_outcomes', []),
                analysis.get('entities_mentioned', []),
                analysis.get('user_sentiment', 'neutral'),
                analysis.get('raw_text_summary', ''),
                json.dumps(chunk_messages),
                trigger_reason
            ))
        
        # Check if we need to clean up old L2 memories
        with duckdb.connect(L2_DB_PATH) as conn:
            count_result = conn.execute("SELECT COUNT(*) FROM l2_episodic_summaries").fetchone()
            if count_result and count_result[0] > config.L2_MAX_MEMORIES:
                # Remove oldest memories
                to_remove = count_result[0] - config.L2_MAX_MEMORIES
                conn.execute("""
                    DELETE FROM l2_episodic_summaries 
                    WHERE l2_id IN (
                        SELECT l2_id FROM l2_episodic_summaries 
                        ORDER BY created_at ASC 
                        LIMIT ?
                    )
                """, (to_remove,))
        
        # Mark old L2 entries as eligible for consolidation
        cutoff_date = datetime.now() - timedelta(days=config.L2_CONSOLIDATION_DAYS)
        with duckdb.connect(L2_DB_PATH) as conn:
            conn.execute("""
                UPDATE l2_episodic_summaries 
                SET eligible_for_consolidation = true 
                WHERE created_at < ? AND eligible_for_consolidation = false
            """, (cutoff_date,))
        
        # Mark range as processed to prevent duplicates
        l2_state.mark_range_processed(chunk_start + 1, chunk_end)
        
        olliePrint_simple(f"L2 summary created: {analysis.get('topic', 'Unknown topic')} (turns {chunk_start + 1}-{chunk_end})")
        return True
        
    except Exception as e:
        olliePrint_simple(f"Failed to create L2 summary: {e}", level='error')
        return False

def query_l2_context(user_message: str) -> str:
    """Query L2 for relevant context based on user message."""
    try:
        # Get embedding for user message
        query_embedding = get_embedding(user_message)
        if query_embedding is None:
            return ""
        
        # Search L2 database
        relevant_summaries = []
        
        with duckdb.connect(L2_DB_PATH) as conn:
            results = conn.execute("""
                SELECT topic, raw_text_summary, created_at, embedding, turn_start, turn_end
                FROM l2_episodic_summaries 
                ORDER BY created_at DESC
            """).fetchall()
            
            for row in results:
                topic, summary, created_at, embedding_list, turn_start, turn_end = row
                if embedding_list:
                    embedding = np.array(embedding_list, dtype=np.float32)
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    
                    if similarity > config.L2_RETRIEVAL_THRESHOLD:
                        relevant_summaries.append({
                            'topic': topic,
                            'summary': summary,
                            'created_at': created_at,
                            'similarity': similarity,
                            'turn_range': f"{turn_start}-{turn_end}"
                        })
        
        if not relevant_summaries:
            return ""
        
        # Sort by similarity and take top results
        relevant_summaries.sort(key=lambda x: x['similarity'], reverse=True)
        top_summaries = relevant_summaries[:config.L2_RETRIEVAL_LIMIT]
        
        # Format for CRAP
        context_lines = []
        for item in top_summaries:
            try:
                dt = item['created_at']
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                formatted_date = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = str(item['created_at'])[:16]
            
            context_lines.append(f"[{formatted_date}] {item['topic']}: {item['summary']} (turns {item['turn_range']})")
        
        return f"""(L2 EPISODIC CONTEXT)
{chr(10).join(context_lines)}
(END L2 EPISODIC CONTEXT)"""
        
    except Exception as e:
        olliePrint_simple(f"Failed to query L2 context: {e}", level='error')
        return ""

def get_l2_consolidation_candidates() -> List[Dict]:
    """Get L2 summaries eligible for L3 consolidation."""
    try:
        with duckdb.connect(L2_DB_PATH) as conn:
            results = conn.execute("""
                SELECT l2_id, topic, raw_text_summary, key_outcomes, entities_mentioned,
                       created_at, turn_start, turn_end, source_messages
                FROM l2_episodic_summaries 
                WHERE eligible_for_consolidation = true
                ORDER BY created_at ASC
                LIMIT ?
            """, (config.SLEEP_CYCLE_L2_CONSOLIDATION_BATCH,)).fetchall()
            
            candidates = []
            for row in results:
                candidates.append({
                    'l2_id': row[0],
                    'topic': row[1],
                    'summary': row[2],
                    'key_outcomes': row[3],
                    'entities_mentioned': row[4],
                    'created_at': row[5],
                    'turn_start': row[6],
                    'turn_end': row[7],
                    'source_messages': json.loads(row[8]) if row[8] else []
                })
            
            return candidates
            
    except Exception as e:
        olliePrint_simple(f"Failed to get L2 consolidation candidates: {e}", level='error')
        return []

def mark_l2_consolidated(l2_ids: List[str]):
    """Mark L2 summaries as consolidated and remove them."""
    try:
        with duckdb.connect(L2_DB_PATH) as conn:
            placeholders = ','.join(['?' for _ in l2_ids])
            conn.execute(f"""
                DELETE FROM l2_episodic_summaries 
                WHERE l2_id IN ({placeholders})
            """, l2_ids)
        
        olliePrint_simple(f"Consolidated and removed {len(l2_ids)} L2 summaries")
        
    except Exception as e:
        olliePrint_simple(f"Failed to mark L2 summaries as consolidated: {e}", level='error')

def process_l2_creation(conversation_history: List[Dict], current_turn: int, user_message: str):
    """Process potential L2 creation based on semantic triggers."""
    try:
        # Get embedding for current user message
        user_embedding = get_embedding(user_message)
        if user_embedding is None:
            olliePrint_simple("L2 trigger skipped: failed to get embedding", level='warning')
            return
        
        # Add to rolling window
        l2_state.add_user_embedding(user_embedding, current_turn)
        
        # Check if L2 summary should be created
        should_create, trigger_reason = l2_state.should_create_l2_summary(user_embedding, current_turn)
        
        if should_create:
            # Find the range of messages to summarize
            # We want to capture messages that still have thinking context (turns 1-3 in F.R.E.D.'s window)
            chunk_start = max(0, l2_state.last_topic_start_turn - 1)  # Start from previous topic
            chunk_end = current_turn - 1  # Up to but not including current turn
            
            if chunk_end > chunk_start:
                olliePrint_simple(f"L2 trigger: {trigger_reason} - creating summary for turns {chunk_start + 1}-{chunk_end}")
                success = create_l2_summary(conversation_history, chunk_start, chunk_end, trigger_reason)
                if not success:
                    olliePrint_simple(f"L2 summary creation failed for turns {chunk_start + 1}-{chunk_end}", level='warning')
            else:
                olliePrint_simple(f"L2 trigger: {trigger_reason} - but insufficient message range", level='warning')
        else:
            # Only log trigger check results if interesting
            if not trigger_reason.startswith("no_trigger"):
                olliePrint_simple(f"L2 trigger check: {trigger_reason}")
                
    except Exception as e:
        olliePrint_simple(f"L2 creation processing failed: {e}", level='error')

# Initialize database on import
init_l2_db() 