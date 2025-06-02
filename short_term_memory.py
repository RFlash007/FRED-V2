import sqlite3
import json
import numpy as np
import requests
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from config import config
import os

# STM Database path
STM_DB_PATH = "short_term_memory.db"

# Thread-safe cache for embeddings
class STMCache:
    def __init__(self, max_size=50):
        self.embeddings = {}  # id -> numpy array
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def add(self, stm_id: int, embedding: np.ndarray):
        with self._lock:
            if len(self.embeddings) >= self.max_size:
                # Remove oldest entry
                oldest_id = min(self.embeddings.keys())
                del self.embeddings[oldest_id]
            self.embeddings[stm_id] = embedding
    
    def get_all(self) -> Dict[int, np.ndarray]:
        with self._lock:
            return self.embeddings.copy()

# Global cache instance
stm_cache = STMCache()

def init_stm_db():
    """Initialize the short-term memory database."""
    try:
        with sqlite3.connect(STM_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS short_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source_messages TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_range TEXT NOT NULL
                )
            """)
        logging.info("STM database initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize STM database: {e}")
        raise

def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for text using Ollama."""
    try:
        response = requests.post(
            config.OLLAMA_EMBED_URL,
            json={"model": config.EMBED_MODEL, "prompt": text},
            timeout=config.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        if len(embedding) != config.EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {config.EMBEDDING_DIM}, got {len(embedding)}")
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logging.error(f"Failed to get embedding: {e}")
        return None

def analyze_conversation_segment(messages: List[Dict], focus_start: int, focus_end: int, context_start: int, context_end: int) -> Optional[str]:
    """Analyze conversation segment with context but focus learning extraction on specific range."""
    try:
        # Format all messages for context
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = msg['role'].upper()
            content = msg['content']
            # Mark the focus range for the LLM
            if context_start + i >= focus_start and context_start + i < focus_end:
                formatted_messages.append(f">>> {role}: {content}")  # Mark focus messages
            else:
                formatted_messages.append(f"{role}: {content}")
        
        messages_text = "\n".join(formatted_messages)
        
        prompt = f"""Analyze this conversation segment and extract 3-5 key learnings ONLY from the messages marked with ">>>" (messages {focus_start}-{focus_end}). 

The other messages provide context but should NOT be analyzed for new learnings.

Focus your learning extraction on:
- User preferences, habits, characteristics, or personality traits
- Important facts, decisions, or information mentioned  
- Emotional context, mood, or interpersonal dynamics
- Context that would be valuable for future conversations

Be concise but include emotional nuance. Output as a simple list.

Conversation (context: messages {context_start}-{context_end}, focus: messages {focus_start}-{focus_end}):
{messages_text}

Key learnings from ONLY the marked ">>>" messages:"""

        response = requests.post(
            config.OLLAMA_GENERATE_URL,
            json={
                "model": config.STM_ANALYSIS_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=config.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        return result if result else None
        
    except Exception as e:
        logging.error(f"Failed to analyze conversation segment: {e}")
        return None

def check_similarity_and_store(content: str, source_messages: List[Dict], message_range: str) -> bool:
    """Check similarity against existing memories and store if unique."""
    try:
        # Get embedding for new content
        new_embedding = get_embedding(content)
        if new_embedding is None:
            return False
        
        # Load existing embeddings
        existing_embeddings = []
        existing_ids = []
        
        with sqlite3.connect(STM_DB_PATH) as conn:
            cursor = conn.execute("SELECT id, embedding FROM short_term_memories ORDER BY created_at DESC")
            for row in cursor:
                stm_id, embedding_blob = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                existing_embeddings.append(embedding)
                existing_ids.append(stm_id)
        
        # Check similarity if we have existing memories
        if existing_embeddings:
            existing_matrix = np.array(existing_embeddings)
            similarities = cosine_similarity([new_embedding], existing_matrix)[0]
            max_similarity = np.max(similarities)
            
            if max_similarity > config.STM_SIMILARITY_THRESHOLD:
                logging.info(f"STM content too similar (similarity: {max_similarity:.3f}), skipping storage")
                return False
        
        # Store the new memory
        embedding_blob = new_embedding.tobytes()
        source_json = json.dumps(source_messages)
        
        with sqlite3.connect(STM_DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO short_term_memories (content, embedding, source_messages, message_range) VALUES (?, ?, ?, ?)",
                (content, embedding_blob, source_json, message_range)
            )
            new_id = cursor.lastrowid
        
        # Add to cache
        stm_cache.add(new_id, new_embedding)
        
        # Check if we need to remove old memories
        with sqlite3.connect(STM_DB_PATH) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM short_term_memories")
            count = cursor.fetchone()[0]
            
            if count > config.STM_MAX_MEMORIES:
                # Remove oldest memories
                to_remove = count - config.STM_MAX_MEMORIES
                conn.execute(
                    "DELETE FROM short_term_memories WHERE id IN (SELECT id FROM short_term_memories ORDER BY created_at ASC LIMIT ?)",
                    (to_remove,)
                )
        
        logging.info(f"Stored new STM: {content[:100]}...")
        return True
        
    except Exception as e:
        logging.error(f"Failed to check similarity and store: {e}")
        return False

def query_stm_context(user_message: str) -> str:
    """Query STM for relevant context based on user message."""
    try:
        # Get embedding for user message
        query_embedding = get_embedding(user_message)
        if query_embedding is None:
            return ""
        
        # Load and compare with stored memories
        relevant_memories = []
        
        with sqlite3.connect(STM_DB_PATH) as conn:
            cursor = conn.execute("SELECT content, embedding, created_at FROM short_term_memories ORDER BY created_at DESC")
            for row in cursor:
                content, embedding_blob, created_at = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                if similarity > 0.3:  # Lower threshold for context retrieval
                    relevant_memories.append((content, created_at, similarity))
        
        if not relevant_memories:
            return ""
        
        # Sort by similarity and take top results
        relevant_memories.sort(key=lambda x: x[2], reverse=True)
        top_memories = relevant_memories[:config.STM_RETRIEVAL_LIMIT]
        
        # Format for F.R.E.D.
        context_lines = []
        for content, created_at, similarity in top_memories:
            # Parse timestamp
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                formatted_date = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = created_at[:16]  # Fallback
            
            context_lines.append(f"[{formatted_date}] {content}")
        
        return f"""(RECENTLY POSSIBLY RELATED CONTEXT)
{chr(10).join(context_lines)}
(END RECENTLY POSSIBLY RELATED CONTEXT)"""
        
    except Exception as e:
        logging.error(f"Failed to query STM context: {e}")
        return ""

def process_stm_analysis(conversation_history: List[Dict], last_analyzed_index: int, current_total: int):
    """Background processing for STM analysis with context-aware non-overlapping."""
    try:
        # Calculate new messages to analyze
        new_messages_count = current_total - last_analyzed_index
        if new_messages_count < config.STM_TRIGGER_INTERVAL:
            return  # Not enough new messages
        
        # Determine focus range (new messages only)
        focus_start = last_analyzed_index + 1
        focus_end = current_total
        
        # Determine context range (more messages for context)
        context_window = max(config.STM_ANALYSIS_WINDOW, new_messages_count + 5)  # At least 5 extra for context
        context_start = max(0, current_total - context_window)
        context_end = current_total
        
        # Get context messages
        context_messages = conversation_history[context_start:context_end]
        
        if not context_messages:
            return
        
        # Create message range description
        message_range = f"focus {focus_start}-{focus_end} (context {context_start+1}-{context_end})"
        
        # Analyze the conversation segment
        logging.info(f"Analyzing STM for {message_range}")
        learnings = analyze_conversation_segment(
            context_messages, 
            focus_start, 
            focus_end, 
            context_start + 1,  # Convert to 1-based indexing for display
            context_end
        )
        
        if learnings:
            # Store learnings with focus messages for reference
            focus_messages = conversation_history[last_analyzed_index:current_total]
            success = check_similarity_and_store(learnings, focus_messages, message_range)
            if success:
                logging.info(f"STM analysis complete for {message_range}")
            else:
                logging.info(f"STM analysis skipped (duplicate) for {message_range}")
        else:
            logging.warning(f"STM analysis failed for {message_range}")
            
    except Exception as e:
        logging.error(f"STM background processing failed: {e}")

# Initialize database on import
init_stm_db() 