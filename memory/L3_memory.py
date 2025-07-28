# Import necessary libraries
import duckdb
import datetime
import requests
import json
import time
import os # For potential environment variables later
import threading
from ollie_print import olliePrint_simple
from contextlib import contextmanager
from config import config, ollama_manager # Import the config and connection manager
from typing import List, Dict, Optional, Tuple, Set, Any
from memory.supersession_helpers import (
    resolve_supersession_chain,
    transfer_pending_edge_tasks, 
    inherit_edges_from_superseded,
    handle_updates_edge_supersession,
    check_edge_exists_through_supersession
)

# --- Configuration ---
# Set default path inside the memory folder
DB_FILE = os.path.join('memory', 'memory.db')  # Default that can be overridden
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
LLM_DECISION_MODEL = os.getenv('LLM_DECISION_MODEL', 'hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M') # As requested
EMBEDDING_DIM = 768 # As specified for nomic-embed-text

# Use centralized Ollama connection manager for all L3 operations

# How many similar nodes to check for automatic edge creation (increased for richer connectivity)
AUTO_EDGE_SIMILARITY_CHECK_LIMIT = 5

# Configuration constants now imported from config.py

# Connection pooling
_connection_lock = threading.Lock()
_thread_connections = threading.local()

@contextmanager
def get_db_connection():
    """Get a database connection with thread-local caching for efficiency."""
    if not hasattr(_thread_connections, 'connection') or _thread_connections.connection is None:
        _thread_connections.connection = duckdb.connect(DB_FILE)
    
    try:
        yield _thread_connections.connection
    except Exception as e:
        # Close connection on error to ensure clean state
        try:
            _thread_connections.connection.close()
        except:
            pass
        _thread_connections.connection = None
        raise e

# Define valid relationship types directly from schema for validation and prompting
VALID_REL_TYPES = [
    'instanceOf', 'relatedTo', 'updates',
    'contains', 'partOf', 'precedes',
    'causes',
    'createdBy', 'hasOwner',
    'locatedAt', 'dependsOn', 'servesPurpose',
    'occursDuring', 'enablesGoal', 'activatesIn', 'contextualAnalog', 'sourceAttribution'
]

# --- Logging Setup ---
# (Replaced standard logging with olliePrint)

# --- Database Initialization ---
def init_db():
    """Initializes the DuckDB database and tables if they don't exist."""
    # This function will fail in environments without duckdb installed
    try:
        with get_db_connection() as con:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS nodes (
                    nodeid BIGINT PRIMARY KEY DEFAULT (CAST(epoch_ms(current_timestamp) AS BIGINT)),
                    type TEXT CHECK (type IN ('Semantic', 'Episodic', 'Procedural')),
                    label TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    last_access TIMESTAMP DEFAULT current_timestamp,
                    superseded_at TIMESTAMP,
                    parent_id BIGINT,
                    embedding FLOAT[{EMBEDDING_DIM}],
                    embed_model TEXT,
                    target_date TIMESTAMP
                );
            """)
            # Use ', '.join for correct SQL syntax in CHECK constraint
            valid_types_sql = "', '".join(VALID_REL_TYPES)
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS edges (
                    sourceid BIGINT NOT NULL,
                    targetid BIGINT NOT NULL,
                    rel_type TEXT CHECK (rel_type IN ('{valid_types_sql}')),
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    PRIMARY KEY (sourceid, targetid, rel_type),
                    FOREIGN KEY (sourceid) REFERENCES nodes(nodeid),
                    FOREIGN KEY (targetid) REFERENCES nodes(nodeid)
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS pending_edge_creation_tasks (
                    task_id BIGINT PRIMARY KEY DEFAULT (CAST(epoch_ms(current_timestamp) AS BIGINT)),
                    node_id_to_process BIGINT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
                    attempts INTEGER DEFAULT 0,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    updated_at TIMESTAMP DEFAULT current_timestamp,
                    FOREIGN KEY (node_id_to_process) REFERENCES nodes(nodeid)
                );
            """)
        # Database initialized (logged by caller if needed)
    except ImportError:
        olliePrint_simple("DuckDB library not found. Database operations will fail.", level='error')
        # Depending on the use case, you might want to raise this error
        # raise ImportError("DuckDB library not found. Please install it.")
    except Exception as e:
         olliePrint_simple(f"Failed to initialize database: {e}", level='error')
         raise

# --- Ollama Interaction Functions ---

def get_embedding(text):
    """Gets the embedding for a given text using the Ollama Embedding API."""
    try:
        # CENTRALIZED CONNECTION: Use connection manager instead of direct client
        response = ollama_manager.embeddings(
            model=EMBED_MODEL,
            prompt=text
        )
        embedding = response.get("embedding", [])
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}")
        return embedding
    except Exception as e:
        olliePrint_simple(f"An unexpected error occurred during embedding: {e}", level='error')
        raise RuntimeError(f"Unexpected error getting embedding: {e}") from e


#Simple ollama call with JSON response
def call_ollama_generate(prompt, model=LLM_DECISION_MODEL):
    """Calls the Ollama chat API, expecting a JSON response."""
    try:
        messages = [
            {"role": "system", "content": config.L3_EDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = ollama_manager.chat_concurrent_safe(
            model=model,
            messages=messages,
            stream=False,
            format="json" # Explicitly request JSON format
        )
        
        response_content = response.get('message', {}).get('content', '')
        if not response_content:
             raise ValueError("Ollama response did not contain any content.")

        # Attempt to parse the JSON string from the content
        try:
            # The 'format="json"' parameter should make the content a valid JSON object already
            parsed_json = json.loads(response_content)
            return parsed_json
        except json.JSONDecodeError as json_e:
            olliePrint_simple(f"Failed to parse JSON response from Ollama: {response_content}. Error: {json_e}", level='error')
            raise ValueError(f"LLM returned invalid JSON: {response_content}") from json_e

    except Exception as e:
        # Catching generic exception as ollama library might raise its own specific errors
        olliePrint_simple(f"An unexpected error occurred during Ollama chat: {e}", level='error')
        # Re-raise as a generic runtime error to signal failure to the caller
        raise RuntimeError(f"Unexpected error during LLM generation: {e}") from e


# --- LLM Decision Functions ---

def discover_relationships_advanced(node_id: int, context_window: int = 5, 
                                   min_confidence: float = config.RELATIONSHIP_CONFIDENCE_THRESHOLD) -> List[Dict[str, Any]]:
    """
    MCP-inspired enhanced relationship discovery with context-aware analysis.
    
    This function provides sophisticated relationship discovery that considers broader
    context and provides confidence scoring for discovered relationships, improving
    upon the existing LLM-based relationship determination.
    
    Args:
        node_id (int): The node ID to discover relationships for
        context_window (int): Number of related nodes to consider for context
        min_confidence (float): Minimum confidence threshold for relationships
        
    Returns:
        List[Dict[str, Any]]: List of discovered relationships with confidence scores:
            - target_node_id: ID of the target node
            - relationship_type: Suggested relationship type
            - confidence: Confidence score (0.0-1.0)
            - reasoning: Explanation for the relationship
            - context_nodes: Nodes that influenced this decision
    """
    try:
        with get_db_connection() as con:
            # Get the source node details
            source_query = """
                SELECT nodeid, type, label, text, created_at
                FROM nodes 
                WHERE nodeid = ? AND superseded_at IS NULL
            """
            source_result = con.execute(source_query, [node_id]).fetchone()
            
            if not source_result:
                return []
            
            source_node = {
                'nodeid': source_result[0],
                'type': source_result[1], 
                'label': source_result[2],
                'text': source_result[3],
                'created_at': source_result[4]
            }
            
            # Get potential target nodes (most similar nodes)
            similar_nodes_query = """
                SELECT nodeid, type, label, text, created_at,
                       array_cosine_similarity(embedding, (
                           SELECT embedding FROM nodes WHERE nodeid = ?
                       )) as similarity
                FROM nodes 
                WHERE nodeid != ? AND superseded_at IS NULL
                AND embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT ?
            """
            
            similar_results = con.execute(similar_nodes_query, 
                                        [node_id, node_id, context_window * 2]).fetchall()
            
            discovered_relationships = []
            
            for target_result in similar_results:
                target_node = {
                    'nodeid': target_result[0],
                    'type': target_result[1],
                    'label': target_result[2], 
                    'text': target_result[3],
                    'created_at': target_result[4],
                    'similarity': target_result[5]
                }
                
                # Skip if relationship already exists
                existing_query = """
                    SELECT COUNT(*) FROM edges 
                    WHERE (sourceid = ? AND targetid = ?) OR (sourceid = ? AND targetid = ?)
                """
                existing_count = con.execute(existing_query, 
                                           [node_id, target_node['nodeid'], 
                                            target_node['nodeid'], node_id]).fetchone()[0]
                
                if existing_count > 0:
                    continue
                
                # Use enhanced LLM analysis with context
                relationship_analysis = determine_edge_type_llm_enhanced(
                    source_node, target_node, similar_results[:context_window]
                )
                
                if relationship_analysis and relationship_analysis.get('confidence', 0.0) >= min_confidence:
                    discovered_relationships.append({
                        'target_node_id': target_node['nodeid'],
                        'relationship_type': relationship_analysis['relationship_type'],
                        'confidence': relationship_analysis['confidence'],
                        'reasoning': relationship_analysis.get('reasoning', ''),
                        'context_nodes': [n[0] for n in similar_results[:context_window]],
                        'similarity_score': target_node['similarity']
                    })
            
            return discovered_relationships
            
    except Exception as e:
        olliePrint_simple(f"Error in advanced relationship discovery for node {node_id}: {e}", level='error')
        return []

def determine_edge_type_llm_enhanced(source_node_info: Dict[str, Any], 
                                   target_node_info: Dict[str, Any],
                                   context_nodes: List[Tuple] = None) -> Optional[Dict[str, Any]]:
    """
    Enhanced LLM-based relationship determination with context awareness and confidence scoring.
    
    This function improves upon the original determine_edge_type_llm by incorporating
    broader context and providing confidence scores for relationship predictions.
    
    Args:
        source_node_info (Dict): Source node information
        target_node_info (Dict): Target node information  
        context_nodes (List[Tuple], optional): Additional context nodes for better analysis
        
    Returns:
        Dict[str, Any]: Relationship analysis with confidence score:
            - relationship_type: Suggested relationship type
            - confidence: Confidence score (0.0-1.0)
            - reasoning: Explanation for the relationship
    """
    try:
        # Prepare context information
        context_info = ""
        if context_nodes:
            context_info = "\n\nRELATED CONTEXT NODES:\n"
            for i, ctx_node in enumerate(context_nodes[:5], 1):  # Increased context limit to 5 nodes
                context_info += f"{i}. {ctx_node[2]} ({ctx_node[1]}): {ctx_node[3][:100]}...\n"
        
        # Enhanced prompt with confidence scoring
        enhanced_prompt = f"""Analyze the relationship between these two memory nodes and provide a confidence score.

SOURCE NODE:
Type: {source_node_info['type']}
Label: {source_node_info['label']}
Text: {source_node_info['text'][:200]}...
Created: {source_node_info['created_at']}

TARGET NODE:
Type: {target_node_info['type']}
Label: {target_node_info['label']}
Text: {target_node_info['text'][:200]}...
Created: {target_node_info['created_at']}{context_info}

VALID RELATIONSHIP TYPES: {', '.join(VALID_REL_TYPES)}

Analyze the semantic relationship between these nodes considering:
1. Content similarity and thematic connections
2. Temporal relationships (creation dates, event sequences)
3. Hierarchical or categorical relationships
4. Causal or dependency relationships
5. Context from related nodes

Respond with ONLY a JSON object containing:
{{
    "relationship_type": "<one of the valid types or null>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation for the relationship and confidence level>"
}}

If no clear relationship exists, use null for relationship_type and low confidence."""
        
        # Get LLM response
        response = ollama_manager.chat_concurrent_safe(
            model=LLM_DECISION_MODEL,
            messages=[{"role": "user", "content": enhanced_prompt}],
            stream=False,
            options={"temperature": 0.1}  # Low temperature for consistent analysis
        )
        
        response_text = response.get('message', {}).get('content', '').strip()
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text)
            
            # Validate response structure
            if not isinstance(analysis, dict):
                return None
                
            relationship_type = analysis.get('relationship_type')
            confidence = analysis.get('confidence', 0.0)
            reasoning = analysis.get('reasoning', '')
            
            # Validate relationship type
            if relationship_type and relationship_type not in VALID_REL_TYPES:
                olliePrint_simple(f"Invalid relationship type suggested: {relationship_type}", level='warning')
                return None
            
            # Validate confidence score
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                confidence = 0.0
            
            return {
                'relationship_type': relationship_type,
                'confidence': float(confidence),
                'reasoning': reasoning
            }
            
        except json.JSONDecodeError:
            olliePrint_simple(f"Failed to parse LLM relationship analysis: {response_text}", level='warning')
            return None
            
    except Exception as e:
        olliePrint_simple(f"Error in enhanced LLM relationship determination: {e}", level='error')
        return None

# Legacy determine_edge_type_llm function removed - functionality now handled by determine_edge_type_llm_enhanced
# All calls have been migrated to use the MCP-inspired enhanced relationship discovery system

# --- Core Memory Functions ---

def add_memory_with_observations(label: str, text: str, memory_type: str, 
                               observations: Optional[List[str]] = None, 
                               parent_id: Optional[int] = None, 
                               target_date: Optional[str] = None, 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    MCP-inspired memory addition with observation-style metadata support.
    
    This function extends the core add_memory functionality with structured observations
    and metadata, similar to industry-standard MCP memory servers while maintaining
    full backward compatibility with the existing system.
    
    Args:
        label (str): Concise label or title for the memory node
        text (str): Main text content of the memory
        memory_type (str): Memory type ('Semantic', 'Episodic', 'Procedural')
        observations (List[str], optional): List of observation statements about this memory
        parent_id (int, optional): ID of parent node for hierarchical relationships
        target_date (str, optional): ISO format date for future events
        metadata (Dict[str, Any], optional): Additional structured metadata
        
    Returns:
        Dict[str, Any]: Enhanced result with observation and metadata information:
            - nodeid: ID of created node
            - success: Whether operation succeeded
            - observations_added: Number of observations processed
            - metadata_keys: List of metadata keys stored
            - relationships_created: Number of automatic relationships created
    """
    # Prepare enhanced text content with observations
    enhanced_text = text
    
    if observations:
        # Append observations in a structured format
        observations_section = "\n\n[OBSERVATIONS]\n"
        for i, obs in enumerate(observations, 1):
            observations_section += f"{i}. {obs}\n"
        enhanced_text += observations_section
    
    if metadata:
        # Append metadata in a structured format
        metadata_section = "\n\n[METADATA]\n"
        for key, value in metadata.items():
            metadata_section += f"{key}: {value}\n"
        enhanced_text += metadata_section
    
    # Call the existing add_memory function with enhanced text
    result = add_memory(label, enhanced_text, memory_type, parent_id, target_date)
    
    # Return enhanced result information
    return {
        'nodeid': result,
        'success': result is not None,
        'observations_added': len(observations) if observations else 0,
        'metadata_keys': list(metadata.keys()) if metadata else [],
        'relationships_created': 0,  # Will be updated by automatic edge creation
        'enhanced_text_length': len(enhanced_text),
        'original_text_length': len(text)
    }

def add_memory(label: str, text: str, memory_type: str, parent_id: Optional[int] = None, 
              target_date: Optional[str] = None, observations: Optional[List[str]] = None, 
              metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Enhanced memory addition with MCP-inspired observation and metadata support.
    
    This unified function combines the original add_memory functionality with structured
    observations and metadata capabilities, providing a single interface for CRAP.
    
    Args:
        label (str): Concise label or title for the memory node
        text (str): Main text content of the memory
        memory_type (str): Memory type ('Semantic', 'Episodic', 'Procedural')
        parent_id (int, optional): ID of parent node for hierarchical relationships
        target_date (str, optional): ISO format date for future events
        observations (List[str], optional): List of observation statements about this memory
        metadata (Dict[str, Any], optional): Additional structured metadata
        
    Returns:
        Optional[int]: ID of created node, or None if creation failed
    """
    # If observations or metadata provided, use enhanced text formatting
    if observations or metadata:
        enhanced_text = text
        
        if observations:
            # Append observations in a structured format
            observations_section = "\n\n[OBSERVATIONS]\n"
            for i, obs in enumerate(observations, 1):
                observations_section += f"{i}. {obs}\n"
            enhanced_text += observations_section
        
        if metadata:
            # Append metadata in a structured format
            metadata_section = "\n\n[METADATA]\n"
            for key, value in metadata.items():
                metadata_section += f"{key}: {value}\n"
            enhanced_text += metadata_section
        
        text = enhanced_text
    olliePrint_simple(f"Attempting to add memory with label: {label}")
    try:
        # 1. Get Embedding
        olliePrint_simple("Generating embedding...")
        embedding = get_embedding(text)
        olliePrint_simple("Embedding generated.")

        # 2. Add Node to DB
        # This block will fail if duckdb is not installed
        try:
            with duckdb.connect(DB_FILE) as con:
                # Use epoch milliseconds for a potentially more unique default ID
                new_id = int(time.time() * 1000)
                con.execute(
                    """
                    INSERT INTO nodes (nodeid, type, label, text, parent_id, embedding, embed_model, target_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING nodeid;
                    """,
                    [new_id, memory_type, label, text, parent_id, embedding, EMBED_MODEL, target_date]
                )
                result = con.fetchone()
                if not result:
                     # This case should ideally not happen if INSERT is successful without error
                     raise RuntimeError("Failed to retrieve nodeid after insertion.")
                new_nodeid = result[0]
                olliePrint_simple(f"Memory node added successfully with ID: {new_nodeid}")

                # Add task for edge creation
                # Using epoch microseconds for task_id to ensure uniqueness and ordering if needed
                task_id = int(time.time() * 1000000) 
                con.execute(
                    """
                    INSERT INTO pending_edge_creation_tasks (task_id, node_id_to_process)
                    VALUES (?, ?);
                    """,
                    [task_id, new_nodeid]
                )
                olliePrint_simple(f"Pending edge creation task added for node {new_nodeid} with task_id {task_id}")

            # End of 'with duckdb.connect...' block, transaction commits automatically on success
            return new_nodeid

        except ImportError:
            olliePrint_simple("DuckDB library not found. Cannot add memory node.", level='error')
            raise ImportError("DuckDB library not found. Cannot add memory node.")
        except duckdb.Error as db_e:
             olliePrint_simple(f"Database error during node insertion: {db_e}", level='error')
             raise # Re-raise critical DB errors


    except (ConnectionError, ValueError, RuntimeError) as e:
        # Errors during classification or embedding are critical
        olliePrint_simple(f"Failed to add memory due to pre-processing error: {e}", level='error')
        # Don't return None, let the exception propagate as per requirement
        raise
    except Exception as e:
         olliePrint_simple(f"Unexpected critical error in add_memory: {e}", level='error')
         raise


def supersede_memory(old_nodeid, new_label, new_text, new_memory_type, target_date=None):
    """Supersedes an old memory node with a new one, adding an 'updates' edge.

    Args:
        old_nodeid: The ID of the node to supersede.
        new_label: The label for the new node.
        new_text: The text content for the new node.
        new_memory_type: The type ('Semantic', 'Episodic', 'Procedural') for the new node.
        target_date: Optional timestamp for future events or activities.

    Raises:
        ValueError: If the old node is not found, already superseded, or invalid memory type provided.
        ImportError: If DuckDB is not found.
        ConnectionError, RuntimeError: For issues during embedding or DB operations.
        duckdb.Error: For database errors.
    """
    olliePrint_simple(f"Attempting to supersede node {old_nodeid} with label: {new_label}")

    # Validate memory type
    if new_memory_type not in ('Semantic', 'Episodic', 'Procedural'):
        raise ValueError(f"Invalid memory type provided: {new_memory_type}")

    try:
        with duckdb.connect(DB_FILE) as con:
            con.begin()
            try:
                # Check if old node exists and get parent_id
                old_node = con.execute("SELECT parent_id FROM nodes WHERE nodeid = ? AND superseded_at IS NULL", [old_nodeid]).fetchone()
                if not old_node:
                    already_superseded = con.execute("SELECT 1 FROM nodes WHERE nodeid = ?", [old_nodeid]).fetchone()
                    if already_superseded:
                         olliePrint_simple(f"Error: Node {old_nodeid} has already been superseded.", level='error')
                         raise ValueError(f"Node {old_nodeid} has already been superseded.")
                    else:
                         olliePrint_simple(f"Error: Node {old_nodeid} not found.", level='error')
                         raise ValueError(f"Node {old_nodeid} not found.")

                old_parent_id = old_node[0]

                # Mark old node as superseded and update last_access
                now = datetime.datetime.now()
                con.execute(
                    "UPDATE nodes SET superseded_at = ?, last_access = ? WHERE nodeid = ?",
                    [now, now, old_nodeid]
                )

                # Get embedding for the new memory (can raise errors)
                olliePrint_simple("Generating embedding for new memory...")
                embedding = get_embedding(new_text)
                olliePrint_simple("Embedding generated.")


                # Insert the new node using the provided new_memory_type
                new_id = int(time.time() * 1000)
                con.execute(
                    """
                    INSERT INTO nodes (nodeid, type, label, text, parent_id, embedding, embed_model, last_access, target_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING nodeid;
                    """,
                    [new_id, new_memory_type, new_label, new_text, old_parent_id, embedding, EMBED_MODEL, now, target_date]
                )
                result = con.fetchone()
                if not result:
                     raise RuntimeError("Failed to insert new node during supersede.")
                new_nodeid = result[0]
                olliePrint_simple(f"Added new node {new_nodeid} (type: {new_memory_type}) to supersede {old_nodeid}")

                # Add 'updates' edge using the consolidated add_edge function
                try:
                    add_edge(sourceid=new_nodeid,
                             targetid=old_nodeid,
                             rel_type='updates', # Explicit type
                             con=con)
                    # Success logging is handled within add_edge
                except (ValueError, ConnectionError, RuntimeError, duckdb.Error) as edge_e:
                     # Log edge error but commit node changes if insertion was successful
                     # add_edge logs its own errors too.
                     olliePrint_simple(f"Failed to add 'updates' edge during supersede (logged by supersede_memory, level='error'): {edge_e}. Node changes will still be committed.")
                except Exception as e:
                    # Catch unexpected errors from add_edge
                    olliePrint_simple(f"Unexpected error adding 'updates' edge during supersede: {e}. Node changes will still be committed.", level='error')


                con.commit()
                olliePrint_simple(f"Node {old_nodeid} successfully superseded by {new_nodeid}")
                return new_nodeid

            except (ConnectionError, ValueError, RuntimeError, duckdb.Error) as e:
                olliePrint_simple(f"Error during supersede operation: {e}", level='error')
                con.rollback()
                raise # Re-raise the critical error
            except Exception as e:
                 olliePrint_simple(f"Unexpected error during supersede: {e}", level='error')
                 con.rollback()
                 raise RuntimeError(f"Unexpected error during supersede: {e}") from e

    except ImportError:
        olliePrint_simple("DuckDB library not found. Cannot supersede memory.", level='error')
        raise ImportError("DuckDB library not found. Cannot supersede memory.")
    except Exception as e:
         olliePrint_simple(f"Unexpected critical error in supersede_memory: {e}", level='error')
         raise


def add_edge(sourceid: int, targetid: int, rel_type: str | None = None, con: duckdb.DuckDBPyConnection | None = None):
    """Adds an edge between two nodes within a provided database connection context.

    If rel_type is None, uses LLM to determine it.

    Args:
        sourceid: The ID of the source node.
        targetid: The ID of the target node.
        rel_type: The relationship type. If None, it will be determined by LLM.
        con: An active DuckDB database connection. This argument is required.

    Raises:
        ValueError: If nodes are invalid, superseded, or rel_type is invalid.
        ConnectionError: If LLM call fails.
        RuntimeError: For unexpected errors during LLM call or DB operation.
        duckdb.Error: For database errors during query execution.
        ImportError: If DuckDB is not installed (should be caught earlier ideally).
        TypeError: If 'con' is not provided.
    """
    if con is None:
         # Explicitly require the connection object
         raise TypeError("add_edge() missing 1 required positional argument: 'con'")

    # No need for close_conn, begin, commit, rollback, or finally block here anymore.
    # Transaction management is handled by the caller.

    try:
        # Check if nodes exist and are not superseded
        source_node = con.execute("SELECT label, text, type FROM nodes WHERE nodeid = ? AND superseded_at IS NULL", [sourceid]).fetchone()
        if not source_node:
            raise ValueError(f"Source node {sourceid} not found or is superseded.")

        # Modify target_node check based on rel_type
        if rel_type == 'updates':
            # For 'updates' relationship, the target (old node) CAN be superseded.
            # We still need to ensure it exists.
            target_node = con.execute("SELECT label, text, type FROM nodes WHERE nodeid = ?", [targetid]).fetchone()
            if not target_node:
                raise ValueError(f"Target node {targetid} (for 'updates' relationship) not found.")
        else:
            # For all other relationships, the target node must NOT be superseded
            target_node = con.execute("SELECT label, text, type FROM nodes WHERE nodeid = ? AND superseded_at IS NULL", [targetid]).fetchone()
            if not target_node:
                raise ValueError(f"Target node {targetid} not found or is superseded (for rel_type '{rel_type}').")

        source_node_info = {"label": source_node[0], "text": source_node[1], "type": source_node[2]}
        target_node_info = {"label": target_node[0], "text": target_node[1], "type": target_node[2]}

        # Determine relationship type if not provided
        determined_rel_type = rel_type # Use a different variable name
        if determined_rel_type is None:
            olliePrint_simple(f"Relationship type not provided for edge {sourceid} -> {targetid}. Determining using enhanced LLM analysis.")
            # This call can raise errors (ConnectionError, ValueError, RuntimeError), which will propagate up
            relationship_analysis = determine_edge_type_llm_enhanced(source_node_info, target_node_info)
            determined_rel_type = relationship_analysis.get('relationship_type', 'relatedTo')
            olliePrint_simple(f"Enhanced LLM determined relationship type: {determined_rel_type} (confidence: {relationship_analysis.get('confidence', 'unknown')})")
        elif determined_rel_type not in VALID_REL_TYPES:
             # Validate explicitly provided type
             raise ValueError(f"Provided relationship type '{determined_rel_type}' is not valid. Must be one of: {VALID_REL_TYPES}")

        # Check again if LLM failed to return a valid type (though determine_edge_type_llm should raise)
        if determined_rel_type not in VALID_REL_TYPES:
             olliePrint_simple(f"Invalid relationship type determined or provided: {determined_rel_type}", level='error')
             raise ValueError(f"Relationship type '{determined_rel_type}' is invalid.")


        # Insert the edge
        now = datetime.datetime.now()
        con.execute(
            "INSERT INTO edges (sourceid, targetid, rel_type, created_at) VALUES (?, ?, ?, ?)",
            [sourceid, targetid, determined_rel_type, now]
        )
        # Update last_access for both nodes involved in the edge
        con.execute("UPDATE nodes SET last_access = ? WHERE nodeid IN (?, ?)", [now, sourceid, targetid])
        olliePrint_simple(f"Successfully added edge ({determined_rel_type}) from {sourceid} to {targetid}")

    except (ValueError, ConnectionError, RuntimeError, duckdb.Error) as e:
        # Log the error, but let the caller handle transaction rollback/commit based on this exception.
        olliePrint_simple(f"Error adding edge from {sourceid} to {targetid} within provided transaction: {e}", level='error')
        raise # Re-raise the error for the caller to handle
    except Exception as e:
         # Catch any other unexpected errors
         olliePrint_simple(f"Unexpected error adding edge: {e}", level='error')
         raise RuntimeError(f"Unexpected error adding edge: {e}") from e


# search_memory_advanced functionality now integrated into unified search_memory function

def search_memory(query_text: str, memory_type: Optional[str] = None, limit: int = 3, 
                 future_events_only: bool = False, include_past_events: bool = True, 
                 use_keyword_search: bool = False, include_connections: bool = False, 
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 filters: Optional[Dict[str, Any]] = None, sort_by: str = "relevance") -> List[Dict[str, Any]]:
    """
    Enhanced memory search with MCP-inspired capabilities while maintaining backward compatibility.
    
    This unified search function combines the original search_memory functionality with
    advanced MCP-inspired features, providing a single, powerful interface for CRAP.
    
    Args:
        query_text (str): The text to search for
        memory_type (str, optional): Filter by memory type ('Semantic', 'Episodic', 'Procedural')
        limit (int): Maximum number of results to return
        future_events_only (bool): Only return memories with future target_date
        include_past_events (bool): Include memories with past target_date
        use_keyword_search (bool): Use keyword instead of semantic search
        include_connections (bool): Include relationship information
        start_date (str, optional): Start date filter (ISO format)
        end_date (str, optional): End date filter (ISO format)
        filters (Dict, optional): Advanced filtering options for enhanced capabilities
        sort_by (str): Sorting method ('relevance', 'date_created', 'date_accessed', 'similarity')
        
    Returns:
        List[Dict[str, Any]]: Search results with enhanced metadata when filters are used
    """
    # If advanced filters are provided, use the enhanced search capabilities
    if filters is not None:
        # Merge legacy parameters into filters for unified processing
        combined_filters = dict(filters)
        combined_filters.update({
            'memory_type': memory_type,
            'future_events_only': future_events_only,
            'include_past_events': include_past_events,
            'use_keyword_search': use_keyword_search,
            'include_connections': include_connections,
            'start_date': start_date,
            'end_date': end_date
        })
        
        # Use advanced search and return just the results for backward compatibility
        advanced_result = search_memory_advanced(query_text, combined_filters, sort_by, limit)
        return advanced_result.get('results', [])
    
    # Original search logic for backward compatibility
    results = []
    node_ids_updated = set() # Track updated nodes to avoid duplicate updates
    now = datetime.datetime.now()

    try:
        with duckdb.connect(DB_FILE) as con:
            if use_keyword_search:
                olliePrint_simple("Performing keyword search.")
                # Basic keyword tokenization (split by space)
                keywords = query_text.lower().split()
                if not keywords:
                    return []

                # Build query for keyword search
                # Search in label and text. Using OR for keywords, AND for conditions.
                # This is a simple approach; more advanced would require FTS5 or similar.
                conditions = ["superseded_at IS NULL"]
                params_keyword = []

                # Keyword matching clauses for label and text
                keyword_clauses = []
                for kw in keywords:
                    keyword_clauses.append("(LOWER(label) LIKE ? OR LOWER(text) LIKE ?)")
                    params_keyword.extend([f"%{kw}%", f"%{kw}%"])
                
                if keyword_clauses:
                    conditions.append(f"({' OR '.join(keyword_clauses)})") # Match any keyword

                if memory_type:
                    conditions.append("type = ?")
                    params_keyword.append(memory_type)
                
                if future_events_only:
                    conditions.append("(target_date IS NOT NULL AND target_date > ?)")
                    params_keyword.append(now)
                elif not include_past_events:
                    conditions.append("(target_date IS NULL OR target_date > ?)")
                    params_keyword.append(now)

                keyword_query = f"""
                SELECT nodeid, label, text, type, created_at, last_access, target_date, 0.0 AS similarity -- Placeholder for similarity
                FROM nodes
                WHERE {' AND '.join(conditions)}
                ORDER BY last_access DESC, created_at DESC -- Prioritize by access/creation for keywords
                LIMIT ?
                """
                params_keyword.append(limit)

                try:
                    keyword_search_results = con.execute(keyword_query, params_keyword).fetchall()
                    results.extend(keyword_search_results)
                    olliePrint_simple(f"Keyword search found {len(keyword_search_results)} nodes.")
                except duckdb.Error as db_err:
                    olliePrint_simple(f"Database error during keyword search: {db_err}", level='error')
                    return [] # Return empty list on DB error during keyword search
                except Exception as e:
                    olliePrint_simple(f"Unexpected error during keyword search: {e}", level='error')
                    return []

            else: # Semantic Search (default)
                try:
                    query_embedding = get_embedding(query_text) # Can raise ConnectionError, ValueError
                except (ConnectionError, ValueError) as e:
                    olliePrint_simple(f"Failed to get embedding for search query: {e}", level='error')
                    return [] # If embedding fails for semantic search, we cannot search

                olliePrint_simple("Performing embedding similarity search.")
                try:
                    base_query = """
                    SELECT nodeid, label, text, type, created_at, last_access, target_date, list_cosine_similarity(embedding, ?) AS similarity
                    FROM nodes
                    WHERE superseded_at IS NULL
                    """
                    params_semantic = [query_embedding]
                    
                    if memory_type:
                        base_query += " AND type = ?"
                        params_semantic.append(memory_type)
                    
                    if future_events_only:
                        base_query += " AND (target_date IS NOT NULL AND target_date > ?)"
                        params_semantic.append(now)
                    elif not include_past_events:
                        base_query += " AND (target_date IS NULL OR target_date > ?)"
                        params_semantic.append(now)

                    # --- NEW ADDITION for date range ---
                    if start_date:
                        base_query += " AND created_at >= ?"
                        params_semantic.append(start_date)
                    if end_date:
                        base_query += " AND created_at <= ?"
                        params_semantic.append(end_date)
                    # --- END NEW ADDITION ---

                    base_query += " ORDER BY similarity DESC LIMIT ?"
                    params_semantic.append(limit)

                    similarity_results = con.execute(base_query, params_semantic).fetchall()
                    results.extend(similarity_results)
                    olliePrint_simple(f"Semantic search found {len(similarity_results)} nodes.")

                except duckdb.Error as db_err:
                     olliePrint_simple(f"Database error during similarity search: {db_err}", level='error')
                     return []
                except Exception as e:
                     olliePrint_simple(f"Unexpected error during similarity search: {e}", level='error')
                     return []

            # Update last_access for all retrieved nodes (from either search type)
            if results:
                node_ids = [row[0] for row in results if row[0] not in node_ids_updated]
                if node_ids: # Check if there are new node_ids to update
                    placeholders = ','.join(['?'] * len(node_ids))
                    con.execute(f"UPDATE nodes SET last_access = ? WHERE nodeid IN ({placeholders})", [now] + node_ids)
                    node_ids_updated.update(node_ids)
                    olliePrint_simple(f"Updated last_access for {len(node_ids)} retrieved nodes.")

            # Format results as dictionaries
            formatted_results = []
            for row in results:
                node_data = {
                    "nodeid": row[0],
                    "label": row[1],
                    "text": row[2],
                    "type": row[3],
                    "created_at": row[4],
                    "last_access": row[5],
                    "target_date": row[6],
                    "similarity": row[7]
                }
                
                # Add edge connections if requested
                if include_connections and node_data["nodeid"]:
                    nodeid = node_data["nodeid"]
                    connections = []
                    
                    # Get outgoing edges for this node
                    try:
                        outgoing_edges = con.execute("""
                            SELECT e.rel_type, e.targetid, n.label, n.type 
                            FROM edges e
                            JOIN nodes n ON e.targetid = n.nodeid
                            WHERE e.sourceid = ? AND n.superseded_at IS NULL
                        """, [nodeid]).fetchall()
                        
                        for edge in outgoing_edges:
                            connections.append({
                                "direction": "outgoing",
                                "rel_type": edge[0],
                                "target_nodeid": edge[1],
                                "target_label": edge[2],
                                "target_type": edge[3]
                            })
                    except Exception as e:
                        olliePrint_simple(f"Error fetching outgoing edges for node {nodeid}: {e}", level='error')
                    
                    # Get incoming edges for this node
                    try:
                        incoming_edges = con.execute("""
                            SELECT e.rel_type, e.sourceid, n.label, n.type 
                            FROM edges e
                            JOIN nodes n ON e.sourceid = n.nodeid
                            WHERE e.targetid = ? AND n.superseded_at IS NULL
                        """, [nodeid]).fetchall()
                        
                        for edge in incoming_edges:
                            connections.append({
                                "direction": "incoming",
                                "rel_type": edge[0],
                                "source_nodeid": edge[1],
                                "source_label": edge[2],
                                "source_type": edge[3]
                            })
                    except Exception as e:
                        olliePrint_simple(f"Error fetching incoming edges for node {nodeid}: {e}", level='error')
                    
                    node_data["connections"] = connections
                
                formatted_results.append(node_data)

    except ImportError:
        olliePrint_simple("DuckDB library not found. Cannot perform memory search.", level='error')
        return []
    except duckdb.Error as e:
        olliePrint_simple(f"Failed to connect to database for memory search: {e}", level='error')
        return []
    except Exception as e:
        olliePrint_simple(f"Unexpected critical error in search_memory: {e}", level='error')
        return []

    return formatted_results


def get_subgraph(center_node_id: int, depth: int = 2, relationship_types: Optional[List[str]] = None,
                max_nodes: int = config.MAX_SUBGRAPH_NODES, include_metadata: bool = True) -> Dict[str, Any]:
    """
    MCP-inspired subgraph retrieval with advanced traversal and filtering capabilities.
    
    This function provides sophisticated subgraph extraction similar to Neo4j MCP servers,
    enabling complex graph queries and relationship analysis while maintaining performance.
    
    Args:
        center_node_id (int): The central node ID to build subgraph around
        depth (int): Maximum traversal depth (capped at MAX_SUBGRAPH_DEPTH)
        relationship_types (List[str], optional): Filter by specific relationship types
        max_nodes (int): Maximum nodes to include (prevents memory issues)
        include_metadata (bool): Include detailed node and edge metadata
        
    Returns:
        Dict[str, Any]: Comprehensive subgraph information:
            - nodes: List of nodes with full details and metadata
            - edges: List of edges with relationship information
            - paths: Notable paths and connection patterns
            - statistics: Subgraph analysis metrics
            - center_node: Detailed information about the central node
    """
    # Ensure depth doesn't exceed maximum
    depth = min(depth, config.MAX_SUBGRAPH_DEPTH)
    
    # Filter relationship types if specified
    valid_rel_types = relationship_types if relationship_types else VALID_REL_TYPES
    
    try:
        with get_db_connection() as con:
            # Get center node details
            center_query = """
                SELECT nodeid, type, label, text, created_at, last_access, 
                       superseded_at, parent_id, target_date
                FROM nodes 
                WHERE nodeid = ? AND superseded_at IS NULL
            """
            center_result = con.execute(center_query, [center_node_id]).fetchone()
            
            if not center_result:
                return {
                    'error': f'Center node {center_node_id} not found or superseded',
                    'nodes': [],
                    'edges': [],
                    'paths': [],
                    'statistics': {},
                    'center_node': None
                }
            
            # Convert center node to dictionary
            center_node = {
                'nodeid': center_result[0],
                'type': center_result[1],
                'label': center_result[2],
                'text': center_result[3],
                'created_at': center_result[4],
                'last_access': center_result[5],
                'superseded_at': center_result[6],
                'parent_id': center_result[7],
                'target_date': center_result[8],
                'distance_from_center': 0
            }
            
            # Collect all nodes and edges through BFS traversal
            visited_nodes = {center_node_id: center_node}
            all_edges = []
            current_level = [center_node_id]
            
            for current_depth in range(depth):
                if len(visited_nodes) >= max_nodes:
                    break
                    
                next_level = []
                
                # Get all edges from current level nodes
                if current_level:
                    placeholders = ','.join(['?' for _ in current_level])
                    rel_filter = f"AND rel_type IN ({','.join(['?' for _ in valid_rel_types])})" if relationship_types else ""
                    
                    edge_query = f"""
                        SELECT DISTINCT e.sourceid, e.targetid, e.rel_type, e.created_at,
                               n1.label as source_label, n2.label as target_label,
                               n1.type as source_type, n2.type as target_type
                        FROM edges e
                        JOIN nodes n1 ON e.sourceid = n1.nodeid
                        JOIN nodes n2 ON e.targetid = n2.nodeid
                        WHERE (e.sourceid IN ({placeholders}) OR e.targetid IN ({placeholders}))
                        AND n1.superseded_at IS NULL AND n2.superseded_at IS NULL
                        {rel_filter}
                        ORDER BY e.created_at DESC
                    """
                    
                    query_params = current_level + current_level
                    if relationship_types:
                        query_params.extend(valid_rel_types)
                    
                    edge_results = con.execute(edge_query, query_params).fetchall()
                    
                    for edge in edge_results:
                        source_id, target_id, rel_type, edge_created = edge[0], edge[1], edge[2], edge[3]
                        
                        # Add edge to collection
                        edge_info = {
                            'sourceid': source_id,
                            'targetid': target_id,
                            'rel_type': rel_type,
                            'created_at': edge_created,
                            'source_label': edge[4],
                            'target_label': edge[5],
                            'source_type': edge[6],
                            'target_type': edge[7]
                        }
                        all_edges.append(edge_info)
                        
                        # Add connected nodes to next level if not visited
                        for node_id in [source_id, target_id]:
                            if node_id not in visited_nodes and len(visited_nodes) < max_nodes:
                                # Get node details
                                node_query = """
                                    SELECT nodeid, type, label, text, created_at, last_access,
                                           superseded_at, parent_id, target_date
                                    FROM nodes
                                    WHERE nodeid = ? AND superseded_at IS NULL
                                """
                                node_result = con.execute(node_query, [node_id]).fetchone()
                                
                                if node_result:
                                    node_info = {
                                        'nodeid': node_result[0],
                                        'type': node_result[1],
                                        'label': node_result[2],
                                        'text': node_result[3],
                                        'created_at': node_result[4],
                                        'last_access': node_result[5],
                                        'superseded_at': node_result[6],
                                        'parent_id': node_result[7],
                                        'target_date': node_result[8],
                                        'distance_from_center': current_depth + 1
                                    }
                                    visited_nodes[node_id] = node_info
                                    next_level.append(node_id)
                
                current_level = next_level
            
            # Generate statistics
            node_types = {}
            relationship_types_count = {}
            
            for node in visited_nodes.values():
                node_type = node['type']
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            for edge in all_edges:
                rel_type = edge['rel_type']
                relationship_types_count[rel_type] = relationship_types_count.get(rel_type, 0) + 1
            
            # Find notable paths (simple implementation)
            notable_paths = []
            if len(visited_nodes) > 2:
                # Find paths of length 2-3 from center
                for edge1 in all_edges:
                    if edge1['sourceid'] == center_node_id:
                        for edge2 in all_edges:
                            if edge2['sourceid'] == edge1['targetid'] and edge2['targetid'] != center_node_id:
                                path = {
                                    'path_length': 2,
                                    'nodes': [center_node_id, edge1['targetid'], edge2['targetid']],
                                    'relationships': [edge1['rel_type'], edge2['rel_type']],
                                    'description': f"{center_node['label']} -> {edge1['target_label']} -> {edge2['target_label']}"
                                }
                                notable_paths.append(path)
                                if len(notable_paths) >= 5:  # Limit paths to prevent overwhelming output
                                    break
                        if len(notable_paths) >= 5:
                            break
            
            return {
                'nodes': list(visited_nodes.values()),
                'edges': all_edges,
                'paths': notable_paths,
                'statistics': {
                    'total_nodes': len(visited_nodes),
                    'total_edges': len(all_edges),
                    'max_depth_reached': depth,
                    'node_type_distribution': node_types,
                    'relationship_type_distribution': relationship_types_count,
                    'center_node_id': center_node_id,
                    'traversal_complete': len(visited_nodes) < max_nodes
                },
                'center_node': center_node
            }
            
    except Exception as e:
        olliePrint_simple(f"Error retrieving subgraph for node {center_node_id}: {e}", level='error')
        return {
            'error': str(e),
            'nodes': [],
            'edges': [],
            'paths': [],
            'statistics': {},
            'center_node': None
        }

def get_graph_data(center_nodeid: int, depth: int = 1) -> Dict[str, Any]:
    """
    Enhanced graph data retrieval with MCP-inspired subgraph capabilities.
    
    This function now uses the advanced get_subgraph functionality while maintaining
    backward compatibility for existing code that expects the old format.
    
    Args:
        center_nodeid (int): The central node ID to build graph around
        depth (int): Maximum traversal depth
        
    Returns:
        Dict[str, Any]: Graph data in backward-compatible format with enhanced information
    """
    # Use the enhanced subgraph retrieval
    subgraph_result = get_subgraph(center_nodeid, depth)
    
    # Handle error cases
    if 'error' in subgraph_result:
        return {'nodes': [], 'edges': []}
    
    # Convert nodes to backward-compatible LIST format (maintains previous behavior expected by Tools.py)
    nodes_list = []
    for node in subgraph_result.get('nodes', []):
        # Provide both the new canonical key ('id') and legacy key ('nodeid')
        nodes_list.append({
            'id': node['nodeid'],          # canonical for downstream callers
            'nodeid': node['nodeid'],      # legacy for older callers
            'type': node['type'],
            'label': node['label'],
            'text': node['text'],
            'created_at': node['created_at'],
            'last_access': node['last_access'],
            'superseded_at': node['superseded_at'],
            'parent_id': node['parent_id'],
            'target_date': node['target_date']
        })
    
    # Convert edges to backward-compatible LIST format with both canonical and legacy keys
    edges_list = []
    for edge in subgraph_result.get('edges', []):
        edges_list.append({
            'source': edge['sourceid'],     # canonical
            'target': edge['targetid'],     # canonical
            'sourceid': edge['sourceid'],   # legacy
            'targetid': edge['targetid'],
            'rel_type': edge['rel_type'],
            'created_at': edge['created_at']
        })
    
    # Final, backward-compatible response
    return {
        'nodes': nodes_list,
        'edges': edges_list,
        'statistics': subgraph_result.get('statistics', {}),
        'paths': subgraph_result.get('paths', [])
    }
    nodes = {}
    edges = []
    # nodes_to_fetch = {center_nodeid} # Will be set after validation
    fetched_nodes = set()
    all_involved_nodes = set() # Track nodes to update last_access

    try:
        with duckdb.connect(DB_FILE) as con:
            now = datetime.datetime.now()

            # --- Start of proposed addition ---
            if not center_nodeid:
                olliePrint_simple(
                    "get_graph_data called without a center_nodeid. The Flask /graph route should provide a default. Returning empty graph.",
                    level='warning'
                )
                return {"nodes": [], "edges": []}

            center_node_check = con.execute(
                "SELECT 1 FROM nodes WHERE nodeid = ? AND superseded_at IS NULL",
                [center_nodeid]
            ).fetchone()
            if not center_node_check:
                olliePrint_simple(
                    f"Provided center_nodeid {center_nodeid} for get_graph_data is not found or is superseded. Returning empty graph.",
                    level='error'
                )
                return {"nodes": [], "edges": []}
            
            nodes_to_fetch = {center_nodeid} # Initialize after validation
            # --- End of proposed addition ---

            for current_depth in range(depth + 1):
                if not nodes_to_fetch:
                    break

                current_batch = list(nodes_to_fetch)
                nodes_to_fetch.clear()
                fetched_nodes.update(current_batch)
                all_involved_nodes.update(current_batch) # Add nodes being fetched

                try:
                    node_placeholders = ','.join(['?'] * len(current_batch))
                    node_results = con.execute(
                        f"SELECT nodeid, type, label, text, created_at, last_access FROM nodes WHERE nodeid IN ({node_placeholders}) AND superseded_at IS NULL",
                        current_batch
                    ).fetchall()

                    for row in node_results:
                        node_id = row[0]
                        if node_id not in nodes: # Avoid duplicates
                            nodes[node_id] = {
                                "id": node_id,
                                "type": row[1],
                                "label": row[2],
                                "text": row[3],
                                "created_at": row[4],
                                "last_access": row[5]
                            }

                    if current_depth < depth:
                        edge_placeholders = ','.join(['?'] * len(current_batch))
                        edge_results = con.execute(
                            f"""
                            SELECT sourceid, targetid, rel_type
                            FROM edges
                            WHERE (sourceid IN ({edge_placeholders}) OR targetid IN ({edge_placeholders}))
                            """,
                            current_batch + current_batch
                        ).fetchall()

                        valid_target_ids = set(nodes.keys()) # Nodes confirmed valid in this fetch

                        for sourceid, targetid, rel_type in edge_results:
                            source_is_valid = sourceid in nodes
                            target_is_valid = targetid in nodes

                            potential_neighbor = None
                            if source_is_valid and targetid not in fetched_nodes and targetid not in nodes_to_fetch:
                               potential_neighbor = targetid
                            elif target_is_valid and sourceid not in fetched_nodes and sourceid not in nodes_to_fetch:
                               potential_neighbor = sourceid

                            if potential_neighbor:
                                 is_neighbor_valid = con.execute("SELECT 1 FROM nodes WHERE nodeid = ? AND superseded_at IS NULL", [potential_neighbor]).fetchone()
                                 if is_neighbor_valid:
                                     nodes_to_fetch.add(potential_neighbor)
                                     all_involved_nodes.add(potential_neighbor)


                            if source_is_valid and target_is_valid:
                               edges.append({"source": sourceid, "target": targetid, "rel_type": rel_type})

                except duckdb.Error as e:
                     olliePrint_simple(f"Database error during graph data retrieval: {e}", level='error')
                     # Continue if possible, graph might be incomplete
                except Exception as e:
                     olliePrint_simple(f"Unexpected error during graph data retrieval: {e}", level='error')

            if all_involved_nodes:
                olliePrint_simple(f"Updating last_access for {len(all_involved_nodes)} nodes involved in graph data.")
                node_ids_list = list(all_involved_nodes)
                id_placeholders = ','.join(['?'] * len(node_ids_list))
                try:
                     con.execute(f"UPDATE nodes SET last_access = ? WHERE nodeid IN ({id_placeholders})", [now] + node_ids_list)
                except duckdb.Error as e:
                     olliePrint_simple(f"Failed to update last_access for graph nodes: {e}", level='error')

    except ImportError:
        olliePrint_simple("DuckDB library not found. Cannot get graph data.", level='error')
        return {"nodes": [], "edges": []}
    except duckdb.Error as e:
        olliePrint_simple(f"Failed to connect to database for graph data: {e}", level='error')
        return {"nodes": [], "edges": []} # Return empty graph on connection error
    except Exception as e:
        olliePrint_simple(f"Unexpected critical error in get_graph_data: {e}", level='error')
        return {"nodes": [], "edges": []}


    return {"nodes": list(nodes.values()), "edges": edges}


def get_all_active_nodes_for_viz():
    """Retrieves all active nodes with their embeddings and total edge counts for visualization."""
    nodes_data = []
    olliePrint_simple("Fetching all active nodes for visualization.")
    try:
        with duckdb.connect(DB_FILE) as con:
            # Fetch all active nodes with their basic details and embeddings
            active_nodes_res = con.execute(
                """
                SELECT nodeid, label, type, embedding
                FROM nodes
                WHERE superseded_at IS NULL;
                """
            ).fetchall()

            if not active_nodes_res:
                olliePrint_simple("No active nodes found for visualization.")
                return []

            for node_row in active_nodes_res:
                nodeid, label, node_type, embedding = node_row
                
                # Calculate total edge count for this node
                edge_count_res = con.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT sourceid AS node_involved FROM edges WHERE targetid = ?
                        UNION ALL
                        SELECT targetid AS node_involved FROM edges WHERE sourceid = ?
                    ) AS all_edges;
                    """,
                    [nodeid, nodeid] # Count where this node is either a source or a target
                ).fetchone()
                
                total_edge_count = edge_count_res[0] if edge_count_res else 0

                # Ensure embedding is a list of floats if it's not already
                # DuckDB might return it as a special type, though list_cosine_similarity implies it's usable
                # For JSON serialization and JS, ensure it's a standard list.
                if embedding is not None and not isinstance(embedding, list):
                    # This step might be redundant if DuckDB already returns a list-like structure
                    # that json.dumps can handle. However, explicit conversion is safer for JS.
                    try:
                        # Assuming embedding is a numpy array or similar that can be converted
                        processed_embedding = [float(e) for e in embedding]
                    except TypeError:
                        olliePrint_simple(f"Embedding for node {nodeid} is not iterable, setting to empty list.", level='warning')
                        processed_embedding = [] # Or handle as an error
                elif embedding is None:
                    processed_embedding = []
                else:
                    processed_embedding = embedding


                nodes_data.append({
                    "nodeid": nodeid,
                    "label": label,
                    "type": node_type,
                    "embedding": processed_embedding, # Store the raw embedding
                    "total_edge_count": total_edge_count
                })
            
            olliePrint_simple(f"Successfully fetched and processed {len(nodes_data)} active nodes for visualization.")
            return nodes_data

    except ImportError:
        olliePrint_simple("DuckDB library not found. Cannot get_all_active_nodes_for_viz.", level='error')
        return []
    except duckdb.Error as e:
        olliePrint_simple(f"Database error in get_all_active_nodes_for_viz: {e}", level='error')
        return []
    except Exception as e:
        olliePrint_simple(f"Unexpected error in get_all_active_nodes_for_viz: {e}", level='error')
        return []


def forget_old_memories(days_old=180):
    """Deletes nodes that were superseded long ago and haven't been accessed."""
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
    deleted_count = 0
    olliePrint_simple(f"Starting forgetting process for memories superseded and not accessed before {cutoff_date}...")

    try:
        with duckdb.connect(DB_FILE) as con:
            con.begin()
            try:
                nodes_to_delete_result = con.execute(
                    """
                    SELECT nodeid FROM nodes
                    WHERE superseded_at IS NOT NULL
                      AND superseded_at < ?
                      AND last_access < ?
                    """,
                    [cutoff_date, cutoff_date]
                ).fetchall()

                nodes_to_delete_ids = [row[0] for row in nodes_to_delete_result]

                if not nodes_to_delete_ids:
                    olliePrint_simple("No old memories found meeting the forgetting criteria.")
                    con.commit()
                    return 0

                olliePrint_simple(f"Identified {len(nodes_to_delete_ids)} nodes to forget.")

                id_placeholders = ",".join(['?'] * len(nodes_to_delete_ids))
                delete_edges_query = f"DELETE FROM edges WHERE sourceid IN ({id_placeholders}) OR targetid IN ({id_placeholders})"
                con.execute(delete_edges_query, nodes_to_delete_ids + nodes_to_delete_ids)
                olliePrint_simple(f"Deleted associated edges for {len(nodes_to_delete_ids)} nodes.")

                delete_nodes_query = f"DELETE FROM nodes WHERE nodeid IN ({id_placeholders})"
                cur = con.execute(delete_nodes_query, nodes_to_delete_ids)
                # Attempt to get row count, fallback to len(ids)
                try:
                    deleted_count = cur.fetchall()[0][0] # DuckDB DELETE RETURNING COUNT(*) might work this way
                except:
                     # If fetchall doesn't work as expected or returns nothing useful
                     olliePrint_simple("Could not get exact deleted row count from cursor, using initial count.", level='warning')
                     deleted_count = len(nodes_to_delete_ids)

                con.commit()
                olliePrint_simple(f"Finished forgetting old memories. Deleted {deleted_count} nodes.")

            except duckdb.Error as e:
                con.rollback()
                olliePrint_simple(f"Database error during forgetting old memories: {e}", level='error')
                # Do not return, let error propagate or be handled by outer block
                raise
            except Exception as e:
                con.rollback()
                olliePrint_simple(f"Unexpected error during forgetting old memories: {e}", level='error')
                raise RuntimeError(f"Unexpected error during forgetting: {e}") from e

    except ImportError:
         olliePrint_simple("DuckDB library not found. Cannot forget old memories.", level='error')
         return 0 # Indicate nothing was done
    except duckdb.Error as e:
         olliePrint_simple(f"Failed to connect to database for forgetting memories: {e}", level='error')
         return 0 # Indicate nothing was done
    except Exception as e:
         olliePrint_simple(f"Unexpected critical error in forget_old_memories: {e}", level='error')
         return 0


    return deleted_count

def process_pending_edges(limit_per_run=10):
    """Processes pending edge creation tasks from the queue.

    Args:
        limit_per_run (int): The maximum number of pending tasks to process in this run.

    Returns:
        dict: A summary of the processing.
    """
    summary = {
        "nodes_processed_this_run": 0,
        "edges_attempted_this_run": 0,
        "edges_succeeded_this_run": 0,
        "nodes_with_errors_this_run": 0,
        "tasks_remaining_after_run": 0
    }
    now_ts = datetime.datetime.now()

    try:
        with duckdb.connect(DB_FILE) as con:
            # Get tasks to process
            pending_tasks_res = con.execute(
                """
                SELECT task_id, node_id_to_process FROM pending_edge_creation_tasks
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT ?;
                """,
                [limit_per_run]
            ).fetchall()

            if not pending_tasks_res:
                olliePrint_simple("No pending edge creation tasks found.")
                # Query for total remaining to ensure accurate count
                remaining_count_res = con.execute("SELECT COUNT(*) FROM pending_edge_creation_tasks WHERE status = 'pending';").fetchone()
                summary["tasks_remaining_after_run"] = remaining_count_res[0] if remaining_count_res else 0
                return summary

            for task_id, node_id_to_process in pending_tasks_res:
                olliePrint_simple(f"Processing task {task_id} for node {node_id_to_process}")
                summary["nodes_processed_this_run"] += 1
                node_had_error = False
                edges_added_for_this_node = 0

                try:
                    # Start a transaction for this specific task's state changes and edge creations
                    con.begin()
                    
                    # Mark task as processing
                    con.execute(
                        """
                        UPDATE pending_edge_creation_tasks
                        SET status = 'processing', attempts = attempts + 1, updated_at = ?
                        WHERE task_id = ?;
                        """,
                        [now_ts, task_id]
                    )
                    # Fetch node details for which edges need to be created
                    source_node_details_res = con.execute(
                        "SELECT label, text, type FROM nodes WHERE nodeid = ? AND superseded_at IS NULL",
                        [node_id_to_process]
                    ).fetchone()

                    if not source_node_details_res:
                        olliePrint_simple(f"Node {node_id_to_process} for task {task_id} not found or is superseded. Marking task as failed.", level='warning')
                        con.execute(
                            "UPDATE pending_edge_creation_tasks SET status = 'failed', last_error = ?, updated_at = ? WHERE task_id = ?;",
                            ["Source node not found or superseded.", now_ts, task_id]
                        )
                        con.commit() # Commit status update for this task
                        summary["nodes_with_errors_this_run"] += 1
                        continue # Move to the next task

                    source_node_info = {"label": source_node_details_res[0], "text": source_node_details_res[1], "type": source_node_details_res[2]}
                    
                    # Find similar nodes (search_memory creates its own connection, which is acceptable here)
                    similar_nodes = search_memory(source_node_info['text'], limit=AUTO_EDGE_SIMILARITY_CHECK_LIMIT + 1)
                    
                    non_self_similar_nodes = [node for node in similar_nodes if node['nodeid'] != node_id_to_process]

                    if not non_self_similar_nodes:
                        olliePrint_simple(f"No similar nodes found (other than self) for node {node_id_to_process}. Task {task_id} completed.")
                        con.execute(
                            "UPDATE pending_edge_creation_tasks SET status = 'completed', updated_at = ? WHERE task_id = ?;",
                            [now_ts, task_id]
                        )
                        con.commit() # Commit status update
                        continue

                    olliePrint_simple(f"Found {len(non_self_similar_nodes)} relevant similar nodes for edge creation for node {node_id_to_process}.")

                    for similar_node in non_self_similar_nodes:
                        similar_nodeid = similar_node['nodeid']
                        # No need to check self-reference again as it's filtered

                        # Check if target node is valid within the current transaction context
                        target_node_valid_res = con.execute(
                            "SELECT label, text, type FROM nodes WHERE nodeid = ? AND superseded_at IS NULL",
                            [similar_nodeid]
                        ).fetchone()

                        if not target_node_valid_res:
                            olliePrint_simple(f"Skipping auto-edge from {node_id_to_process} to {similar_nodeid} as target is missing or superseded.", level='warning')
                            continue
                        
                        target_node_info = {"label": target_node_valid_res[0], "text": target_node_valid_res[1], "type": target_node_valid_res[2]}

                        olliePrint_simple(f"Analyzing relationship between node {node_id_to_process} and similar node {similar_nodeid} using enhanced context analysis...")
                        
                        relationship_analysis = determine_edge_type_llm_enhanced(source_node_info, target_node_info) # Can raise
                        determined_rel_type = relationship_analysis.get('relationship_type', 'relatedTo')
                        confidence = relationship_analysis.get('confidence', 'unknown')
                        olliePrint_simple(f"Enhanced analysis: relationship from {node_id_to_process} to {similar_nodeid} = {determined_rel_type} (confidence: {confidence})")
                        
                        summary["edges_attempted_this_run"] += 1
                        
                        # Handle supersession if LLM determines 'updates' relationship
                        if determined_rel_type == 'updates':
                            # Check if edge already exists to avoid duplicate supersession
                            if not check_edge_exists_through_supersession(node_id_to_process, similar_nodeid, 'updates', con):
                                olliePrint_simple(f"LLM determined supersession: node {node_id_to_process} updates node {similar_nodeid}")
                                handle_updates_edge_supersession(node_id_to_process, similar_nodeid, con)
                                # Create the 'updates' edge
                                add_edge(sourceid=node_id_to_process,
                                         targetid=similar_nodeid,
                                         rel_type='updates',
                                         con=con)
                                summary["edges_succeeded_this_run"] += 1
                                edges_added_for_this_node += 1
                            else:
                                olliePrint_simple(f"Supersession edge already exists: {node_id_to_process} -> {similar_nodeid}")
                        else:
                            # Regular edge creation with duplicate checking
                            if not check_edge_exists_through_supersession(node_id_to_process, similar_nodeid, determined_rel_type, con):
                                add_edge(sourceid=node_id_to_process,
                                         targetid=similar_nodeid,
                                         rel_type=determined_rel_type,
                                         con=con)
                                summary["edges_succeeded_this_run"] += 1
                                edges_added_for_this_node += 1
                            else:
                                olliePrint_simple(f"Edge already exists: {node_id_to_process} -[{determined_rel_type}]-> {similar_nodeid}")
                        
                        if edges_added_for_this_node >= AUTO_EDGE_SIMILARITY_CHECK_LIMIT:
                            break
                    
                    # If loop completes without error for this node
                    con.execute(
                        "UPDATE pending_edge_creation_tasks SET status = 'completed', updated_at = ? WHERE task_id = ?;",
                        [now_ts, task_id]
                    )
                    con.commit() # Commit successful processing of this task

                except (ValueError, ConnectionError, RuntimeError, duckdb.Error) as e:
                    olliePrint_simple(f"Error processing task {task_id} for node {node_id_to_process}: {e}", level='error')
                    if con.in_transaction: #.is_active for newer duckdb versions
                        con.rollback() # Rollback if error occurred mid-transaction for this task
                    
                    # Re-open connection/transaction for final status update if needed, or do it in a new one.
                    # For simplicity, we'll assume the main connection 'con' is still usable or re-establish for this update.
                    # However, a robust way is to ensure 'con' is always valid or use a new short-lived one for error update.
                    try:
                        with duckdb.connect(DB_FILE) as err_con: # Fresh connection for error update
                             err_con.execute(
                                "UPDATE pending_edge_creation_tasks SET status = 'failed', last_error = ?, attempts = attempts + 1, updated_at = ? WHERE task_id = ?;",
                                [str(e)[:1024], now_ts, task_id] # Limit error message length
                            ) # attempts already incremented when set to 'processing'
                    except Exception as update_err:
                        olliePrint_simple(f"Critical: Failed to even mark task {task_id} as failed: {update_err}", level='error')

                    summary["nodes_with_errors_this_run"] += 1
                    node_had_error = True # To prevent marking as completed outside
                except Exception as e_unexpected: # Catch any other unexpected Python error
                    olliePrint_simple(f"Unexpected Python error processing task {task_id} for node {node_id_to_process}: {e_unexpected}", level='error')
                    if con.in_transaction:
                        con.rollback()
                    try:
                        with duckdb.connect(DB_FILE) as err_con:
                            err_con.execute(
                                "UPDATE pending_edge_creation_tasks SET status = 'failed', last_error = ?, attempts = attempts + 1, updated_at = ? WHERE task_id = ?;",
                                [f"Unexpected: {str(e_unexpected)[:1000]}", now_ts, task_id]
                            )
                    except Exception as update_err:
                        olliePrint_simple(f"Critical: Failed to mark task {task_id} as failed after unexpected error: {update_err}", level='error')
                    summary["nodes_with_errors_this_run"] += 1
                    node_had_error = True


            # After processing the batch, query for remaining tasks
            remaining_count_res = con.execute("SELECT COUNT(*) FROM pending_edge_creation_tasks WHERE status = 'pending';").fetchone()
            summary["tasks_remaining_after_run"] = remaining_count_res[0] if remaining_count_res else 0

    except ImportError:
        olliePrint_simple("DuckDB library not found. Cannot process pending edges.", level='error')
        # tasks_remaining_after_run will be 0, which is not ideal but reflects no processing happened.
    except duckdb.Error as e:
        olliePrint_simple(f"Database connection/query error in process_pending_edges: {e}", level='error')
    except Exception as e:
        olliePrint_simple(f"Unexpected critical error in process_pending_edges: {e}", level='error')

    olliePrint_simple(f"Pending edge processing run summary: {summary}")
    return summary


def clear_all_memory(force=False):
    """Deletes ALL nodes and edges from the database by dropping and re-initializing tables.

    Warning: This action is irreversible.

    Args:
        force (bool): Set to True to bypass safety checks for production environments.
                     Default is False, which prevents deletion if environment is not development.

    Returns:
        bool: True if deletion and re-initialization was successful, False otherwise.
    """
    # Safety check for production environments
    if not force:
        env = os.getenv('FRED_ENV', 'production').lower()
        if env != 'development' and env != 'test':
            olliePrint_simple("Attempted to clear memory in non-development environment! Set force=True to override.", level='warning')
            return False
            
    olliePrint_simple("Attempting to clear ALL memory by dropping and re-initializing tables!", level='warning')
    try:
        with duckdb.connect(DB_FILE) as con:
            con.begin() # Start transaction
            try:
                olliePrint_simple("Dropping existing 'edges' table (if exists)...")
                con.execute("DROP TABLE IF EXISTS edges;")
                olliePrint_simple("Dropped 'edges' table.")
                
                olliePrint_simple("Dropping existing 'nodes' table (if exists)...")
                con.execute("DROP TABLE IF EXISTS nodes;")
                olliePrint_simple("Dropped 'nodes' table.")
                
                con.commit() # Commit drops
                olliePrint_simple("Tables dropped. Now re-initializing schema...")
                
                # Re-initialize the schema
                init_db() # This will create the tables again
                
                olliePrint_simple("Successfully cleared and re-initialized all memory.", level='warning')
                return True
            except duckdb.Error as e:
                con.rollback() # Rollback on error
                olliePrint_simple(f"Database error during memory clearing (drop/re-init, level='error'): {e}")
                return False
            except Exception as e:
                con.rollback()
                olliePrint_simple(f"Unexpected error during memory clearing (drop/re-init, level='error'): {e}")
                return False

    except ImportError:
         olliePrint_simple("DuckDB library not found. Cannot clear memory.", level='error')
         return False
    except duckdb.Error as e:
         olliePrint_simple(f"Failed to connect to database for clearing memory: {e}", level='error')
         return False
    except Exception as e:
         olliePrint_simple(f"Unexpected critical error in clear_all_memory: {e}", level='error')
         return False

# --- Initialization ---
# Only initialize DB automatically when this script is run directly
try:
    import duckdb
    if __name__ == "__main__":
        olliePrint_simple("Librarian script loaded. Initializing database...")
        init_db()
        olliePrint_simple("Database ready.")
        # Add test calls here if desired, wrapped in try/except ImportError
except ImportError:
    olliePrint_simple("DuckDB not found during import. Database functionality is disabled.", level='warning')
    if __name__ == "__main__":
        olliePrint_simple("Librarian script loaded, but DuckDB is missing. Database functionality disabled.")
