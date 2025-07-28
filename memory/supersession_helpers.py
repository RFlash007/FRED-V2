"""
Background Supersession Helper Functions
========================================

Helper functions for handling LLM-triggered supersession during background edge creation.
These functions handle edge inheritance, task transfer, and supersession chain resolution.
"""

import datetime
import duckdb
from ollie_print import olliePrint_simple


def resolve_supersession_chain(node_id: int, con: duckdb.DuckDBPyConnection) -> int:
    """Follow supersession chain to get current active node.
    
    Args:
        node_id: The starting node ID to resolve
        con: Active database connection
        
    Returns:
        int: The current active node ID (end of supersession chain)
    """
    current = node_id
    visited = set()
    
    while current not in visited:
        visited.add(current)
        
        # Check if this node was superseded by looking for 'updates' relationship
        newer_node = con.execute("""
            SELECT sourceid FROM edges 
            WHERE targetid = ? AND rel_type = 'updates'
            AND sourceid IN (SELECT nodeid FROM nodes WHERE superseded_at IS NULL)
        """, [current]).fetchone()
        
        if newer_node:
            current = newer_node[0]
        else:
            break
    
    return current


def transfer_pending_edge_tasks(old_node_id: int, new_node_id: int, con: duckdb.DuckDBPyConnection):
    """Transfer pending edge creation tasks from superseded node to new node.
    
    Args:
        old_node_id: The superseded node ID
        new_node_id: The new node ID that supersedes the old one
        con: Active database connection
    """
    # Check if we need to add supersession tracking columns
    try:
        con.execute("""
            UPDATE pending_edge_creation_tasks 
            SET node_id_to_process = ?
            WHERE node_id_to_process = ? AND status = 'pending'
        """, [new_node_id, old_node_id])
        
        updated_tasks = con.execute("SELECT changes()").fetchone()[0]
        if updated_tasks > 0:
            olliePrint_simple(f"Transferred {updated_tasks} pending edge tasks from node {old_node_id} to {new_node_id}")
    except duckdb.Error as e:
        olliePrint_simple(f"Error transferring pending tasks: {e}", level='error')


def inherit_edges_from_superseded(old_node_id: int, new_node_id: int, con: duckdb.DuckDBPyConnection):
    """Inherit all edges from superseded node to new node.
    
    Args:
        old_node_id: The superseded node ID
        new_node_id: The new node ID that supersedes the old one
        con: Active database connection
    """
    try:
        # Inherit outgoing edges: old_node → X becomes new_node → X
        con.execute("""
            INSERT OR IGNORE INTO edges (sourceid, targetid, rel_type, created_at)
            SELECT ?, targetid, rel_type, current_timestamp
            FROM edges 
            WHERE sourceid = ? AND targetid != ? AND rel_type != 'updates'
        """, [new_node_id, old_node_id, new_node_id])
        
        outgoing_inherited = con.execute("SELECT changes()").fetchone()[0]
        
        # Inherit incoming edges: X → old_node becomes X → new_node
        con.execute("""
            INSERT OR IGNORE INTO edges (sourceid, targetid, rel_type, created_at)
            SELECT sourceid, ?, rel_type, current_timestamp
            FROM edges 
            WHERE targetid = ? AND sourceid != ? AND rel_type != 'updates'
        """, [new_node_id, old_node_id, new_node_id])
        
        incoming_inherited = con.execute("SELECT changes()").fetchone()[0]
        
        if outgoing_inherited > 0 or incoming_inherited > 0:
            olliePrint_simple(f"Inherited {outgoing_inherited} outgoing + {incoming_inherited} incoming edges from node {old_node_id} to {new_node_id}")
    
    except duckdb.Error as e:
        olliePrint_simple(f"Error inheriting edges: {e}", level='error')


def handle_updates_edge_supersession(source_node_id: int, target_node_id: int, con: duckdb.DuckDBPyConnection):
    """Handle supersession when LLM determines 'updates' relationship.
    
    This function is called when the background edge creation determines that
    source_node_id 'updates' target_node_id, triggering full supersession logic.
    
    Args:
        source_node_id: The new node that supersedes the old one
        target_node_id: The old node being superseded
        con: Active database connection
    """
    olliePrint_simple(f"Processing supersession: node {source_node_id} updates node {target_node_id}")
    
    try:
        # 1. Mark old node as superseded
        con.execute("""
            UPDATE nodes SET superseded_at = current_timestamp, last_access = current_timestamp 
            WHERE nodeid = ?
        """, [target_node_id])
        
        # 2. Transfer pending edge tasks from old to new node
        transfer_pending_edge_tasks(target_node_id, source_node_id, con)
        
        # 3. Inherit all edges from old node to new node
        inherit_edges_from_superseded(target_node_id, source_node_id, con)
        
        olliePrint_simple(f"Supersession complete: node {target_node_id} superseded by node {source_node_id}")
        
    except Exception as e:
        olliePrint_simple(f"Error during supersession handling: {e}", level='error')
        raise


def check_edge_exists_through_supersession(sourceid: int, targetid: int, rel_type: str, con: duckdb.DuckDBPyConnection) -> bool:
    """Check if edge already exists including through supersession chains.
    
    Args:
        sourceid: Source node ID
        targetid: Target node ID
        rel_type: Relationship type
        con: Active database connection
        
    Returns:
        bool: True if edge already exists (directly or through supersession)
    """
    try:
        # Resolve both nodes through supersession chains
        resolved_source = resolve_supersession_chain(sourceid, con)
        resolved_target = resolve_supersession_chain(targetid, con)
        
        # Check if edge already exists between resolved nodes
        existing = con.execute("""
            SELECT 1 FROM edges 
            WHERE sourceid = ? AND targetid = ? AND rel_type = ?
        """, [resolved_source, resolved_target, rel_type]).fetchone()
        
        return existing is not None
        
    except Exception as e:
        olliePrint_simple(f"Error checking edge existence: {e}", level='error')
        return False
