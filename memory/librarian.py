import duckdb
import datetime
import requests
import json
import time

DB_FILE = 'memory.db'
OLLAMA_API_URL = 'http://localhost:11434/api/embeddings'
EMBED_MODEL = 'nomic-embed-text'
LLM_MODEL = 'llama3.1:8b'
EMBEDDING_DIM = 768 # As specified for nomic-embed-text

def init_db():
    """Initializes the DuckDB database and tables if they don't exist."""
    with duckdb.connect(DB_FILE) as con:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS nodes (
                nodeid BIGINT PRIMARY KEY DEFAULT (CAST(strftime(current_timestamp, '%s%f') AS BIGINT)),
                type TEXT CHECK (type IN ('Semantic', 'Episodic')),
                label TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT current_timestamp,
                last_access TIMESTAMP DEFAULT current_timestamp,
                superseded_at TIMESTAMP,
                parent_id BIGINT,
                embedding FLOAT[{EMBEDDING_DIM}],
                embed_model TEXT
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                sourceid BIGINT NOT NULL,
                targetid BIGINT NOT NULL,
                rel_type TEXT CHECK (rel_type IN (
                    'instanceOf', 'relatedTo', 'updates', 
                    'contains', 'partOf', 'precedes',
                    'causes',
                    'createdBy', 'hasOwner',
                    'locatedAt', 'dependsOn', 'servesPurpose',
                    'occursDuring', 'enablesGoal', 'activatesIn', 'contextualAnalog', 'sourceAttribution' 
                )),
                created_at TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (sourceid, targetid, rel_type),
                FOREIGN KEY (sourceid) REFERENCES nodes(nodeid),
                FOREIGN KEY (targetid) REFERENCES nodes(nodeid)
            );
        """)

def get_embedding(text):
    """Gets the embedding for a given text using the Ollama API."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": EMBED_MODEL, "prompt": text}
        )
        response.raise_for_status()
        # Ensure the embedding has the correct dimension
        embedding = response.json().get("embedding", [])
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}")
        return embedding
    except requests.exceptions.RequestException as e:
        print(f"Error contacting Ollama: {e}")
        return None
    except ValueError as e:
        print(f"Embedding error: {e}")
        return None

def classify_memory_type(text):
    """Classifies memory as Semantic or Episodic (LLM based)"""
    # Use an LLM to classify the memory type
    prompt = f"""
    Classify the following memory as either "Semantic" or "Episodic":
    {text}
    """
    response = requests.post(OLLAMA_API_URL, json={"model": "llama3.1:8b", "prompt": prompt})


def add_memory(label, text, memory_type=None, parent_id=None):
    """Adds a new memory node."""
    if memory_type is None:
        memory_type = classify_memory_type(text)

    embedding = get_embedding(text)
    if embedding is None:
        print("Failed to get embedding. Memory not added.")
        return None

    with duckdb.connect(DB_FILE) as con:
        # Use epoch milliseconds for a potentially more unique default ID
        new_id = int(time.time() * 1000)
        con.execute(
            """
            INSERT INTO nodes (nodeid, type, label, text, parent_id, embedding, embed_model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING nodeid;
            """,
            [new_id, memory_type, label, text, parent_id, embedding, EMBED_MODEL]
        )
        result = con.fetchone()
        return result[0] if result else None


def supersede_memory(old_nodeid, new_label, new_text):
    """Supersedes an old memory node with a new one."""
    with duckdb.connect(DB_FILE) as con:
        con.begin()
        try:
            # Check if old node exists
            old_node = con.execute("SELECT type, parent_id FROM nodes WHERE nodeid = ?", [old_nodeid]).fetchone()
            if not old_node:
                print(f"Error: Node {old_nodeid} not found.")
                con.rollback()
                return None

            old_type, old_parent_id = old_node

            # Mark old node as superseded and update last_access
            now = datetime.datetime.now()
            con.execute(
                "UPDATE nodes SET superseded_at = ?, last_access = ? WHERE nodeid = ?",
                [now, now, old_nodeid]
            )

            # Add the new memory
            new_memory_type = classify_memory_type(new_text)
            embedding = get_embedding(new_text)
            if embedding is None:
                print("Failed to get embedding. Supersede failed.")
                con.rollback()
                return None

            # Use epoch milliseconds for a potentially more unique default ID
            new_id = int(time.time() * 1000)
            con.execute(
                """
                INSERT INTO nodes (nodeid, type, label, text, parent_id, embedding, embed_model, last_access)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING nodeid;
                """,
                [new_id, new_memory_type, new_label, new_text, old_parent_id, embedding, EMBED_MODEL, now]
            )
            result = con.fetchone()
            if not result:
                print("Failed to insert new node. Supersede failed.")
                con.rollback()
                return None
            new_nodeid = result[0]

            # Add 'updates' edge from new to old
            add_edge(new_nodeid, old_nodeid, 'updates', con=con)

            # Update last_access on the new node as well (redundant but safe)
            con.execute("UPDATE nodes SET last_access = ? WHERE nodeid = ?", [now, new_nodeid])

            con.commit()
            return new_nodeid
        except Exception as e:
            print(f"Error during supersede: {e}")
            con.rollback()
            return None


def add_edge(sourceid, targetid, rel_type, con=None):
    """Adds an edge between two nodes."""
    close_conn = False
    if con is None:
        con = duckdb.connect(DB_FILE)
        close_conn = True

    try:
        now = datetime.datetime.now()
        con.execute(
            "INSERT INTO edges (sourceid, targetid, rel_type, created_at) VALUES (?, ?, ?, ?)",
            [sourceid, targetid, rel_type, now]
        )
        # Update last_access for both nodes involved in the edge
        con.execute("UPDATE nodes SET last_access = ? WHERE nodeid IN (?, ?)", [now, sourceid, targetid])
        if close_conn:
            con.commit() # Commit if we opened the connection here
    except Exception as e:
        print(f"Error adding edge: {e}")
        if close_conn:
            con.rollback() # Rollback if we opened the connection here
    finally:
        if close_conn:
            con.close()


def search_memory(query_text, memory_type=None, limit=10):
    """Searches memory nodes using cosine similarity and keyword fallback."""
    query_embedding = get_embedding(query_text)
    results = []

    with duckdb.connect(DB_FILE) as con:
        now = datetime.datetime.now()

        # Cosine Similarity Search (if embedding was successful)
        if query_embedding:
            base_query = """
            SELECT nodeid, label, text, type, created_at, last_access, list_cosine_similarity(embedding, ?) AS similarity
            FROM nodes
            WHERE superseded_at IS NULL
            """
            params = [query_embedding]
            if memory_type:
                base_query += " AND type = ?"
                params.append(memory_type)

            base_query += " ORDER BY similarity DESC LIMIT ?"
            params.append(limit)

            results = con.execute(base_query, params).fetchall()

            # Update last_access for retrieved nodes
            if results:
                node_ids = [row[0] for row in results]
                con.execute(f"UPDATE nodes SET last_access = ? WHERE nodeid IN ({','.join(['?']*len(node_ids))})", [now] + node_ids)


        # Keyword Fallback Search (if no embedding or fewer results than limit)
        if len(results) < limit:
            keyword_limit = limit - len(results)
            # Basic keyword search using LIKE
            keywords = query_text.split()
            like_clauses = " OR ".join([f"text LIKE '%{keyword}%'" for keyword in keywords])

            fallback_query = f"""
            SELECT nodeid, label, text, type, created_at, last_access, 0.0 AS similarity
            FROM nodes
            WHERE superseded_at IS NULL AND ({like_clauses})
            """
            params = []
            if memory_type:
                fallback_query += " AND type = ?"
                params.append(memory_type)

            # Exclude nodes already found by similarity search
            if results:
                found_ids = [row[0] for row in results]
                fallback_query += f" AND nodeid NOT IN ({','.join(['?']*len(found_ids))})"
                params.extend(found_ids)

            fallback_query += " ORDER BY last_access DESC LIMIT ?" # Prioritize recently accessed if only keyword match
            params.append(keyword_limit)


            try:
                fallback_results = con.execute(fallback_query, params).fetchall()
                 # Update last_access for keyword-retrieved nodes
                if fallback_results:
                    node_ids = [row[0] for row in fallback_results]
                    # Filter out IDs already updated by similarity search
                    node_ids_to_update = [nid for nid in node_ids if nid not in [r[0] for r in results]]
                    if node_ids_to_update:
                         con.execute(f"UPDATE nodes SET last_access = ? WHERE nodeid IN ({','.join(['?']*len(node_ids_to_update))})", [now] + node_ids_to_update)

                results.extend(fallback_results)

            except duckdb.BinderException as e:
                 print(f"Keyword search binder error (likely missing params): {e}")
            except Exception as e:
                 print(f"Error during keyword fallback search: {e}")


    # Format results as dictionaries
    formatted_results = [
        {
            "nodeid": row[0],
            "label": row[1],
            "text": row[2],
            "type": row[3],
            "created_at": row[4],
            "last_access": row[5],
            "similarity": row[6] if len(row)>6 else 0.0 # handle keyword results missing similarity
        } for row in results
    ]
    return formatted_results


def get_graph_data(center_nodeid, depth=1):
    """Retrieves nodes and edges for the mind map within a certain depth from a center node."""
    nodes = {}
    edges = []
    nodes_to_fetch = {center_nodeid}
    fetched_nodes = set()

    with duckdb.connect(DB_FILE) as con:
        now = datetime.datetime.now()

        for _ in range(depth + 1):
            if not nodes_to_fetch:
                break

            current_batch = list(nodes_to_fetch)
            nodes_to_fetch.clear()
            fetched_nodes.update(current_batch)

            # Fetch node details for the current batch
            node_results = con.execute(
                f"SELECT nodeid, type, label, text, created_at, last_access FROM nodes WHERE nodeid IN ({','.join(['?']*len(current_batch))}) AND superseded_at IS NULL",
                current_batch
            ).fetchall()

            for row in node_results:
                 node_id = row[0]
                 if node_id not in nodes: # Avoid duplicates if fetched via different paths
                     nodes[node_id] = {
                        "id": node_id, # d3 expects 'id'
                        "type": row[1],
                        "label": row[2],
                        "text": row[3],
                        "created_at": row[4],
                        "last_access": row[5]
                    }


            # Fetch edges connected to the current batch and find next layer of nodes
            edge_results = con.execute(
                f"""
                SELECT sourceid, targetid, rel_type
                FROM edges
                WHERE (sourceid IN ({','.join(['?']*len(current_batch))}) OR targetid IN ({','.join(['?']*len(current_batch))}))
                """,
                current_batch + current_batch # Need placeholders for both source and target IN clauses
            ).fetchall()

            for sourceid, targetid, rel_type in edge_results:
                # Add edge if both nodes are within the fetched set so far (or will be)
                # Check against fetched_nodes AND the nodes dictionary keys to be safe
                source_in_scope = sourceid in fetched_nodes or sourceid in nodes
                target_in_scope = targetid in fetched_nodes or targetid in nodes

                # Only add edge if both nodes are valid and not superseded
                source_node_valid = con.execute("SELECT 1 FROM nodes WHERE nodeid = ? AND superseded_at IS NULL", [sourceid]).fetchone()
                target_node_valid = con.execute("SELECT 1 FROM nodes WHERE nodeid = ? AND superseded_at IS NULL", [targetid]).fetchone()


                if source_node_valid and target_node_valid:
                     edges.append({"source": sourceid, "target": targetid, "rel_type": rel_type}) # d3 expects source/target

                     # Add neighbours to the next fetch list if not already fetched/queued
                     if sourceid not in fetched_nodes and sourceid not in nodes_to_fetch:
                         nodes_to_fetch.add(sourceid)
                     if targetid not in fetched_nodes and targetid not in nodes_to_fetch:
                         nodes_to_fetch.add(targetid)

        # Update last_access for all nodes included in the graph
        if nodes:
             node_ids = list(nodes.keys())
             con.execute(f"UPDATE nodes SET last_access = ? WHERE nodeid IN ({','.join(['?']*len(node_ids))})", [now] + node_ids)


    return {"nodes": list(nodes.values()), "edges": edges}


def forget_old_memories(days_old=180):
    """Deletes nodes that were superseded long ago and haven't been accessed."""
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
    deleted_count = 0
    with duckdb.connect(DB_FILE) as con:
        try:
            con.begin()

            # 1. Identify nodes to be deleted
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
                print("No old memories found to forget.")
                con.commit() # Commit even if nothing to delete
                return 0

            # 2. Delete edges connected to these nodes
            # Need to format the list of IDs for the IN clause
            id_placeholders = ",".join(['?'] * len(nodes_to_delete_ids))
            delete_edges_query = f"DELETE FROM edges WHERE sourceid IN ({id_placeholders}) OR targetid IN ({id_placeholders})"
            # Double the list of IDs because placeholders appear twice
            deleted_edges_result = con.execute(delete_edges_query, nodes_to_delete_ids + nodes_to_delete_ids).fetchall()
            print(f"Deleted associated edges for {len(nodes_to_delete_ids)} nodes.") # Note: fetchall() on DELETE returns empty list

            # 3. Delete the nodes themselves
            # Edges are deleted manually now, not via cascade
            delete_nodes_query = f"""
                DELETE FROM nodes
                WHERE nodeid IN ({id_placeholders})
                RETURNING COUNT(*);
                """
            result = con.execute(delete_nodes_query, nodes_to_delete_ids)
            fetched_result = result.fetchone()
            if fetched_result:
                deleted_count = fetched_result[0]
            else:
                 deleted_count = 0 # Should not happen if nodes_to_delete_ids was not empty
            
            con.commit()
            print(f"Successfully forgot {deleted_count} old memories.")
        except Exception as e:
            con.rollback()
            print(f"Error during forgetting old memories: {e}")
    return deleted_count

# Initialize DB on first import
init_db() 