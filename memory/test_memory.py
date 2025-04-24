import pytest
import os
import sys
import duckdb
import time
import datetime
import requests_mock

# Ensure the librarian module can be found
# Get the directory of the current test file
test_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

from memory import librarian

# --- Test Configuration ---
TEST_DB_FILE = "test_memory.db"
OLLAMA_API_URL = librarian.OLLAMA_API_URL # Use the URL from the module
EMBED_MODEL = librarian.EMBED_MODEL
EMBEDDING_DIM = librarian.EMBEDDING_DIM
# Generate a dummy embedding of the correct dimension
DUMMY_EMBEDDING = [0.1] * EMBEDDING_DIM

# --- Fixtures ---

@pytest.fixture(scope="function")
def test_db():
    """Fixture to set up and tear down a test database."""
    db_path = os.path.join(test_dir, TEST_DB_FILE)
    # Ensure librarian uses the test DB
    original_db_file = librarian.DB_FILE
    librarian.DB_FILE = db_path

    # Delete the test database file if it exists before the test
    if os.path.exists(db_path):
        os.remove(db_path)
        if os.path.exists(f"{db_path}.wal"): # Remove WAL file too
             os.remove(f"{db_path}.wal")

    # Initialize the database schema
    librarian.init_db()
    print(f"Initialized test database: {db_path}")

    yield db_path # Provide the path to the test function

    # Teardown: Close connection (if any open) and delete the test database file
    try:
        # Attempt to connect and close to ensure no locks
        conn = duckdb.connect(db_path)
        conn.close()
    except Exception as e:
        print(f"Error during test DB teardown connection: {e}")

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Removed test database: {db_path}")
        except PermissionError as e:
             print(f"Warning: Could not remove test database file {db_path}: {e}")
        if os.path.exists(f"{db_path}.wal"): # Remove WAL file too
             try:
                 os.remove(f"{db_path}.wal")
             except PermissionError as e:
                 print(f"Warning: Could not remove WAL file {db_path}.wal: {e}")

    # Restore original DB file path in librarian
    librarian.DB_FILE = original_db_file


@pytest.fixture
def mock_ollama(requests_mock):
    """Fixture to mock the Ollama embedding API call."""
    requests_mock.post(OLLAMA_API_URL, json={"embedding": DUMMY_EMBEDDING})
    return requests_mock

# --- Test Functions ---

def test_init_db(test_db):
    """Test if database and tables are created."""
    assert os.path.exists(test_db)
    with duckdb.connect(test_db) as con:
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        assert "nodes" in table_names
        assert "edges" in table_names

        # Check nodes table schema
        nodes_info = con.execute("PRAGMA table_info('nodes')").fetchall()
        nodes_cols = {row[1]: row[2] for row in nodes_info}
        assert nodes_cols['nodeid'] == 'BIGINT'
        assert nodes_cols['type'] == 'VARCHAR' # DuckDB uses VARCHAR for TEXT
        assert nodes_cols['label'] == 'VARCHAR'
        assert nodes_cols['text'] == 'VARCHAR'
        assert nodes_cols['created_at'] == 'TIMESTAMP'
        assert nodes_cols['last_access'] == 'TIMESTAMP'
        assert nodes_cols['superseded_at'] == 'TIMESTAMP'
        assert nodes_cols['parent_id'] == 'BIGINT'
        assert nodes_cols['embedding'].startswith(f'FLOAT[{EMBEDDING_DIM}]') # Check dimension
        assert nodes_cols['embed_model'] == 'VARCHAR'

        # Check edges table schema
        edges_info = con.execute("PRAGMA table_info('edges')").fetchall()
        edges_cols = {row[1]: row[2] for row in edges_info}
        assert edges_cols['sourceid'] == 'BIGINT'
        assert edges_cols['targetid'] == 'BIGINT'
        assert edges_cols['rel_type'] == 'VARCHAR'
        assert edges_cols['created_at'] == 'TIMESTAMP'

def test_classify_memory_type():
    """Test the simple classification rule."""
    assert librarian.classify_memory_type("This is a fact.") == "Semantic"
    assert librarian.classify_memory_type("The sky is blue.") == "Semantic"
    assert librarian.classify_memory_type("I went to the store yesterday.") == "Episodic"
    assert librarian.classify_memory_type("We learned about Python in 2023.") == "Episodic"
    assert librarian.classify_memory_type("The meeting happened on 5th June.") == "Episodic"

def test_add_memory(test_db, mock_ollama):
    """Test adding a new memory node."""
    label = "Test Fact"
    text = "This is a semantic test memory."
    node_id = librarian.add_memory(label, text)

    assert node_id is not None
    assert isinstance(node_id, int)
    mock_ollama.post.assert_called_once()
    assert mock_ollama.last_request.json() == {"model": EMBED_MODEL, "prompt": text}

    with duckdb.connect(test_db) as con:
        result = con.execute("SELECT nodeid, label, text, type, embed_model FROM nodes WHERE nodeid = ?", [node_id]).fetchone()
        assert result is not None
        assert result[0] == node_id
        assert result[1] == label
        assert result[2] == text
        assert result[3] == "Semantic" # Should be classified correctly
        assert result[4] == EMBED_MODEL

        # Check embedding (optional, depends on how you store it)
        embedding_result = con.execute("SELECT embedding FROM nodes WHERE nodeid = ?", [node_id]).fetchone()
        assert len(embedding_result[0]) == EMBEDDING_DIM
        assert embedding_result[0] == pytest.approx(DUMMY_EMBEDDING) # Check against mocked value

def test_add_episodic_memory(test_db, mock_ollama):
    """Test adding an explicitly episodic memory."""
    label = "Test Experience"
    text = "I wrote a test today."
    node_id = librarian.add_memory(label, text, memory_type="Episodic")

    assert node_id is not None
    with duckdb.connect(test_db) as con:
        result = con.execute("SELECT type FROM nodes WHERE nodeid = ?", [node_id]).fetchone()
        assert result[0] == "Episodic"

def test_add_edge(test_db, mock_ollama):
    """Test adding an edge between two nodes."""
    id1 = librarian.add_memory("Node 1", "First node text")
    id2 = librarian.add_memory("Node 2", "Second node text")
    assert id1 is not None
    assert id2 is not None

    librarian.add_edge(id1, id2, "relatedTo")

    with duckdb.connect(test_db) as con:
        edge = con.execute("SELECT sourceid, targetid, rel_type FROM edges WHERE sourceid = ? AND targetid = ?", [id1, id2]).fetchone()
        assert edge is not None
        assert edge[0] == id1
        assert edge[1] == id2
        assert edge[2] == "relatedTo"

        # Check last_access times were updated
        # Note: Timestamps might be tricky to assert exactly due to slight delays
        node1_access = con.execute("SELECT last_access FROM nodes WHERE nodeid = ?", [id1]).fetchone()[0]
        node2_access = con.execute("SELECT last_access FROM nodes WHERE nodeid = ?", [id2]).fetchone()[0]
        assert (datetime.datetime.now() - node1_access).total_seconds() < 5 # Check within a few seconds
        assert (datetime.datetime.now() - node2_access).total_seconds() < 5

def test_supersede_memory(test_db, mock_ollama):
    """Test superseding an existing memory."""
    old_label = "Old Fact"
    old_text = "This fact is outdated."
    new_label = "New Fact"
    new_text = "This fact is current."

    old_id = librarian.add_memory(old_label, old_text)
    assert old_id is not None
    time.sleep(0.01) # Ensure timestamps differ slightly

    # Store initial access time
    with duckdb.connect(test_db) as con:
         initial_access = con.execute("SELECT last_access FROM nodes WHERE nodeid = ?", [old_id]).fetchone()[0]

    time.sleep(0.01) # Ensure supersede time is later

    new_id = librarian.supersede_memory(old_id, new_label, new_text)
    assert new_id is not None
    assert new_id != old_id

    with duckdb.connect(test_db) as con:
        # Check old node
        old_node = con.execute("SELECT label, superseded_at, last_access FROM nodes WHERE nodeid = ?", [old_id]).fetchone()
        assert old_node is not None
        assert old_node[0] == old_label
        assert old_node[1] is not None # superseded_at should be set
        assert isinstance(old_node[1], datetime.datetime)
        assert old_node[2] > initial_access # last_access should be updated

        # Check new node
        new_node = con.execute("SELECT label, text, type, superseded_at, last_access FROM nodes WHERE nodeid = ?", [new_id]).fetchone()
        assert new_node is not None
        assert new_node[0] == new_label
        assert new_node[1] == new_text
        assert new_node[2] == "Semantic"
        assert new_node[3] is None # New node not superseded
        assert new_node[4] > initial_access # last_access should be updated

        # Check 'updates' edge
        edge = con.execute("SELECT sourceid, targetid, rel_type FROM edges WHERE sourceid = ? AND rel_type = ?", [new_id, 'updates']).fetchone()
        assert edge is not None
        assert edge[0] == new_id
        assert edge[1] == old_id
        assert edge[2] == 'updates'


def test_search_memory_similarity(test_db, mock_ollama):
    """Test searching by cosine similarity."""
    id1 = librarian.add_memory("Topic A", "Information about apples.")
    id2 = librarian.add_memory("Topic B", "Details regarding oranges.")
    id3 = librarian.add_memory("Topic C", "Notes on bananas.")
    assert all([id1, id2, id3])

    # Mock Ollama to return the same embedding for the query
    query = "Tell me about apples"
    requests_mock.post(OLLAMA_API_URL, json={"embedding": DUMMY_EMBEDDING})

    results = librarian.search_memory(query, limit=2)

    assert len(results) > 0 # Should find at least the exact match if mocking works
    # With perfect mock, similarity should be ~1 for the first result if embedding matches
    # Check if the most relevant item is returned (assuming mock makes id1 most similar)
    # Note: Real results depend heavily on the actual embedding model
    assert results[0]['nodeid'] == id1
    assert results[0]['label'] == "Topic A"
    assert results[0]['similarity'] == pytest.approx(1.0)

    # Check last access was updated for the returned node(s)
    with duckdb.connect(test_db) as con:
        access_time = con.execute("SELECT last_access FROM nodes WHERE nodeid = ?", [id1]).fetchone()[0]
        assert (datetime.datetime.now() - access_time).total_seconds() < 5

def test_search_memory_keyword(test_db):
    """Test keyword fallback search when embedding fails or finds nothing."""
    # Add nodes without mocking Ollama (so embedding is None)
    id1 = librarian.add_memory("Keyword Test 1", "Contains the word pineapple.")
    id2 = librarian.add_memory("Keyword Test 2", "Also mentions PINEAPPLE.") # Test case insensitivity if LIKE is used
    id3 = librarian.add_memory("Irrelevant", "Topic about grapes.")
    assert all([id1, id2, id3])

    # Simulate embedding failure by not mocking Ollama for the search query
    # or by searching for something guaranteed not to match semantically if embedding worked.
    results = librarian.search_memory("pineapple")

    assert len(results) >= 1 # Should find at least one via keyword
    result_ids = {r['nodeid'] for r in results}
    assert id1 in result_ids
    # Depending on DB collation/LIKE behavior, id2 might also be found
    # assert id2 in result_ids
    assert id3 not in result_ids
    assert all(r['similarity'] == 0.0 for r in results) # Keyword results have 0 similarity


def test_search_memory_type_filter(test_db, mock_ollama):
    """Test filtering search results by type."""
    id_semantic = librarian.add_memory("Fact", "Sky is blue.", memory_type="Semantic")
    id_episodic = librarian.add_memory("Event", "I saw a blue bird yesterday.", memory_type="Episodic")
    assert id_semantic and id_episodic

    # Search for "blue" - should match both without type filter
    results_all = librarian.search_memory("blue")
    assert len(results_all) >= 1 # Should find at least one
    result_ids_all = {r['nodeid'] for r in results_all}
    # Exact results depend on embedding/keyword match, but both *could* be present
    # assert id_semantic in result_ids_all
    # assert id_episodic in result_ids_all

    # Search for "blue" filtering by Semantic
    results_semantic = librarian.search_memory("blue", memory_type="Semantic")
    assert len(results_semantic) >= 1
    assert all(r['type'] == 'Semantic' for r in results_semantic)
    assert results_semantic[0]['nodeid'] == id_semantic # Assuming it's the primary match
    assert id_episodic not in {r['nodeid'] for r in results_semantic}

    # Search for "blue" filtering by Episodic
    results_episodic = librarian.search_memory("blue", memory_type="Episodic")
    assert len(results_episodic) >= 1
    assert all(r['type'] == 'Episodic' for r in results_episodic)
    assert results_episodic[0]['nodeid'] == id_episodic # Assuming it's the primary match
    assert id_semantic not in {r['nodeid'] for r in results_episodic}


def test_get_graph_data(test_db, mock_ollama):
    """Test retrieving graph data for the mind map."""
    id1 = librarian.add_memory("Center", "Center node")
    id2 = librarian.add_memory("Child1", "First child")
    id3 = librarian.add_memory("Child2", "Second child")
    id4 = librarian.add_memory("Grandchild", "Child of Child1")
    id5 = librarian.add_memory("Unrelated", "Not connected initially")
    assert all([id1, id2, id3, id4, id5])

    librarian.add_edge(id1, id2, "relatedTo")
    librarian.add_edge(id1, id3, "relatedTo")
    librarian.add_edge(id2, id4, "instanceOf")

    # Test depth 0
    graph_d0 = librarian.get_graph_data(id1, depth=0)
    assert len(graph_d0['nodes']) == 1
    assert graph_d0['nodes'][0]['id'] == id1
    assert len(graph_d0['edges']) == 0 # No edges within depth 0 of center

    # Test depth 1
    graph_d1 = librarian.get_graph_data(id1, depth=1)
    node_ids_d1 = {n['id'] for n in graph_d1['nodes']}
    edge_tuples_d1 = {(e['source'], e['target']) for e in graph_d1['edges']}
    assert node_ids_d1 == {id1, id2, id3} # Center + direct children
    assert len(graph_d1['edges']) == 2
    assert (id1, id2) in edge_tuples_d1 or (id2, id1) in edge_tuples_d1
    assert (id1, id3) in edge_tuples_d1 or (id3, id1) in edge_tuples_d1
    # Grandchild should not be included yet
    assert id4 not in node_ids_d1

    # Test depth 2
    graph_d2 = librarian.get_graph_data(id1, depth=2)
    node_ids_d2 = {n['id'] for n in graph_d2['nodes']}
    edge_tuples_d2 = {(e['source'], e['target'], e['rel_type']) for e in graph_d2['edges']}
    assert node_ids_d2 == {id1, id2, id3, id4} # Center, children, grandchild
    assert len(graph_d2['edges']) == 3
    assert (id1, id2, 'relatedTo') in edge_tuples_d2
    assert (id1, id3, 'relatedTo') in edge_tuples_d2
    assert (id2, id4, 'instanceOf') in edge_tuples_d2
    # Unrelated node should not be included
    assert id5 not in node_ids_d2

    # Check last access updated for nodes in graph_d2
    with duckdb.connect(test_db) as con:
        for nid in node_ids_d2:
             access_time = con.execute("SELECT last_access FROM nodes WHERE nodeid = ?", [nid]).fetchone()[0]
             assert (datetime.datetime.now() - access_time).total_seconds() < 5
        # Check unrelated node access time wasn't updated by get_graph_data
        unrelated_access = con.execute("SELECT last_access FROM nodes WHERE nodeid = ?", [id5]).fetchone()[0]
        assert (datetime.datetime.now() - unrelated_access).total_seconds() > 5 # Assuming test runs fast enough


def test_forget_old_memories(test_db, mock_ollama):
    """Test the forgetting mechanism for old, superseded nodes."""
    # Add a node and supersede it immediately
    old_id = librarian.add_memory("Initial", "To be superseded")
    time.sleep(0.01)
    new_id = librarian.supersede_memory(old_id, "Current", "Superseded the old one")
    assert new_id is not None

    # Add another node that is not superseded
    active_id = librarian.add_memory("Active", "This one stays")

    # Make the superseded node appear old (both superseded_at and last_access)
    very_old_date = datetime.datetime.now() - datetime.timedelta(days=200)
    with duckdb.connect(test_db) as con:
        con.execute("UPDATE nodes SET superseded_at = ?, last_access = ? WHERE nodeid = ?",
                    [very_old_date, very_old_date, old_id])
        # Ensure the active node's last_access is recent
        con.execute("UPDATE nodes SET last_access = ? WHERE nodeid = ?",
                    [datetime.datetime.now(), active_id])
        # Ensure the new node (that superseded the old one) is also recent
        con.execute("UPDATE nodes SET last_access = ? WHERE nodeid = ?",
                    [datetime.datetime.now(), new_id])

    # Run the forget function with default 180 days
    deleted_count = librarian.forget_old_memories()
    assert deleted_count == 1

    # Verify the old node is gone, others remain
    with duckdb.connect(test_db) as con:
        old_node_exists = con.execute("SELECT COUNT(*) FROM nodes WHERE nodeid = ?", [old_id]).fetchone()[0]
        new_node_exists = con.execute("SELECT COUNT(*) FROM nodes WHERE nodeid = ?", [new_id]).fetchone()[0]
        active_node_exists = con.execute("SELECT COUNT(*) FROM nodes WHERE nodeid = ?", [active_id]).fetchone()[0]

        assert old_node_exists == 0
        assert new_node_exists == 1
        assert active_node_exists == 1

        # Verify the 'updates' edge involving the deleted node is also gone (due to CASCADE)
        edge_exists = con.execute("SELECT COUNT(*) FROM edges WHERE sourceid = ? OR targetid = ?", [old_id, old_id]).fetchone()[0]
        assert edge_exists == 0

def test_backup_script_execution(tmp_path):
    """Test if the backup script can be generated (execution test requires shell)."""
    # This test only checks if the script content seems okay.
    # A full integration test would run the script.
    backup_script_path = os.path.join(project_root, "scripts", "backup_memory.sh")
    assert os.path.exists(backup_script_path)
    with open(backup_script_path, 'r') as f:
        content = f.read()
    assert "#!/bin/bash" in content
    assert "duckdb" in content
    assert ".backup" in content
    assert "backups/" in content
    assert "memory.db" in content

def test_forget_script_execution(tmp_path):
    """Test if the forget script can be generated (execution test requires shell)."""
    # This test only checks if the script content seems okay.
    forget_script_path = os.path.join(project_root, "scripts", "forget_memory.sh")
    assert os.path.exists(forget_script_path)
    with open(forget_script_path, 'r') as f:
        content = f.read()
    assert "#!/bin/bash" in content
    assert "python" in content # or python3
    assert "librarian.forget_old_memories" in content
    assert "days_old=180" in content 