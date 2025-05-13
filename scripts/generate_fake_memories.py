#!/usr/bin/env python
"""
Fake Memory Generator - TEST VERSION (Multi-Level Deterministic)

This script creates a more complex, deterministic graph with primary and secondary
connections to test F.R.E.D. memory visualization.
It uses ONLY valid memory types: 'Semantic', 'Episodic', 'Procedural'.
And ONLY valid relationship types.
The "F.R.E.D. Memory Core" node itself is not created here as a database entity;
app.py injects it dynamically into the graph data sent to the frontend.
This script focuses on creating the memories that will orbit this conceptual core.
"""

import os
import sys
import datetime

# Add the parent directory to sys.path to import librarian
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import memory.librarian as lib

# Initialize database
DB_PATH = os.path.join(parent_dir, 'memory', 'memory.db')
lib.DB_FILE = DB_PATH

print(f"Setting memory database path to: {DB_PATH}")
lib.init_db()

# Clear existing memories
print("Clearing existing memories...")
if not lib.clear_all_memory(force=True): # Force clear for test data generation
    print("CRITICAL: Failed to clear memory. Aborting fake memory generation.")
    sys.exit(1)

# Function to create specific memories and connections
def generate_multi_level_test_graph():
    print("Generating a multi-level deterministic test graph...")
    
    node_ids = {} # Use a dict to store nodes by name for clarity

    # --- Create Memory Nodes (20 nodes to test more complex visualization) ---
    # Using valid memory types: 'Semantic', 'Episodic', 'Procedural'
    node_data_to_create = [
        # Primary Nodes (will orbit F.R.E.D. directly)
        ('AI_Concepts', "AI Concepts", "Core ideas in Artificial Intelligence including machine learning, neural networks, and natural language processing.", "Semantic"),
        ('Project_FRED_Dev', "Project F.R.E.D. Development", "Timeline and key milestones in F.R.E.D.'s development process, from initial concept to current implementation.", "Episodic"),
        ('Home_Automation', "Home Automation System", "Complete smart home setup with various integrated devices, control systems and automation rules.", "Semantic"),
        ('Personal_Schedule', "Personal Schedule", "Daily routines, appointments, and important time-based events.", "Episodic"),
        ('Knowledge_Base', "Knowledge Base Structure", "Organization and access patterns for F.R.E.D.'s internal knowledge representation.", "Semantic"),
        
        # Secondary Nodes (will orbit primary nodes)
        ('NLP_Topic', "Natural Language Processing", "Understanding and generating human language through computational methods.", "Semantic"),
        ('CV_Topic', "Computer Vision", "Techniques for acquiring, processing, and analyzing digital images and video.", "Semantic"),
        ('ML_Models', "Machine Learning Models", "Various algorithms and architectures for pattern recognition and prediction.", "Semantic"),
        
        ('Coding_Sessions', "F.R.E.D. Coding Sessions", "Records of significant programming work on F.R.E.D. components.", "Episodic"),
        ('Bug_Fixes', "Bug Fixes History", "Tracking of software issues encountered and their resolutions.", "Episodic"),
        ('Feature_Planning', "Feature Planning", "Roadmap and prioritization of upcoming F.R.E.D. capabilities.", "Procedural"),
        
        ('SmartLights_Control', "Smart Lighting System", "Procedures for controlling Philips Hue and other lighting systems.", "Procedural"),
        ('HVAC_Automation', "HVAC Control System", "Temperature and climate control automation procedures.", "Procedural"),
        ('Security_System', "Home Security System", "Monitoring and managing home security devices and alerts.", "Procedural"),
        
        ('Morning_Routine', "Morning Routine", "Sequence of regular morning activities and automation triggers.", "Episodic"),
        ('Work_Schedule', "Work Schedule", "Regular working hours and meeting patterns.", "Episodic"),
        
        ('Memory_Organization', "Memory Organization", "Structure and classification system for F.R.E.D.'s memory nodes.", "Semantic"),
        ('Query_Patterns', "Knowledge Query Patterns", "Common methods of retrieving and relating information.", "Procedural"),
        
        # Tertiary Nodes (will orbit secondary nodes)
        ('Speech_Recognition', "Speech Recognition", "Converting spoken language to text through audio processing.", "Procedural"),
        ('Language_Models', "Language Models", "Transformer-based architectures for language understanding and generation.", "Semantic")
    ]

    for key, label, text, mem_type in node_data_to_create:
        try:
            node_ids[key] = lib.add_memory(label=label, text=text, memory_type=mem_type)
            print(f"Created node '{label}' ({key}) with ID: {node_ids[key]}")
        except Exception as e:
            print(f"ERROR creating node '{label}': {e}")
            node_ids[key] = None # Ensure key exists but is None if creation failed

    print("\n--- Created Memory Nodes ---")
    success_count = sum(1 for nid in node_ids.values() if nid is not None)
    print(f"Successfully created {success_count} out of {len(node_data_to_create)} memory nodes.")

    # --- Create Edges ---
    # Structure: Primary nodes have many connections, secondary fewer, tertiary fewest
    edge_data_to_create = [
        # Connect primary nodes to each other
        ('AI_Concepts', 'Project_FRED_Dev', 'relatedTo'),
        ('AI_Concepts', 'Knowledge_Base', 'dependsOn'),
        ('Project_FRED_Dev', 'Home_Automation', 'enablesGoal'),
        ('Project_FRED_Dev', 'Personal_Schedule', 'servesPurpose'),
        ('Home_Automation', 'Personal_Schedule', 'activatesIn'),
        ('Knowledge_Base', 'AI_Concepts', 'contains'),
        
        # Connect secondary nodes to primary nodes
        # AI Concepts connections
        ('AI_Concepts', 'NLP_Topic', 'contains'),
        ('AI_Concepts', 'CV_Topic', 'contains'),
        ('AI_Concepts', 'ML_Models', 'contains'),
        
        # Project FRED connections
        ('Project_FRED_Dev', 'Coding_Sessions', 'contains'),
        ('Project_FRED_Dev', 'Bug_Fixes', 'contains'),
        ('Project_FRED_Dev', 'Feature_Planning', 'dependsOn'),
        
        # Home Automation connections
        ('Home_Automation', 'SmartLights_Control', 'contains'),
        ('Home_Automation', 'HVAC_Automation', 'contains'),
        ('Home_Automation', 'Security_System', 'contains'),
        
        # Personal Schedule connections
        ('Personal_Schedule', 'Morning_Routine', 'contains'),
        ('Personal_Schedule', 'Work_Schedule', 'contains'),
        
        # Knowledge Base connections
        ('Knowledge_Base', 'Memory_Organization', 'contains'),
        ('Knowledge_Base', 'Query_Patterns', 'dependsOn'),
        
        # Connect tertiary nodes to secondary nodes
        ('NLP_Topic', 'Speech_Recognition', 'enablesGoal'),
        ('NLP_Topic', 'Language_Models', 'contains'),
        
        # Cross-connections between different hierarchies
        ('ML_Models', 'Language_Models', 'contains'),
        ('Speech_Recognition', 'SmartLights_Control', 'activatesIn'),
        ('SmartLights_Control', 'Morning_Routine', 'partOf'),
        ('HVAC_Automation', 'Work_Schedule', 'activatesIn'),
        ('Feature_Planning', 'Knowledge_Base', 'dependsOn'),
        ('Query_Patterns', 'Speech_Recognition', 'servesPurpose'),
        ('Security_System', 'Morning_Routine', 'precedes')
    ]

    print("\n--- Creating Edges Between Memory Nodes ---")
    edge_creation_successful_count = 0
    edge_creation_failed_count = 0
    try:
        with lib.duckdb.connect(lib.DB_FILE) as con:
            for source_key, target_key, rel_type in edge_data_to_create:
                source_id = node_ids.get(source_key)
                target_id = node_ids.get(target_key)
                
                if source_id and target_id:
                    try:
                        lib.add_edge(sourceid=source_id, targetid=target_id, rel_type=rel_type, con=con)
                        print(f"Created edge: {source_key} --({rel_type})--> {target_key}")
                        edge_creation_successful_count += 1
                    except Exception as e:
                        print(f"ERROR creating edge from '{source_key}' ({source_id}) to '{target_key}' ({target_id}) with rel_type '{rel_type}': {e}")
                        edge_creation_failed_count += 1
                else:
                    print(f"Skipping edge from '{source_key}' to '{target_key}' due to missing node(s).")
                    edge_creation_failed_count += 1
            
            if edge_creation_successful_count > 0:
                 print(f"  Successfully created {edge_creation_successful_count} edges.")
            if edge_creation_failed_count > 0:
                 print(f"  Failed to create {edge_creation_failed_count} edges. See errors above.")
            if edge_creation_successful_count == 0 and edge_creation_failed_count == 0:
                 print("  No edges were attempted or created (possibly due to all nodes failing creation).")

    except Exception as e:
        print(f"  Major error during edge creation phase: {e}")
        
    return [nid for nid in node_ids.values() if nid is not None]

# Create the multi-level test graph
created_node_ids = generate_multi_level_test_graph()

print("\nMulti-level deterministic test graph generation complete!")
if created_node_ids:
    print(f"Successfully created {len(created_node_ids)} memory nodes.")
    
    # Count edges to inform user about test data size
    try:
        with lib.duckdb.connect(lib.DB_FILE) as con:
            edge_count = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            print(f"Successfully created {edge_count} edges between memory nodes.")
            
            # Print nodes with most connections for reference
            print("\nTop 5 most connected nodes (these should orbit F.R.E.D. directly):")
            most_connected = con.execute("""
                SELECT n.nodeid, n.label, COUNT(e.sourceid) + COUNT(e2.targetid) as edge_count
                FROM nodes n
                LEFT JOIN edges e ON n.nodeid = e.sourceid
                LEFT JOIN edges e2 ON n.nodeid = e2.targetid
                GROUP BY n.nodeid, n.label
                ORDER BY edge_count DESC
                LIMIT 5
            """).fetchall()
            
            for i, (node_id, label, count) in enumerate(most_connected, 1):
                print(f"{i}. {label} - {count} connections")
    except Exception as e:
        print(f"Error retrieving statistics: {e}")
else:
    print("No memory nodes were successfully created. Please check the logs for errors.")
    
print("\nYou can now restart app.py and view the memory visualization in your browser.") 