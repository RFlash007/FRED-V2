import os
import shutil
import sys

# Adjust path to import from parent directory (memory module)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from memory.hybrid_memory import HybridMemory

def run_association_test():
    """Tests the memory association logic."""
    storage_dir = "./memory_store"
    test_threshold = 0.75 # Set a threshold for this test

    print("--- Starting Association Test ---")

    # 1. Clear existing memory store
    if os.path.exists(storage_dir):
        print(f"Clearing existing memory store: {storage_dir}")
        try:
            shutil.rmtree(storage_dir)
        except OSError as e:
            print(f"Error removing directory {storage_dir}: {e}")
            return # Exit if we can't clear the directory
    os.makedirs(storage_dir, exist_ok=True)

    # 2. Initialize HybridMemory
    print(f"Initializing HybridMemory with threshold: {test_threshold}")
    try:
        memory = HybridMemory(storage_dir=storage_dir, similarity_threshold=test_threshold)
    except Exception as e:
        print(f"Error initializing HybridMemory: {e}")
        return

    # 3. Add test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",  # 0
        "A fast, dark-colored fox leaps above a sleeping canine.", # 1 (Similar to 0)
        "The weather today is sunny and warm.", # 2
        "It's a beautiful day with clear skies.", # 3 (Similar to 2)
        "Automotive engineering requires precision.", # 4
        "Building cars needs careful manufacturing.", # 5 (Similar to 4)
        "The stock market experienced volatility.", # 6 (Less similar to others)
        "Fred is the central concept.", # 7 (Should be linked to Fred)
    ]

    print("\nAdding test sentences to memory...")
    node_ids = []
    for sentence in test_sentences:
        try:
            node = memory.add_memory(content=sentence)
            print(f"  Added: '{sentence[:30]}...' (ID: {node.id})")
            if node.content != "Fred": # Don't include Fred in manual pairwise comparison list
                 node_ids.append(node.id)
        except Exception as e:
            print(f"Error adding sentence '{sentence}': {e}")

    # Add Fred explicitly if it wasn't created (should have been by __init__)
    if not memory.central_node_id:
        print("[WARN] Central node 'Fred' was not found/created during init. Attempting add.")
        try:
            fred_node = memory.add_memory(content="Fred", metadata={"isCentral": True})
            memory.central_node_id = fred_node.id
            print(f"  Added Fred node with ID: {fred_node.id}")
        except Exception as e:
            print(f"Error adding Fred node: {e}")

    num_nodes = len(node_ids)
    print(f"\n--- Calculating Pairwise Similarities ({num_nodes} nodes) ---")

    associations_to_add = []

    # 4. Manually iterate, calculate similarity, report
    for i in range(num_nodes):
        node_id_a = node_ids[i]
        node_a = memory.get_node(node_id_a)
        if not node_a or not node_a.embedding:
            print(f"Skipping node {node_id_a} (missing data)")
            continue

        for j in range(i + 1, num_nodes):
            node_id_b = node_ids[j]
            node_b = memory.get_node(node_id_b)
            if not node_b or not node_b.embedding:
                print(f"Skipping node {node_id_b} (missing data)")
                continue

            try:
                similarity = memory._calculate_similarity(node_a.embedding, node_b.embedding)
                print(f"  Sim('{node_a.content[:20]}...', '{node_b.content[:20]}...') = {similarity:.4f}")

                # 5. Check threshold and mark for association
                if similarity >= memory.similarity_threshold:
                    print(f"    -> Similarity >= {memory.similarity_threshold}. Marking for association.")
                    associations_to_add.append((node_id_a, node_id_b, similarity))

            except Exception as e:
                print(f"Error calculating similarity between {node_id_a} and {node_id_b}: {e}")

    # 6. Create associations
    print(f"\n--- Creating {len(associations_to_add)} Associations ---")
    for id_a, id_b, sim in associations_to_add:
        try:
            success = memory.add_association(id_a, id_b, sim)
            if success:
                print(f"  Associated: {id_a} <-> {id_b} (Sim: {sim:.4f})")
            else:
                print(f"  Failed to associate: {id_a} <-> {id_b}")
        except Exception as e:
             print(f"Error adding association between {id_a} and {id_b}: {e}")

    # 7. Report final associations
    print("\n--- Final Node Associations ---")
    all_nodes = list(memory.nodes.values()) # Include Fred now
    for node in all_nodes:
        associations = node.get_associations_with_strengths()
        print(f"\nNode: '{node.content[:40]}...' (ID: {node.id}) has {len(associations)} associations:")
        # Print associations sorted by strength
        sorted_assocs = sorted(associations.items(), key=lambda item: item[1], reverse=True)
        for target_id, strength in sorted_assocs:
            target_node = memory.get_node(target_id)
            target_content = target_node.content[:40] if target_node else "[Missing Node]"
            print(f"  -> {target_id} ('{target_content}...') - Strength: {strength:.4f}")
        if not associations:
             print("  (No associations)")

    print("\n--- Association Test Complete ---")

if __name__ == "__main__":
    run_association_test() 