#!/bin/bash

# Simple script to forget old, superseded memories

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go up one level to the project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Path to the Python script/module containing the forget function
# Assuming the Flask app structure where librarian is in memory/
PYTHON_SCRIPT="$PROJECT_ROOT/memory/run_forget.py"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null;
then
    echo "Error: python command not found."
    exit 1
fi

# Determine which python command to use
PYTHON_CMD=$(command -v python3 || command -v python)

# Create a temporary Python script to call the librarian function
cat << EOF > "$PYTHON_SCRIPT"
import sys
import os
from ollie_print import olliePrint

# Add the project root to the Python path to find the memory module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    # Ensure correct DB path relative to the script
    from memory import librarian
    db_path = os.path.join(project_root, 'memory', 'memory.db')
    if not os.path.exists(db_path):
        olliePrint(f"Error: Database file not found at {db_path}", level='error')
        sys.exit(1)
    librarian.DB_FILE = db_path
    olliePrint(f"Running forget_old_memories (cutoff: 180 days)... Database: {librarian.DB_FILE}")
    deleted_count = librarian.forget_old_memories(days_old=180)
    olliePrint(f"Forget script finished. Deleted {deleted_count} nodes.", level='success')
    sys.exit(0)
except ImportError as e:
    olliePrint(f"Error importing librarian module: {e}", level='error')
    olliePrint("Ensure the script is run from the project root or the Python path is set correctly.", level='warning')
    sys.exit(1)
except Exception as e:
    olliePrint(f"An error occurred: {e}", level='error')
    sys.exit(1)
EOF

echo "Executing forget script via Python..."

# Execute the Python script
"$PYTHON_CMD" "$PYTHON_SCRIPT"

# Clean up the temporary Python script
rm "$PYTHON_SCRIPT"

if [ $? -eq 0 ]; then
    echo "Forget script execution successful."
else
    echo "Error: Forget script execution failed."
    exit 1
fi

exit 0 