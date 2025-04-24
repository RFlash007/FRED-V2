#!/bin/bash

# Simple backup script for the memory database

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go up one level to the project root (assuming scripts/ is in the root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DB_FILE="$PROJECT_ROOT/memory/memory.db"
BACKUP_DIR="$PROJECT_ROOT/backups"
TIMESTAMP=$(date +%F_%H%M)
BACKUP_FILE="$BACKUP_DIR/${TIMESTAMP}.db"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Check if duckdb command exists
if ! command -v duckdb &> /dev/null
then
    echo "Error: duckdb command not found. Please install DuckDB CLI."
    exit 1
fi

# Check if database file exists
if [ ! -f "$DB_FILE" ]; then
    echo "Error: Database file not found at $DB_FILE"
    exit 1
fi

echo "Backing up $DB_FILE to $BACKUP_FILE..."

# Run the backup command
duckdb "$DB_FILE" ".backup '$BACKUP_FILE'"

if [ $? -eq 0 ]; then
    echo "Backup successful."
else
    echo "Error: Backup failed."
    exit 1
fi

exit 0 