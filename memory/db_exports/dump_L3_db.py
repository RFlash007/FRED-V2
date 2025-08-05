#!/usr/bin/env python
"""
Utility script to export the contents of the L3 DuckDB (knowledge graph)
into a human-readable JSON file located in the same directory.

Usage
-----
python dump_L3_db.py [output_filename]

If *output_filename* is omitted, a timestamped file like
``L3_dump_20250729_120305.json`` is created automatically.
"""

from __future__ import annotations

import duckdb
import json
import os
import sys
from datetime import datetime
from contextlib import closing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ollie_print import olliePrint_simple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The L3 DB lives two directories up (memory/memory.db)
DB_PATH = os.path.join(SCRIPT_DIR, "..", "memory.db")

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Could not locate L3 database at {DB_PATH}.")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fetch_table_as_dict(conn: duckdb.DuckDBPyConnection, table_name: str):
    """Return *table_name* as a list of dicts for JSON serialisation."""
    result = conn.execute(f"SELECT * FROM {table_name}")
    col_names = [desc[0] for desc in result.description]
    return [dict(zip(col_names, row)) for row in result.fetchall()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Determine output filename
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
        if not out_file.lower().endswith(".json"):
            out_file += ".json"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"L3_dump_{ts}.json"

    out_path = os.path.join(SCRIPT_DIR, out_file)

    with closing(duckdb.connect(DB_PATH)) as conn:
        tables = conn.execute("SHOW TABLES").fetchall()
        if not tables:
            olliePrint_simple("No tables found in the database.")
            return

        db_dump: dict[str, list[dict]] = {}
        for (tbl,) in tables:
            db_dump[tbl] = fetch_table_as_dict(conn, tbl)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(db_dump, f, ensure_ascii=False, indent=2, default=str)

    olliePrint_simple(f"Export complete → {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        olliePrint_simple(f"Error: {exc}", level='error')
