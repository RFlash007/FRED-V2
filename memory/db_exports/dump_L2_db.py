#!/usr/bin/env python
"""
Utility script to export the contents of the L2 DuckDB (episodic cache)
into a human-readable JSON file placed in the same directory.

Usage
-----
python dump_L2_db.py [output_filename]

If *output_filename* is omitted, one is generated automatically with the
current timestamp, e.g. ``L2_dump_20250729_120305.json``.
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
# The database lives one directory above this script (memory/L2_episodic_cache.db)
DB_PATH = os.path.join(SCRIPT_DIR, "..", "L2_episodic_cache.db")

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Could not locate L2 database at {DB_PATH}.")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fetch_table_as_dict(conn: duckdb.DuckDBPyConnection, table_name: str):
    """Return *table_name* as a list of dicts suitable for JSON serialisation."""
    query = f"SELECT * FROM {table_name}"
    result = conn.execute(query)
    col_names = [desc[0] for desc in result.description]
    rows = result.fetchall()
    return [dict(zip(col_names, row)) for row in rows]


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
        out_file = f"L2_dump_{ts}.json"

    out_path = os.path.join(SCRIPT_DIR, out_file)

    with closing(duckdb.connect(DB_PATH)) as conn:
        tables = conn.execute("SHOW TABLES").fetchall()
        if not tables:
            olliePrint_simple("No tables found in the database.")
            return

        # Build a JSON-serialisable dict: {table_name: [row_dicts]}
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
