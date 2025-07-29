#!/usr/bin/env python
"""Manual trigger for F.R.E.D.'s sleep-cycle.

This script invokes the `tool_trigger_sleep_cycle` function defined in
`Tools.py`, prints a human-readable summary to stdout, and writes a full
JSON report to the current directory (or a user-supplied filename).

Usage
-----
python trigger_sleep_cycle.py [output_filename]

If *output_filename* is omitted, a timestamped file such as
``sleep_cycle_20250729_121530.json`` is generated automatically.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# Ensure project root is on sys.path when script run from sub-directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from Tools import tool_trigger_sleep_cycle  # noqa: E402
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"[trigger_sleep_cycle] Failed to import Tools: {exc}\n")
    sys.exit(1)


def main() -> None:  # pragma: no cover
    # ---------------------------------------------------------------------
    # Run sleep cycle
    # ---------------------------------------------------------------------
    result = tool_trigger_sleep_cycle()

    if not result.get("success", False):
        sys.stderr.write(f"[trigger_sleep_cycle] Sleep cycle failed: {result.get('error')}\n")
        sys.exit(1)

    # Pretty print summary
    print("==== F.R.E.D. Sleep Cycle Summary ====")
    print(result.get("summary", "<no summary provided>"))
    print("======================================\n")

    # ------------------------------------------------------------------
    # Determine output filename
    # ------------------------------------------------------------------
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
        if not out_file.lower().endswith(".json"):
            out_file += ".json"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"sleep_cycle_{ts}.json"

    out_path = os.path.join(SCRIPT_DIR, out_file)

    # Dump full result
    try:
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2, default=str)
        print(f"[trigger_sleep_cycle] Detailed report written to: {out_path}")
    except (IOError, OSError) as io_err:  # pragma: no cover
        sys.stderr.write(f"[trigger_sleep_cycle] Failed to write report: {io_err}\n")


if __name__ == "__main__":
    main()
