#!/usr/bin/env python3
"""
Interactive CLI for debugging F.R.E.D. search tools
--------------------------------------------------
This utility lets you:
  1. Choose one of the high-level search helpers defined in `Tools.py`.
  2. Enter a query (or URL for `webpage`).
  3. View the *exact* JSON string that the LLM receives inside
     `arch_delve_research.py` (minified with no indentation), followed by a
     pretty-printed version for easier human inspection.

Run with no arguments for an interactive prompt, or:
  python debug_search_cli.py <tool> <query>
Examples:
  python debug_search_cli.py general "python list comprehension"
  python debug_search_cli.py webpage https://example.com
"""
from __future__ import annotations

import json
import sys
from typing import Callable, Dict

from Tools import (
    tool_search_general,
    tool_search_news,
    tool_search_academic,
    tool_search_forums,
    tool_read_webpage,
)

TOOL_FUNC_MAP: Dict[str, Callable[..., dict]] = {
    "general": tool_search_general,
    "news": tool_search_news,
    "academic": tool_search_academic,
    "forums": tool_search_forums,
    "webpage": tool_read_webpage,
}


def _execute(tool: str, query_or_url: str) -> None:
    """Execute the chosen search helper and print results like the model sees."""
    func = TOOL_FUNC_MAP[tool]

    # Call with correct param name
    if tool == "webpage":
        result = func(url=query_or_url)
    else:
        result = func(query=query_or_url)

    # The model receives the raw json string via json.dumps(result)
    raw_json = json.dumps(result)

    print("\n================ RAW TOOL OUTPUT (as seen by the model) ================")
    print(raw_json)
    print("==================== HUMAN-FRIENDLY PRETTY PRINT ====================")
    print(json.dumps(result, indent=2))
    print("======================================================================\n")


def interactive_mode() -> None:
    """Loop, letting the user pick a tool and enter queries."""
    print("F.R.E.D. Search Debug CLI. Type 'exit' to quit.\n")
    while True:
        tool = input("Choose tool [general, news, academic, forums, webpage]: ").strip().lower()
        if tool in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        if tool not in TOOL_FUNC_MAP:
            print("Invalid tool. Try again.\n")
            continue

        query = input("Enter query (or URL for webpage): ").strip()
        if not query:
            print("Empty query. Try again.\n")
            continue

        _execute(tool, query)


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if len(argv) >= 2:
        tool = argv[0].lower()
        query = " ".join(argv[1:])
        if tool not in TOOL_FUNC_MAP:
            print(f"Unknown tool '{tool}'. Valid options: {', '.join(TOOL_FUNC_MAP)}")
            sys.exit(1)
        _execute(tool, query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main() 