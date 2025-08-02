"""Command-line tool for FRED's web search pipeline.

This script allows you to run a web search using the existing
:func:`intelligent_search` function and prints the summary
produced by the GIST model. Provide the search query as a
command-line argument or interactively when prompted.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import config as cfg_module

# Ensure required config attributes exist for web_search_core
cfg_module.OLLAMA_BASE_URL = cfg_module.Config.OLLAMA_BASE_URL
cfg_module.LLM_GENERATION_OPTIONS = cfg_module.Config.LLM_GENERATION_OPTIONS

from web_search_core import intelligent_search


def run_search(query: str) -> Dict[str, Any]:
    """Execute a search query through FRED's web search pipeline.

    Parameters
    ----------
    query:
        Text of the search query.

    Returns
    -------
    dict
        Result dictionary returned by :func:`intelligent_search`.
    """
    return intelligent_search(query)


def main() -> None:
    """Parse arguments and run the search."""
    parser = argparse.ArgumentParser(
        description="Run a web search through FRED's pipeline and get a gist summary.",
    )
    parser.add_argument("query", nargs="?", help="Search query to run")
    args = parser.parse_args()

    query = args.query or input("Enter your search query: ")
    results = run_search(query)

    print("\n=== Gist Summary ===\n")
    print(results.get("summary", "No summary generated."))

    links = results.get("links", [])
    if links:
        print("\n=== Top Links ===\n")
        for link in links:
            title = link.get("title", "")
            url = link.get("url", "")
            print(f"- {title}\n  {url}\n")


if __name__ == "__main__":
    main()

