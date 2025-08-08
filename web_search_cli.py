"""Command-line tool for FRED's web search pipeline.

This script allows you to run a web search using the existing
:func:`intelligent_search` function and prints the summary
produced by the GIST model. Provide the search query as a
command-line argument or interactively when prompted.
"""

from __future__ import annotations

import argparse
import logging
import os
import json
from datetime import datetime
from typing import Any, Dict

import config as cfg_module

# No-op logger to fully silence CLI output unless explicit prints are added
class _NoOpLogger:
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass

logger = _NoOpLogger()

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
    logger.info(f"Starting search for query: {query}")
    results = intelligent_search(query)
    
    # Log the search results summary
    if 'links' in results:
        logger.info(f"Found {len(results['links'])} relevant links")
    if 'extracted_content' in results:
        logger.info(f"Successfully extracted content from {len(results['extracted_content'])} pages")
    
    return results


def save_results_to_file(query: str, results: Dict[str, Any]) -> str:
    """Save search results to a JSON file with timestamp.
    
    Args:
        query: The search query
        results: Search results dictionary
        
    Returns:
        str: Path to the saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs("search_results", exist_ok=True)
    
    # Create a filename with timestamp and sanitized query
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
    filename = f"search_results/{timestamp}_{safe_query}.json"
    
    # Save results to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    return filename

def main() -> None:
    """Parse arguments and run the search."""
    parser = argparse.ArgumentParser(
        description="Run a web search through FRED's pipeline and get a gist summary.",
    )
    parser.add_argument("query", nargs="?", help="Search query to run")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-generated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable console output (silent by default)")
    args = parser.parse_args()

    # Logging is disabled globally; verbosity flag has no effect

    query = args.query or input("Enter your search query: ")
    
    try:
        results = run_search(query)
        
        # Print summary
        if args.verbose:
            print("=== Gist Summary ===")
            summary = results.get("summary", "No summary generated.")
            print(summary)
            print()

        # Print top links with ranking
        links = results.get("links", [])
        if args.verbose and links:
            print("=== Top Links ===")
            for i, link in enumerate(links, 1):
                title = link.get("title", "No title")
                url = link.get("url", "")
                score = link.get("score", link.get("relevance_score", 0))
                print(f"{i}. {title} (Relevance: {score:.2f})")
                print(f"   {url}")
            print()
        
        # Save results to file
        output_file = args.output or save_results_to_file(query, results)
        if args.verbose:
            print(f"Search results saved to: {os.path.abspath(output_file)}")
        
    except Exception as e:
        if args.verbose:
            print(f"An error occurred during search: {str(e)}")
        raise


if __name__ == "__main__":
    main()
