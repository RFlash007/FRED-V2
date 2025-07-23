import re
import time
import json
import datetime
from decimal import Decimal # Added import for Decimal
from duckduckgo_search import DDGS # Uncommented DuckDuckGo import
# Import the librarian module
import memory.L3_memory as L3
from config import config, ollama_manager
from ollie_print import olliePrint
import uuid
import concurrent.futures
import threading
import io
import sys
from trafilatura import fetch_url, extract
# Use centralized Ollama connection manager for all tool functions

# Explicitly export search functions for direct import
__all__ = [
    'tool_search_general',
    'tool_search_news',
    'tool_search_academic',
    'tool_search_forums',
    'tool_read_webpage',
    'tool_code_interpreter'
]

# Change logging level to ERROR to reduce console clutter (handled by olliePrint)

# --- Utility Functions ---
def parse_target_date(target_date: str | None) -> datetime.datetime | None:
    """Parse ISO format date string to datetime object."""
    if not target_date:
        return None
    
    try:
        # Try full datetime format first (YYYY-MM-DDTHH:MM:SS)
        return datetime.datetime.fromisoformat(target_date)
    except ValueError:
        try:
            # Try date-only format (YYYY-MM-DD)
            return datetime.datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid target_date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

def validate_memory_type(memory_type: str, field_name: str = "memory_type") -> None:
    """Validate memory type is one of the allowed values."""
    if memory_type not in ('Semantic', "Episodic", "Procedural"):
        raise ValueError(f"Invalid {field_name} '{memory_type}' provided to tool.")

def convert_datetime_for_json(obj: dict) -> dict:
    """Convert datetime and Decimal objects to JSON-compatible types."""
    for key, value in obj.items():
        if isinstance(value, datetime.datetime):
            obj[key] = value.isoformat()
        elif isinstance(value, Decimal):
            obj[key] = float(value)
    return obj

# --- Search Tool Implementations (moved early) ---

def tool_search_general(query: str) -> dict:
    """Specialized tool for general web searches with Brave-first and SearchAPI fallback."""
    is_valid, validation_message = validate_research_input(query)
    if not is_valid:
        return {"success": False, "error": f"Input validation failed: {validation_message}"}

    olliePrint(f"Executing general search for: '{query}'", show_banner=False)
    all_results = []

    # Step 1: Try Brave first
    brave_results = search_brave(query)
    if brave_results and 'web' in brave_results:
        all_results.extend(brave_results['web'])
        olliePrint("Brave search successful, using results.", level='info')
    else:
        olliePrint("Brave search failed or returned no results, falling back to SearchAPI.", level='warning')
        # Step 2: Fall back to SearchAPI if Brave fails
        searchapi_results = search_searchapi(query)
        if searchapi_results and 'web' in searchapi_results:
            all_results.extend(searchapi_results['web'])
            olliePrint("SearchAPI fallback successful.", level='info')
        else:
            olliePrint("SearchAPI fallback also failed.", level='error')

    # Deduplicate results
    unique_results = {result['url']: result for result in all_results}.values()
    return {"success": True, "results": list(unique_results)}

def tool_search_news(query: str) -> dict:
    """Specialized tool for news searches with Brave-first and SearchAPI fallback."""
    is_valid, validation_message = validate_research_input(query)
    if not is_valid:
        return {"success": False, "error": f"Input validation failed: {validation_message}"}

    olliePrint(f"Executing news search for: '{query}'", show_banner=False)
    all_results = []

    # Step 1: Try Brave first
    brave_results = search_brave(query)
    if brave_results and 'news' in brave_results:
        all_results.extend(brave_results['news'])
        olliePrint("Brave news search successful, using results.", level='info')
    else:
        olliePrint("Brave news search failed or returned no results, falling back to SearchAPI.", level='warning')
        # Step 2: Fall back to SearchAPI if Brave fails
        searchapi_results = search_news_apis(query)
        if searchapi_results and isinstance(searchapi_results, list):
            all_results.extend(searchapi_results)
            olliePrint("SearchAPI news fallback successful.", level='info')
        else:
            olliePrint("SearchAPI news fallback also failed.", level='error')

    # Deduplicate results
    unique_results = {result['url']: result for result in all_results}.values()
    return {"success": True, "results": list(unique_results)}

def tool_search_academic(query: str) -> dict:
    """Enhanced academic search with arXiv, Semantic Scholar, and PubMed Central."""
    is_valid, validation_message = validate_research_input(query)
    if not is_valid:
        return {"success": False, "error": f"Input validation failed: {validation_message}"}

    olliePrint(f"Executing academic search for: '{query}'", show_banner=False)
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_source = {
            executor.submit(search_arxiv, query): "arXiv",
            executor.submit(search_semantic_scholar, query): "Semantic Scholar",
            executor.submit(search_pubmed, query): "PubMed Central"
        }
        for future in concurrent.futures.as_completed(future_to_source):
            try:
                source_results = future.result()
                if source_results:
                    # Enhance Semantic Scholar results with Unpaywall
                    if future_to_source[future] == "Semantic Scholar":
                        source_results = [enhance_with_unpaywall(paper) for paper in source_results]
                    all_results.extend(source_results)
            except Exception as e:
                olliePrint(f"Academic search source {future_to_source[future]} failed: {e}", level='warning')

    unique_results = {result['url']: result for result in all_results}.values()
    return {"success": True, "results": list(unique_results)}

def tool_search_forums(query: str) -> dict:
    """Specialized tool for forum and community discussion searches."""
    is_valid, validation_message = validate_research_input(query)
    if not is_valid:
        return {"success": False, "error": f"Input validation failed: {validation_message}"}

    olliePrint(f"Executing forum search for: '{query}'", show_banner=False)
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_source = {
            executor.submit(search_reddit, query): "Reddit",
            executor.submit(search_stackoverflow, query): "Stack Overflow",
            executor.submit(search_youtube_transcripts, query): "YouTube"
        }
        for future in concurrent.futures.as_completed(future_to_source):
            try:
                source_results = future.result()
                if source_results:
                    all_results.extend(source_results)
            except Exception as e:
                olliePrint(f"Forum search source {future_to_source[future]} failed: {e}", level='warning')

    unique_results = {result['url']: result for result in all_results}.values()
    return {"success": True, "results": list(unique_results)}

# --- Tool Definitions ---
AVAILABLE_TOOLS = [
    {
        "name": "add_memory",
        "description": "Adds a new memory node to the knowledge graph. Use for new information, facts, events, or procedures.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "A concise label or title for the memory node."
                },
                "text": {
                    "type": "string",
                    "description": "The detailed text content of the memory."
                },
                "memory_type": {
                    "type": "string",
                    "description": "The type of memory.",
                    "enum": ["Semantic", "Episodic", "Procedural"]
                },
                "parent_id": {
                    "type": ["integer", "null"],
                    "description": "Optional. The ID of a parent node if this memory is hierarchically related."
                },
                "target_date": {
                    "type": ["string", "null"],
                    "description": "Optional. ISO format date (YYYY-MM-DD) or datetime (YYYY-MM-DDTHH:MM:SS) for future events or activities."
                }
            },
            "required": ["label", "text", "memory_type"]
        }
    },
    {
        "name": "supersede_memory",
        "description": "Replaces a specific old memory node with new, corrected information. Use ONLY after the user explicitly confirms an update to resolve a conflict you identified.",
        "parameters": {
            "type": "object",
            "properties": {
                "old_nodeid": {
                    "type": "integer",
                    "description": "The NodeID of the specific memory to replace"
                },
                "new_label": {
                    "type": "string",
                    "description": "A concise label/title for the new, replacing memory."
                },
                "new_text": {
                    "type": "string",
                    "description": "The full, corrected text content for the new memory."
                },
                "new_memory_type": {
                    "type": "string",
                    "description": "The classification ('Semantic', 'Episodic', 'Procedural') for the new memory content.",
                    "enum": ["Semantic", "Episodic", "Procedural"]
                },
                "target_date": {
                    "type": ["string", "null"],
                    "description": "Optional. ISO format date (YYYY-MM-DD) or datetime (YYYY-MM-DDTHH:MM:SS) for future events or activities."
                }
            },
            "required": ["old_nodeid", "new_label", "new_text", "new_memory_type"]
        }
    },
    {
        "name": "search_memory",
        "description": "Searches the knowledge graph for memories relevant to a query text using semantic similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The text to search for relevant memories."
                },
                "memory_type": {
                    "type": ["string", "null"],
                    "description": "Optional. Filter search results to a specific memory type ('Semantic', 'Episodic', 'Procedural').",
                    "enum": ["Semantic", "Episodic", "Procedural", None]
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional. The maximum number of search results to return. Defaults to 10.",
                    "default": 10
                },
                "future_events_only": {
                    "type": "boolean",
                    "description": "Optional. If true, only return memories with a target_date in the future.",
                    "default": False
                },
                "use_keyword_search": {
                    "type": "boolean",
                    "description": "Optional. If true, performs a keyword-based search instead of semantic. Defaults to false (semantic search).",
                    "default": False
                }
            },
            "required": ["query_text"]
        }
    },
    {
        "name": "search_web_information",
        "description": "Searches the web for information using DuckDuckGo. This tool retrieves current information from the internet. It combines results from general web search and news search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The text to search for."
                }
            },
            "required": ["query_text"]
        }
    },
    {
        "name": "get_node_by_id",
        "description": "Retrieves a specific memory node by its ID, along with its connections to other nodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "nodeid": {
                    "type": "integer",
                    "description": "The ID of the node to retrieve."
                }
            },
            "required": ["nodeid"]
        }
    },
    {
        "name": "get_graph_data",
        "description": "Retrieves a subgraph centered around a specific node, showing its connections up to a certain depth.",
        "parameters": {
            "type": "object",
            "properties": {
                "center_nodeid": {
                    "type": "integer",
                    "description": "The ID of the node to center the graph around."
                },
                "depth": {
                    "type": "integer",
                    "description": "Optional. How many levels of connections to retrieve. Defaults to 1.",
                    "default": 1
                }
            },
            "required": ["center_nodeid"]
        }
    },
    {
        "name": "enroll_person",
        "description": "Learns and remembers a new person's face. Use when the user introduces someone (e.g., 'This is Sarah', 'My name is Ian'). Requires an active camera feed from the Pi glasses.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the person to enroll."
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "update_knowledge_graph_edges",
        "description": "Processes pending edge creation tasks. Iteratively builds connections for recently added memories based on semantic similarity and LLM-based relationship determination.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit_per_run": {
                    "type": "integer",
                    "description": "Optional. The maximum number of pending memory nodes to process for edge creation in this run. Defaults to 5.",
                    "default": 5
                }
            },
            "required": []
        }
    },
    {
        "name": "read_webpage",
        "description": "Extract text from webpages or PDFs.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The complete URL of the webpage or PDF to read and extract content from."
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "search_general",
        "description": "General web search for broad topics, documentation, or official sources using search engines like Brave and DuckDuckGo.",
        "parameters": {
            "type": "object",
            "properties": { "query": { "type": "string", "description": "The search query."}},
            "required": ["query"]
        }
    },
    {
        "name": "search_news",
        "description": "Search for recent news articles and current events from news-specific sources.",
        "parameters": {
            "type": "object",
            "properties": { "query": { "type": "string", "description": "The search query for news."}},
            "required": ["query"]
        }
    },
    {
        "name": "search_academic",
        "description": "Search for academic papers, research articles, and scholarly publications from sources like arXiv and Semantic Scholar.",
        "parameters": {
            "type": "object",
            "properties": { "query": { "type": "string", "description": "The search query for academic content."}},
            "required": ["query"]
        }
    },
    {
        "name": "search_forums",
        "description": "Search community discussion platforms like Reddit and Stack Overflow for user-generated content and opinions.",
        "parameters": {
            "type": "object",
            "properties": { "query": { "type": "string", "description": "The search query for forum discussions."}},
            "required": ["query"]
        }
    }
]

# --- Tool Implementations ---

# Code interpreter tool
def tool_code_interpreter(code: str) -> dict:
    """Executes Python code and returns the output or errors."""
    olliePrint(f"Executing code interpreter tool", show_banner=False)
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, {})
        output = redirected_output.getvalue()
        return {"output": output}
    except Exception as e:
        return {"error": str(e)}
    finally:
        sys.stdout = old_stdout
# Wrapper for adding memory
def tool_add_memory(label: str, text: str, memory_type: str, parent_id: int | None = None, target_date: str | None = None):
    """Adds a new memory node to the knowledge graph."""
    try:
        olliePrint(f"Adding memory: '{label}' ({memory_type})", show_banner=False)
        validate_memory_type(memory_type)
        parsed_target_date = parse_target_date(target_date)
        
        node_id = L3.add_memory(label=label, text=text, memory_type=memory_type, parent_id=parent_id, target_date=parsed_target_date)
        return f"Memory '{label}' added with ID {node_id}"
    except Exception as e:
        olliePrint(f"Error adding memory: {e}", level='error')
        return f"Failed to add memory: {e}"

# Wrapper for superseding memory
def tool_supersede_memory(old_nodeid: int, new_label: str, new_text: str, new_memory_type: str, target_date: str | None = None):
    """Supersedes an existing memory with a new one."""
    try:
        olliePrint(f"Superseding memory {old_nodeid} with '{new_label}' ({new_memory_type})", show_banner=False)
        validate_memory_type(new_memory_type, "new_memory_type")
        parsed_target_date = parse_target_date(target_date)
        
        new_id = L3.supersede_memory(old_nodeid=old_nodeid, new_label=new_label, new_text=new_text, new_memory_type=new_memory_type, target_date=parsed_target_date)
        return f"Memory superseded. New ID: {new_id}"
    except ValueError as e:
        error_message = f"Cannot supersede memory {old_nodeid}: {e}"
        olliePrint(error_message, level='error')
        return error_message
    except Exception as e:
        olliePrint(f"Error superseding memory: {e}", level='error')
        return f"Failed to supersede memory: {e}"

# Wrapper for searching memory
def tool_search_memory(query_text: str, memory_type: str | None = None, limit: int = None, future_events_only: bool = False, use_keyword_search: bool = False):
    """Searches the knowledge graph for relevant memories."""
    try:
        olliePrint(f"Searching memories: '{query_text}' (type: {memory_type}, limit: {limit})", show_banner=False)
        # Validate memory_type if provided
        if memory_type is not None:
            validate_memory_type(memory_type, "memory_type filter")
        
        # Use config default if no limit specified
        search_limit = limit if limit is not None else config.MEMORY_SEARCH_LIMIT
        
        # Pass include_connections=True to get edge data in a single efficient query
        results = L3.search_memory(
            query_text=query_text, 
            memory_type=memory_type, 
            limit=search_limit, 
            future_events_only=future_events_only, 
            use_keyword_search=use_keyword_search,
            include_connections=True  # Always include connections in the tool interface
        )
        
        # Convert datetime and Decimal objects to JSON-compatible types
        for result in results:
            convert_datetime_for_json(result)
            
        return {"success": True, "results": results}
    except Exception as e:
        olliePrint(f"Error searching memory: {e}", level='error')
        return {"success": False, "error": str(e)}

# Wrapper for searching L2 memory
def tool_search_l2_memory(query_text: str, limit: int = 5):
    """Searches L2 Episodic Cache for recent conversation summaries."""
    try:
        olliePrint(f"Searching L2 memory: '{query_text}' (limit: {limit})", show_banner=False)
        import memory.L2_memory as L2
        
        # Use L2's existing search functionality
        results = L2.search_l2_memory(query_text, limit)
        
        if results:
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "topic": result.get('topic', ''),
                    "summary": result.get('raw_text_summary', ''),
                    "turns": f"{result.get('turn_start', 0)}-{result.get('turn_end', 0)}",
                    "sentiment": result.get('user_sentiment', 'neutral'),
                    "entities": result.get('entities_mentioned', [])
                })
            return {"success": True, "results": formatted_results}
        else:
            return {"success": True, "results": []}
    except Exception as e:
        olliePrint(f"Error searching L2 memory: {e}", level='error')
        return {"success": False, "error": str(e)}

# Research System Error Handling and Rate Limiting
# ================================================

class ResearchRateLimiter:
    """Simple rate limiter for research APIs."""
    
    def __init__(self):
        self.request_counts = {}
        self.last_reset = {}
    
    def can_make_request(self, api_name: str, limit: int, window_minutes: int = 60) -> bool:
        """Check if we can make a request to the given API."""
        import time
        
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        # Reset counter if window has passed
        if api_name not in self.last_reset or (current_time - self.last_reset[api_name]) > window_seconds:
            self.request_counts[api_name] = 0
            self.last_reset[api_name] = current_time
        
        # Check if under limit
        return self.request_counts.get(api_name, 0) < limit
    
    def record_request(self, api_name: str):
        """Record that a request was made to the API."""
        self.request_counts[api_name] = self.request_counts.get(api_name, 0) + 1

# Global rate limiter instance
research_rate_limiter = ResearchRateLimiter()

def safe_api_call(api_name: str, api_function, *args, **kwargs):
    """Safely execute an API call with error handling and rate limiting."""
    try:
        # Check rate limits (basic implementation)
        rate_limits = {
            'BraveSearch': (config.BRAVE_SEARCH_REQUEST_LIMIT, 43200),  # Monthly limit
            'SearchAPI': (config.SEARCHAPI_REQUEST_LIMIT, 43200),      # Monthly limit
            'YouTube': (config.YOUTUBE_API_QUOTA_LIMIT, 1440),         # Daily limit
            'Reddit': (config.REDDIT_REQUEST_LIMIT, 1),                # Per minute
            'StackOverflow': (config.STACKOVERFLOW_REQUEST_LIMIT, 1440),  # Daily
            'NewsAPI': (config.NEWS_API_REQUEST_LIMIT, 1440),          # Daily
            'Jina': (config.JINA_REQUEST_LIMIT, 43200),                # Monthly (30 days)
        }
        
        if api_name in rate_limits:
            limit, window_minutes = rate_limits[api_name]
            if not research_rate_limiter.can_make_request(api_name, limit, window_minutes):
                olliePrint(f"⚠️ Rate limit reached for {api_name}, skipping...", level='warning')
                return None
        
        # Execute the API function
        result = api_function(*args, **kwargs)
        
        # Record successful request
        if api_name in rate_limits:
            research_rate_limiter.record_request(api_name)
        
        return result
        
    except Exception as e:
        olliePrint(f"❌ Safe API call failed for {api_name}: {e}", level='error')
        return None

def validate_research_input(query_or_url: str) -> tuple[bool, str]:
    """Validate research input and return validation status."""
    if not query_or_url or not query_or_url.strip():
        return False, "Empty query or URL provided"
    
    # Check for extremely long inputs
    if len(query_or_url) > 500:
        return False, "Query or URL too long (max 500 characters)"
    
    # Check for potentially malicious URLs
    if query_or_url.startswith(('http://', 'https://')):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(query_or_url)
            if not parsed.netloc:
                return False, "Invalid URL format"
        except Exception:
            return False, "Invalid URL format"
    
    return True, "Valid input"

def create_error_summary(failed_sources: list, phase_1_sources: list, phase_2_sources: list) -> dict:
    """Create a comprehensive error summary for the research results."""
    total_attempted = len(failed_sources) + len(phase_1_sources) + len(phase_2_sources)
    success_rate = (len(phase_1_sources) + len(phase_2_sources)) / max(total_attempted, 1) * 100
    
    return {
        "total_sources_attempted": total_attempted,
        "successful_sources": len(phase_1_sources) + len(phase_2_sources),
        "failed_sources_count": len(failed_sources),
        "success_rate_percent": round(success_rate, 1),
        "phase_1_success": len(phase_1_sources),
        "phase_2_success": len(phase_2_sources),
        "critical_sources_available": len(phase_1_sources) > 0,  # At least core sources worked
        "research_quality": "High" if success_rate >= 80 else "Medium" if success_rate >= 50 else "Basic"
    }

def enhance_research_results_with_metadata(results: dict) -> dict:
    """Enhance research results with additional metadata and quality indicators."""
    try:
        # Add error analysis
        error_summary = create_error_summary(
            results["execution_summary"]["failed_sources"],
            results["execution_summary"]["phase_1_sources"], 
            results["execution_summary"]["phase_2_sources"]
        )
        results["execution_summary"]["error_analysis"] = error_summary
        
        # Add content quality metrics
        content_length = len(results.get("content", ""))
        total_links = results["execution_summary"]["total_links_found"]
        
        results["execution_summary"]["content_metrics"] = {
            "content_length_chars": content_length,
            "has_substantial_content": content_length > 1000,
            "link_diversity": len([cat for cat, links in results["suggested_links"].items() if links]),
            "total_categories_with_links": len([cat for cat, links in results["suggested_links"].items() if links]),
            "academic_sources_found": len(results["suggested_links"].get("academic_papers", [])),
            "social_sources_found": len(results["suggested_links"].get("forums", [])) + len(results["suggested_links"].get("social_media", []))
        }
        
        # Add research recommendations
        recommendations = []
        if error_summary["success_rate_percent"] < 50:
            recommendations.append("Consider checking API configurations for failed sources")
        if total_links < 10:
            recommendations.append("Query might be too specific - consider broader terms")
        if len(results["suggested_links"].get("academic_papers", [])) == 0:
            recommendations.append("No academic sources found - try more scholarly terms")
        if error_summary["phase_1_success"] == 0:
            recommendations.append("Core research sources failed - check network connectivity")
        
        results["execution_summary"]["recommendations"] = recommendations
        
        return results
        
    except Exception as e:
        olliePrint(f"Failed to enhance research results: {e}", level='warning')
        return results

# Update the main search function to include enhanced error handling
def search_web_information(query_or_url: str) -> dict:
    """Enhanced version of search_web_information with comprehensive error handling."""
    try:
        # Enhanced logging
        olliePrint(f"\n[WEB SEARCH] Starting search for: '{query_or_url}'", show_banner=False)
        olliePrint(f"[WEB SEARCH] Query type: {'URL' if query_or_url.startswith(('http://', 'https://')) else 'Query'}", show_banner=False)
        olliePrint(f"[WEB SEARCH] Query length: {len(query_or_url)} characters", show_banner=False)
        
        # Input validation
        is_valid, validation_message = validate_research_input(query_or_url)
        if not is_valid:
            error_result = {
                "success": False, 
                "error": f"Input validation failed: {validation_message}"
            }
            olliePrint(f"[WEB SEARCH] ❌ Validation failed: {validation_message}", level='error')
            return error_result
        
        # Execute main search
        olliePrint(f"[WEB SEARCH] ✅ Input valid, executing comprehensive search...", show_banner=False)
        result = execute_comprehensive_search(query_or_url)
        
        # Enhanced logging of results
        if result.get("success"):
            olliePrint(f"[WEB SEARCH] ✅ Search successful", show_banner=False)
            if "results" in result:
                content_length = len(result["results"].get("content", ""))
                links_count = sum(len(links) for links in result["results"].get("suggested_links", {}).values())
                olliePrint(f"[WEB SEARCH] Content length: {content_length} chars, Links found: {links_count}", show_banner=False)
                
                # Enhance successful results
                result["results"] = enhance_research_results_with_metadata(result["results"])
        else:
            olliePrint(f"[WEB SEARCH] ❌ Search failed: {result.get('error', 'Unknown error')}", level='error')
        
        return result
        
    except Exception as e:
        error_msg = f"Research system error: {str(e)}"
        olliePrint(f"[WEB SEARCH] ❌ Exception occurred: {e}", level='error')
        olliePrint(f"❌ Enhanced research system error: {e}", level='error')
        return {
            "success": False, 
            "error": error_msg,
            "fallback_suggestion": "Try a simpler query or check system configuration"
        }

def execute_comprehensive_search(query_or_url: str) -> dict:
    """
    Internal comprehensive search function with hybrid execution engine.
    
    Supports both search queries and direct URL analysis. Uses a two-phase approach:
    - Phase 1: Parallel execution of core research sources (web, academic, news)
    - Phase 2: Sequential execution of social sources (forums, social media, videos)
    
    Args:
        query_or_url (str): Search query or URL to analyze
        
    Returns:
        dict: Comprehensive research results with categorized content and links
    """
    try:
        olliePrint(f"🔬 Advanced research initiated: '{query_or_url[:80]}...'", show_banner=False)
        
        # Determine input type
        is_url = query_or_url.startswith(('http://', 'https://'))
        
        # Initialize result structure
        research_results = {
            "input_type": "url" if is_url else "query",
            "input_value": query_or_url,
            "content": "",
            "suggested_links": {
                "academic_papers": [],
                "related_articles": [],
                "official_sources": [],
                "news_articles": [],
                "documentation": [],
                "forums": [],
                "social_media": [],
                "video_transcripts": []
            },
            "execution_summary": {
                "phase_1_sources": [],
                "phase_2_sources": [],
                "failed_sources": [],
                "total_links_found": 0,
                "execution_time": 0
            }
        }
        
        start_time = time.time()
        
        if is_url:
            # Direct URL analysis
            research_results = execute_url_analysis(query_or_url, research_results)
        else:
            # Comprehensive search query processing
            research_results = execute_hybrid_search(query_or_url, research_results)
        
        # Calculate execution time
        research_results["execution_summary"]["execution_time"] = round(time.time() - start_time, 2)
        
        # Count total links found
        total_links = sum(len(links) for links in research_results["suggested_links"].values())
        research_results["execution_summary"]["total_links_found"] = total_links
        
        olliePrint(f"✅ Research completed: {total_links} links found in {research_results['execution_summary']['execution_time']}s", show_banner=False)
        
        return {"success": True, "results": research_results}
        
    except Exception as e:
        olliePrint(f"❌ Research system error: {e}", level='error')
        return {"success": False, "error": str(e)}

def execute_hybrid_search(query: str, results: dict) -> dict:
    """Execute hybrid search with Phase 1 (parallel core) and Phase 2 (sequential social)."""
    import concurrent.futures
    import threading
    
    # Phase 1: Parallel Core Sources
    olliePrint("📡 Phase 1: Executing core research sources...", show_banner=False)
    
    if config.RESEARCH_ENABLE_PARALLEL_CORE:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit core research tasks
            future_to_source = {
                executor.submit(safe_api_call, 'BraveSearch', search_brave, query): "Brave",
                executor.submit(search_duckduckgo, query): "DuckDuckGo",
                executor.submit(safe_api_call, 'SearchAPI', search_searchapi, query): "SearchAPI",
                executor.submit(search_arxiv, query): "arXiv",
                executor.submit(search_semantic_scholar, query): "Semantic Scholar",
                executor.submit(search_news_apis, query): "News APIs"
            }
            
            # Collect results
            try:
                for future in concurrent.futures.as_completed(future_to_source, timeout=config.RESEARCH_CORE_TIMEOUT):
                    source_name = future_to_source[future]
                    try:
                        source_results = future.result()
                        if source_results:
                            merge_source_results(results, source_results, source_name)
                            results["execution_summary"]["phase_1_sources"].append(source_name)
                    except Exception as e:
                        olliePrint(f"⚠️ {source_name} failed: {e}", level='warning')
                        results["execution_summary"]["failed_sources"].append(f"{source_name}: {str(e)}")
            except concurrent.futures.TimeoutError:
                olliePrint(f"❌ Phase 1: Research timed out after {config.RESEARCH_CORE_TIMEOUT} seconds. Some sources may not have completed.", level='error')
                # Add any remaining futures to failed sources
                for future in future_to_source:
                    if not future.done():
                        results["execution_summary"]["failed_sources"].append(f"{future_to_source[future]}: Timed out")
    
    # Phase 2: Sequential Social Sources
    olliePrint("💬 Phase 2: Executing social & supplementary sources...", show_banner=False)
    
    if config.RESEARCH_ENABLE_SOCIAL_SOURCES:
        social_sources = [
            ("Reddit", search_reddit),
            ("Stack Overflow", search_stackoverflow),
        ]
        
        if config.RESEARCH_ENABLE_VIDEO_TRANSCRIPTS:
            social_sources.append(("YouTube", search_youtube_transcripts))
        
        for source_name, search_func in social_sources:
            try:
                time.sleep(config.RESEARCH_REQUEST_DELAY)  # Rate limiting
                source_results = search_func(query)
                if source_results:
                    merge_source_results(results, source_results, source_name)
                    results["execution_summary"]["phase_2_sources"].append(source_name)
            except Exception as e:
                olliePrint(f"⚠️ {source_name} failed: {e}", level='warning')
                results["execution_summary"]["failed_sources"].append(f"{source_name}: {str(e)}")
    
    return results

def execute_url_analysis(url: str, results: dict) -> dict:
    """Analyze a specific URL and extract content + related links."""
    try:
        # Extract content from URL
        content = extract_webpage_content(url)
        if content:
            results["content"] = content["text"]
            
            # Extract and categorize links from the webpage
            if "links" in content:
                categorized_links = categorize_extracted_links(content["links"], url)
                for category, links in categorized_links.items():
                    if category in results["suggested_links"]:
                        results["suggested_links"][category].extend(links[:config.RESEARCH_MAX_WEB_ARTICLES])
        
        results["execution_summary"]["phase_1_sources"].append("Direct URL Analysis")
        
    except Exception as e:
        olliePrint(f"❌ URL analysis failed: {e}", level='error')
        results["execution_summary"]["failed_sources"].append(f"URL Analysis: {str(e)}")
    
    return results

def merge_source_results(main_results: dict, source_results: dict, source_name: str):
    """Merge results from a source into the main results structure."""
    # Merge content
    if source_results.get("content"):
        if main_results["content"]:
            main_results["content"] += f"\n\n--- {source_name} ---\n{source_results['content']}"
        else:
            main_results["content"] = f"--- {source_name} ---\n{source_results['content']}"
    
    # Merge categorized links
    for category, links in source_results.get("links", {}).items():
        if category in main_results["suggested_links"] and links:
            # Apply limits per category
            current_count = len(main_results["suggested_links"][category])
            limit_map = {
                "academic_papers": config.RESEARCH_MAX_ACADEMIC_PAPERS,
                "related_articles": config.RESEARCH_MAX_WEB_ARTICLES,
                "news_articles": config.RESEARCH_MAX_NEWS_ARTICLES,
                "forums": config.RESEARCH_MAX_FORUM_POSTS,
                "social_media": config.RESEARCH_MAX_SOCIAL_POSTS,
                "video_transcripts": config.RESEARCH_MAX_VIDEO_TRANSCRIPTS,
                "documentation": config.RESEARCH_MAX_DOCUMENTATION,
                "official_sources": config.RESEARCH_MAX_WEB_ARTICLES
            }
            
            limit = limit_map.get(category, 10)
            remaining_slots = max(0, limit - current_count)
            
            if remaining_slots > 0:
                main_results["suggested_links"][category].extend(links[:remaining_slots])

# New tool to get a node by ID with its connections
def tool_get_node_by_id(nodeid: int):
    """Gets a specific memory node by its ID."""
    try:
        olliePrint(f"Getting memory node: {nodeid}", show_banner=False)
        # Use get_graph_data with depth=1 to get the node and its immediate connections
        graph_data = L3.get_graph_data(nodeid, depth=1)
        
        # Find the target node in the nodes list
        target_node = next((n for n in graph_data.get('nodes', []) if n['id'] == nodeid), None)
        
        if not target_node:
            return {"success": False, "error": f"Node with ID {nodeid} not found or is superseded."}
        
        # Process datetime objects for JSON compatibility
        convert_datetime_for_json(target_node)
        
        # Extract connections specific to this node
        connections = []
        for edge in graph_data.get('edges', []):
            if edge['source'] == nodeid:
                # Outgoing connection
                target_node_info = next((n for n in graph_data.get('nodes', []) if n['id'] == edge['target']), None)
                if target_node_info:
                    connections.append({
                        "direction": "outgoing",
                        "rel_type": edge['rel_type'],
                        "target_nodeid": edge['target'],
                        "target_label": target_node_info.get('label', ''),
                        "target_type": target_node_info.get('type', '')
                    })
            elif edge['target'] == nodeid:
                # Incoming connection
                source_node_info = next((n for n in graph_data.get('nodes', []) if n['id'] == edge['source']), None)
                if source_node_info:
                    connections.append({
                        "direction": "incoming",
                        "rel_type": edge['rel_type'],
                        "source_nodeid": edge['source'],
                        "source_label": source_node_info.get('label', ''),
                        "source_type": source_node_info.get('type', '')
                    })
        
        # Add connections to the result
        result = {
            "node": target_node,
            "connections": connections
        }
        
        return {"success": True, "result": result}
    except Exception as e:
        olliePrint(f"Error getting node: {e}", level='error')
        return {"success": False, "error": str(e)}

# New tool to get graph data
def tool_get_graph_data(center_nodeid: int, depth: int = 1):
    """Gets graph data for visualization."""
    try:
        olliePrint(f"Getting graph data (center: {center_nodeid}, depth: {depth})", show_banner=False)
        graph_data = L3.get_graph_data(center_nodeid, depth=depth)
        
        # Process datetime objects for JSON compatibility
        for node in graph_data.get('nodes', []):
            convert_datetime_for_json(node)
        
        return {"success": True, "result": graph_data}
    except Exception as e:
        olliePrint(f"Error getting graph data: {e}", level='error')
        return {"nodes": [], "links": []}

def tool_get_subgraph(center_node_id: int, depth: int = 2, relationship_types: list = None, 
                     max_nodes: int = 100, include_metadata: bool = True):
    """MCP-inspired subgraph retrieval with advanced traversal and relationship analysis."""
    try:
        olliePrint(f"Getting subgraph (center: {center_node_id}, depth: {depth}, max_nodes: {max_nodes})", show_banner=False)
        
        subgraph_data = L3.get_subgraph(
            center_node_id=center_node_id,
            depth=depth,
            relationship_types=relationship_types,
            max_nodes=max_nodes,
            include_metadata=include_metadata
        )
        
        # Process datetime objects for JSON compatibility
        for node in subgraph_data.get('nodes', []):
            convert_datetime_for_json(node)
        for edge in subgraph_data.get('edges', []):
            convert_datetime_for_json(edge)
        
        return {"success": True, "result": subgraph_data}
    except Exception as e:
        olliePrint(f"Error getting subgraph: {e}", level='error')
        return {"success": False, "error": str(e)}

def tool_add_memory_with_observations(label: str, text: str, memory_type: str, 
                                     observations: list = None, parent_id: int = None,
                                     target_date: str = None, metadata: dict = None):
    """MCP-inspired memory addition with structured observations and metadata support."""
    try:
        olliePrint(f"Adding memory with observations: '{label}' ({memory_type})", show_banner=False)
        validate_memory_type(memory_type)
        parsed_target_date = parse_target_date(target_date)
        
        result = L3.add_memory_with_observations(
            label=label,
            text=text,
            memory_type=memory_type,
            observations=observations,
            parent_id=parent_id,
            target_date=parsed_target_date,
            metadata=metadata
        )
        
        return {
            "success": result.get('success', False),
            "node_id": result.get('nodeid'),
            "observations_added": result.get('observations_added', 0),
            "metadata_keys": result.get('metadata_keys', []),
            "message": f"Memory '{label}' added with ID {result.get('nodeid')} ({result.get('observations_added', 0)} observations, {len(result.get('metadata_keys', []))} metadata keys)"
        }
    except Exception as e:
        olliePrint(f"Error adding memory with observations: {e}", level='error')
        return {"success": False, "error": str(e)}

def tool_discover_relationships_advanced(node_id: int, context_window: int = 5, 
                                        min_confidence: float = 0.7):
    """MCP-inspired enhanced relationship discovery with context-aware analysis and confidence scoring."""
    try:
        olliePrint(f"Discovering advanced relationships for node {node_id} (context: {context_window}, min_confidence: {min_confidence})", show_banner=False)
        
        relationships = L3.discover_relationships_advanced(
            node_id=node_id,
            context_window=context_window,
            min_confidence=min_confidence
        )
        
        # Process datetime objects for JSON compatibility in relationship data
        for rel in relationships:
            if 'context_nodes' in rel and isinstance(rel['context_nodes'], list):
                rel['context_nodes'] = [int(nid) for nid in rel['context_nodes']]  # Ensure node IDs are integers
        
        return {
            "success": True,
            "relationships_found": len(relationships),
            "relationships": relationships,
            "message": f"Found {len(relationships)} potential relationships for node {node_id} above confidence threshold {min_confidence}"
        }
    except Exception as e:
        olliePrint(f"Error discovering advanced relationships: {e}", level='error')
        return {"success": False, "error": str(e)}

def tool_enroll_person(name: str):
    """Enrolls a new person by capturing their face from the live camera feed."""
    try:
        olliePrint(f"Received request to enroll new person: {name}")
        # Local imports to prevent circular dependencies and access services
        import asyncio
        from vision_service import vision_service
        from persona_service import persona_service
        # This local import is critical to avoid a circular dependency loop at startup
        from webrtc_server import request_frame_from_client

        if not vision_service.pi_connected:
            return "Enrollment failed: No camera client is connected."

        async def enroll_async_wrapper():
            # Request fresh capture from Pi for enrollment
            from webrtc_server import send_capture_request_to_pi
            
            olliePrint(f"Requesting fresh image capture for enrollment of '{name}'")
            
            # For enrollment, we need to handle this differently since we need immediate response
            # This is a simplified approach - in production you might want a more sophisticated callback system
            success = await send_capture_request_to_pi()
            
            if success:
                # Wait for image to be processed by vision service and use current frame
                await asyncio.sleep(3)  # Give time for capture and processing
                
                # Get the latest processed image data from vision service
                if hasattr(vision_service, 'last_processed_image_array'):
                    frame_array = vision_service.last_processed_image_array
                    result = persona_service.enroll_person(frame_array, name)
                    return result
                else:
                    return "Enrollment failed: Could not get processed image from vision service."
            else:
                return "Enrollment failed: Could not request image capture from Pi."
        
        # Run the async enrollment function in a new event loop
        return asyncio.run(enroll_async_wrapper())

    except Exception as e:
        olliePrint(f"Error during enrollment tool execution: {e}", level='error')
        return f"An unexpected error occurred during enrollment: {str(e)}"

def tool_update_knowledge_graph_edges(limit_per_run: int = 3):
    """Updates edges in the knowledge graph."""
    try:
        olliePrint(f"Updating graph edges (limit: {limit_per_run})", show_banner=False)
        if not isinstance(limit_per_run, int) or limit_per_run <= 0:
            raise ValueError("limit_per_run must be a positive integer.")
        
        summary = L3.process_pending_edges(limit_per_run=limit_per_run)
        # Ensure all parts of the summary are JSON serializable, though they should be.
        # datetime objects are handled by librarian if any were returned directly.
        return {"success": True, "result": summary}
    except Exception as e:
        olliePrint(f"Error updating edges: {e}", level='error')
        return "Failed to update edges"

def tool_add_task_to_agenda(task_description: str, priority: int = 2):
    """Add a research task to the agenda for future processing."""
    try:
        import memory.agenda_system as agenda
        
        olliePrint(f"Adding task to agenda (priority {priority}): {task_description[:100]}...", show_banner=False)
        
        task_id = agenda.add_task_to_agenda(task_description, priority)
        if task_id:
            return {"success": True, "task_id": task_id, "message": f"Task added to agenda with ID: {task_id}"}
        else:
            return {"success": False, "error": "Failed to add task to agenda"}
        
    except Exception as e:
        olliePrint(f"Error adding task to agenda: {e}", level='error')
        return {"success": False, "error": str(e)}

def tool_trigger_sleep_cycle():
    """Trigger the sleep cycle to process agenda tasks and consolidate memories."""
    try:
        import memory.agenda_system as agenda
        import memory.L2_memory as L2
        from datetime import datetime
        
        olliePrint("Initiating sleep cycle...", show_banner=False)
        
        sleep_summary = {
            "start_time": datetime.now().isoformat(),
            "agenda_tasks_processed": 0,
            "agenda_tasks_completed": 0,
            "agenda_tasks_failed": 0,
            "l2_consolidations": 0,
            "edge_processing": 0,
            "notifications_created": 0
        }
        
        # Step 1: Process Agenda Tasks
        olliePrint("Processing agenda tasks...")
        pending_tasks = agenda.get_pending_agenda_tasks()
        sleep_summary["agenda_tasks_processed"] = len(pending_tasks)
        
        for task in pending_tasks:
            try:
                success = agenda.process_agenda_task(task)
                if success:
                    sleep_summary["agenda_tasks_completed"] += 1
                else:
                    sleep_summary["agenda_tasks_failed"] += 1
            except Exception as e:
                olliePrint(f"Failed to process agenda task {task.get('task_id', 'unknown')}: {e}", level='error')
                sleep_summary["agenda_tasks_failed"] += 1
        
        # Step 2: L2 to L3 Consolidation
        olliePrint("Consolidating L2 memories to L3...")
        l2_candidates = L2.get_l2_consolidation_candidates()
        
        consolidated_ids = []
        for candidate in l2_candidates:
            try:
                # Create L3 episodic memory from L2 summary
                result = L3.add_memory(
                    label=f"Consolidated: {candidate['topic']}",
                    text=f"Topic: {candidate['topic']}\n\nSummary: {candidate['summary']}\n\nKey Outcomes: {', '.join(candidate.get('key_outcomes', []))}\n\nEntities: {', '.join(candidate.get('entities_mentioned', []))}\n\nOriginal Turns: {candidate['turn_start']}-{candidate['turn_end']}",
                    memory_type="Episodic"
                )
                
                if result:
                    consolidated_ids.append(candidate['l2_id'])
                    sleep_summary["l2_consolidations"] += 1
                    olliePrint(f"Consolidated L2 {candidate['l2_id']} to L3 node {result}")
                
            except Exception as e:
                olliePrint(f"Failed to consolidate L2 candidate {candidate['l2_id']}: {e}", level='error')
        
        # Remove consolidated L2 entries
        if consolidated_ids:
            L2.mark_l2_consolidated(consolidated_ids)
        
        # Step 3: Process pending graph edges
        olliePrint("Processing pending knowledge graph edges...")
        try:
            edge_summary = L3.process_pending_edges(config.SLEEP_CYCLE_L2_CONSOLIDATION_BATCH)
            if isinstance(edge_summary, dict):
                sleep_summary["edge_processing"] = edge_summary.get("edges_created", 0)
        except Exception as e:
            olliePrint(f"Edge processing failed: {e}", level='error')
        
        # Step 4: Check for new notifications
        notifications = agenda.get_pending_notifications()
        sleep_summary["notifications_created"] = len(notifications)
        
        sleep_summary["end_time"] = datetime.now().isoformat()
        
        # Create summary message
        summary_msg = f"""Sleep cycle completed:
- Processed {sleep_summary['agenda_tasks_processed']} agenda tasks ({sleep_summary['agenda_tasks_completed']} completed, {sleep_summary['agenda_tasks_failed']} failed)
- Consolidated {sleep_summary['l2_consolidations']} L2 memories to L3
- Created {sleep_summary['edge_processing']} new knowledge connections
- {sleep_summary['notifications_created']} new notifications ready"""
        
        olliePrint("Sleep cycle completed successfully")
        
        return {
            "success": True, 
            "summary": summary_msg,
            "details": sleep_summary,
            "notifications": [notif['message'] for notif in notifications[:3]]  # Preview first 3
        }
        
    except Exception as e:
        olliePrint(f"Error during sleep cycle: {e}", level='error')
        return {"success": False, "error": str(e)}
    
def gist_summarize_source(source: str) -> str:
    """Summarize the content of a source using Ollama with G.I.S.T. model."""
    try:
        messages = [
            {"role": "system", "content": config.GIST_SYSTEM_PROMPT},
            {"role": "user", "content": config.GIST_USER_PROMPT.format(source=source)}
        ]
        
        response = ollama_manager.chat_concurrent_safe(
            model=config.GIST_SUMMARY_MODEL,
            messages=messages,
            stream=False
        )
        
        response_content = response.get('message', {}).get('content', '')
        return re.sub(r'<think>.*?</think>', '', response_content.strip(), flags=re.DOTALL)
    except Exception as e:
        olliePrint(f"G.I.S.T. summarization failed: {e}", level='error')
        return f"Error summarizing source: {e}"

def tool_read_webpage(url: str) -> dict:
    """Extract text from webpages or PDFs."""
    error_prefix = f"Failed to read URL '{url}'. **MOVE ON TO A DIFFERENT LINK**."
    if url.lower().endswith('.pdf'):
        try:
            import pdfplumber
            import requests
            response = requests.get(url)
            response.raise_for_status()
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            if not text:
                return {"success": False, "error": f"{error_prefix} Reason: PDF contained no extractable text."}
            return {"success": True, "url": url, "content": gist_summarize_source(text), "links_found": len(text.split())}
        except Exception as e:
            return {"success": False, "error": f"{error_prefix} Reason: PDF extraction failed. Details: {e}"}
    else:
        try:
            downloaded = fetch_url(url)
            if not downloaded:
                return {"success": False, "error": f"{error_prefix} Reason: Failed to download webpage content."}
            content = extract(downloaded, include_links=False)
            if not content:
                return {"success": False, "error": f"{error_prefix} Reason: No main content could be extracted from the webpage."}
            return {"success": True, "url": url, "content": gist_summarize_source(content), "links_found": len(content.split())}
        except Exception as e:
            return {"success": False, "error": f"{error_prefix} Reason: General error during webpage processing. Details: {e}"}

# --- Tool Registry (finalized) ---
TOOL_FUNCTIONS = {
    "add_memory": tool_add_memory,
    "supersede_memory": tool_supersede_memory,
    "search_memory": tool_search_memory,
    "search_l2_memory": tool_search_l2_memory,
    # High-level search helpers
    "search_general": tool_search_general,
    "search_news": tool_search_news,
    "search_academic": tool_search_academic,
    "search_forums": tool_search_forums,
    # Graph & memory helpers
    "get_node_by_id": tool_get_node_by_id,
    "get_graph_data": tool_get_graph_data,
    "get_subgraph": tool_get_subgraph,
    "add_memory_with_observations": tool_add_memory_with_observations,
    "discover_relationships_advanced": tool_discover_relationships_advanced,
    "enroll_person": tool_enroll_person,
    "update_knowledge_graph_edges": tool_update_knowledge_graph_edges,
    "addTaskToAgenda": tool_add_task_to_agenda,
    "triggerSleepCycle": tool_trigger_sleep_cycle,
    "read_webpage": tool_read_webpage,
    "code_interpreter": tool_code_interpreter,
}

# --- Tool Execution Logic ---
# Simplified execution logic, assuming args are already parsed dicts by handle_tool_calls
def handle_tool_calls(tool_calls):
    """Handles multiple tool calls and returns results."""
    if not tool_calls:
        return []
    
    results = []
    for tool_call in tool_calls:
        tool_call_id = tool_call.get('id', '')
        function_name = tool_call.get('function', {}).get('name', '')
        tool_args = tool_call.get('function', {}).get('arguments', {})
        
        if not function_name:
            olliePrint("Tool call missing function name", level='error')
            continue
            
        if not tool_call_id:
            olliePrint(f"Tool '{function_name}' missing ID - generating one", level='warning')
            tool_call_id = str(uuid.uuid4())
        
        olliePrint(f"Executing tool: {function_name}", show_banner=False)
        
        try:
            # Convert arguments if they're a string
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    olliePrint(f"Invalid JSON arguments for '{function_name}': {tool_args}", level='error')
                    continue
            
            # Execute the tool function
            if function_name in TOOL_FUNCTIONS:
                function = TOOL_FUNCTIONS[function_name]
                content = function(**tool_args)
                # Convert content to string if it's not already
                if not isinstance(content, str):
                    content = json.dumps(content, indent=2)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content
                })
            else:
                olliePrint(f"Tool '{function_name}' not found", level='error')
                
        except TypeError as e:
            olliePrint(f"TypeError executing {function_name}: {e}", level='error')
        except Exception as e:
            olliePrint(f"Error executing {function_name}: {e}", level='error')
    
    return results

# Individual API Integration Functions
# ===================================

def search_brave(query: str) -> dict:
    """Search Brave Search API and return structured results for web and news."""
    try:
        import requests
        
        if not config.BRAVE_SEARCH_API_KEY:
            olliePrint(f"[BRAVE] No API key configured, skipping", show_banner=False)
            return None
        
        olliePrint(f"[BRAVE] Starting search for: '{query}'", show_banner=False)
        
        web_results = []
        news_results = []
        seen_urls = set()
        
        headers = {
            'X-Subscription-Token': config.BRAVE_SEARCH_API_KEY,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip'
        }
        
        # Web search
        web_params = {
            'q': query,
            'count': config.RESEARCH_MAX_WEB_ARTICLES,
            'country': 'us',
            'search_lang': 'en',
            'safesearch': 'off'
        }
        
        try:
            web_response = requests.get(
                config.BRAVE_SEARCH_API_URL,
                headers=headers,
                params=web_params,
                timeout=config.RESEARCH_CORE_TIMEOUT
            )
            web_response.raise_for_status()
            web_data = web_response.json()
            
            if 'web' in web_data and 'results' in web_data['web']:
                for result in web_data['web']['results']:
                    title = result.get('title', 'No Title')
                    snippet = result.get('description', '')
                    url = result.get('url', '')
                    if url and url not in seen_urls:
                        web_results.append({"title": title, "snippet": snippet, "url": url, "source": "Brave Web"})
                        seen_urls.add(url)
            olliePrint(f"[BRAVE] Web search completed: {len(web_results)} results", show_banner=False)
            
        except Exception as e:
            olliePrint(f"[BRAVE] Web search failed: {e}", level='warning')
        
        # News search
        news_params = {
            'q': query,
            'count': config.RESEARCH_MAX_NEWS_ARTICLES,
            'country': 'us',
            'search_lang': 'en'
        }
        
        try:
            news_response = requests.get(
                config.BRAVE_NEWS_API_URL,
                headers=headers,
                params=news_params,
                timeout=config.RESEARCH_CORE_TIMEOUT
            )
            news_response.raise_for_status()
            news_data = news_response.json()
            
            if 'results' in news_data:
                for result in news_data['results']:
                    title = result.get('title', 'No Title')
                    snippet = result.get('description', '')
                    url = result.get('url', '')
                    if url and url not in seen_urls:
                        news_results.append({"title": title, "snippet": snippet, "url": url, "source": "Brave News"})
                        seen_urls.add(url)
            olliePrint(f"[BRAVE] News search completed: {len(news_results)} results", show_banner=False)
            
        except Exception as e:
            olliePrint(f"[BRAVE] News search failed: {e}", level='warning')
        
        if web_results or news_results:
            return {"web": web_results, "news": news_results}
        else:
            olliePrint(f"[BRAVE] No results obtained", show_banner=False)
            return None
        
    except Exception as e:
        olliePrint(f"Brave search failed: {e}", level='warning')
        return None

def search_duckduckgo(query: str) -> dict:
    """Search DuckDuckGo and return structured results for web and news."""
    try:
        import time
        import random
        
        olliePrint(f"[DUCKDUCKGO] Starting search for: '{query}' (fallback mode)", show_banner=False)
        
        time.sleep(random.uniform(2.0, 4.0))
        
        try:
            ddgs = DDGS(timeout=config.WEB_SEARCH_TIMEOUT)
        except Exception as e:
            olliePrint(f"[DUCKDUCKGO] Failed to initialize DDGS: {e}", level='warning')
            return None
        
        web_results = []
        news_results = []
        
        # Web search with retry
        raw_web_results = None
        for attempt in range(2):
            try:
                raw_web_results = ddgs.text(
                    keywords=query, region="us-en", safesearch="off", max_results=config.RESEARCH_MAX_WEB_ARTICLES
                )
                if raw_web_results: break
            except Exception as e:
                olliePrint(f"[DUCKDUCKGO] Web search attempt {attempt+1} failed: {e}", level='warning')
                time.sleep(3)
        
        if raw_web_results:
            for result in raw_web_results:
                url = result.get('href')
                if url:
                    web_results.append({
                        "title": result.get('title', 'No Title'),
                        "snippet": result.get('body', ''),
                        "url": url,
                        "source": "DuckDuckGo Web"
                    })
        
        # News search with retry
        raw_news_results = None
        for attempt in range(2):
            try:
                raw_news_results = ddgs.news(
                    keywords=query, region="us-en", safesearch="off", max_results=config.RESEARCH_MAX_NEWS_ARTICLES
                )
                if raw_news_results: break
            except Exception as e:
                olliePrint(f"[DUCKDUCKGO] News search attempt {attempt+1} failed: {e}", level='warning')
                time.sleep(2)

        if raw_news_results:
            for result in raw_news_results:
                url = result.get('url')
                if url:
                    news_results.append({
                        "title": result.get('title', 'No Title'),
                        "snippet": result.get('body', ''),
                        "url": url,
                        "source": "DuckDuckGo News"
                    })

        olliePrint(f"[DUCKDUCKGO] Search completed: {len(web_results)} web, {len(news_results)} news results.", show_banner=False)
        if web_results or news_results:
            return {"web": web_results, "news": news_results}
        else:
            return None
        
    except Exception as e:
        olliePrint(f"DuckDuckGo search failed: {e}", level='warning')
        return None

def search_searchapi(query: str) -> dict:
    """Search using SearchAPI.io and return structured results."""
    try:
        import requests
        
        if not config.SEARCHAPI_API_KEY:
            olliePrint(f"[SEARCHAPI] No API key configured, skipping", show_banner=False)
            return None
        
        olliePrint(f"[SEARCHAPI] Starting search for: '{query}'", show_banner=False)
        
        results = []
        
        search_params = {
            'engine': 'duckduckgo', 'q': query, 'api_key': config.SEARCHAPI_API_KEY, 'num': config.RESEARCH_MAX_WEB_ARTICLES
        }
        
        try:
            response = requests.get(config.SEARCHAPI_BASE_URL, params=search_params, timeout=config.RESEARCH_CORE_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Process organic and news results
            for result_type in ['organic_results', 'news_results']:
                if result_type in data:
                    for result in data[result_type]:
                        url = result.get('link')
                        if url:
                            results.append({
                                "title": result.get('title', 'No Title'),
                                "snippet": result.get('snippet', ''),
                                "url": url,
                                "source": "SearchAPI"
                            })
                            
            olliePrint(f"[SEARCHAPI] Search completed: {len(results)} results", show_banner=False)
            return {"web": results, "news": []} if results else None # SearchAPI doesn't differentiate well
            
        except Exception as e:
            olliePrint(f"[SEARCHAPI] Search failed: {e}", level='warning')
            return None
        
    except Exception as e:
        olliePrint(f"SearchAPI search failed: {e}", level='warning')
        return None

def search_arxiv(query: str) -> list:
    """Search arXiv with explicit PDF URLs."""
    import requests
    import xml.etree.ElementTree as ET
    
    params = {'search_query': f'all:{query}', 'start': 0, 'max_results': config.RESEARCH_MAX_ACADEMIC_PAPERS}
    response = requests.get(config.ARXIV_API_BASE, params=params, timeout=config.RESEARCH_CORE_TIMEOUT)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    results = []
    for entry in root.findall('atom:entry', ns):
        url = entry.find('atom:id', ns).text
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
        summary = entry.find('atom:summary', ns).text.strip()
        pdf_url = f"https://arxiv.org/pdf/{url.split('/')[-1]}"  # Explicit PDF URL
        
        results.append({
            "title": title,
            "snippet": f"Authors: {', '.join(authors[:3])}. Abstract: {summary[:300]}...",
            "url": url,
            "pdf_url": pdf_url,  # Add PDF link
            "source": "arXiv"
        })
    return results

def search_semantic_scholar(query: str) -> list:
    """Search Semantic Scholar and return a list of structured results."""
    try:
        import requests
        
        url = f"{config.SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        params = {
            'query': query, 'limit': config.RESEARCH_MAX_ACADEMIC_PAPERS, 'fields': 'title,abstract,authors,url,year,citationCount'
        }
        
        response = requests.get(url, params=params, timeout=config.RESEARCH_CORE_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for paper in data.get('data', []):
            url = paper.get('url')
            if url:
                authors = [author.get('name', '') for author in paper.get('authors', [])]
                snippet = f"Authors: {', '.join(authors[:3])} ({paper.get('year', '')}). Citations: {paper.get('citationCount', 0)}. Abstract: {paper.get('abstract', '')[:250]}..."
                results.append({
                    "title": paper.get('title', 'No Title'), "snippet": snippet, "url": url, "source": "Semantic Scholar"
                })
        
        olliePrint(f"[SEMANTIC SCHOLAR] Success. Found {len(results)} academic papers.", show_banner=False)
        return results
        
    except Exception as e:
        olliePrint(f"Semantic Scholar search failed: {e}", level='warning')
        return []

def search_news_apis(query: str) -> list:
    """Search NewsAPI and return a list of structured results."""
    try:
        import requests
        results = []
        
        if config.NEWS_API_KEY:
            try:
                params = {
                    'q': query, 'apiKey': config.NEWS_API_KEY, 'pageSize': config.RESEARCH_MAX_NEWS_ARTICLES, 'language': 'en'
                }
                response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=config.RESEARCH_CORE_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                for article in data.get('articles', []):
                    url = article.get('url')
                    if url:
                        snippet = f"Source: {article.get('source', {}).get('name', '')}. Published: {article.get('publishedAt', '')}. {article.get('description', '')}"
                        results.append({
                            "title": article.get('title', 'No Title'), "snippet": snippet, "url": url, "source": "NewsAPI"
                        })
            except Exception as e:
                olliePrint(f"NewsAPI failed: {e}", level='warning')
        
        olliePrint(f"[NEWSAPI] Success. Found {len(results)} news articles.", show_banner=False)
        return results
        
    except Exception as e:
        olliePrint(f"News APIs search failed: {e}", level='warning')
        return []

def search_reddit(query: str) -> list:
    """Search Reddit and return a list of structured results."""
    try:
        import requests
        results = []
        
        if config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET:
            try:
                auth_response = requests.post(
                    'https://www.reddit.com/api/v1/access_token',
                    auth=(config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET),
                    data={'grant_type': 'client_credentials'},
                    headers={'User-Agent': config.REDDIT_USER_AGENT},
                    timeout=config.RESEARCH_SOCIAL_TIMEOUT
                )
                auth_response.raise_for_status()
                token = auth_response.json()['access_token']
                
                headers = {'User-Agent': config.REDDIT_USER_AGENT, 'Authorization': f'bearer {token}'}
                params = {'q': query, 'sort': 'relevance', 'limit': config.RESEARCH_MAX_FORUM_POSTS, 'type': 'link'}
                
                search_response = requests.get('https://oauth.reddit.com/search', headers=headers, params=params, timeout=config.RESEARCH_SOCIAL_TIMEOUT)
                search_response.raise_for_status()
                data = search_response.json()
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    permalink = f"https://reddit.com{post_data.get('permalink', '')}"
                    if permalink:
                        snippet = f"r/{post_data.get('subreddit', '')} | Score: {post_data.get('score', 0)}. {post_data.get('selftext', '')[:200]}..."
                        results.append({
                            "title": post_data.get('title', 'No Title'), "snippet": snippet, "url": permalink, "source": "Reddit"
                        })
            except Exception as e:
                olliePrint(f"Reddit API failed: {e}", level='warning')
        
        olliePrint(f"[REDDIT] Success. Found {len(results)} relevant posts.", show_banner=False)
        return results
        
    except Exception as e:
        olliePrint(f"Reddit search failed: {e}", level='warning')
        return []

def search_stackoverflow(query: str) -> list:
    """Search Stack Overflow and return a list of structured results."""
    try:
        import requests
        
        params = {
            'order': 'desc', 'sort': 'relevance', 'intitle': query, 'site': 'stackoverflow', 'pagesize': config.RESEARCH_MAX_FORUM_POSTS
        }
        if config.STACKOVERFLOW_API_KEY:
            params['key'] = config.STACKOVERFLOW_API_KEY
        
        response = requests.get('https://api.stackexchange.com/2.3/search/advanced', params=params, timeout=config.RESEARCH_SOCIAL_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for question in data.get('items', []):
            link = question.get('link')
            if link:
                tags_str = ", ".join(question.get('tags', [])[:5])
                snippet = f"Tags: {tags_str} | Score: {question.get('score', 0)}, Answers: {question.get('answer_count', 0)}."
                results.append({
                    "title": question.get('title', 'No Title'), "snippet": snippet, "url": link, "source": "Stack Overflow"
                })
        
        olliePrint(f"[STACKOVERFLOW] Success. Found {len(results)} relevant questions.", show_banner=False)
        return results
        
    except Exception as e:
        olliePrint(f"Stack Overflow search failed: {e}", level='warning')
        return []

def search_youtube_transcripts(query: str) -> list:
    """Search YouTube, get transcripts, and return a list of structured results."""
    try:
        import requests
        results = []
        
        if not config.YOUTUBE_API_KEY:
            olliePrint(f"[YOUTUBE] No API key configured, skipping", show_banner=False)
            return []
        
        olliePrint(f"[YOUTUBE] Starting search for: '{query}'", show_banner=False)
        
        try:
            search_params = {
                'part': 'snippet', 'q': query, 'type': 'video', 'maxResults': config.RESEARCH_MAX_VIDEO_TRANSCRIPTS,
                'key': config.YOUTUBE_API_KEY
            }
            search_response = requests.get('https://www.googleapis.com/youtube/v3/search', params=search_params, timeout=config.RESEARCH_SOCIAL_TIMEOUT)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            try:
                from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
                transcript_api_available = True
            except ImportError:
                transcript_api_available = False
                olliePrint(f"[YOUTUBE] WARNING: youtube-transcript-api not installed.", level='warning')
            
            for item in search_data.get('items', []):
                video_id = item['id']['videoId']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                title = item['snippet']['title']
                
                transcript_text = "[Transcript not available]"
                if transcript_api_available:
                    try:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
                        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
                        transcript_text = full_transcript[:1000] + "..." if len(full_transcript) > 1000 else full_transcript
                    except (TranscriptsDisabled, NoTranscriptFound):
                        transcript_text = "[Transcript disabled or not found]"
                    except Exception:
                        transcript_text = "[Error retrieving transcript]"
                
                snippet = f"Channel: {item['snippet']['channelTitle']}. Description: {item['snippet']['description'][:150]}... Transcript: {transcript_text[:200]}..."
                results.append({"title": title, "snippet": snippet, "url": video_url, "source": "YouTube"})
                
            olliePrint(f"[YOUTUBE] Search completed: {len(results)} videos found", show_banner=False)
            
        except Exception as e:
            olliePrint(f"YouTube API failed: {e}", level='warning')
        
        return results
        
    except Exception as e:
        olliePrint(f"YouTube search failed: {e}", level='warning')
        return []

def extract_webpage_content(url: str) -> dict:
    """Extract content from a webpage using Jina Reader or fallback methods."""
    try:
        import requests
        from urllib.parse import urljoin, urlparse
        
        # Try Jina Reader API first
        if config.JINA_API_KEY:
            try:
                headers = {
                    'Authorization': f'Bearer {config.JINA_API_KEY}',
                    'Content-Type': 'application/json'
                }
                
                jina_url = f"https://r.jina.ai/{url}"
                response = requests.get(jina_url, headers=headers, timeout=config.RESEARCH_CORE_TIMEOUT)
                response.raise_for_status()
                
                return {
                    "text": response.text,
                    "links": extract_links_from_text(response.text, url)
                }
                
            except Exception as e:
                olliePrint(f"Jina Reader failed: {e}", level='warning')
        
        # Fallback: Basic webpage extraction
        try:
            response = requests.get(url, timeout=config.RESEARCH_CORE_TIMEOUT, headers={
                'User-Agent': 'F.R.E.D. Research Assistant v2.0'
            })
            response.raise_for_status()
            
            # Basic HTML content extraction (you might want to use BeautifulSoup for better parsing)
            content = response.text
            
            # Simple text extraction (remove HTML tags)
            import re
            text_content = re.sub(r'<[^>]+>', '', content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            return {
                "text": text_content[:2000],  # Limit content length
                "links": extract_links_from_html(content, url)
            }
            
        except Exception as e:
            olliePrint(f"Basic webpage extraction failed: {e}", level='warning')
            return None
        
    except Exception as e:
        olliePrint(f"Webpage content extraction failed: {e}", level='warning')
        return None

def extract_links_from_html(html_content: str, base_url: str) -> list:
    """Extract links from HTML content."""
    try:
        import re
        from urllib.parse import urljoin, urlparse
        
        # Find all href attributes
        link_pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(link_pattern, html_content)
        
        absolute_links = []
        for link in matches:
            try:
                absolute_url = urljoin(base_url, link)
                parsed = urlparse(absolute_url)
                
                # Only include HTTP/HTTPS links
                if parsed.scheme in ['http', 'https']:
                    absolute_links.append(absolute_url)
                    
            except:
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in absolute_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links[:20]  # Limit number of links
        
    except Exception as e:
        olliePrint(f"Link extraction failed: {e}", level='warning')
        return []

def extract_links_from_text(text_content: str, base_url: str) -> list:
    """Extract links from plain text content."""
    try:
        import re
        from urllib.parse import urljoin
        
        # Find URLs in text
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+[^\s.,;:!?<>"{}|\\^`[\])]'
        matches = re.findall(url_pattern, text_content)
        
        return matches[:20]  # Limit number of links
        
    except Exception as e:
        olliePrint(f"Text link extraction failed: {e}", level='warning')
        return []

def categorize_extracted_links(links: list, source_url: str) -> dict:
    """Categorize extracted links into appropriate categories."""
    categorized = {
        "academic_papers": [],
        "related_articles": [],
        "official_sources": [],
        "news_articles": [],
        "documentation": [],
        "forums": [],
        "social_media": [],
        "video_transcripts": []
    }
    
    for link in links:
        try:
            link_lower = link.lower()
            
            # Academic papers
            if any(domain in link_lower for domain in ['arxiv.org', 'scholar.google', 'researchgate', 'pubmed', 'doi.org']):
                categorized["academic_papers"].append(link)
            
            # Social media
            elif any(domain in link_lower for domain in ['reddit.com', 'facebook.com', 'linkedin.com']):
                categorized["social_media"].append(link)
            
            # Video platforms
            elif any(domain in link_lower for domain in ['youtube.com', 'vimeo.com']):
                categorized["video_transcripts"].append(link)
            
            # Forums
            elif any(domain in link_lower for domain in ['stackoverflow.com', 'stackexchange.com', 'forum']):
                categorized["forums"].append(link)
            
            # News
            elif any(domain in link_lower for domain in ['news', 'bbc.com', 'cnn.com', 'reuters.com', 'npr.org']):
                categorized["news_articles"].append(link)
            
            # Official sources
            elif any(domain in link_lower for domain in ['.gov', '.edu', '.org']):
                categorized["official_sources"].append(link)
            
            # Documentation
            elif any(keyword in link_lower for keyword in ['docs', 'documentation', 'guide', 'manual', 'wiki']):
                categorized["documentation"].append(link)
            
            # Default to related articles
            else:
                categorized["related_articles"].append(link)
                
        except:
            continue
    
    return categorized

def search_pubmed(query: str) -> list:
    """Search PubMed Central for biomedical papers."""
    import requests
    
    params = {
        'term': query,
        'retmax': config.RESEARCH_MAX_ACADEMIC_PAPERS,
        'retmode': 'json'
    }
    response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc", params=params)
    papers = response.json().get('esearchresult', {}).get('idlist', [])
    
    results = []
    for paper_id in papers:
        results.append({
            "title": f"PMC{paper_id}",
            "snippet": f"PubMed Central ID: {paper_id}",
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{paper_id}",
            "pdf_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{paper_id}/pdf/",
            "source": "PubMed Central"
        })
    return results

def fetch_unpaywall(doi: str) -> str | None:
    """Fetch open-access PDF URL from Unpaywall."""
    import requests
    try:
        response = requests.get(f"https://api.unpaywall.org/v2/{doi}?email=YOUR_EMAIL@example.com", timeout=10)
        return response.json().get("best_oa_location", {}).get("url")
    except Exception:
        return None

def enhance_with_unpaywall(paper: dict) -> dict:
    """Add PDF URL if Unpaywall finds one."""
    if not paper.get("pdf_url") and "doi.org" in paper.get("url", ""):
        doi = paper["url"].split("doi.org/")[-1]
        pdf_url = fetch_unpaywall(doi)
        if pdf_url:
            paper["pdf_url"] = pdf_url
    return paper
