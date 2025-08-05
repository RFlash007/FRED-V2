"""
F.R.E.D. Tool Schemas
=====================

This module contains all tool schemas for F.R.E.D. and its various agents.
Separating schemas from the main configuration improves organization and makes
them easier to manage, version, and test.
"""

# ============================================================================
# 12. TOOL SCHEMAS - CONSOLIDATED FROM LEGACY FILES
# ============================================================================
# All tool schemas organized by functional area and model usage
# Previously scattered across Tools.py, memory/crap.py, app.py, arch_delve_research.py

# --- Core Memory Management Tools ---
# Used by: C.R.A.P. (enhanced) and Tools.py for compatibility
MEMORY_TOOLS = [
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
        "name": "get_subgraph",
        "description": "Extract connected memory network around central node. Use for analyzing relationship patterns, finding paths between concepts, or understanding context clusters.",
        "parameters": {
            "type": "object",
            "properties": {
                "center_node_id": {
                    "type": "integer",
                    "description": "The central node ID to build subgraph around."
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth (capped at 5).",
                    "default": 2,
                    "maximum": 5
                },
                "relationship_types": {
                    "type": "array",
                    "description": "Optional. Filter by specific relationship types.",
                    "items": {"type": "string"}
                },
                "max_nodes": {
                    "type": "integer",
                    "description": "Maximum nodes to include (prevents memory issues).",
                    "default": 50,
                    "maximum": 100
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include detailed node and edge metadata.",
                    "default": True
                }
            },
            "required": ["center_node_id"]
        }
    }
]

# --- M.A.D. (Memory Addition Daemon) Tools ---
# Focused, minimal toolset for dedicated memory creation flows
MAD_TOOLS = [
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
        "name": "add_memory_with_observations",
        "description": "Enhanced memory addition supporting structured observations and arbitrary metadata for richer context.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Concise label/title for the memory."},
                "text": {"type": "string", "description": "Full textual content of the memory."},
                "memory_type": {
                    "type": "string",
                    "description": "Memory category.",
                    "enum": ["Semantic", "Episodic", "Procedural"]
                },
                "observations": {
                    "type": ["array", "null"],
                    "description": "Optional list of structured observations to attach to the memory.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "confidence": {"type": ["number", "null"], "description": "Confidence in this observation (0.0-1.0)."},
                            "source": {"type": ["string", "null"], "description": "Source of the information."},
                            "context": {"type": ["string", "null"], "description": "Context about when/where this was learned."},
                            "note": {"type": ["string", "null"], "description": "Free-form note or detail."}
                        },
                        "additionalProperties": True
                    }
                },
                "parent_id": {"type": ["integer", "null"], "description": "Optional parent node ID for hierarchical linkage."},
                "target_date": {"type": ["string", "null"], "description": "Optional ISO date/datetime for scheduling or future events."},
                "metadata": {
                    "type": ["object", "null"],
                    "description": "Optional arbitrary key/value metadata to persist with the memory.",
                    "additionalProperties": True
                }
            },
            "required": ["label", "text", "memory_type"]
        }
    }
]

# --- Research & Web Search Tools ---
RESEARCH_TOOLS = [
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
    },
    {
        "name": "read_webpage",
        "description": "Extract text from webpages or PDFs. Use after a search to read promising sources.",
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
        "name": "search_web_information",
        "description": "Searches the web for information using DuckDuckGo. This tool retrieves current information from the internet, combining results from general web and news searches.",
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
    }
]

# --- Agent Management Tools ---
AGENT_MANAGEMENT_TOOLS = [
    {
        "name": "addTaskToAgenda",
        "description": "Add a research task to the agenda for future processing during sleep cycles. Use when the user wants information that requires recent data you don't possess, or complex research that should be done later.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Detailed description of the research task or information needed."
                },
                "priority": {
                    "type": "integer",
                    "description": "Task priority: 1 (important) or 2 (normal). Defaults to 2.",
                    "enum": [1, 2],
                    "default": 2
                }
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "triggerSleepCycle",
        "description": "Initiate the sleep cycle to process agenda tasks, consolidate L2 memories to L3, and perform background maintenance. This will block F.R.E.D. temporarily while processing.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# --- Research Pipeline Control Tools ---
PIPELINE_CONTROL_TOOLS = [
    {
        "name": "complete_research",
        "description": "Signal that the research is 100% complete and all objectives have been met.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# --- Utility & System Tools ---
UTILITY_TOOLS = [
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
        "name": "code_interpreter",
        "description": "Executes Python code and returns the output or errors.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute."
                }
            },
            "required": ["code"]
        }
    }
]

# ============================================================================
# MODEL-SPECIFIC TOOL MAPPINGS
# ============================================================================
FRED_TOOLS = AGENT_MANAGEMENT_TOOLS.copy()
DELVE_TOOLS = RESEARCH_TOOLS.copy()
ARCH_TOOLS = PIPELINE_CONTROL_TOOLS.copy()
AVAILABLE_TOOLS = (MEMORY_TOOLS + MAD_TOOLS + RESEARCH_TOOLS + UTILITY_TOOLS).copy()

# ============================================================================
# TOOL SCHEMA VALIDATION & UTILITIES
# ============================================================================
def get_tool_set(agent_type: str) -> list:
    """
    Get the appropriate tool set for a specific agent type.
    
    Args:
        agent_type: One of 'FRED', 'DELVE', 'ARCH'
        
    Returns:
        list: Tool schema list for the specified agent
    """
    mappings = {
        'FRED': FRED_TOOLS,
        'DELVE': DELVE_TOOLS,
        'ARCH': ARCH_TOOLS
    }
    return mappings.get(agent_type, [])

def get_all_tool_names() -> set:
    """
    Get set of all unique tool names across all tool schemas.
    
    Returns:
        set: All unique tool names
    """
    all_tools = (MEMORY_TOOLS + MAD_TOOLS + RESEARCH_TOOLS +
                AGENT_MANAGEMENT_TOOLS + PIPELINE_CONTROL_TOOLS +
                UTILITY_TOOLS)
    return {tool['name'] for tool in all_tools}