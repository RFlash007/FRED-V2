"""
A.R.C.H./D.E.L.V.E. Iterative Research System
Advanced research director/analyst conversation system for F.R.E.D.'s agenda processing
"""

import os
import json
import uuid
import ollama
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re

from config import config
from ollie_print import olliePrint_simple

# Import memory tools for complete_research functionality
from Tools import tool_add_memory, tool_read_webpage, TOOL_FUNCTIONS

class ArchDelveState:
    """State management for A.R.C.H./D.E.L.V.E. research conversations with thinking removal."""
    
    def __init__(self, task_id: str, original_task: str):
        self.task_id = task_id
        self.original_task = original_task
        self.conversation_history = []
        self.arch_context = []  # A.R.C.H.'s thinking context (last 5 messages)
        self.delve_context = []  # D.E.L.V.E.'s thinking context (last 5 messages)
        self.research_complete = False
        self.final_findings = ""
        self._lock = threading.Lock()
        
        # Create conversation storage directory
        self.conversation_dir = Path(config.ARCH_DELVE_CONVERSATION_STORAGE_PATH) / task_id
        self.conversation_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_think_content(self, text):
        """Extract thinking content from <think>...</think> tags."""
        if not text:
            return ""
        matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
        return '\n'.join(matches).strip()
    
    def strip_think_tags(self, text):
        """Remove <think>...</think> blocks from text."""
        if not text:
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    def add_conversation_turn(self, role: str, content: str, model_type: str, thinking: str = ""):
        """Add a turn to conversation history with thinking context management."""
        with self._lock:
            # Create full turn record
            turn = {
                'role': role,
                'content': content,
                'model_type': model_type,  # 'arch' or 'delve'
                'thinking': thinking,
                'timestamp': datetime.now().isoformat()
            }
            
            self.conversation_history.append(turn)
            
            # Manage thinking context per model (5 messages each)
            if model_type == 'arch':
                self.arch_context.append(turn)
                if len(self.arch_context) > config.ARCH_DELVE_MAX_CONVERSATION_MESSAGES:
                    self.arch_context = self.arch_context[-config.ARCH_DELVE_MAX_CONVERSATION_MESSAGES:]
            elif model_type == 'delve':
                self.delve_context.append(turn)
                if len(self.delve_context) > config.ARCH_DELVE_MAX_CONVERSATION_MESSAGES:
                    self.delve_context = self.delve_context[-config.ARCH_DELVE_MAX_CONVERSATION_MESSAGES:]
    
    def get_context_for_model(self, model_type: str) -> List[Dict]:
        """Get conversation context for specific model with only final outputs for cross-model communication."""
        with self._lock:
            if model_type == 'arch':
                context = self.arch_context.copy()
            elif model_type == 'delve':
                context = self.delve_context.copy()
            else:
                return []
            
            # Prepare messages with clean final outputs only
            messages = []
            for turn in context:
                if turn['role'] == 'user':
                    messages.append({"role": "user", "content": turn['content']})
                elif turn['role'] == 'assistant':
                    # Always use clean content without thinking blocks for inter-model communication
                    clean_content = turn['content']
                    messages.append({"role": "assistant", "content": clean_content})
            
            return messages
    
    def save_conversation_state(self):
        """Save current conversation state to storage."""
        try:
            # Save full conversation log
            conversation_file = self.conversation_dir / "full_conversation.json"
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'task_id': self.task_id,
                    'original_task': self.original_task,
                    'conversation_history': self.conversation_history,
                    'research_complete': self.research_complete,
                    'final_findings': self.final_findings,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            # Save model contexts separately
            arch_context_file = self.conversation_dir / "arch_context.json"
            with open(arch_context_file, 'w', encoding='utf-8') as f:
                json.dump(self.arch_context, f, indent=2, ensure_ascii=False)
            
            delve_context_file = self.conversation_dir / "delve_context.json"
            with open(delve_context_file, 'w', encoding='utf-8') as f:
                json.dump(self.delve_context, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            olliePrint_simple(f"Failed to save A.R.C.H./D.E.L.V.E. conversation state: {e}", level='error')

def log_tool_result(tool_name: str, arguments: dict, result: dict):
    """Logs the output of a tool call in a readable format."""
    print("\n" + "="*25 + f" 📖 TOOL RESULT: {tool_name} " + "="*25)
    
    query = arguments.get('query', arguments.get('url', 'N/A'))
    print(f"  ➡️  Input: {query}")

    if not isinstance(result, dict) or 'success' not in result:
        print(f"  ❌ Status: Failed (Invalid result format)")
        print(f"  Raw Result: {result}")
        print("="*70)
        return

    success = result.get('success', False)
    status = "✅ Success" if success else "❌ Failed"
    print(f"  Status: {status}")

    if not success:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    else:
        # Handle structured search results from any search tool
        if "results" in result and isinstance(result["results"], list):
            search_results = result["results"]
            print(f"\n  📄 Found {len(search_results)} results:")
            print("-" * 40)
            if not search_results:
                print("   (No results returned)")
            else:
                for i, res in enumerate(search_results[:5]): # Limit print to first 5
                    print(f"  {i+1}. {res.get('title', 'No Title')}")
                    print(f"     URL: {res.get('url', 'N/A')}")
                    print(f"     Snippet: {res.get('snippet', 'N/A')[:200]}...")
            print("-" * 40)

        # Handle read_webpage
        elif tool_name == "read_webpage":
            content = result.get("content", "")
            print("\n  📄 Page Content:")
            print("-" * 40)
            print(content if content.strip() else "   (No content extracted)")
            print("-" * 40)
            print(f"  🔗 Links Found on Page: {result.get('links_found', 0)}")

    print("="*70)

# Global storage for active research sessions
active_research_sessions: Dict[str, ArchDelveState] = {}

def create_research_session(task_id: str, original_task: str) -> ArchDelveState:
    """Create a new A.R.C.H./D.E.L.V.E. research session."""
    session = ArchDelveState(task_id, original_task)
    active_research_sessions[task_id] = session
    
    olliePrint_simple(f"[A.R.C.H./D.E.L.V.E.] Research session created: {task_id}")
    olliePrint_simple(f"   Task: {original_task[:100]}...")
    
    return session

def prepare_arch_messages(session: ArchDelveState) -> List[Dict]:
    """Prepare messages for A.R.C.H. with system prompt and task injection."""
    current_time_iso = datetime.now().isoformat()
    current_date_readable = datetime.now().strftime("%B %d, %Y")
    print(f"[SYSTEM TIME] {current_time_iso}")
    
    messages = [
        {
            "role": "system", 
            "content": config.ARCH_SYSTEM_PROMPT.format(
                original_task=session.original_task,
                current_date_time=current_time_iso,
                current_date=current_date_readable
            )
        }
    ]
    
    # Add conversation context
    context_messages = session.get_context_for_model('arch')
    messages.extend(context_messages)
    
    # Add task reinforcement if starting new session
    if len(context_messages) == 0:
        messages.append({
            "role": "user",
            "content": config.ARCH_TASK_PROMPT.format(original_task=session.original_task)
        })
    
    return messages

def prepare_delve_messages(session: ArchDelveState, arch_instruction: str) -> List[Dict]:
    """Prepare messages for D.E.L.V.E. with system prompt and director instruction."""
    current_time_iso = datetime.now().isoformat()
    current_date_readable = datetime.now().strftime("%B %d, %Y")
    messages = [
        {"role": "system", "content": config.DELVE_SYSTEM_PROMPT.format(
            current_date_time=current_time_iso,
            current_date=current_date_readable
        )}
    ]
    
    # Add conversation context (D.E.L.V.E.'s own context only)
    context_messages = session.get_context_for_model('delve')
    messages.extend(context_messages)
    
    # Add current instruction from A.R.C.H.
    messages.append({
        "role": "user",
        "content": arch_instruction
    })
    
    # Simplified debug output
    print(f"\n[D.E.L.V.E. CONTEXT] {len(context_messages)} previous messages | Instruction: '{arch_instruction}'")
    
    return messages

def run_arch_iteration(session: ArchDelveState, ollama_client: ollama.Client) -> Tuple[str, bool]:
    """Run A.R.C.H. iteration and return (response, is_complete)."""
    try:
        messages = prepare_arch_messages(session)
        
        # A.R.C.H. tools (only complete_research)
        arch_tools = [
            {
                "name": "complete_research",
                "description": "Submit comprehensive research findings when 100% confident task is complete",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "comprehensive_findings": {
                            "type": "string", 
                            "description": "Compile concise academic report with Executive Summary → Methodology → Core Findings → Analysis → Conclusions → Sources"
                        }
                    },
                    "required": ["comprehensive_findings"]
                }
            }
        ]
        
        response = ollama_client.chat(
            model=config.ARCH_MODEL,
            messages=messages,
            tools=arch_tools,
            stream=False,
            options=config.THINKING_MODE_OPTIONS
        )
        
        response_message = response.get('message', {})
        raw_content = response_message.get('content', '')
        tool_calls = response_message.get('tool_calls')
        
        # Extract thinking and get clean content for D.E.L.V.E.
        thinking = session.extract_think_content(raw_content)
        clean_content = session.strip_think_tags(raw_content)
        
        # Show A.R.C.H. thinking and instruction
        print(f"\n[A.R.C.H. THINKING]:\n{thinking}")
        print(f"\n[A.R.C.H. → D.E.L.V.E.]:\n{clean_content}")
        print("-" * 70)
        
        # Check for completion tool call
        research_complete = False
        if tool_calls:
            for tool_call in tool_calls:
                print(f"\n[A.R.C.H. TOOL CALL DEBUG]:")
                print(f"  Full tool_call: {tool_call}")
                
                # Safely extract tool name
                tool_name = None
                try:
                    tool_name = tool_call.get('function', {}).get('name')
                    if not tool_name:
                        print(f"  ERROR: No tool name found in tool_call")
                        continue
                except Exception as e:
                    print(f"  ERROR: Failed to extract tool name: {e}")
                    continue
                
                if tool_name == 'complete_research':
                    research_complete = True
                    
                    # Parse arguments with improved error handling
                    try:
                        raw_arguments = tool_call.get('function', {}).get('arguments', {})
                        
                        print(f"  Raw arguments: {raw_arguments}")
                        print(f"  Arguments type: {type(raw_arguments)}")
                        
                        if isinstance(raw_arguments, str):
                            # Try to clean up common JSON formatting issues
                            cleaned_args = raw_arguments.strip()
                            if cleaned_args.startswith('"') and cleaned_args.endswith('"'):
                                # Remove outer quotes if present
                                cleaned_args = cleaned_args[1:-1]
                            
                            # Try to parse the JSON
                            try:
                                arguments = json.loads(cleaned_args)
                                print(f"  Parsed arguments: {arguments}")
                            except json.JSONDecodeError as json_err:
                                print(f"  JSON parse error: {json_err}")
                                # Try manual extraction for comprehensive_findings
                                import re
                                findings_match = re.search(r'"comprehensive_findings"\s*:\s*"([^"]*)"', cleaned_args)
                                if findings_match:
                                    arguments = {"comprehensive_findings": findings_match.group(1)}
                                    print(f"  Manual extraction successful: {arguments}")
                                else:
                                    print(f"  Manual extraction failed")
                                    arguments = {"comprehensive_findings": "Error: No findings provided in tool call"}
                        elif isinstance(raw_arguments, dict):
                            arguments = raw_arguments
                            print(f"  Using dict arguments: {arguments}")
                        else:
                            print(f"  Unexpected arguments format: {type(raw_arguments)}")
                            arguments = {"comprehensive_findings": "Error: Invalid arguments format"}
                        
                        # Get findings and validate
                        findings = arguments.get('comprehensive_findings', '')
                        if not findings or findings.strip() == '':
                            print(f"  ERROR: Empty findings provided!")
                            findings = "Error: No comprehensive findings provided. A.R.C.H. completed research without providing findings."
                        
                        print(f"  Findings length: {len(findings)} characters")
                        
                        session.final_findings = findings
                        session.research_complete = True
                        
                        print(f"[A.R.C.H.] Research complete! Findings: {len(findings)} chars")
                        
                    except Exception as e:
                        print(f"  Failed to parse arguments: {e}")
                        import traceback
                        traceback.print_exc()
                        findings = f"Error parsing completion arguments: {str(e)}"
                        session.final_findings = findings
                        session.research_complete = True
                    
                    break
                else:
                    print(f"  ERROR: Unknown tool name: {tool_name}")
                    print(f"  Available tools: complete_research")
                    print(f"  A.R.C.H. should only use complete_research tool!")
        
        # Handle empty A.R.C.H. responses
        if not clean_content.strip():
            if tool_calls:
                print(f"[ERROR] A.R.C.H. used tools without providing delegation instructions!")
                return "ERROR: A.R.C.H. completed research without delegating to D.E.L.V.E. first", False
            else:
                print(f"[ERROR] A.R.C.H. provided no instruction!")
                return "ERROR: No instruction provided", False
        
        # Store A.R.C.H.'s response
        session.add_conversation_turn('assistant', clean_content, 'arch', thinking)
        session.save_conversation_state()
        
        return clean_content, research_complete
        
    except Exception as e:
        olliePrint_simple(f"A.R.C.H. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        return f"A.R.C.H. system error: {str(e)}", False

def run_delve_iteration(session: ArchDelveState, arch_instruction: str, ollama_client: ollama.Client) -> str:
    """Run D.E.L.V.E. iteration and return response."""
    try:
        messages = prepare_delve_messages(session, arch_instruction)
        
        # D.E.L.V.E. tools (specialized web search)
        delve_tools = [
            {
                "name": "search_general",
                "description": "General web search for broad topics, documentation, or official sources.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}
            },
            {
                "name": "search_news",
                "description": "Search for recent news articles and current events.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query for news"}}, "required": ["query"]}
            },
            {
                "name": "search_academic",
                "description": "Search for academic papers and research articles.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query for academic content"}}, "required": ["query"]}
            },
            {
                "name": "search_forums",
                "description": "Search forums and community discussion platforms like Reddit and Stack Overflow.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query for forum discussions"}}, "required": ["query"]}
            },
            {
                "name": "read_webpage",
                "description": "Extract full content from a specific webpage URL. Use after a search to read promising sources.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The complete URL of the webpage to read."}}, "required": ["url"]}
            }
        ]
        
        # Store A.R.C.H.'s instruction first
        session.add_conversation_turn('user', arch_instruction, 'delve')
        
        # DELVE can now use unlimited tool iterations for thorough research
        assistant_response = ""
        raw_thinking = ""
        
        # Continue until DELVE stops using tools naturally
        while True:
            response = ollama_client.chat(
                model=config.DELVE_MODEL,
                messages=messages,
                tools=delve_tools,
                stream=False,
                options=config.THINKING_MODE_OPTIONS
            )
            
            response_message = response.get('message', {})
            raw_content = response_message.get('content', '')
            tool_calls = response_message.get('tool_calls')
            
            # Extract thinking
            current_thinking = session.extract_think_content(raw_content)
            if current_thinking:
                raw_thinking += current_thinking + "\n"
            
            clean_content = session.strip_think_tags(raw_content)
            
            # Show D.E.L.V.E.'s full response
            if current_thinking:
                print(f"\n[D.E.L.V.E. THINKING]:\n{current_thinking}")
            if clean_content:
                print(f"\n[D.E.L.V.E. RESPONSE]:\n{clean_content}")
            if tool_calls:
                print(f"\n[D.E.L.V.E. TOOL CALLS] {len(tool_calls)} calls")
            print("=" * 70)
            
            if 'role' not in response_message:
                response_message['role'] = 'assistant'
            
            # Preserve thinking for next iteration if there are tools
            if tool_calls and current_thinking:
                response_message['content'] = raw_content
            
            messages.append(response_message)
            
            if tool_calls:
                # Execute web search tools
                tool_results_for_next_iteration = []

                for i, tool_call in enumerate(tool_calls):
                    print(f"\n[TOOL CALL {i+1}]")
                    
                    tool_name = tool_call.get('function', {}).get('name')
                    if not tool_name:
                        print(f"  ERROR: No tool name found")
                        tool_results_for_next_iteration.append({
                            "role": "tool", "tool_call_id": tool_call.get('id'),
                            "content": json.dumps({"success": False, "error": "Tool name missing."})
                        })
                        continue

                    # Parse arguments safely
                    try:
                        raw_arguments = tool_call.get('function', {}).get('arguments', {})
                        if isinstance(raw_arguments, str):
                            arguments = json.loads(raw_arguments)
                        elif isinstance(raw_arguments, dict):
                            arguments = raw_arguments
                        else:
                            raise ValueError(f"Unexpected arguments format: {type(raw_arguments)}")
                    except Exception as e:
                        print(f"  ERROR: Failed to parse arguments for {tool_name}: {e}")
                        tool_results_for_next_iteration.append({
                            "role": "tool", "tool_call_id": tool_call.get('id'),
                            "content": json.dumps({"success": False, "error": f"Argument parsing failed: {e}"})
                        })
                        continue

                    # Execute the tool call using the TOOL_FUNCTIONS registry
                    if tool_name in TOOL_FUNCTIONS:
                        try:
                            tool_function = TOOL_FUNCTIONS[tool_name]
                            print(f"  Executing {tool_name} with args: {arguments}")
                            
                            result = tool_function(**arguments)
                            log_tool_result(tool_name, arguments, result)
                            
                            tool_output_content = json.dumps(result)
                            
                        except Exception as e:
                            print(f"  ERROR: Tool {tool_name} execution failed: {e}")
                            import traceback
                            traceback.print_exc()
                            tool_output_content = json.dumps({"success": False, "error": f"Tool execution failed: {str(e)}"})
                    else:
                        print(f"  ERROR: Unknown tool name: {tool_name}")
                        tool_output_content = json.dumps({"success": False, "error": f"Tool '{tool_name}' not found."})

                    tool_results_for_next_iteration.append({
                        "role": "tool",
                        "content": tool_output_content,
                        "tool_call_id": tool_call.get('id', 'unknown')
                    })

                messages.extend(tool_results_for_next_iteration)
            else:
                # No tools, DELVE is finished with research
                if clean_content:
                    assistant_response = clean_content
                break
        
        # Store D.E.L.V.E.'s response
        session.add_conversation_turn('assistant', assistant_response, 'delve', raw_thinking.strip())
        session.save_conversation_state()
        
        return assistant_response
        
    except Exception as e:
        olliePrint_simple(f"D.E.L.V.E. iteration failed: {e}", level='error')
        import traceback
        traceback.print_exc()
        return f"D.E.L.V.E. system error: {str(e)}"

def conduct_iterative_research(task_id: str, original_task: str) -> Dict:
    """Main function to conduct iterative A.R.C.H./D.E.L.V.E. research."""
    try:
        olliePrint_simple(f"[RESEARCH] Starting A.R.C.H./D.E.L.V.E. investigation...")
        olliePrint_simple(f"   Task ID: {task_id}")
        olliePrint_simple(f"   Objective: {original_task[:100]}...")
        
        # Create research session
        session = create_research_session(task_id, original_task)
        
        # Initialize Ollama client
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        
        # Iterative research loop
        iteration_count = 0
        max_iterations = config.ARCH_DELVE_MAX_RESEARCH_ITERATIONS
        
        while iteration_count < max_iterations:
            iteration_count += 1
            olliePrint_simple(f"[RESEARCH] Iteration {iteration_count}/{max_iterations}")
            
            # A.R.C.H. provides direction
            arch_response, is_complete = run_arch_iteration(session, client)
            
            if is_complete:
                olliePrint_simple(f"[RESEARCH] A.R.C.H. declared research complete!")
                break
            
            # A.R.C.H. and D.E.L.V.E. responses are now logged inside their respective functions
            # D.E.L.V.E. executes research
            delve_response = run_delve_iteration(session, arch_response, client)
        
        # Check if research was completed
        if session.research_complete:
            olliePrint_simple(f"[SUCCESS] Research completed in {iteration_count} iterations")
            return {
                'success': True,
                'task_id': task_id,
                'findings': session.final_findings,
                'conversation_path': str(session.conversation_dir),
                'iterations': iteration_count
            }
        else:
            olliePrint_simple(f"[WARNING] Research reached max iterations without completion", level='warning')
            # Force completion with current conversation
            final_findings = f"Research conducted over {iteration_count} iterations. Investigation covered multiple aspects of: {original_task}. See conversation log for detailed findings."
            
            return {
                'success': False,
                'task_id': task_id,
                'findings': final_findings,
                'conversation_path': str(session.conversation_dir),
                'iterations': iteration_count,
                'reason': 'max_iterations_reached'
            }
        
    except Exception as e:
        olliePrint_simple(f"Research system error: {e}", level='error')
        return {
            'success': False,
            'task_id': task_id,
            'findings': f"Research system encountered an error: {str(e)}",
            'conversation_path': None,
            'iterations': 0,
            'reason': 'system_error'
        }
    
    finally:
        # Clean up session
        if task_id in active_research_sessions:
            del active_research_sessions[task_id]

def synthesize_research_to_memory(research_result: Dict, original_task: str) -> str:
    """Convert research findings to L3 memory node using S.A.G.E. synthesis."""
    try:
        olliePrint_simple("[S.A.G.E.] Synthesizing research findings for L3 memory...")
        
        # Prepare S.A.G.E. synthesis prompt
        synthesis_prompt = config.SAGE_SYNTHESIS_PROMPT.format(
            original_task=original_task,
            research_findings=research_result['findings']
        )
        
        # Call S.A.G.E. for synthesis
        response = requests.post(
            config.OLLAMA_GENERATE_URL,
            json={
                "model": config.SAGE_MODEL,
                "prompt": f"System: {config.SAGE_SYSTEM_PROMPT}\n\nUser: {synthesis_prompt}",
                "stream": False,
                "format": "json"
            },
            timeout=config.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        
        response_text = response.json().get("response", "").strip()
        if not response_text:
            olliePrint_simple("[S.A.G.E.] No response from synthesis model", level='error')
            return ""
        
        try:
            synthesis_result = json.loads(response_text)
        except json.JSONDecodeError as e:
            olliePrint_simple(f"[S.A.G.E.] Failed to parse synthesis JSON: {e}", level='error')
            return ""
        
        # Extract synthesized components
        memory_type = synthesis_result.get('memory_type', 'Semantic')
        memory_label = synthesis_result.get('label', f"Research: {original_task[:80]}...")
        memory_text = synthesis_result.get('text', research_result['findings'])
        
        olliePrint_simple(f"[S.A.G.E.] Synthesis complete - Type: {memory_type}")
        
        # Create optimized memory node
        result = tool_add_memory(
            label=memory_label,
            text=memory_text,
            memory_type=memory_type
        )
        
        # tool_add_memory returns a string, not a dict
        if result and "added with ID" in result:
            # Extract node ID from success message: "Memory 'label' added with ID 12345"
            try:
                node_id = result.split("ID ")[-1]
                olliePrint_simple(f"[S.A.G.E.] Optimized memory created: {node_id}")
                return node_id
            except:
                olliePrint_simple(f"[S.A.G.E.] Memory created but couldn't extract ID: {result}")
                return "created"
        else:
            olliePrint_simple(f"[S.A.G.E.] Failed to create memory node: {result}", level='error')
            return ""
            
    except Exception as e:
        olliePrint_simple(f"[S.A.G.E.] Synthesis system error: {e}", level='error')
        # Fallback to simple storage if S.A.G.E. fails
        olliePrint_simple("[S.A.G.E.] Falling back to direct storage...", level='warning')
        try:
            result = tool_add_memory(
                label=f"Research: {original_task[:100]}...",
                text=research_result['findings'],
                memory_type="Semantic"
            )
            if result and "added with ID" in result:
                try:
                    return result.split("ID ")[-1]
                except:
                    return "created"
        except Exception as fallback_error:
            olliePrint_simple(f"[S.A.G.E.] Fallback also failed: {fallback_error}", level='error')
        return "" 