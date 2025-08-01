#!/usr/bin/env python3
"""A.R.C.H./D.E.L.V.E. research demo (not part of automated tests)."""

import pytest
pytest.skip("manual integration script", allow_module_level=True)  # Skip in automated environments

import sys
import uuid
import time
import json
import os
from datetime import datetime
from pathlib import Path
import re

# Add current directory to path for imports
sys.path.append('.')

from memory.arch_delve_research import conduct_enhanced_iterative_research, synthesize_research_to_memory
from ollie_print import olliePrint

# **STREAMING MODE ENABLED** - All model outputs will be streamed in real-time
ENABLE_STREAMING = True
DEBUG_MODE = True

def print_banner():
    """Print the F.R.E.D. research system banner."""
    streaming_note = " [🔴 LIVE STREAMING]" if ENABLE_STREAMING else ""
    debug_note = " [DEBUG MODE]" if DEBUG_MODE else ""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              F.R.E.D. A.R.C.H./D.E.L.V.E. Research Test{streaming_note}{debug_note}                ║
║                                                                              ║
║  A.R.C.H. (Adaptive Research Command Hub) - Research Director               ║
║  D.E.L.V.E. (Data Extraction and Logical Verification Engine) - Analyst     ║
║  S.A.G.E. (Synthesis and Archive Generation Engine) - Memory Synthesis      ║
║  V.E.T. (Verification & Evidence Triangulation) - Quality Assessor          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    
    if ENABLE_STREAMING:
        print("🔴 LIVE STREAMING MODE: Model responses will appear in real-time as they're generated")
        print("💭 You'll see thinking, reasoning, and responses as they happen")
        print("⚡ This provides full transparency into the AI research process")
        print("-" * 80)
    
    if DEBUG_MODE:
        print("🔍 DEBUG MODE: Detailed system information and diagnostics enabled")
        print("⚙️  Tool calls, database operations, and state changes will be shown")
        print("-" * 80)

def get_user_input():
    """Get research prompt from user with some guidance."""
    print("\n🔬 Research Query Input")
    print("=" * 50)
    print("Enter your research question or topic. Examples:")
    print("• 'Latest developments in quantum computing 2024'")
    print("• 'Best practices for Python async programming'") 
    print("• 'Current state of AI safety research'")
    print("• 'How to implement WebRTC for real-time communication'")
    
    if DEBUG_MODE:
        print("• 'simple test query' (for debugging)")
    
    if ENABLE_STREAMING:
        print("\n🔴 Note: With streaming enabled, you'll see agents 'thinking' and responding live!")
    
    print()
    
    while True:
        query = input("📝 Research Query: ").strip()
        if query:
            return query
        print("❌ Please enter a valid research query.")

def display_research_progress(task_id: str):
    """Display progress information."""
    print(f"\n🚀 Initiating Research Session")
    print(f"   Task ID: {task_id}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if ENABLE_STREAMING:
        print(f"   🔴 Live Streaming: ENABLED")
        print("   💭 You'll see real-time thinking from each agent:")
        print("      • A.R.C.H. strategic planning and direction")
        print("      • D.E.L.V.E. web searches and data extraction")
        print("      • V.E.T. evidence verification and analysis")
        print("      • S.A.G.E. final synthesis and memory integration")
    
    print("\n📡 A.R.C.H./D.E.L.V.E. Research Pipeline Active...")
    print("   • A.R.C.H. will direct the research strategy")
    print("   • D.E.L.V.E. will execute comprehensive web searches")
    print("   • V.E.T. will verify and triangulate evidence")
    print("   • S.A.G.E. will synthesize findings into F.R.E.D.'s memory")
    
    if DEBUG_MODE:
        print("\n🔍 DEBUG MODE ACTIVE:")
        print("   • Model requests and responses will be shown")
        print("   • Tool calls and results will be displayed")
        print("   • System state changes will be logged")
        print("   • Error details will be verbose")
    
    print("\n⏳ Research in progress... (this may take several minutes)")
    print("-" * 70)

def debug_print(message: str, level: str = "DEBUG"):
    """Print debug messages only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

def stream_print(content: str, agent_name: str = "MODEL", end_chunk: bool = False):
    """Print streaming content with agent identification."""
    if ENABLE_STREAMING:
        if end_chunk:
            print(f"\n[{agent_name} COMPLETE] ✅")
            print("-" * 60)
        else:
            # Print content without newline for streaming effect
            print(content, end='', flush=True)

def display_results(result: dict, original_task: str):
    """Display comprehensive research results."""
    print("\n" + "=" * 70)
    print("🎯 RESEARCH RESULTS")
    print("=" * 70)
    
    print(f"✅ Success: {result.get('success', False)}")
    print(f"🔄 Iterations: {result.get('iterations', 0)}")
    print(f"📊 Enhancement: {result.get('enhancement', 'N/A')}")
    
    if ENABLE_STREAMING:
        print(f"🔴 Streaming: All model responses were shown in real-time")
    
    if DEBUG_MODE:
        print(f"🔍 Debug Info:")
        print(f"   • VET Reports: {result.get('vet_reports_count', 0)}")
        print(f"   • Global Citations: {result.get('global_citations_count', 0)}")
        print(f"   • Completion Reason: {result.get('reason', 'Unknown')}")
    
    if result.get('conversation_path'):
        print(f"💾 Conversation Log Path: {result['conversation_path']}")
        print(f"   • enhanced_research_findings.txt  (human-readable report)")
        print(f"   • enhanced_research_summary.json (complete metadata)")
        print(f"   • full_conversation.json (complete model exchange)")
        if DEBUG_MODE:
            print(f"   • research_events.jsonl  (tool calls, synthesis logs)")
    
    print(f"\n📋 FINAL RESEARCH REPORT:")
    print("-" * 40)
    findings = result.get('findings', 'No findings available')

    # Improved regex to avoid overlapping matches
    sections = re.split(
        r'(?m)^(### [A-Z ]+ ###)$',  # Matches section headers like "### EXECUTIVE SUMMARY ###"
        findings
    )
    
    if len(sections) > 1:
        # Skip the first empty element if present
        sections = [s.strip() for s in sections if s.strip()]
        
        # Iterate through sections in pairs (header, content)
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i]
                content = sections[i + 1]
                
                print(f"\n{header}")
                print_word_wrapped(content)  # Helper function to wrap text
    else:
        # Fallback for unstructured findings
        print_word_wrapped(findings)

    # Display any issues
    if not result.get('success'):
        reason = result.get('reason', 'unknown')
        print(f"\n⚠️  Research Status: {reason}")
        if reason == 'max_iterations_reached':
            print("   The research reached the maximum iteration limit.")
            print("   Results may be incomplete but still contain valuable information.")

def print_word_wrapped(text: str, line_length: int = 80):
    """Helper function to print text with word wrapping."""
    words = text.split()
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > line_length:
            print("   " + " ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        print("   " + " ".join(current_line))

def test_memory_synthesis(result: dict, original_task: str):
    """Test the S.A.G.E. memory synthesis component."""
    if not result.get('success'):
        print(f"\n❌ Skipping memory synthesis due to research failure")
        return
    
    print(f"\n🧠 S.A.G.E. Memory Synthesis")
    print("-" * 40)
    print("   Analyzing research findings...")
    print("   Creating optimized L3 memory structure...")
    
    if ENABLE_STREAMING:
        print("🔴 LIVE: S.A.G.E. memory synthesis (streaming enabled)")
    
    try:
        memory_result = synthesize_research_to_memory(result, original_task)
        
        if memory_result:
            print(f"✅ Memory synthesis successful!")
            print(f"   L3 Memory Node ID: {memory_result}")
            print("   Research findings have been integrated into F.R.E.D.'s knowledge graph.")
        else:
            print(f"❌ Memory synthesis failed")
            print("   Research findings were not stored in long-term memory.")
            
    except Exception as e:
        print(f"❌ Memory synthesis error: {e}")

def main():
    """Main test script execution."""
    print_banner()
    
    try:
        # Get research query from user
        query = get_user_input()
        
        # Generate unique task ID
        task_id = f"test_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Display progress info
        display_research_progress(task_id)
        
        # **ENABLE STREAMING FOR ALL AGENTS** 
        if ENABLE_STREAMING:
            print(f"\n🔴 INITIATING LIVE STREAMING MODE")
            print("=" * 60)
            print("All agent responses will appear in real-time...")
            print("=" * 60)
        
        # Conduct the research with streaming enabled
        result = conduct_enhanced_iterative_research(task_id, query, enable_streaming=ENABLE_STREAMING)
        
        # Display results
        display_results(result, query)
        
        # Test memory synthesis
        print(f"\n🤔 Would you like to synthesize these findings into F.R.E.D.'s memory? (y/n): ", end="")
        if input().lower().startswith('y'):
            test_memory_synthesis(result, query)
        else:
            print("   Skipping memory synthesis.")
        
        # Final summary
        print(f"\n✨ Research Session Complete!")
        print(f"   Query: {query}")
        print(f"   Status: {'Success' if result.get('success') else 'Partial'}")
        
        if ENABLE_STREAMING:
            print(f"   🔴 Streaming: All agent responses were displayed in real-time")
        
        if result.get('conversation_path'):
            print(f"\n📁 All artifacts for this session are located in:")
            print(f"   {result['conversation_path']}")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Research interrupted by user.")
        print("   Any partial results have been saved.")
        
    except Exception as e:
        print(f"\n❌ Test script error: {e}")
        print("   Check F.R.E.D. configuration and dependencies.")
    
    finally:
        print(f"\n👋 Research test session ended.")

if __name__ == "__main__":
    main() 