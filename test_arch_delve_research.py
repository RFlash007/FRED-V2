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
from ollie_print import olliePrint, olliePrint_simple, olliePrint_concise, olliePrint_raw

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
    olliePrint_simple(banner)
    
    if ENABLE_STREAMING:
        olliePrint_concise("🔴 LIVE STREAMING MODE: Model responses will appear in real-time as they're generated")
        olliePrint_concise("💭 You'll see thinking, reasoning, and responses as they happen")
        olliePrint_concise("⚡ This provides full transparency into the AI research process")
        olliePrint_concise("-" * 80)
    
    if DEBUG_MODE:
        olliePrint_concise("🔍 DEBUG MODE: Detailed system information and diagnostics enabled")
        olliePrint_concise("⚙️  Tool calls, database operations, and state changes will be shown")
        olliePrint_concise("-" * 80)

def get_user_input():
    """Get research prompt from user with some guidance."""
    olliePrint_concise("\n🔬 Research Query Input")
    olliePrint_concise("=" * 50)
    olliePrint_concise("Enter your research question or topic. Examples:")
    olliePrint_concise("• 'Latest developments in quantum computing 2024'")
    olliePrint_concise("• 'Best practices for Python async programming'") 
    olliePrint_concise("• 'Current state of AI safety research'")
    olliePrint_concise("• 'How to implement WebRTC for real-time communication'")
    
    if DEBUG_MODE:
        olliePrint_concise("• 'simple test query' (for debugging)")
    
    if ENABLE_STREAMING:
        olliePrint_concise("\n🔴 Note: With streaming enabled, you'll see agents 'thinking' and responding live!")
    
    olliePrint_concise("")
    
    while True:
        query = input("📝 Research Query: ").strip()
        if query:
            return query
        olliePrint_concise("❌ Please enter a valid research query.")

def display_research_progress(task_id: str):
    """Display progress information."""
    olliePrint_concise(f"\n🚀 Initiating Research Session")
    olliePrint_concise(f"   Task ID: {task_id}")
    olliePrint_concise(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if ENABLE_STREAMING:
        olliePrint_concise(f"   🔴 Live Streaming: ENABLED")
        olliePrint_concise("   💭 You'll see real-time thinking from each agent:")
        olliePrint_concise("      • A.R.C.H. strategic planning and direction")
        olliePrint_concise("      • D.E.L.V.E. web searches and data extraction")
        olliePrint_concise("      • V.E.T. evidence verification and analysis")
        olliePrint_concise("      • S.A.G.E. final synthesis and memory integration")
    
    olliePrint_concise("\n📡 A.R.C.H./D.E.L.V.E. Research Pipeline Active...")
    olliePrint_concise("   • A.R.C.H. will direct the research strategy")
    olliePrint_concise("   • D.E.L.V.E. will execute comprehensive web searches")
    olliePrint_concise("   • V.E.T. will verify and triangulate evidence")
    olliePrint_concise("   • S.A.G.E. will synthesize findings into F.R.E.D.'s memory")
    
    if DEBUG_MODE:
        olliePrint_concise("\n🔍 DEBUG MODE ACTIVE:")
        olliePrint_concise("   • Model requests and responses will be shown")
        olliePrint_concise("   • Tool calls and results will be displayed")
        olliePrint_concise("   • System state changes will be logged")
        olliePrint_concise("   • Error details will be verbose")
    
    olliePrint_concise("\n⏳ Research in progress... (this may take several minutes)")
    olliePrint_concise("-" * 70)

def debug_print(message: str, level: str = "DEBUG"):
    """Print debug messages only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        olliePrint_concise(f"[{timestamp}] [{level}] {message}", level='debug')

def stream_print(content: str, agent_name: str = "MODEL", end_chunk: bool = False):
    """Print streaming content with agent identification."""
    if ENABLE_STREAMING:
        if end_chunk:
            olliePrint_concise(f"\n[{agent_name} COMPLETE] ✅")
            olliePrint_concise("-" * 60)
        else:
            # Streaming without newline
            olliePrint_raw(content, end='', flush=True)

def display_results(result: dict, original_task: str):
    """Display comprehensive research results."""
    olliePrint_concise("\n" + "=" * 70)
    olliePrint_concise("🎯 RESEARCH RESULTS")
    olliePrint_concise("=" * 70)
    
    olliePrint_concise(f"✅ Success: {result.get('success', False)}")
    olliePrint_concise(f"🔄 Iterations: {result.get('iterations', 0)}")
    olliePrint_concise(f"📊 Enhancement: {result.get('enhancement', 'N/A')}")
    
    if ENABLE_STREAMING:
        olliePrint_concise(f"🔴 Streaming: All model responses were shown in real-time")
    
    if DEBUG_MODE:
        olliePrint_concise(f"🔍 Debug Info:")
        olliePrint_concise(f"   • VET Reports: {result.get('vet_reports_count', 0)}")
        olliePrint_concise(f"   • Global Citations: {result.get('global_citations_count', 0)}")
        olliePrint_concise(f"   • Completion Reason: {result.get('reason', 'Unknown')}")
    
    if result.get('conversation_path'):
        olliePrint_concise(f"💾 Conversation Log Path: {result['conversation_path']}")
        olliePrint_concise(f"   • enhanced_research_findings.txt  (human-readable report)")
        olliePrint_concise(f"   • enhanced_research_summary.json (complete metadata)")
        olliePrint_concise(f"   • full_conversation.json (complete model exchange)")
        if DEBUG_MODE:
            olliePrint_concise(f"   • research_events.jsonl  (tool calls, synthesis logs)")
    
    olliePrint_concise(f"\n📋 FINAL RESEARCH REPORT:")
    olliePrint_concise("-" * 40)
    
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
                
                olliePrint_concise(f"\n{header}")
                print_word_wrapped(content)  # Helper function to wrap text
    else:
        # Fallback for unstructured findings
        print_word_wrapped(findings)

    # Display any issues
    if not result.get('success'):
        reason = result.get('reason', 'unknown')
        olliePrint_concise(f"\n⚠️  Research Status: {reason}")
        if reason == 'max_iterations_reached':
            olliePrint_concise("   The research reached the maximum iteration limit.")
            olliePrint_concise("   Results may be incomplete but still contain valuable information.")

def print_word_wrapped(text: str, line_length: int = 80):
    """Helper function to print text with word wrapping."""
    words = text.split()
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > line_length:
            olliePrint_concise("   " + " ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        olliePrint_concise("   " + " ".join(current_line))

def test_memory_synthesis(result: dict, original_task: str):
    """Test the S.A.G.E. memory synthesis component."""
    if not result.get('success'):
        olliePrint_concise(f"\n❌ Skipping memory synthesis due to research failure")
        return
    
    olliePrint_concise(f"\n🧠 S.A.G.E. Memory Synthesis")
    olliePrint_concise("-" * 40)
    olliePrint_concise("   Analyzing research findings...")
    olliePrint_concise("   Creating optimized L3 memory structure...")
    
    if ENABLE_STREAMING:
        olliePrint_concise("🔴 LIVE: S.A.G.E. memory synthesis (streaming enabled)")
    
    try:
        memory_result = synthesize_research_to_memory(result, original_task)
        
        if memory_result:
            olliePrint_concise(f"✅ Memory synthesis successful!")
            olliePrint_concise(f"   L3 Memory Node ID: {memory_result}")
            olliePrint_concise("   Research findings have been integrated into F.R.E.D.'s knowledge graph.")
        else:
            olliePrint_concise(f"❌ Memory synthesis failed")
            olliePrint_concise("   Research findings were not stored in long-term memory.")
            
    except Exception as e:
        olliePrint_concise(f"❌ Memory synthesis error: {e}")

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
            olliePrint_concise(f"\n🔴 INITIATING LIVE STREAMING MODE")
            olliePrint_concise("=" * 60)
            olliePrint_concise("All agent responses will appear in real-time...")
            olliePrint_concise("=" * 60)
        
        # Conduct the research with streaming enabled
        result = conduct_enhanced_iterative_research(task_id, query, enable_streaming=ENABLE_STREAMING)
        
        # Display results
        display_results(result, query)
        
        # Test memory synthesis
        olliePrint_raw(f"\n🤔 Would you like to synthesize these findings into F.R.E.D.'s memory? (y/n): ", end="")
        if input().lower().startswith('y'):
            test_memory_synthesis(result, query)
        else:
            olliePrint_concise("   Skipping memory synthesis.")
        
        # Final summary
        olliePrint_concise(f"\n✨ Research Session Complete!")
        olliePrint_concise(f"   Query: {query}")
        olliePrint_concise(f"   Status: {'Success' if result.get('success') else 'Partial'}")
        
        if ENABLE_STREAMING:
            olliePrint_concise(f"   🔴 Streaming: All agent responses were displayed in real-time")
        
        if result.get('conversation_path'):
            olliePrint_concise(f"\n📁 All artifacts for this session are located in:")
            olliePrint_concise(f"   {result['conversation_path']}")
        
    except KeyboardInterrupt:
        olliePrint_concise(f"\n\n⏹️  Research interrupted by user.")
        olliePrint_concise("   Any partial results have been saved.")
        
    except Exception as e:
        olliePrint_concise(f"\n❌ Test script error: {e}")
        olliePrint_concise("   Check F.R.E.D. configuration and dependencies.")
    
    finally:
        olliePrint_concise(f"\n👋 Research test session ended.")

if __name__ == "__main__":
    main()