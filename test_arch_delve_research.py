#!/usr/bin/env python3
"""
A.R.C.H./D.E.L.V.E. Research System Test Script
Simple console interface for testing the iterative research system
"""

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

from memory.arch_delve_research import conduct_iterative_research, synthesize_research_to_memory
from ollie_print import olliePrint

def print_banner():
    """Print the F.R.E.D. research system banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    F.R.E.D. A.R.C.H./D.E.L.V.E. Research Test               ║
║                                                                              ║
║  A.R.C.H. (Adaptive Research Command Hub) - Research Director               ║
║  D.E.L.V.E. (Data Extraction and Logical Verification Engine) - Analyst     ║
║  S.A.G.E. (Synthesis and Archive Generation Engine) - Memory Synthesis      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def get_user_input():
    """Get research prompt from user with some guidance."""
    print("\n🔬 Research Query Input")
    print("=" * 50)
    print("Enter your research question or topic. Examples:")
    print("• 'Latest developments in quantum computing 2024'")
    print("• 'Best practices for Python async programming'") 
    print("• 'Current state of AI safety research'")
    print("• 'How to implement WebRTC for real-time communication'")
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
    print("\n📡 A.R.C.H./D.E.L.V.E. Research Pipeline Active...")
    print("   • A.R.C.H. will direct the research strategy")
    print("   • D.E.L.V.E. will execute comprehensive web searches")
    print("   • S.A.G.E. will synthesize findings into F.R.E.D.'s memory")
    print("\n⏳ Research in progress... (this may take several minutes)")
    print("-" * 70)

def display_results(result: dict, original_task: str):
    """Display comprehensive research results."""
    print("\n" + "=" * 70)
    print("🎯 RESEARCH RESULTS")
    print("=" * 70)
    
    print(f"✅ Success: {result.get('success', False)}")
    print(f"🔄 Iterations: {result.get('iterations', 0)}")
    
    if result.get('conversation_path'):
        print(f"💾 Conversation Log: {result['conversation_path']}")
    
    print(f"\n📋 FINAL RESEARCH REPORT:")
    print("-" * 40)
    findings = result.get('findings', 'No findings available')

    # Try to parse the report sections for cleaner output
    sections = re.split(
        r'(Executive Summary|Methodology|Core Findings|Analysis|Conclusions|Sources):', 
        findings
    )
    
    if len(sections) > 1:
        # The first element is usually empty, so skip it.
        # Iterate in pairs of (header, content)
        for i in range(1, len(sections), 2):
            header = sections[i]
            content = sections[i+1].strip()
            
            print(f"\n\n### {header.upper()} ###")
            
            # Word wrap the content for readability
            words = content.split(' ')
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > 80:  # 80 char line limit
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                print(f"   {line}")
    else:
        # Fallback for unstructured findings
        print(f"   {findings}")
    
    # Display any issues
    if not result.get('success'):
        reason = result.get('reason', 'unknown')
        print(f"\n⚠️  Research Status: {reason}")
        if reason == 'max_iterations_reached':
            print("   The research reached the maximum iteration limit.")
            print("   Results may be incomplete but still contain valuable information.")

def test_memory_synthesis(result: dict, original_task: str):
    """Test the S.A.G.E. memory synthesis component."""
    if not result.get('success'):
        print(f"\n❌ Skipping memory synthesis due to research failure")
        return
    
    print(f"\n🧠 S.A.G.E. Memory Synthesis")
    print("-" * 40)
    print("   Analyzing research findings...")
    print("   Creating optimized L3 memory structure...")
    
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

def save_complete_research_log(task_id: str, original_task: str, result: dict):
    """Save complete research findings to agenda_conversations directory for test script."""
    try:
        # Create agenda_conversations directory if it doesn't exist
        agenda_dir = Path("memory/agenda_conversations")
        agenda_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test-specific subdirectory
        test_dir = agenda_dir / task_id
        test_dir.mkdir(exist_ok=True)
        
        # Prepare complete research log
        complete_log = {
            "test_session": True,
            "task_id": task_id,
            "original_task": original_task,
            "timestamp": datetime.now().isoformat(),
            "research_result": result,
            "complete_research_findings": result.get('findings', ''),
            "success": result.get('success', False),
            "iterations": result.get('iterations', 0),
            "conversation_path": result.get('conversation_path', ''),
            "test_metadata": {
                "script_version": "test_arch_delve_research.py",
                "session_type": "manual_test",
                "findings_length": len(result.get('findings', '')),
                "research_complete": True
            }
        }
        
        # Save complete research log
        log_file = test_dir / "complete_research_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(complete_log, f, indent=2, ensure_ascii=False)
        
        # Also save just the findings in a readable format
        findings_file = test_dir / "research_findings.txt"
        with open(findings_file, 'w', encoding='utf-8') as f:
            f.write(f"Research Task: {original_task}\n")
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Success: {result.get('success', False)}\n")
            f.write(f"Iterations: {result.get('iterations', 0)}\n")
            f.write("=" * 80 + "\n")
            f.write("COMPLETE RESEARCH FINDINGS:\n")
            f.write("=" * 80 + "\n")
            f.write(result.get('findings', 'No findings available'))
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\n📁 Complete research log saved:")
        print(f"   JSON: {log_file}")
        print(f"   Text: {findings_file}")
        
        return str(test_dir)
        
    except Exception as e:
        print(f"❌ Failed to save complete research log: {e}")
        return None

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
        
        # Conduct the research
        result = conduct_iterative_research(task_id, query)
        
        # Display results
        display_results(result, query)
        
        # Save complete research log to agenda_conversations
        save_complete_research_log(task_id, query, result)
        
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
        print(f"   Duration: Check conversation logs for detailed timeline")
        
        if result.get('conversation_path'):
            print(f"\n📁 Full conversation logs available at:")
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