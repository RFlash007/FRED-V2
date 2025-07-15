#!/usr/bin/env python3
"""
Test script for the new FRED multi-agent architecture
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import gate
from agents.remind import RemindAgent
from agents.scout import ScoutAgent
from agents.pivot import PivotAgent
from agents.synapse import SynapseAgent

def test_gate_routing():
    """Test G.A.T.E. routing system"""
    print("=== Testing G.A.T.E. Routing ===")
    
    test_queries = [
        "What did we talk about yesterday?",
        "What's the weather today?", 
        "Remind me to call mom tomorrow",
        "This is John, my friend"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = gate.run_gate_analysis(query, [])
            print(f"Result: {result[:200]}...")
        except Exception as e:
            print(f"Error: {e}")

def test_reminder_agent():
    """Test R.E.M.I.N.D. agent"""
    print("\n=== Testing R.E.M.I.N.D. Agent ===")
    
    remind_agent = RemindAgent()
    
    test_messages = [
        "Remind me to call mom tomorrow",
        "Don't forget to buy groceries",
        "Thanks, got it!"
    ]
    
    for message in test_messages:
        print(f"\nMessage: {message}")
        try:
            result = remind_agent.process_conversation_turn(message)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

def test_scout_agent():
    """Test S.C.O.U.T. agent"""
    print("\n=== Testing S.C.O.U.T. Agent ===")
    
    scout_agent = ScoutAgent()
    
    test_query = "What's the current weather?"
    print(f"\nQuery: {test_query}")
    try:
        result = scout_agent.search_and_assess(test_query)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

def test_synapse_agent():
    """Test S.Y.N.A.P.S.E. agent"""
    print("\n=== Testing S.Y.N.A.P.S.E. Agent ===")
    
    synapse_agent = SynapseAgent()
    
    mock_agent_outputs = {
        "crap": {"memories": "Previous conversation about weather"},
        "scout": {"search_result": "Current weather is sunny", "confidence_score": 85},
        "remind": {"detected_reminders": [], "active_reminders": []}
    }
    
    try:
        result = synapse_agent.synthesize_context(
            agent_outputs=mock_agent_outputs,
            l2_summaries=[{"summary": "Recent weather discussion"}],
            user_query="What's the weather like?",
            visual_context=""
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("FRED Multi-Agent Architecture Test")
    print("=" * 40)
    
    try:
        test_gate_routing()
        test_reminder_agent()
        test_scout_agent()
        test_synapse_agent()
        print("\n=== All Tests Completed ===")
    except Exception as e:
        print(f"Test suite error: {e}")
