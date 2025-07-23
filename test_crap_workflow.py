"""
CRAP Implicit Memory Workflow Test Suite
========================================

This test script validates CRAP's ability to automatically manage FRED's memory
behind the scenes. In FRED's architecture, memory operations are completely
implicit and agentic:

- User: "My favorite color is purple"
- FRED: "That's a nice choice!" (responds naturally)
- CRAP: (silently stores: User prefers purple colors)

- User: "What's my favorite color?"
- CRAP: (silently retrieves and provides context)
- FRED: "Your favorite color is purple" (appears to remember naturally)

This tests whether CRAP can autonomously:
1. Detect when information should be stored (without being told)
2. Retrieve relevant context when needed (proactively)
3. Update memories when corrections occur (seamlessly)
4. Maintain FRED's illusion of natural memory

FRED remains completely unaware that CRAP is managing his knowledge.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import CRAP and related modules
try:
    from memory import crap
    from config import config
    from ollie_print import olliePrint_simple
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the FRED-V2 directory")
    sys.exit(1)

class CrapImplicitScenario:
    """Represents an implicit memory test scenario for CRAP analysis."""
    
    def __init__(self, name: str, user_message: str, conversation_history: List[Dict], 
                 expected_memory_action: str, description: str, 
                 fred_context_needed: bool = False):
        self.name = name
        self.user_message = user_message
        self.conversation_history = conversation_history
        self.expected_memory_action = expected_memory_action  # "STORE", "RETRIEVE", "UPDATE", "ANALYZE", "NONE"
        self.description = description
        self.fred_context_needed = fred_context_needed  # Does FRED need memory context to respond naturally?

class CrapWorkflowTester:
    """Test suite for CRAP memory management workflow."""
    
    def __init__(self):
        self.scenarios = self._create_test_scenarios()
        self.results = []
    
    def _create_test_scenarios(self) -> List[CrapImplicitScenario]:
        """Create realistic implicit memory scenarios reflecting FRED's natural conversation flow."""
        
        scenarios = [
            # Test 1: User Shares Preference (CRAP should silently store)
            CrapImplicitScenario(
                name="User States Preference",
                user_message="I really like dark purple themes for my apps",
                conversation_history=[
                    {"role": "user", "content": "I really like dark purple themes for my apps"},
                    {"role": "assistant", "content": "Dark purple is a great choice for UI themes! It's easy on the eyes and looks professional."}
                ],
                expected_memory_action="STORE",
                description="CRAP should silently detect and store user's theme preference without FRED being aware",
                fred_context_needed=False
            ),
            
            # Test 2: User Asks About Previous Topic (CRAP should retrieve context)
            CrapImplicitScenario(
                name="Recall Previous Discussion",
                user_message="What was that Python library we discussed last week?",
                conversation_history=[
                    {"role": "user", "content": "What was that Python library we discussed last week?"},
                    {"role": "assistant", "content": "You were asking about FastAPI! We talked about how it's great for building REST APIs with automatic documentation."}
                ],
                expected_memory_action="RETRIEVE",
                description="CRAP should search memory and provide context so FRED can respond naturally about FastAPI",
                fred_context_needed=True
            ),
            
            # Test 3: User Corrects Information (CRAP should seamlessly update)
            CrapImplicitScenario(
                name="Natural Information Correction",
                user_message="Actually, I was wrong earlier - my favorite color is blue, not green",
                conversation_history=[
                    {"role": "user", "content": "Actually, I was wrong earlier - my favorite color is blue, not green"},
                    {"role": "assistant", "content": "Got it! Blue is a wonderful color choice."}
                ],
                expected_memory_action="UPDATE",
                description="CRAP should find existing color preference and update it without FRED knowing",
                fred_context_needed=False
            ),
            
            # Test 4: User Shares Process Casually (CRAP should store procedure)
            CrapImplicitScenario(
                name="Casual Process Sharing",
                user_message="Here's how I deploy my React apps: npm build, upload to Vercel, then update DNS",
                conversation_history=[
                    {"role": "user", "content": "Here's how I deploy my React apps: npm build, upload to Vercel, then update DNS"},
                    {"role": "assistant", "content": "That's a solid deployment workflow! Vercel makes React deployments really smooth."}
                ],
                expected_memory_action="STORE",
                description="CRAP should silently store user's deployment procedure for future reference",
                fred_context_needed=False
            ),
            
            # Test 5: User Shares Complex Research (CRAP should analyze and store)
            CrapImplicitScenario(
                name="Complex Research Findings",
                user_message="I finished researching quantum computing applications. The key breakthrough is that quantum algorithms can solve certain optimization problems exponentially faster than classical computers, particularly in cryptography and drug discovery",
                conversation_history=[
                    {"role": "user", "content": "I finished researching quantum computing applications. The key breakthrough is that quantum algorithms can solve certain optimization problems exponentially faster than classical computers, particularly in cryptography and drug discovery"},
                    {"role": "assistant", "content": "That's fascinating research! The implications for both cryptography and drug discovery are significant."}
                ],
                expected_memory_action="ANALYZE",
                description="CRAP should analyze this complex research finding, potentially discover relationships, then store structured memory",
                fred_context_needed=False
            ),
            
            # Test 6: Asking About Relationships (Should use advanced tools)
            CrapImplicitScenario(
                name="Relationship Query",
                user_message="How does my Python work connect to my AI projects?",
                conversation_history=[
                    {"role": "user", "content": "How does my Python work connect to my AI projects?"},
                    {"role": "assistant", "content": "Let me analyze the connections between your Python and AI work..."}
                ],
                expected_memory_action="ANALYZE",
                description="CRAP should search for related memories and analyze relationships",
                fred_context_needed=True
            ),
            
            # Test 7: User Shares Major Life Event (CRAP should store episodic memory)
            CrapImplicitScenario(
                name="Major Life Event",
                user_message="I got promoted to Senior Developer at my company today! Starting January 15th, 2025",
                conversation_history=[
                    {"role": "user", "content": "I got promoted to Senior Developer at my company today! Starting January 15th, 2025"},
                    {"role": "assistant", "content": "Congratulations on your promotion! That's excellent news."}
                ],
                expected_memory_action="STORE",
                description="CRAP should silently store this significant episodic event with date context",
                fred_context_needed=False
            ),
            
            # Test 8: User Asks About Previous Learning (CRAP should retrieve context)
            CrapImplicitScenario(
                name="Learning History Query",
                user_message="What did I learn about React hooks in our previous sessions?",
                conversation_history=[
                    {"role": "user", "content": "What did I learn about React hooks in our previous sessions?"},
                    {"role": "assistant", "content": "We covered useState, useEffect, and custom hooks! You were particularly interested in how useEffect handles cleanup functions."}
                ],
                expected_memory_action="RETRIEVE",
                description="CRAP should search for and provide context about React hooks learning history",
                fred_context_needed=True
            ),
            
            # Test 9: User Casually Updates Major Preference (CRAP should seamlessly update)
            CrapImplicitScenario(
                name="Major Tool Switch",
                user_message="Remember when I said I use VS Code? Well, I switched to Neovim last month and I'm loving it",
                conversation_history=[
                    {"role": "user", "content": "Remember when I said I use VS Code? Well, I switched to Neovim last month and I'm loving it"},
                    {"role": "assistant", "content": "That's a significant change in your development setup! Neovim has some great features for power users."}
                ],
                expected_memory_action="UPDATE",
                description="CRAP should find old editor preference and seamlessly update it without FRED knowing",
                fred_context_needed=False
            ),
            
            # Test 10: General Question (CRAP should recognize no memory action needed)
            CrapImplicitScenario(
                name="General Question",
                user_message="What's the weather like today?",
                conversation_history=[
                    {"role": "user", "content": "What's the weather like today?"},
                    {"role": "assistant", "content": "I don't have access to current weather information, but you could check your local weather app!"}
                ],
                expected_memory_action="NONE",
                description="CRAP should recognize this general question doesn't require memory management",
                fred_context_needed=False
            )
        ]
        
        return scenarios
    
    def run_test(self, scenario: CrapImplicitScenario) -> Dict[str, Any]:
        """Run a single test scenario and analyze CRAP's implicit memory management."""
        
        print(f"\n{'='*60}")
        print(f"🧪 IMPLICIT MEMORY TEST: {scenario.name}")
        print(f"📝 Description: {scenario.description}")
        print(f"💬 User Message: {scenario.user_message}")
        print(f"🎯 Expected Memory Action: {scenario.expected_memory_action}")
        print(f"🧠 FRED Context Needed: {'Yes' if scenario.fred_context_needed else 'No'}")
        print(f"{'='*60}")
        
        try:
            # --- Intercept real tool calls by monkeypatching handle_tool_calls ---
            olliePrint_simple(f"[TEST] Running REAL CRAP analysis for: {scenario.name}")
        
            import importlib
            crap_module = importlib.import_module('memory.crap')
            recorded_tools: list[str] = []
            original_handle = getattr(crap_module, 'handle_tool_calls', None)
            
            def _dummy_handle(tool_calls):
                """Record tool call names without executing underlying logic."""
                nonlocal recorded_tools
                for tc in tool_calls or []:
                    name = tc.get('function', {}).get('name', 'unknown')
                    recorded_tools.append(name)
                # Return empty list to mimic no additional assistant messages
                return []
            
            # Patch
            if original_handle is not None:
                crap_module.handle_tool_calls = _dummy_handle  # type: ignore
            
            try:
                # Call CRAP's actual analysis function
                crap_response = crap_module.run_crap_analysis(scenario.user_message, scenario.conversation_history)
            finally:
                # Restore original handler to avoid side-effects
                if original_handle is not None:
                    crap_module.handle_tool_calls = original_handle  # type: ignore
            
            # Determine which tools were invoked: prefer recorded_tools, fallback to keyword extraction
            actual_tools_used = recorded_tools if recorded_tools else self._extract_tools_from_crap_response(crap_response)
            
            # Analyze what CRAP actually did vs what we expected
            analysis = self._analyze_crap_behavior(scenario, actual_tools_used, crap_response)
            
            result = {
                "scenario_name": scenario.name,
                "user_message": scenario.user_message,
                "expected_memory_action": scenario.expected_memory_action,
                "fred_context_needed": scenario.fred_context_needed,
                "actual_crap_response": crap_response,
                "actual_tools_used": actual_tools_used,
                "analysis": analysis,
                "test_status": analysis.get("test_result", "UNKNOWN"),
                "notes": "Real CRAP test - actual tool calls analyzed"
            }
            
        except Exception as e:
            olliePrint_simple(f"❌ Test failed with error: {e}", level='error')
            result = {
                "scenario_name": scenario.name,
                "test_status": "ERROR",
                "error": str(e)
            }
        
        return result
    
    def _extract_tools_from_crap_response(self, crap_response: str) -> List[str]:
        """Extract tool calls from CRAP's response text."""
        
        tools_used = []
        
        # Look for common tool patterns in CRAP's response
        # Note: This assumes CRAP returns structured memory context with tool info
        response_lower = crap_response.lower()
        
        # Check for memory operations mentioned in response
        # Detect potential add_memory actions using a broader set of verbs
        if any(k in response_lower for k in [
            "stored", "store", "saved", "remembered", "recorded", "add", "added", "captured", "ingested"
        ]):
            tools_used.append("add_memory")
        
        # Detect search_memory usage with broader keywords and explicit tool mention
        if any(k in response_lower for k in [
            "retrieved", "found", "searched", "search_memory", "lookup", "queried"
        ]):
            tools_used.append("search_memory")
            
        # Detect supersede_memory (updates) using wider range of verbs
        if any(k in response_lower for k in [
            "updated", "update", "superseded", "supersede", "corrected", "replaced", "overwrote"
        ]):
            tools_used.append("supersede_memory")
            
        if "relationships" in response_lower or "connections" in response_lower:
            tools_used.append("discover_relationships_advanced")
            
        if "subgraph" in response_lower or "network" in response_lower:
            tools_used.append("get_subgraph")
            
        if "observations" in response_lower or "structured" in response_lower:
            tools_used.append("add_memory_with_observations")
        
        return tools_used
    
    def _analyze_crap_behavior(self, scenario: CrapImplicitScenario, 
                               actual_tools: List[str], 
                               crap_response: str) -> Dict[str, Any]:
        """Analyze CRAP's actual behavior vs expected implicit memory action."""
        
        analysis = {
            "expected_action": scenario.expected_memory_action,
            "actual_tools_detected": actual_tools,
            "crap_response_length": len(crap_response),
            "behavior_assessment": "",
            "test_result": "UNKNOWN",
            "detailed_analysis": "",
            "memory_context_provided": "(MEMORY CONTEXT)" in crap_response
        }
        
        # Analyze based on expected memory action
        if scenario.expected_memory_action == "STORE":
            if any(tool in actual_tools for tool in ["add_memory", "add_memory_with_observations"]):
                analysis["test_result"] = "PASSED"
                analysis["behavior_assessment"] = "CRAP correctly identified information to store"
            else:
                analysis["test_result"] = "FAILED"
                analysis["behavior_assessment"] = "CRAP missed information that should be stored"
                
        elif scenario.expected_memory_action == "RETRIEVE":
            if "search_memory" in actual_tools or analysis["memory_context_provided"]:
                analysis["test_result"] = "PASSED"
                analysis["behavior_assessment"] = "CRAP correctly retrieved relevant memory context"
            else:
                analysis["test_result"] = "FAILED"
                analysis["behavior_assessment"] = "CRAP failed to retrieve expected memory context"
                
        elif scenario.expected_memory_action == "UPDATE":
            if "supersede_memory" in actual_tools or ("search_memory" in actual_tools and "add_memory" in actual_tools):
                analysis["test_result"] = "PASSED"
                analysis["behavior_assessment"] = "CRAP correctly updated existing memory"
            else:
                analysis["test_result"] = "FAILED"
                analysis["behavior_assessment"] = "CRAP failed to update existing information"
                
        elif scenario.expected_memory_action == "ANALYZE":
            if any(tool in actual_tools for tool in ["discover_relationships_advanced", "get_subgraph", "add_memory_with_observations"]):
                analysis["test_result"] = "PASSED"
                analysis["behavior_assessment"] = "CRAP correctly used advanced analysis tools"
            else:
                analysis["test_result"] = "FAILED"
                analysis["behavior_assessment"] = "CRAP failed to use advanced analysis for complex scenario"
                
        elif scenario.expected_memory_action == "NONE":
            if len(actual_tools) == 0 and not analysis["memory_context_provided"]:
                analysis["test_result"] = "PASSED"
                analysis["behavior_assessment"] = "CRAP correctly identified no memory action needed"
            else:
                analysis["test_result"] = "FAILED"
                analysis["behavior_assessment"] = "CRAP unnecessarily performed memory operations"
        
        # Add detailed analysis
        analysis["detailed_analysis"] = f"""
Scenario: {scenario.name}
Expected: {scenario.expected_memory_action}
Tools Used: {actual_tools}
FRED Context Needed: {scenario.fred_context_needed}
Memory Context Provided: {analysis['memory_context_provided']}
Response Length: {analysis['crap_response_length']} chars
"""
        
        return analysis
    
    def _analyze_scenario(self, scenario: CrapImplicitScenario) -> Dict[str, Any]:
        """Analyze CRAP's expected implicit memory behavior for this scenario."""
        
        analysis = {
            "conversation_pattern": "",
            "implicit_reasoning": "",
            "memory_action_needed": "",
            "context_for_fred": "",
            "behind_scenes_operation": ""
        }
        
        message = scenario.user_message.lower()
        
        # Analyze conversation pattern and implicit memory behavior
        if "i like" in message or "my favorite" in message or "i prefer" in message:
            analysis["conversation_pattern"] = "User Shares Personal Preference"
            analysis["implicit_reasoning"] = "User casually mentions preference, CRAP should silently store for future reference"
            analysis["memory_action_needed"] = "STORE"
            analysis["context_for_fred"] = "No context needed - FRED responds naturally"
            analysis["behind_scenes_operation"] = "add_memory() with Semantic type for preference"
            
        elif "actually" in message and ("not" in message or "wrong" in message):
            analysis["conversation_pattern"] = "User Corrects Previous Information"
            analysis["implicit_reasoning"] = "User naturally corrects mistake, CRAP should find and update old memory"
            analysis["memory_action_needed"] = "UPDATE"
            analysis["context_for_fred"] = "No context needed - FRED acknowledges naturally"
            analysis["behind_scenes_operation"] = "search_memory() + supersede_memory() seamlessly"
            
        elif any(word in message for word in ["what was", "what did", "remember when", "we discussed"]):
            analysis["conversation_pattern"] = "User Asks About Past Topic"
            analysis["implicit_reasoning"] = "User expects FRED to remember, CRAP should provide relevant context"
            analysis["memory_action_needed"] = "RETRIEVE"
            analysis["context_for_fred"] = "Memory context provided so FRED can respond naturally"
            analysis["behind_scenes_operation"] = "search_memory() to find relevant past conversations"
            
        elif "here's how" in message or "steps" in message or "process" in message:
            analysis["conversation_pattern"] = "User Casually Shares Process"
            analysis["implicit_reasoning"] = "User mentions workflow naturally, CRAP should store procedure for later"
            analysis["memory_action_needed"] = "STORE"
            analysis["context_for_fred"] = "No context needed - FRED responds about the process"
            analysis["behind_scenes_operation"] = "add_memory() with Procedural type for workflow"
            
        elif "research" in message or "finding" in message or len(scenario.user_message) > 100:
            analysis["conversation_pattern"] = "User Shares Complex Research"
            analysis["implicit_reasoning"] = "Complex finding requires advanced storage and relationship analysis"
            analysis["memory_action_needed"] = "ANALYZE"
            analysis["context_for_fred"] = "No context needed - FRED discusses research naturally"
            analysis["behind_scenes_operation"] = "add_memory_with_observations() + discover_relationships_advanced()"
            
        elif "promoted" in message or "got" in message or "started" in message:
            analysis["conversation_pattern"] = "User Shares Life Event"
            analysis["implicit_reasoning"] = "Significant personal event should be stored with date context"
            analysis["memory_action_needed"] = "STORE"
            analysis["context_for_fred"] = "No context needed - FRED congratulates naturally"
            analysis["behind_scenes_operation"] = "add_memory() with Episodic type and date metadata"
            
        elif "how does" in message and "connect" in message:
            analysis["conversation_pattern"] = "User Asks About Relationships"
            analysis["implicit_reasoning"] = "User wants connection analysis, CRAP should discover relationships"
            analysis["memory_action_needed"] = "ANALYZE"
            analysis["context_for_fred"] = "Relationship context provided for natural response"
            analysis["behind_scenes_operation"] = "search_memory() + get_subgraph() for connection analysis"
            
        elif "weather" in message or "what time" in message:
            analysis["conversation_pattern"] = "General Information Query"
            analysis["implicit_reasoning"] = "Generic question doesn't involve personal memory or learning"
            analysis["memory_action_needed"] = "NONE"
            analysis["context_for_fred"] = "No memory context needed"
            analysis["behind_scenes_operation"] = "No memory operations required"
            
        else:
            analysis["conversation_pattern"] = "Unrecognized Pattern"
            analysis["implicit_reasoning"] = "Pattern unclear, default to search for safety"
            analysis["memory_action_needed"] = "RETRIEVE"
            analysis["context_for_fred"] = "Search for any relevant context"
            analysis["behind_scenes_operation"] = "search_memory() as safe default"
        
        return analysis
    
    def _print_single_result(self, result: Dict[str, Any]):
        """Prints the detailed analysis of a single test result."""
        if result.get("analysis"):
            analysis = result["analysis"]
            print(f"\n🧪 Scenario: {result['scenario_name']}")
            print(f"   - Expected Action: {result['expected_memory_action']}")
            print(f"   - Tools Used: {result.get('actual_tools_used', [])}")
            
            test_result = analysis.get('test_result', 'UNKNOWN')
            if test_result == 'PASSED':
                print(f"   - ✅ Result: PASSED")
            elif test_result == 'FAILED':
                print(f"   - ❌ Result: FAILED")
            else:
                print(f"   - ❓ Result: UNKNOWN")
                
            if test_result == 'FAILED' and analysis.get('detailed_analysis'):
                print(f"     - 🔍 Details: {analysis['detailed_analysis'].strip()}")
        
        elif result.get('test_status') == 'ERROR':
            print(f"\n❌ Scenario: {result['scenario_name']} - ERROR")
            print(f"   - Error: {result.get('error', 'Unknown error')}")

    def run_all_tests(self):
        """Run all test scenarios and print results as they complete."""
        results = []
        
        for i, scenario in enumerate(self.scenarios):
            print(f"\n{'='*80}")
            print(f"--- Running Test {i+1}/{len(self.scenarios)}: {scenario.name} ---")
            result = self.run_test(scenario)
            self._print_single_result(result)
            results.append(result)
            
        self._print_summary(results) # Keep the final summary
        self.results = results
        return results
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print test summary."""
        
        print(f"\n{'='*80}")
        print("📊 TEST SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(results)
        simulated_tests = sum(1 for r in results if r.get("test_status") == "SIMULATED")
        error_tests = sum(1 for r in results if r.get("test_status") == "ERROR")
        
        print(f"Total Tests: {total_tests}")
        print(f"Simulated Tests: {simulated_tests}")
        print(f"Error Tests: {error_tests}")
        
        print(f"\n{'='*60}")
        print("📋 DETAILED ANALYSIS")
        print(f"{'='*60}")
        
        # Count test results
        passed = failed = error = 0
        
        for result in results:
            if result.get("analysis"):
                analysis = result["analysis"]
                print(f"\n🧪 {result['scenario_name']}")
                print(f"   Expected Action: {result['expected_memory_action']}")
                print(f"   Tools Used: {result.get('actual_tools_used', [])}")
                print(f"   FRED Context Needed: {'Yes' if result['fred_context_needed'] else 'No'}")
                print(f"   Memory Context Provided: {'Yes' if analysis.get('memory_context_provided', False) else 'No'}")
                print(f"   Assessment: {analysis.get('behavior_assessment', 'Unknown')}")
                
                # Show test result with appropriate emoji
                test_result = analysis.get('test_result', 'UNKNOWN')
                if test_result == 'PASSED':
                    print(f"   ✅ TEST PASSED")
                    passed += 1
                elif test_result == 'FAILED':
                    print(f"   ❌ TEST FAILED")
                    failed += 1
                else:
                    print(f"   ❓ TEST RESULT UNKNOWN")
                    
                # Show detailed analysis for failed tests
                if test_result == 'FAILED' and analysis.get('detailed_analysis'):
                    print(f"   🔍 Details: {analysis['detailed_analysis'].strip()}")
            
            elif result.get('test_status') == 'ERROR':
                print(f"\n❌ {result['scenario_name']} - ERROR")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                error += 1
        
        # Print summary statistics
        total = passed + failed + error
        print(f"\n{'='*60}")
        print(f"📊 TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} (✅)")
        print(f"Failed: {failed} (❌)")
        print(f"Errors: {error} (⚠️)")
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\n{'='*80}")
        print("💡 CRAP TESTING NOTES")
        print(f"{'='*80}")
        print("This is a REAL CRAP testing suite that:")
        print("✅ Makes actual calls to CRAP's run_crap_analysis function")
        print("✅ Analyzes CRAP's real memory management decisions")
        print("✅ Validates implicit memory behavior (STORE, RETRIEVE, UPDATE, ANALYZE, NONE)")
        print("✅ Tests CRAP's ability to operate transparently behind FRED")
        print("⚠️  Note: Tool call detection is based on response text analysis")
        print("⚠️  For deeper validation, consider adding tool call interception")
        
    def save_results(self, results: List[Dict[str, Any]], filename: str = "crap_test_results.json"):
        """Save test results to JSON file."""
        
        output_path = os.path.join(os.path.dirname(__file__), filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test results saved to: {output_path}")

def main():
    """Main test function."""
    
    print("🧠 CRAP Memory Management Workflow Test Suite")
    print("=" * 50)
    print("This script tests CRAP's decision-making for L3 memory management")
    print("across various conversation scenarios.\n")
    
    # Initialize tester
    tester = CrapWorkflowTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Save results
    tester.save_results(results)
    
    print(f"\n🏁 Testing complete! Check results for CRAP's decision-making patterns.")

if __name__ == "__main__":
    main()
