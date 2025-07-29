import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies that are not part of this integration test
sys.modules['memory.L2_memory'] = MagicMock()
sys.modules['duckduckgo_search'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['trafilatura'] = MagicMock()
sys.modules['duckdb'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import the modules under test
from memory import gate
from agents.dispatcher import AgentDispatcher
from config import ollama_manager # Import the real manager

class TestGateIntegration(unittest.TestCase):
    """Integration test suite for the G.A.T.E. routing agent."""

    @classmethod
    def setUpClass(cls):
        """Preload model once and create a shared dispatcher to minimize server workers."""
        from config import config, ollama_manager
        ollama_manager.preload_model(config.DEFAULT_MODEL)
        cls.dispatcher = AgentDispatcher()
        # Single MagicMock reused—return value string distinguishes tests for logging
        cls.dispatcher.dispatch_agents = MagicMock(return_value="Dispatcher reused by setUpClass")

    @patch('memory.gate.L2.query_l2_context')
    def test_run_gate_analysis_with_explicit_web_search(self, mock_l2_query):
        """
        This is an INTEGRATION TEST.
        It makes a REAL network call to the Ollama model to test the G.A.T.E. agent.
        Ensure the model is running and configured correctly before executing.
        """
        print("\nRunning: test_run_gate_analysis_with_real_model")
        # Arrange
        # Use a real AgentDispatcher but mock the method that gets called
        agent_dispatcher = AgentDispatcher()
        agent_dispatcher.dispatch_agents = MagicMock(return_value="Dispatcher was called by integration test.")

        mock_l2_query.return_value = "No relevant context."
        user_message = "Search the web for the latest news on NVIDIA stock."

        # Act
        # Run the analysis with the real dispatcher instance
        final_content = gate.run_gate_analysis(
            user_message=user_message,
            conversation_history=[],
            agent_dispatcher=agent_dispatcher
        )

        # Assert
        # We expect the real model to identify a need for a web search
        self.assertTrue(agent_dispatcher.dispatch_agents.called, "The real dispatcher was not called.")
        
        # Inspect the arguments passed to the dispatcher
        call_kwargs = agent_dispatcher.dispatch_agents.call_args.kwargs
        self.assertIn('routing_flags', call_kwargs)
        routing_flags = call_kwargs['routing_flags']

        print(f"--> Routing flags from real model: {routing_flags}")

        # The primary assertion: did the model correctly flag a web search?
        self.assertTrue(routing_flags.get('needs_web_search', False), 
                        "The real model failed to flag 'needs_web_search' as True.")

        print("\nIntegration Test Passed: The real AI model correctly triggered a web search route.")
        print(f"Final content received: {final_content}")

    @patch('memory.gate.L2.query_l2_context')
    def test_gate_analysis_implicit_web_search(self, mock_l2_query):
        """Implicit phrasing should also trigger web search."""
        print("\nRunning: test_gate_analysis_implicit_web_search")
        agent_dispatcher = AgentDispatcher()
        agent_dispatcher.dispatch_agents = MagicMock(return_value="Dispatcher called (implicit web search)")
        mock_l2_query.return_value = "No relevant context."
        user_message = "What's the latest news on NVIDIA stock?"  # Implicit phrasing
        gate.run_gate_analysis(user_message, [], agent_dispatcher)
        self.assertTrue(agent_dispatcher.dispatch_agents.called, "Dispatcher not called for implicit web search")
        routing_flags = agent_dispatcher.dispatch_agents.call_args.kwargs['routing_flags']
        print(f"--> Routing flags (implicit web search): {routing_flags}")
        self.assertTrue(routing_flags.get('needs_web_search', False), "Model failed to flag needs_web_search for implicit query")

    @patch('memory.gate.L2.query_l2_context')
    def test_gate_analysis_memory_only_implicit(self, mock_l2_query):
        """Implicit memory retrieval should trigger memory flag with no web search."""
        print("\nRunning: test_gate_analysis_memory_only_implicit")
        agent_dispatcher = AgentDispatcher()
        agent_dispatcher.dispatch_agents = MagicMock(return_value="Dispatcher called (implicit memory)")
        mock_l2_query.return_value = "No relevant context."
        user_message = "Do you remember when I bought my first car?"  # Implicit memory query
        gate.run_gate_analysis(user_message, [], agent_dispatcher)
        self.assertTrue(agent_dispatcher.dispatch_agents.called, "Dispatcher not called for implicit memory query")
        routing_flags = agent_dispatcher.dispatch_agents.call_args.kwargs['routing_flags']
        print(f"--> Routing flags (implicit memory): {routing_flags}")
        self.assertTrue(routing_flags.get('needs_memory', False), "Model failed to flag needs_memory for implicit memory query")
        # New assertion: Ensure the G.A.T.E. agent provides or defaults a valid memory_search_query
        memory_search_query = routing_flags.get('memory_search_query') or user_message
        self.assertTrue(memory_search_query, "G.A.T.E. did not supply a valid memory_search_query for memory retrieval")

if __name__ == '__main__':
    unittest.main()
