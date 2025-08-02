import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock external dependencies before any other imports
sys.modules['duckduckgo_search'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['trafilatura'] = MagicMock()
sys.modules['duckdb'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['memory.L2_memory'] = MagicMock()

# Now, import the modules to be tested
from memory import gate
from agents.dispatcher import AgentDispatcher

class TestGateAgent(unittest.TestCase):
    """Test suite for the G.A.T.E. routing agent using dependency injection."""

    @patch('memory.gate.L2.query_l2_context')
    @patch('memory.gate.ollama_manager.chat_concurrent_safe')
    def test_route_to_memory_only(self, mock_ollama, mock_l2_query):
        """Tests routing to memory by passing a mock dispatcher."""
        print("\nRunning: test_route_to_memory_only")
        # Arrange
        mock_dispatcher = MagicMock(spec=AgentDispatcher)
        mock_dispatcher.dispatch_agents.return_value = "Dispatcher processed memory task."
        expected_flags = {
            "needs_memory": True,
            "web_search_strategy": {
                "needed": False,
                "search_priority": "quick",
                "search_query": ""
            }
        }
        mock_ollama.return_value = {'message': {'content': json.dumps(expected_flags)}}
        mock_l2_query.return_value = "No relevant context."

        # Act
        gate.run_gate_analysis("Do you remember...", [], mock_dispatcher)

        # Assert
        self.assertTrue(mock_dispatcher.dispatch_agents.called, "Dispatcher was not called.")
        call_kwargs = mock_dispatcher.dispatch_agents.call_args.kwargs
        self.assertIn('routing_flags', call_kwargs)
        self.assertEqual(call_kwargs['routing_flags']['needs_memory'], True)
        self.assertEqual(
            call_kwargs['routing_flags']['web_search_strategy']['needed'],
            False
        )
        print("PASSED: Correctly routed to memory agent.")

    @patch('memory.gate.L2.query_l2_context')
    @patch('memory.gate.ollama_manager.chat_concurrent_safe')
    def test_route_to_multiple_agents(self, mock_ollama, mock_l2_query):
        """Tests routing to multiple agents by passing a mock dispatcher."""
        print("\nRunning: test_route_to_multiple_agents")
        # Arrange
        mock_dispatcher = MagicMock(spec=AgentDispatcher)
        mock_dispatcher.dispatch_agents.return_value = "Dispatcher processed multi-agent task."
        expected_flags = {
            "needs_memory": True,
            "needs_pi_tools": False,
            "web_search_strategy": {
                "needed": True,
                "search_priority": "quick",
                "search_query": ""
            }
        }
        mock_ollama.return_value = {'message': {'content': json.dumps(expected_flags)}}
        mock_l2_query.return_value = "User mentioned AI hardware."

        # Act
        gate.run_gate_analysis("Find news on AI hardware...", [], mock_dispatcher)

        # Assert
        self.assertTrue(mock_dispatcher.dispatch_agents.called, "Dispatcher was not called.")
        call_kwargs = mock_dispatcher.dispatch_agents.call_args.kwargs
        self.assertIn('routing_flags', call_kwargs)
        self.assertEqual(call_kwargs['routing_flags']['needs_memory'], True)
        self.assertEqual(
            call_kwargs['routing_flags']['web_search_strategy']['needed'],
            True
        )
        self.assertEqual(call_kwargs['routing_flags']['needs_pi_tools'], False)
        print("PASSED: Correctly routed to multiple agents.")

    @patch('memory.gate.L2.query_l2_context')
    @patch('memory.gate.ollama_manager.chat_concurrent_safe')
    def test_malformed_json_fallback(self, mock_ollama, mock_l2_query):
        """Tests fallback behavior with malformed JSON by passing a mock dispatcher."""
        print("\nRunning: test_malformed_json_fallback")
        # Arrange
        mock_dispatcher = MagicMock(spec=AgentDispatcher)
        mock_dispatcher.dispatch_agents.return_value = "Dispatcher processed with default flags."
        mock_ollama.return_value = {'message': {'content': '{"invalid json'}}
        mock_l2_query.return_value = "No relevant context."

        # Act
        gate.run_gate_analysis("A generic message", [], mock_dispatcher)

        # Assert
        self.assertTrue(mock_dispatcher.dispatch_agents.called, "Dispatcher was not called.")
        call_kwargs = mock_dispatcher.dispatch_agents.call_args.kwargs
        self.assertIn('routing_flags', call_kwargs)
        self.assertEqual(call_kwargs['routing_flags'], gate._get_default_routing_flags())
        print("PASSED: Correctly used default flags on JSON error.")

    @patch('memory.gate.L2.query_l2_context')
    @patch('memory.gate.ollama_manager.chat_concurrent_safe')
    def test_llm_failure_fallback(self, mock_ollama, mock_l2_query):
        """Tests fallback behavior on LLM failure by passing a mock dispatcher."""
        print("\nRunning: test_llm_failure_fallback")
        # Arrange
        mock_dispatcher = MagicMock(spec=AgentDispatcher)
        mock_dispatcher.dispatch_agents.return_value = "Dispatcher processed with default flags."
        mock_ollama.side_effect = Exception("Ollama connection failed")
        mock_l2_query.return_value = "No relevant context."

        # Act
        gate.run_gate_analysis("Another generic message", [], mock_dispatcher)

        # Assert
        self.assertTrue(mock_dispatcher.dispatch_agents.called, "Dispatcher was not called.")
        call_kwargs = mock_dispatcher.dispatch_agents.call_args.kwargs
        self.assertIn('routing_flags', call_kwargs)
        self.assertEqual(call_kwargs['routing_flags'], gate._get_default_routing_flags())
        print("PASSED: Correctly used default flags on LLM failure.")

if __name__ == '__main__':
    unittest.main()
