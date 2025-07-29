import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies before importing the module under test
# We mock these to prevent errors from missing libraries and to isolate the test
# to the interaction between crap.py and the real ollama_manager.
sys.modules['memory.L2_memory'] = MagicMock()
sys.modules['duckduckgo_search'] = MagicMock()
sys.modules['duckdb'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['trafilatura'] = MagicMock()

# Now import the module under test
from memory.crap import run_crap_analysis, crap_state
from config import ollama_manager # Import the real manager

class TestCrapIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Preload the model once for all tests in this class to avoid spawning extra workers."""
        from config import config, ollama_manager
        ollama_manager.preload_model(config.DEFAULT_MODEL)



    @patch('memory.L2_memory.query_l2_context')
    @patch('memory.crap.handle_tool_calls')
    def test_run_crap_analysis_with_real_model(self, mock_handle_tool_calls, mock_query_l2_context):
        """
        This is an INTEGRATION TEST.
        It makes a REAL network call to the Ollama model.
        Ensure the model is running and configured correctly before executing.
        """
        # Arrange: Define mock return values to create a realistic context
        mock_handle_tool_calls.return_value = [{'content': 'Your favorite color is blue.'}]
        mock_query_l2_context.return_value = "Recent topics of conversation include art, design, and personal preferences."

        # Act: Run the analysis with a user message that should trigger a memory search
        user_message = "what is my favorite color?"
        conversation_history = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]
        
        final_context = run_crap_analysis(user_message, conversation_history)

        # Assert: Verify that the model's response triggered our tool handler
        self.assertTrue(mock_handle_tool_calls.called, "handle_tool_calls was not called by the C.R.A.P. analysis")
        
        # Optional: More detailed assertion on what it was called with
        call_args, _ = mock_handle_tool_calls.call_args
        tool_calls_list = call_args[0]
        self.assertIsInstance(tool_calls_list, list)
        self.assertGreater(len(tool_calls_list), 0, "No tool calls were made")
        
        first_tool_call = tool_calls_list[0]
        self.assertIn('name', first_tool_call)
        self.assertEqual(first_tool_call['name'], 'search_memory')

        print("\nIntegration Test Passed: The real AI model correctly generated a 'search_memory' tool call.")
        print(f"Final context received: {final_context}")

    @patch('memory.L2_memory.query_l2_context')
    @patch('memory.crap.handle_tool_calls')
    def test_run_crap_analysis_implicit_search(self, mock_handle_tool_calls, mock_query_l2_context):
        """
        Implicit phrasing should also trigger a memory search. The AI model should infer intent without explicit instruction.
        """
        print("\nRunning: test_run_crap_analysis_implicit_search")
        # Arrange
        mock_handle_tool_calls.return_value = [{'content': 'Your favorite color is blue.'}]
        mock_query_l2_context.return_value = "No relevant context."

        # Act
        user_message = "Whats my favorite color?"  # Implicit phrasing
        conversation_history = []
        run_crap_analysis(user_message, conversation_history)

        # Assert
        self.assertTrue(mock_handle_tool_calls.called, "handle_tool_calls was not invoked for implicit search query")
        first_call_args, _ = mock_handle_tool_calls.call_args
        tool_calls_list = first_call_args[0]
        self.assertGreater(len(tool_calls_list), 0, "No tool calls generated for implicit query")
        self.assertEqual(tool_calls_list[0]['name'], 'search_memory', "Expected 'search_memory' tool call was not generated")

        print("PASSED: Implicit query correctly triggered a 'search_memory' tool call.")

if __name__ == '__main__':
    unittest.main()
