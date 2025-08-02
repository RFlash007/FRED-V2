import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import pytest
pytest.skip("Skipping integration tests that require running models", allow_module_level=True)

# Add the project root to the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock external dependencies not required for this integration test
sys.modules['memory.L2_memory'] = MagicMock()
sys.modules['memory.L3_memory'] = MagicMock()
sys.modules['duckduckgo_search'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['trafilatura'] = MagicMock()
sys.modules['duckdb'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Import the module under test after setting up mocks
from agents.mad import MADAgent  # noqa: E402
from config import ollama_manager  # noqa: E402


class TestMadIntegration(unittest.TestCase):
    """Integration tests for the Memory Addition Daemon (M.A.D.) agent."""

    @classmethod
    def setUpClass(cls):
        """Preload the model once and prepare shared MADAgent instance."""
        from config import config
        ollama_manager.preload_model(config.MAD_OLLAMA_MODEL)

        # Patch the actual memory-writing Tools so we don't write to a real database
        cls.add_memory_patcher = patch('Tools.tool_add_memory', MagicMock(return_value={"success": True, "id": 1}))
        cls.add_memory_obs_patcher = patch('Tools.tool_add_memory_with_observations', MagicMock(return_value={"success": True, "id": 2}))
        cls.add_memory_patcher.start()
        cls.add_memory_obs_patcher.start()

        cls.mad_agent = MADAgent()

    @classmethod
    def tearDownClass(cls):
        """Stop patchers started in setUpClass."""
        cls.add_memory_patcher.stop()
        cls.add_memory_obs_patcher.stop()

    def _assert_memory_created(self, result_dict):
        """Helper assertion to verify at least one memory was created."""
        self.assertTrue(result_dict.get('success'), "M.A.D. analysis failed")
        tool_results = result_dict.get('tool_results', [])
        self.assertTrue(tool_results, "No tool results returned by M.A.D.")
        self.assertTrue(
            any(tr.get('tool') in ('add_memory', 'add_memory_with_observations') for tr in tool_results),
            "No memory-creation tool call detected",
        )

    def test_explicit_memory_addition(self):
        """Explicit user instruction should trigger add_memory tool call."""
        user_message = "Add to memory that my favorite color is blue."
        fred_response = "Certainly, I'll remember that."

        result = self.mad_agent.analyze_turn(user_message, fred_response, [])
        self._assert_memory_created(result)
        print("PASSED: Explicit memory request generated add_memory tool call.")

    def test_implicit_memory_addition(self):
        """Model should infer memory addition without explicit instruction."""
        user_message = "By the way, I absolutely love the color blue."
        fred_response = "Noted!"

        result = self.mad_agent.analyze_turn(user_message, fred_response, [])
        self._assert_memory_created(result)
        print("PASSED: Implicit user preference triggered add_memory tool call.")


if __name__ == '__main__':
    unittest.main()
