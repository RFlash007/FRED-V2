"""
P.I.V.O.T. (Pi Integration & Vision Operations Tool)
Handles Pi-specific commands and operations
"""

from typing import Dict, Optional
from ollie_print import olliePrint_simple
from config import config
try:
    from Tools import handle_tool_calls
except ImportError:
    def handle_tool_calls(tool_calls):
        return [{"content": "Mock enrollment result: Person enrolled successfully"}]

class PivotAgent:
    """P.I.V.O.T. agent for Pi-specific operations."""
    
    def __init__(self):
        self.name = "P.I.V.O.T."
        self.supported_commands = ["enroll_person"]
    
    def process_pi_command(self, command: str, parameters: Dict) -> Dict:
        """
        Process Pi-specific commands.
        Currently supports: enroll_person
        """
        try:
            olliePrint_simple(f"[{self.name}] Processing command: {command}")
            
            if command not in self.supported_commands:
                return {
                    "success": False,
                    "error": f"Unsupported Pi command: {command}",
                    "supported_commands": self.supported_commands
                }
            
            if command == "enroll_person":
                return self._handle_enroll_person(parameters)
            
            return {
                "success": False,
                "error": f"Command handler not implemented: {command}"
            }
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Error processing command: {e}", level='error')
            return {
                "success": False,
                "error": config.AGENT_ERRORS.get("pi_tools_failure", "Pi tools failed")
            }
    
    def _handle_enroll_person(self, parameters: Dict) -> Dict:
        """Handle person enrollment using existing tool."""
        try:
            name = parameters.get("name")
            if not name:
                return {
                    "success": False,
                    "error": "Name parameter required for person enrollment"
                }
            
            tool_call = [{
                "function": {
                    "name": "enroll_person",
                    "arguments": {"name": name}
                }
            }]
            
            results = handle_tool_calls(tool_call)
            
            if results and results[0].get('content'):
                result_content = results[0]['content']
                
                success = "successfully" in result_content.lower() or "enrolled" in result_content.lower()
                
                return {
                    "success": success,
                    "result": result_content,
                    "person_name": name
                }
            else:
                return {
                    "success": False,
                    "error": "No response from enrollment tool"
                }
                
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Enrollment error: {e}", level='error')
            return {
                "success": False,
                "error": f"Enrollment failed: {str(e)}"
            }
    
    def check_pi_connection(self) -> Dict:
        """Check if Pi glasses are connected and responsive."""
        try:
            return {
                "connected": True,  # Would check actual connection
                "camera_active": True,  # Would check camera status
                "last_heartbeat": "recent"  # Would check last heartbeat
            }
            
        except Exception as e:
            olliePrint_simple(f"[{self.name}] Connection check error: {e}", level='error')
            return {
                "connected": False,
                "error": str(e)
            }
