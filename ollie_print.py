"""
Stub ollie_print module for silent operation
All logging functions are no-ops to ensure silent runtime
"""

def olliePrint(*args, **kwargs):
    """No-op logging function for silent operation"""
    pass

def olliePrint_simple(*args, **kwargs):
    """No-op simple logging function for silent operation"""
    pass

def olliePrint_warning(*args, **kwargs):
    """No-op warning logging function for silent operation"""
    pass

def olliePrint_error(*args, **kwargs):
    """No-op error logging function for silent operation"""
    pass

def olliePrint_debug(*args, **kwargs):
    """No-op debug logging function for silent operation"""
    pass

def log_model_io(*args, **kwargs):
    """No-op model I/O logging function for silent operation"""
    pass

def banner(*args, **kwargs):
    """No-op banner function for silent operation"""
    return ""

def startup_block(*args, **kwargs):
    """No-op startup block function for silent operation"""
    return ""
