import re
from ollie_print import olliePrint_simple

def strip_think_tags(text):
    """Remove <think>...</think> blocks from text."""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
