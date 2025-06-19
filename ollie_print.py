import os
import sys
from datetime import datetime

COLOR_CODES = {
    'info': '\033[34m',  # blue
    'success': '\033[32m',
    'warning': '\033[33m',
    'error': '\033[31m'
}
RESET = '\033[0m'
COMMENTS = {
    'info': 'Systems green across the board.',
    'success': 'Mission accomplished!',
    'warning': 'Caution: power conduit unstable.',
    'error': 'Critical failure detected!'
}

def olliePrint(message, level='info', module=None):
    """Print a colorized FRED-style message with banner."""
    level = level.lower()
    color = COLOR_CODES.get(level, COLOR_CODES['info'])
    if not sys.stdout.isatty():
        color = ''
    banner_module = module or os.path.splitext(os.path.basename(sys._getframe(1).f_code.co_filename))[0]
    timestamp = datetime.now().strftime('%H:%M:%S')
    header = f"{banner_module} [{timestamp}] - Designed by Ollie-Tec"
    bar = '-' * len(header)
    msg = f"{color}{message}{RESET if sys.stdout.isatty() else ''}"
    print(f"{bar}\n{header}\n{bar}\n{msg} {COMMENTS.get(level, '')}")
