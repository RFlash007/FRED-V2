import time
import builtins as _b

_RESET = '\033[0m'

# Dark Purple + Anthropic-inspired Color Scheme
_COLOR_MAP = {
    # Errors - clean red (as requested)
    'CRITICAL': '\033[38;2;220;53;69m',        # Anthropic red: #DC3545
    'ERROR': '\033[38;2;220;53;69m',           # Anthropic red: #DC3545
    'ARCTEC': '\033[38;2;220;53;69m',          # Anthropic red: #DC3545
    
    # Warnings - muted purple-gray
    'WARNING': '\033[38;2;156;139;192m',       # Muted purple-gray: #9C8BC0
    
    # Success states - Anthropic green (harmonizes with purple)
    'SUCCESS': '\033[38;2;52;168;83m',         # Anthropic green: #34A853
    'ARMLINK': '\033[38;2;52;168;83m',         # Anthropic green: #34A853
    'SHELTER': '\033[38;2;52;168;83m',         # Anthropic green: #34A853
    
    # Media/Audio - warm purple
    'AUDIO': '\033[38;2;183;110;255m',         # Warm purple: #B76EFF
    'OPTICS': '\033[38;2;183;110;255m',        # Warm purple: #B76EFF
    'VISION': '\033[38;2;183;110;255m',        # Warm purple: #B76EFF
    
    # Network/System - deep purple (main identity color)
    'NETWORK': '\033[38;2;139;69;235m',        # Deep purple: #8B45EB
    'MAINFRAME': '\033[38;2;139;69;235m',      # Deep purple: #8B45EB
    'DEBUG': '\033[38;2;139;69;235m',          # Deep purple: #8B45EB
    
    # Branding - signature purple
    'OLLIE-TEC': '\033[38;2;139;69;235m',      # Deep purple: #8B45EB
    'VAULT-TEC': '\033[38;2;139;69;235m',      # Deep purple: #8B45EB
}

_orig_print = _b.print

def ollietec_print(*args, **kwargs):
    colored_args = []
    for arg in args:
        text = str(arg)
        color = ''
        for token, col in _COLOR_MAP.items():
            if token in text:
                color = col
                break
        if color:
            text = f"{color}{text}{_RESET}"
        colored_args.append(text)
    _orig_print(*colored_args, **kwargs)


def apply_theme():
    _b.print = ollietec_print


def banner(component: str) -> str:
    """Return a formatted banner string for the given component with Anthropic colors."""
    # Use Anthropic-inspired colors
    anthropic_orange = '\033[38;2;241;122;83m'  # Anthropic orange: #F17A53
    warm_orange = '\033[38;2;255;159;67m'       # Warm orange: #FF9F43
    clean_green = '\033[38;2;52;168;83m'        # Clean green: #34A853
    neutral_gray = '\033[38;2;108;117;125m'     # Clean gray: #6C757D
    
    lines = [
        clean_green + '═' * 60 + _RESET,
        f'{anthropic_orange}  🛰️  OLLIE-TEC™ {component} ONLINE  🛰️{_RESET}',
        f'{warm_orange}  ArcTec Labs x ShelterNet Interface{_RESET}',
        f'{neutral_gray}  Boot: {time.strftime("%Y-%m-%d %H:%M:%S")}{_RESET}',
        clean_green + '═' * 60 + _RESET,
    ]
    return '\n'.join(lines)
