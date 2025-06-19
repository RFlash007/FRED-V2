import time
import builtins as _b

_RESET = '\033[0m'
_COLOR_MAP = {
    'CRITICAL': '\033[91m',
    'ERROR': '\033[91m',
    'WARNING': '\033[93m',
    'SUCCESS': '\033[92m',
    'DEBUG': '\033[90m',
    'AUDIO': '\033[95m',
    'NETWORK': '\033[96m',
    'OPTICS': '\033[96m',
    'ARMLINK': '\033[92m',
    'MAINFRAME': '\033[95m',
    'SHELTER': '\033[96m',
    'OLLIE-TEC': '\033[95m',
    'ARCTEC': '\033[91m',
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
    """Return a formatted banner string for the given component."""
    lines = [
        '\033[92m' + '═' * 60 + '\033[0m',
        f'\033[95m  🛰️  OLLIE-TEC™ {component} ONLINE  🛰️\033[0m',
        '\033[96m  ArcTec Labs x ShelterNet Interface\033[0m',
        f'\033[93m  Boot: {time.strftime("%Y-%m-%d %H:%M:%S")}\033[0m',
        '\033[92m' + '═' * 60 + '\033[0m',
    ]
    return '\n'.join(lines)


def startup_block(component: str, info_lines: list[str]) -> str:
    """Return a banner joined with additional startup info lines."""
    return '\n'.join([banner(component), *info_lines])
