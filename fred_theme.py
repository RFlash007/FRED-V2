# fred_theme.py
"""
Light-weight terminal theming without external packages.

Adds concise Vault-Tec / Stark-Industries prefixes & ANSI colours while
filtering noisy log lines.  Works on most modern terminals; on Windows 10+
we enable VT-100 escape sequence support via ctypes.
"""

from __future__ import annotations

import builtins
import os
from typing import List

# ---------------------------------------------------------------------------
#  ANSI COLOURS & WINDOWS SUPPORT
# ---------------------------------------------------------------------------

ANSI_RESET = "\033[0m"
ANSI_BRIGHT = "\033[1m"

_THEME_COLOURS = {
    "pi": ANSI_BRIGHT + "\033[36m",   # Bright Cyan
    "main": ANSI_BRIGHT + "\033[32m", # Bright Green
}

_PREFIXES = {
    "pi": "⟦PIP-BOY⟧",
    "main": "⟦F.R.E.D.⟧",
}


def _enable_windows_ansi():
    """Attempt to enable ANSI escape sequence processing on Windows."""
    if os.name != "nt":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        # Fallback: no colour but still functionally works
        _THEME_COLOURS["pi"] = ""
        _THEME_COLOURS["main"] = ""


_enable_windows_ansi()

# ---------------------------------------------------------------------------
#  SPAM FILTER
# ---------------------------------------------------------------------------

_SKIP_TERMS: List[str] = [
    "PortAudio status: input overflow",
    "[DEBUG]",
    "[VITAL-MONITOR]",
    "Audio level:",
]


def extend_skip_terms(terms: List[str]):
    """Public helper to silence extra noisy fragments during runtime."""
    _SKIP_TERMS.extend(terms)


def _should_skip(text: str) -> bool:
    return any(term in text for term in _SKIP_TERMS)


# Backup original print so we can still call it inside our wrapper
_original_print = builtins.print  # type: ignore[assignment]


def apply_theme(mode: str = "pi") -> None:
    """Activate themed, concise printing (mode = 'pi' or 'main')."""

    colour = _THEME_COLOURS.get(mode, ANSI_BRIGHT + "\033[37m")  # Bright White default
    prefix = _PREFIXES.get(mode, "⟦SYS⟧")

    def themed_print(*args, **kwargs):  # type: ignore[override]
        if not args:
            return _original_print(*args, **kwargs)

        first_arg = str(args[0])
        if _should_skip(first_arg):
            return  # Suppressed

        message = " ".join(str(a) for a in args)
        _original_print(f"{colour}{prefix} {message}{ANSI_RESET}", **kwargs)

    builtins.print = themed_print  # Monkey-patch global print

    # Greeting banner (concise)
    builtins.print(f"{colour}{prefix} Terminal styling engaged – ready for wasteland operations{ANSI_RESET}") 