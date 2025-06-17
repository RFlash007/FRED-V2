# fred_theme.py
"""
F.R.E.D. Terminal Theming Utility
Provides a lightweight way to style and declutter console output for both the
Raspberry Pi client (Pip-Boy) and the main F.R.E.D. server.

Usage (MUST be invoked as early as possible in the top-level script):

    from fred_theme import apply_theme
    apply_theme("pi")      # For the Pi glasses client
    # OR
    apply_theme("main")    # For the main server

This call monkey-patches the built-in print() so that:
  • Repetitive / noisy messages are filtered.
  • Each line gets a Vault-Tec/Stark-Industries styled prefix & colour.
  • ANSI colours are automatically reset (thanks to colorama).

The approach keeps the existing code-base almost unchanged – no massive refactor
of hundreds of print() calls – yet the user sees a MUCH more concise and themed
output.
"""

from __future__ import annotations

import builtins
import re
from typing import List
from colorama import init, Fore, Style

# Initialise colour support (Windows included)
init(autoreset=True)

# ---------------------------------------------------------------------------
#  INTERNAL HELPERS
# ---------------------------------------------------------------------------

# Skip very verbose / repetitive lines coming from lower-level libs that we
# cannot easily silence otherwise.  You can extend this list at runtime by
# calling `extend_skip_terms([...])`.
_SKIP_TERMS: List[str] = [
    "PortAudio status: input overflow",  # sounddevice buffer overflows
    "[DEBUG]",                            # generic debug noise
    "[VITAL-MONITOR]",                   # frequent heart-beats
    "Audio level:",                       # continuous VAD level prints
]

_THEME_COLOURS = {
    "pi": Fore.CYAN + Style.BRIGHT,    # Pip-Boy / field unit
    "main": Fore.GREEN + Style.BRIGHT, # Mainframe in the vault
}

_PREFIXES = {
    "pi": "⟦PIP-BOY⟧",
    "main": "⟦F.R.E.D.⟧",
}

_original_print = builtins.print  # Backup original


def extend_skip_terms(terms: List[str]):
    """Allow other modules to add extra noisy fragments to be filtered."""
    _SKIP_TERMS.extend(terms)


def _should_skip(text: str) -> bool:
    return any(term in text for term in _SKIP_TERMS)


def apply_theme(mode: str = "pi"):  # mode should be 'pi' or 'main'
    """Patch built-in print so that logs are themed and spam reduced."""

    colour = _THEME_COLOURS.get(mode, Fore.WHITE + Style.BRIGHT)
    prefix = _PREFIXES.get(mode, "⟦SYS⟧")

    def themed_print(*args, **kwargs):  # type: ignore[override]
        if not args:
            return _original_print(*args, **kwargs)

        # Convert first positional arg to string for filtering
        first_arg_str = str(args[0])
        if _should_skip(first_arg_str):
            return  # Silenced

        # Assemble message – we keep the rest of *args untouched to avoid side-effects
        message = " ".join(str(a) for a in args)
        themed_message = f"{colour}{prefix} {message}{Style.RESET_ALL}"
        _original_print(themed_message, **kwargs)

    # Monkey-patch the global print for this interpreter process
    builtins.print = themed_print

    # Optional welcome banner (concise)
    banner = f"{colour}{prefix} Terminal styling engaged – ready for wasteland operations{Style.RESET_ALL}"
    _original_print(banner) 