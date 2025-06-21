"""
F.R.E.D. Enhanced Logging System - Pi Client Edition
Designed by Ollie-Tec™ - Post-Apocalyptic Computing Division

A comprehensive logging solution that combines Stark Industries sophistication
with Vault-Tec's retro-futuristic charm. Features robust ANSI color support
with graceful degradation for various terminal environments.
"""

import os
import sys
import platform
from datetime import datetime
from typing import Optional, Dict, Any

# === ANSI Color Codes & Styling ===
class ANSIColors:
    """ANSI escape sequences for terminal styling with Raspberry Pi support"""
    
    # Reset
    RESET = '\033[0m'
    
    # Standard Colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright Colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background Colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    # Text Styling
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'

# === Anthropic-inspired Color Scheme ===
FRED_COLORS = {
    # Core system messages - Anthropic's signature orange
    'info': '\033[38;2;241;122;83m',           # Anthropic orange: #F17A53
    'debug': '\033[38;2;241;122;83m',          # Anthropic orange: #F17A53
    'network': '\033[38;2;241;122;83m',        # Anthropic orange: #F17A53
    'mainframe': '\033[38;2;241;122;83m',      # Anthropic orange: #F17A53
    
    # Success states - muted green
    'success': '\033[38;2;52;168;83m',         # Clean green: #34A853
    'armlink': '\033[38;2;52;168;83m',         # Clean green: #34A853
    'shelter': '\033[38;2;52;168;83m',         # Clean green: #34A853
    
    # Media/Audio - warm secondary orange
    'audio': '\033[38;2;255;159;67m',          # Warm orange: #FF9F43
    'optics': '\033[38;2;255;159;67m',         # Warm orange: #FF9F43
    
    # Warnings - amber
    'warning': '\033[38;2;255;193;7m',         # Amber warning: #FFC107
    
    # Errors - keep red but use Anthropic's red tone
    'error': '\033[38;2;220;53;69m',           # Clean red: #DC3545
    'critical': '\033[48;2;220;53;69m\033[38;2;255;255;255m',  # Red background, white text
    
    # System text - neutral gray
    'system': '\033[38;2;108;117;125m',        # Clean gray: #6C757D
}

# === F.R.E.D. Personality Comments ===
FRED_COMMENTS = {
    'info': [
        "ArmLink systems operational.",
        "Field diagnostics nominal.",
        "Pi interface operating within parameters.",
        "Mobile ShelterNet protocols active.",
        "Field data streams flowing smoothly."
    ],
    'success': [
        "Field mission accomplished!",
        "Objective complete, returning to base.",
        "Another successful field operation.",
        "Target acquired and processed.",
        "Field victory confirmed."
    ],
    'warning': [
        "Caution: field hazard detected.",
        "Warning: environmental anomaly observed.",
        "Attention: field irregularity detected.",
        "Alert: potential wasteland threat.",
        "Advisory: proceed with tactical vigilance."
    ],
    'error': [
        "Field equipment malfunction!",
        "ArmLink system failure detected.",
        "Emergency field protocols activated.",
        "Catastrophic field error encountered.",
        "Immediate field intervention required."
    ],
    'critical': [
        "⚠️ FIELD SHELTER BREACH ⚠️",
        "🚨 FIELD DEFCON 1 ACTIVATED 🚨",
        "💀 ARMLINK CORE FAILURE 💀",
        "⛔ FIELD INTEGRITY COMPROMISED ⛔",
        "🔥 EMERGENCY FIELD EVAC REQUIRED 🔥"
    ],
    'debug': [
        "Field diagnostic mode active.",
        "Analyzing ArmLink matrices.",
        "Field debug protocols engaged.",
        "Technical field scan in progress.",
        "Detailed field analysis available."
    ],
    'audio': [
        "Field audio systems online.",
        "Voice communication ready.",
        "Field communication channels clear.",
        "Sound processing active.",
        "Audio transmission stable."
    ],
    'network': [
        "Field network protocols established.",
        "ArmLink communication secured.",
        "Field data transmission successful.",
        "Connection integrity verified.",
        "Mobile network topology stable."
    ],
    'optics': [
        "Field visual sensors calibrated.",
        "Pi camera systems operational.",
        "Field image processing active.",
        "Optical field analysis complete.",
        "Visual reconnaissance data acquired."
    ],
    'armlink': [
        "ArmLink field connection secure.",
        "Field operative protocols established.",
        "Mobile interface active.",
        "Field unit responding to mainframe.",
        "Mobile field operations nominal."
    ],
    'mainframe': [
        "Connection to mainframe established.",
        "Central command link active.",
        "Primary server communication active.",
        "Command hub responding.",
        "Mainframe control link operational."
    ],
    'shelter': [
        "Field ShelterNet security active.",
        "Mobile protected environment confirmed.",
        "Field safe zone protocols enabled.",
        "Portable vault systems operational.",
        "Mobile secure facility status green."
    ]
}

# === Terminal Capability Detection ===
def detect_terminal_capabilities() -> Dict[str, bool]:
    """Detect terminal capabilities for optimal rendering on Pi"""
    capabilities = {
        'ansi_colors': False,
        'unicode': False,
        'wide_terminal': False
    }
    
    # Check if we're in a TTY
    if not (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()):
        return capabilities
    
    # Raspberry Pi typically runs Linux
    if platform.system() == 'Linux':
        capabilities['ansi_colors'] = True
        capabilities['unicode'] = True
        try:
            columns = os.get_terminal_size().columns
            capabilities['wide_terminal'] = columns >= 80
        except:
            capabilities['wide_terminal'] = True
    
    # Check environment variables
    term = os.environ.get('TERM', '').lower()
    if 'color' in term or 'ansi' in term or term in ['xterm', 'xterm-256color', 'screen']:
        capabilities['ansi_colors'] = True
    
    return capabilities

# Global terminal capabilities
_TERM_CAPS = detect_terminal_capabilities()

def get_random_comment(level: str) -> str:
    """Get a random F.R.E.D. personality comment for the given level"""
    import random
    comments = FRED_COMMENTS.get(level, FRED_COMMENTS.get('info', ['System ready.']))
    return random.choice(comments)

def colorize_text(text: str, color: str, bold: bool = False) -> str:
    """Apply color and styling to text with terminal capability detection"""
    if not _TERM_CAPS['ansi_colors']:
        return text
    
    color_code = FRED_COLORS.get(color, ANSIColors.WHITE)
    style = ANSIColors.BOLD if bold else ''
    
    return f"{style}{color_code}{text}{ANSIColors.RESET}"

def create_banner(module_name: str, timestamp: str, level: str = 'info') -> str:
    """Create an enhanced F.R.E.D./Vault-Tec themed banner for Pi"""
    
    # Vault-Tec inspired banner characters
    if _TERM_CAPS['unicode']:
        top_char = '╔'
        bottom_char = '╚'
        side_char = '║'
        line_char = '═'
        corner_tr = '╗'
        corner_br = '╝'
    else:
        top_char = '+'
        bottom_char = '+'
        side_char = '|'
        line_char = '='
        corner_tr = '+'
        corner_br = '+'
    
    # Create header content
    ollie_brand = "OLLIE-TEC™ ARMLINK"
    tech_division = "Field Operations Division"
    system_id = f"Module: {module_name.upper()}"
    timestamp_str = f"Timestamp: {timestamp}"
    
    # Calculate banner width (smaller for Pi terminals)
    content_lines = [ollie_brand, tech_division, system_id, timestamp_str]
    max_width = max(len(line) for line in content_lines) + 4
    banner_width = max(max_width, 50)  # Smaller default for Pi
    
    # Color scheme based on level
    if level in ['error', 'critical']:
        banner_color = 'error'
        accent_color = 'critical'
    elif level == 'warning':
        banner_color = 'warning'
        accent_color = 'warning'
    elif level == 'success':
        banner_color = 'success'
        accent_color = 'success'
    else:
        banner_color = 'info'
        accent_color = 'armlink'  # Use armlink color for Pi
    
    # Build banner
    line_fill = line_char * (banner_width - 2)
    top_line = colorize_text(f"{top_char}{line_fill}{corner_tr}", banner_color, bold=True)
    bottom_line = colorize_text(f"{bottom_char}{line_fill}{corner_br}", banner_color, bold=True)
    
    # Content lines
    banner_lines = [top_line]
    
    # Add branded header
    ollie_line = f"{side_char} {colorize_text(ollie_brand, accent_color, bold=True):^{banner_width-4}} {side_char}"
    tech_line = f"{side_char} {colorize_text(tech_division, 'system'):^{banner_width-4}} {side_char}"
    banner_lines.extend([
        colorize_text(ollie_line, banner_color),
        colorize_text(tech_line, banner_color)
    ])
    
    # Add separator
    sep_line = colorize_text(f"{side_char}{line_char * (banner_width-2)}{side_char}", banner_color)
    banner_lines.append(sep_line)
    
    # Add system info
    system_line = f"{side_char} {system_id:<{banner_width-4}} {side_char}"
    time_line = f"{side_char} {timestamp_str:<{banner_width-4}} {side_char}"
    banner_lines.extend([
        colorize_text(system_line, banner_color),
        colorize_text(time_line, banner_color)
    ])
    
    banner_lines.append(bottom_line)
    
    return '\n'.join(banner_lines)

def olliePrint(message: Any, level: str = 'info', module: Optional[str] = None, 
               show_banner: bool = True, show_comment: bool = True) -> None:
    """
    Enhanced F.R.E.D. logging function for Pi Client with Stark Industries meets Vault-Tec theming
    
    Args:
        message: The message to log (any type, will be converted to string)
        level: Log level (info, success, warning, error, critical, debug, etc.)
        module: Module name override (auto-detected from call stack if not provided)
        show_banner: Whether to show the decorative banner (default: True)
        show_comment: Whether to show F.R.E.D. personality comment (default: True)
    """
    
    # Convert message to string
    msg_str = str(message)
    
    # Normalize level
    level = level.lower().strip()
    
    # Auto-detect module name if not provided
    if module is None:
        try:
            frame = sys._getframe(1)
            filename = os.path.basename(frame.f_code.co_filename)
            module = os.path.splitext(filename)[0]
        except:
            module = 'PI_CLIENT'
    
    # Create timestamp
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    # Create and print banner if requested
    if show_banner:
        banner = create_banner(module, timestamp, level)
        print(banner)
    
    # Colorize main message
    colored_message = colorize_text(msg_str, level, bold=True)
    
    # Add F.R.E.D. personality comment if requested
    if show_comment:
        comment = get_random_comment(level)
        comment_colored = colorize_text(f" → {comment}", 'system')
        full_message = f"{colored_message}{comment_colored}"
    else:
        full_message = colored_message
    
    # Print the message
    print(full_message)
    
    # Add spacing for readability
    print()

# === Specialized Logging Functions ===
def olliePrint_info(message: Any, module: Optional[str] = None) -> None:
    """Log an info message with F.R.E.D. theming"""
    olliePrint(message, 'info', module)

def olliePrint_success(message: Any, module: Optional[str] = None) -> None:
    """Log a success message with F.R.E.D. theming"""
    olliePrint(message, 'success', module)

def olliePrint_warning(message: Any, module: Optional[str] = None) -> None:
    """Log a warning message with F.R.E.D. theming"""
    olliePrint(message, 'warning', module)

def olliePrint_error(message: Any, module: Optional[str] = None) -> None:
    """Log an error message with F.R.E.D. theming"""
    olliePrint(message, 'error', module)

def olliePrint_critical(message: Any, module: Optional[str] = None) -> None:
    """Log a critical error message with F.R.E.D. theming"""
    olliePrint(message, 'critical', module)

def olliePrint_debug(message: Any, module: Optional[str] = None) -> None:
    """Log a debug message with F.R.E.D. theming"""
    olliePrint(message, 'debug', module)

# === Banner-only Functions ===
def olliePrint_simple(message: Any, level: str = 'info') -> None:
    """Print a simple colored message without banner or thematic comments"""
    olliePrint(message, level, show_banner=False, show_comment=False)

def olliePrint_quiet(message: Any, level: str = 'info') -> None:
    """Print a message without banner or comment"""
    olliePrint(message, level, show_banner=False, show_comment=False)

# === Startup Banner Function ===
def startup_block(component: str, info_lines: list) -> str:
    """Create a startup banner with system information for Pi"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create enhanced startup banner
    if _TERM_CAPS['unicode']:
        border_chars = {'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝', 'h': '═', 'v': '║'}
    else:
        border_chars = {'tl': '+', 'tr': '+', 'bl': '+', 'br': '+', 'h': '=', 'v': '|'}
    
    # Calculate width (smaller for Pi)
    all_lines = [f"🛰️  OLLIE-TEC™ {component} ONLINE  🛰️", 
                 "Field Operations Division", 
                 f"Boot: {timestamp}"] + info_lines
    width = max(len(line) for line in all_lines) + 4
    width = max(width, 60)  # Smaller default for Pi
    
    # Build banner
    lines = []
    
    # Top border
    top_border = colorize_text(
        f"{border_chars['tl']}{border_chars['h'] * (width-2)}{border_chars['tr']}", 
        'success', bold=True
    )
    lines.append(top_border)
    
    # Title
    title = f"🛰️  OLLIE-TEC™ {component} ONLINE  🛰️"
    title_line = f"{border_chars['v']} {colorize_text(title, 'armlink', bold=True):^{width-4}} {border_chars['v']}"
    lines.append(colorize_text(title_line, 'success'))
    
    # Subtitle
    subtitle = "Field Operations Division"
    subtitle_line = f"{border_chars['v']} {colorize_text(subtitle, 'network'):^{width-4}} {border_chars['v']}"
    lines.append(colorize_text(subtitle_line, 'success'))
    
    # Separator
    sep = f"{border_chars['v']}{border_chars['h'] * (width-2)}{border_chars['v']}"
    lines.append(colorize_text(sep, 'success'))
    
    # Boot time
    boot_line = f"{border_chars['v']} {colorize_text(f'Boot: {timestamp}', 'warning'):<{width-4}} {border_chars['v']}"
    lines.append(colorize_text(boot_line, 'success'))
    
    # Info lines
    for info in info_lines:
        info_line = f"{border_chars['v']} {info:<{width-4}} {border_chars['v']}"
        lines.append(colorize_text(info_line, 'success'))
    
    # Bottom border
    bottom_border = colorize_text(
        f"{border_chars['bl']}{border_chars['h'] * (width-2)}{border_chars['br']}", 
        'success', bold=True
    )
    lines.append(bottom_border)
    
    return '\n'.join(lines)

# === Legacy Compatibility ===
# Keep the old function name for backward compatibility
def banner(component: str) -> str:
    """Legacy banner function - redirects to startup_block"""
    return startup_block(component, [])

# === Testing Function ===
if __name__ == "__main__":
    print("=== F.R.E.D. Enhanced Logging System Test - Pi Client ===\n")
    
    # Test terminal capabilities
    print("Terminal Capabilities:")
    for cap, status in _TERM_CAPS.items():
        print(f"  {cap}: {'✓' if status else '✗'}")
    print()
    
    # Test different log levels
    test_levels = ['info', 'success', 'warning', 'error', 'critical', 'debug']
    
    for level in test_levels:
        olliePrint(f"This is a {level} message test from Pi", level, 'PI_TEST')
    
    # Test startup banner
    print("\n" + startup_block("PI GLASSES", [
        "🔧 ArmLink systems nominal",
        "🌐 Field network connections established", 
        "⚡ Mobile power levels optimal",
        "📷 Pi camera systems ready"
    ])) 