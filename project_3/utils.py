"""
Utility functions
"""

from datetime import datetime

def print_bold(text: str) -> None:
    """Print a bold header text."""
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{BOLD}{text}{RESET}")

def get_dataset_folder_name(dt):
    """Create a unique dataset folder name based on key parameters and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dt_{dt}_{timestamp}"

