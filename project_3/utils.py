"""
Utility functions
"""

import torch
import json
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

def get_experiment_name(config):
    """Create a unique experiment name based on key parameters and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ace_fno_m{config['modes']}_w{config['width']}_d{config['depth']}_{timestamp}"

def save_config(config, save_dir):
    """Save configuration to a JSON file"""
    config_copy = {
        'model_config': {k: str(v) if isinstance(v, torch.device) else v 
                        for k, v in config['model_config'].items()},
        'training_config': {k: str(v) if isinstance(v, torch.device) else v 
                          for k, v in config['training_config'].items()}
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_copy, f, indent=4)
