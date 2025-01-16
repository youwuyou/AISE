"""
Utility functions used for training and evaluating the ACE foundation model of project 3
"""

import torch
import numpy as np

import json
from datetime import datetime
from typing import Optional, Union
from pathlib import Path
from model import AllenCahnFNO
import matplotlib.pyplot as plt

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
    
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config_copy, f, indent=4)

def load_base_model(checkpoint_dir: str) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(checkpoint_dir)
    
    with open(checkpoint_dir / 'training_config.json', 'r') as f:
        config_dict = json.load(f)
    
    model_config = config_dict['model_config']
    model_args = {k: v for k, v in model_config.items()}
    model = AllenCahnFNO(**model_args)
    
    model.load_state_dict(torch.load(checkpoint_dir / 'base_model.pth', weights_only=True))    
    model = model.to(device)
    model.eval()
    return model

def plot_training_history(experiment_dir: Path, save_dir: Optional[Path] = None) -> None:
    experiment_dir = Path(experiment_dir)
    save_dir = Path(save_dir) if save_dir else experiment_dir
    save_dir.mkdir(exist_ok=True)
    
    with open(experiment_dir / 'training_config.json', 'r') as f:
        config = json.load(f)
        history = config['training_history']
    
    curriculum_steps = config.get('training_config', {}).get('curriculum_steps')
    if not curriculum_steps:
        curriculum_steps = []
    if not isinstance(curriculum_steps, list):
        raise TypeError("'curriculum_steps' should be a list.")

    gray_colors = ['#666666', '#787878', '#8A8A8A', '#9C9C9C', '#AEAEAE']
    line_styles = ['--', ':', '-.', (0, (5, 10)), (0, (3, 5, 1, 5))]

    plt.figure(figsize=(12, 7))
    
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)

    combined_losses = history['train_loss'] + history['val_loss']
    data_min, data_max = min(combined_losses), max(combined_losses)
    magnitude_min = np.floor(np.log10(data_min))
    magnitude_max = np.ceil(np.log10(data_max))
    ymin = 10 ** magnitude_min
    ymax = 10 ** magnitude_max
    
    print(f"Data range: {data_min:.2e} to {data_max:.2e}")
    print(f"Plot limits: {ymin:.2e} to {ymax:.2e}")
    
    added_labels = set()
    for i, (epoch, val) in enumerate(curriculum_steps):
        if isinstance(val, list):
            val = val[-1]
        epoch = int(epoch)
        epsilon_val = float(val)
        
        color = gray_colors[i % len(gray_colors)]
        line_style = line_styles[i % len(line_styles)]
        
        label = f'ε = {epsilon_val}' if epsilon_val not in added_labels else None
        if label:
            added_labels.add(epsilon_val)

        plt.vlines(x=epoch, ymin=ymin, ymax=ymax, colors=color, linestyles=line_style,
                   label=label, linewidth=1.5, alpha=0.8, zorder=1)

    plt.title('Training History - ACE Foundation Model', pad=20, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    
    plt.grid(True, which='major', linestyle='-', alpha=0.2)
    plt.grid(True, which='minor', linestyle=':', alpha=0.1)
    plt.ylim(ymin, ymax)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    plt.savefig(save_dir / 'training_history.png', bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    plt.close()