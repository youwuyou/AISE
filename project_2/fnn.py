"""
Utility functions used for training and evaluating the simple FNN model
"""

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    """Simple NN used for approximating spatiotemporal solution u(x, t)"""
    def __init__(self, width = 64, activation_fun = "Tanh", device="cuda"):
        super(Net, self).__init__()
        self.width = width

        # Layers
        self.fc0 = nn.Linear(2, self.width)
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, 1)

        if activation_fun == "Tanh":
            self.activation = nn.Tanh()
        elif activation_fun == "GELU":
            self.activation = nn.GELU()
        print(f"Activation function set to {activation_fun}")
           
        self.to(device)
        
    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        X = self.fc0(X)
        X = self.activation(X)
        X = self.fc1(X)
        X = self.activation(X)
        X = self.fc2(X)
        return X

def get_experiment_name(width, learning_rate):
    """Create a unique experiment name based on key parameters and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pde_sol_w{width}_lr{learning_rate}_{timestamp}"

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

def load_model(checkpoint_dir: str) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(checkpoint_dir)
    
    with open(checkpoint_dir / 'training_config.json', 'r') as f:
        config_dict = json.load(f)
    
    model_config = config_dict['model_config']
    model_args = {k: v for k, v in model_config.items()}
    model = Net(**model_args)
    
    model.load_state_dict(torch.load(checkpoint_dir / 'model.pth', weights_only=True))    
    model = model.to(device)
    model.eval()
    return model