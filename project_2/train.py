"""
This file contains routines used for approximating the provided dataset.
- We approximate spatiotemporal solution u(x,t) using simple feed forward neural network (FNN)
- The trained neural networks are stored under `checkpoints/system_X` for system X ∈ {1,2}
- We can then use automatic differentiation on the trained neural network in order to approximate derivatives
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

from fnn import (
Net,
get_experiment_name,
save_config,
load_model
)


def main(system=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Configure system and create experiment name
    if system == 1:
        path = 'data/1.npz'
        name = "Burgers' Equation" 

        # Model config
        activation_fun = "Tanh"
        width = 64

        # Training config
        num_epochs = 300
        batch_size = 64
        learning_rate = 1e-3
        weight_decay  = 1e-4

    elif system == 2:
        path = 'data/2.npz'
        name = "KdV Equation"

        # Model config
        activation_fun = "GELU"
        width = 64

        # Training config
        num_epochs = 300
        # batch_size = 128
        batch_size = 64
        learning_rate = 1e-3
        weight_decay  = 1e-4

    experiment_name = get_experiment_name(width, learning_rate)
    save_dir = Path(f'checkpoints/system_{system}') / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    data = np.load(path)
    u, x, t = data['u'], data['x'], data['t']

    print(f"Shape of u: {u.shape}")
    print(f"Shape of x: {x.shape}") 
    print(f"Shape of t: {t.shape}")

    # Create meshgrid if needed
    if x.ndim == 1 and t.ndim == 1:
        X, T = np.meshgrid(x, t, indexing='ij')
    else:
        X, T = x, t

    # Prepare tensors
    x_tensor = torch.tensor(X.ravel(), dtype=torch.float32).unsqueeze(1).to(device)
    t_tensor = torch.tensor(T.ravel(), dtype=torch.float32).unsqueeze(1).to(device)
    u_tensor = torch.tensor(u.ravel(), dtype=torch.float32).unsqueeze(1).to(device)

    # Create dataloader
    dataset = TensorDataset(x_tensor, t_tensor, u_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and training
    model = Net(width=width, activation_fun=activation_fun).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Save full configuration
    full_config = {
        'model_config': {
            'width': width,
            'activation_fun': activation_fun,
            'device': str(device)
        },
        'training_config': {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        }
    }
    save_config(full_config, save_dir)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_t, batch_u in dataloader:
            optimizer.zero_grad()
            u_pred = model(batch_x, batch_t)
            loss = criterion(u_pred, batch_u)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        
        # Print every epoch
        if epoch % 1 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.8f}")

    # Save model
    model_path = save_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    
    main(system=args.system)
