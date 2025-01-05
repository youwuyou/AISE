"""
This file contains routines used for approximating the provided dataset.
- We approximate spatiotemporal solution u(x,t) using neural network
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


class Net(nn.Module):
    """Simple NN used for approximating spatiotemporal solution u(x, t)"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.fc(inp)


def main(system = 1):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Specify dataset to load
    if system == 1:
        path = 'data/1.npz'
        name = "Burgers' Equation"    
    else:
        path = 'data/2.npz'
        name = "KdV Equation"

    data = np.load(path)
    u = data['u']        # Shape: (256, 101)
    x = data['x']        # Shape: (256, 1) or (256,)
    t = data['t']        # Shape: (1, 101) or (101,)

    print(f"Shape of u: {u.shape}")
    print(f"Shape of x: {x.shape}")
    print(f"Shape of t: {t.shape}")


    # Data preprocessing
    # Prepare meshgrid
    if x.ndim == 1 and t.ndim == 1:
        X, T = np.meshgrid(x, t, indexing='ij')  # Shape: (256, 101)
    else:
        X, T = x, t  # Assuming x and t are already meshgridded

    # Flatten the data
    X_flat = X.ravel()
    T_flat = T.ravel()
    u_flat = u.ravel()

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(1).to(device)
    t_tensor = torch.tensor(T_flat, dtype=torch.float32).unsqueeze(1).to(device)
    u_tensor = torch.tensor(u_flat, dtype=torch.float32).unsqueeze(1).to(device)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(x_tensor, t_tensor, u_tensor)
    batch_size = 1024
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)   


    # Initialize model, loss function, and optimizer
    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train model
    num_epochs = 1000
    print_every = 100

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
        
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.8f}")

    # Store trained model that approximate the solution underlying the dataset
    checkpoint_dir = f'checkpoints/system_{system}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(checkpoint_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

    print(f"Model saved at {model_path}")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    
    main(system=args.system)
