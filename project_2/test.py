"""
This file contains routines for testing out the quality of the approximation of the PDE solution u(x,t)
- the approximation is done using simple NN, with code defined under `train.py`
- the evaluation is done in a governing equation-agnostic manner, where we barely do comparison against the dataset
"""

import os
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import shutil

from train import Net

def main(system=1):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory for the current system
    results_dir = f"results/system_{system}"
    os.makedirs(results_dir, exist_ok=True)


    # Load the saved model
    model_path = f'checkpoints/system_{system}/model.pth'
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Initialize model, loss function, and optimizer
    model = Net().to(device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Model loaded from {model_path}")


    # Evaluation: Preparation of Dataset
    if system == 1:
        path = 'data/1.npz'
        name = "Burgers' Equation"
    elif system == 2:
        path = 'data/2.npz'
        name = "KdV Equation"
    print(f"Testing on dataset loaded from {path}")

    data = np.load(path)
    u = data['u']        
    x = data['x']
    t = data['t']

    # Prepare meshgrid
    if x.ndim == 1 and t.ndim == 1:
        X, T = np.meshgrid(x, t, indexing='ij')  # Shape: (256, 101)
    else:
        X, T = x, t  # Assuming x and t are already meshgridded

    X_min, X_max = X.min(), X.max()
    T_min, T_max = T.min(), T.max()

    # TODO: for 2D data
    # y_min, y_max = Y.min(), Y.max()

    # Flatten the data
    X_flat = X.ravel()
    T_flat = T.ravel()
    u_flat = u.ravel()

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(1).to(device)
    t_tensor = torch.tensor(T_flat, dtype=torch.float32).unsqueeze(1).to(device)
    u_tensor = torch.tensor(u_flat, dtype=torch.float32).unsqueeze(1).to(device)


    # Evaluation: Compute Predictions and Derivatives
    model.eval()
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy().reshape(u.shape)

    # Plot the predicted solution as a heatmap
    plt.figure(figsize=(8, 6))
    extent = [X_min, X_max, T_min, T_max]
    plt.imshow(u_pred.T, extent=extent, aspect='auto', cmap='coolwarm', origin='lower')
    plt.colorbar(label='Predicted u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Neural Network Approximation for {name}')
    plt.savefig(f'{results_dir}/approximate_sol_heatmap.png')
    plt.close()

    # Temporary directory to save frames
    frames_dir = os.path.join(results_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Generate frames
    for i in range(u.shape[1]):
        plt.figure(figsize=(10, 6))
        plt.plot(X[:, i], u[:, i], label='True u', color='blue')
        plt.plot(X[:, i], u_pred[:, i], '--', label='Approximate u (NN)', color='red')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f'u at t = {t[0, i]:.2f}' if t.ndim > 1 else f'u at t = {t[i]:.2f}')
        plt.legend()
        plt.grid(True)
        
        # Save the frame
        plt.savefig(f'{frames_dir}/frame_{i:04d}.png')
        plt.close()

    # Create GIF
    gif_path = os.path.join(results_dir, 'solution_comparison.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
        for i in range(u.shape[1]):
            writer.append_data(imageio.imread(f'{frames_dir}/frame_{i:04d}.png'))

    # Optional: Clean up frames directory
    shutil.rmtree(frames_dir)

    print(f"GIF saved as '{gif_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    
    main(system=args.system)