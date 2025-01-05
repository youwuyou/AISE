"""
This file contains routines used for compute approximation of both spatial and temporal derivatives of PDE solution
- we assume the PDE solution is represented with a neural network that is sufficiently trained on the dataset
- we use automatic differentiation for computing the derivatives
- routines here can be utlized by the feature library to build the matrix used in the PDE-Find method
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


# Function to compute derivatives
def compute_derivatives(model, x, t, derivatives):
    """
    Computes specified derivatives of the model output.

    Args:
        model: The neural network model.
        x: Spatial input tensor.
        t: Temporal input tensor.
        derivatives: List of tuples specifying the derivatives to compute.
                     Example: [("t", 1), ("x", 2)] to compute ∂u/∂t and ∂²u/∂x².

    Returns:
        A dictionary with the derivative labels as keys and computed tensors as values.
    """
    results = {}
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)

    for var, order in derivatives:
        current_derivative = u
        for _ in range(order):
            grad_outputs = torch.ones_like(current_derivative)
            if var == "x":
                current_derivative = torch.autograd.grad(current_derivative, x, grad_outputs=grad_outputs, create_graph=True)[0]
            elif var == "t":
                current_derivative = torch.autograd.grad(current_derivative, t, grad_outputs=grad_outputs, create_graph=True)[0]
            else:
                raise ValueError(f"Invalid variable: {var}. Must be 'x' or 't'.")
        results[f"d{order}u/d{var}{order}"] = current_derivative

    return results







def main(system=1):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory for the current system
    results_dir = f"results/system_{system}"
    os.makedirs(results_dir, exist_ok=True)    

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

    # Load the saved model
    model_path = f'checkpoints/system_{system}/model.pth'
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Initialize model, loss function, and optimizer
    model = Net().to(device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Model loaded from {model_path}")

    derivatives_to_compute = [("t", 1), ("x", 1), ("x", 2), ("x", 3)]  # Compute u_t, u_x, u_xx, u_xxx
    derivatives = compute_derivatives(model, x_tensor, t_tensor, derivatives_to_compute)

    # Access specific derivatives
    u_t = derivatives["d1u/dt1"]
    u_x = derivatives["d1u/dx1"]
    u_xx = derivatives["d2u/dx2"]
    u_xxx = derivatives["d3u/dx3"]

    # Compute uu_x as a PyTorch tensor
    uu_x = u_tensor * u_x

    # Move derivatives to CPU and reshape
    u_t_np = u_t.detach().cpu().numpy().reshape(u.shape)
    u_x_np = u_x.detach().cpu().numpy().reshape(u.shape)
    u_xx_np = u_xx.detach().cpu().numpy().reshape(u.shape)
    uu_x_np = uu_x.detach().cpu().numpy().reshape(u.shape)
    if system == 2:
        u_xxx_np = u_xxx.detach().cpu().numpy().reshape(u.shape)

    # Plot a specific time snapshot and its derivatives
    snapshot = u.shape[1] // 5  # Choose a specific time snapshot, e.g., 1/4th of the total time

    plt.figure(figsize=(18, 12))

    # Evaluation: Compute Predictions and Derivatives
    model.eval()
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy().reshape(u.shape)


    # Plot True vs Predicted u
    plt.subplot(3, 2, 1)
    plt.plot(X[:, snapshot], u[:, snapshot], label='True u')
    plt.plot(X[:, snapshot], u_pred[:, snapshot], '--', label='Predicted u')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'u at t = {t[0, snapshot]:.2f}' if t.ndim >1 else f'u at t = {t[snapshot]:.2f}')
    plt.legend()

    # Plot u_t
    plt.subplot(3, 2, 2)
    plt.plot(X[:, snapshot], u_t_np[:, snapshot], label='u_t', color='orange')
    plt.xlabel('x')
    plt.ylabel('u_t')
    plt.title('First Derivative w.r.t t (u_t)')
    plt.legend()

    # Plot u_x
    plt.subplot(3, 2, 3)
    plt.plot(X[:, snapshot], u_x_np[:, snapshot], label='u_x', color='green')
    plt.xlabel('x')
    plt.ylabel('u_x')
    plt.title('First Derivative w.r.t x (u_x)')
    plt.legend()

    # Plot u_xx
    plt.subplot(3, 2, 4)
    plt.plot(X[:, snapshot], u_xx_np[:, snapshot], label='u_xx', color='red')
    plt.xlabel('x')
    plt.ylabel('u_xx')
    plt.title('Second Derivative w.r.t x (u_xx)')
    plt.legend()

    # Plot u * u_x
    plt.subplot(3, 2, 5)
    plt.plot(X[:, snapshot], uu_x_np[:, snapshot], label='u * u_x', color='purple')
    plt.xlabel('x')
    plt.ylabel('u * u_x')
    plt.title('Nonlinear Term (u * u_x)')
    plt.legend()

    # Plot u_xxx (only for KdV)
    if system == 2 and 'u_xxx_np' in locals():
        plt.subplot(3, 2, 6)
        plt.plot(X[:, snapshot], u_xxx_np[:, snapshot], label='u_xxx', color='brown')
        plt.xlabel('x')
        plt.ylabel('u_xxx')
        plt.title('Third Derivative w.r.t x (u_xxx)')
        plt.legend()
    else:
        # If not KdV, leave the subplot empty with a note
        plt.subplot(3, 2, 6)
        plt.axis('off')  # Hide the subplot
        plt.title('u_xxx not applicable for Burgers\' Equation')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/derivatives.png')
    plt.close()

    if system == 1:
        # Compute the residual: u_t + uu_x - u_xx
        residual = u_t_np + uu_x_np - 0.1 * u_xx_np

        # Plot the comparison between u_t + uu_x and u_xx
        plt.figure(figsize=(10, 6))

        # Plot u_t + uu_x
        plt.plot(X[:, snapshot], (u_t_np + uu_x_np)[:, snapshot], label='$u_t + u u_x$', color='darkblue')

        # Plot u_xx
        plt.plot(X[:, snapshot], 0.1 * u_xx_np[:, snapshot], '--', label='$u_{xx}$', color='crimson')

        # Plot the residual
        plt.plot(X[:, snapshot], residual[:, snapshot], ':', label='Residual $(u_t + uu_x - 0.1 * u_{xx})$', color='green')

        # Labels and legend
        plt.xlabel('x')
        plt.ylabel('Value')
        plt.title('Comparison of $u_t + uu_x$ and $-u_{xx}$')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.savefig(f'{results_dir}/check_sol.png')
        plt.close()

    elif system == 2:
        # Compute the residual: u_t + 6uu_x  = -u_xxx
        residual = u_t_np + 6 * uu_x_np + u_xxx_np

        # Plot the comparison between u_t + uu_x and u_xx
        plt.figure(figsize=(10, 6))

        # Plot u_t + uu_x
        plt.plot(X[:, snapshot], (u_t_np + 6 * uu_x_np)[:, snapshot], label='$u_t + 6 u u_x$', color='darkblue')

        # Plot u_xxx
        plt.plot(X[:, snapshot], -u_xxx_np[:, snapshot], '--', label='$-u_{xxx}$', color='crimson')

        # Plot the residual
        plt.plot(X[:, snapshot], residual[:, snapshot], ':', label='Residual $(u_t + 6 uu_x + u_{xxx})$', color='green')

        # Labels and legend
        plt.xlabel('x')
        plt.ylabel('Value')
        plt.title('Comparison of $u_t + 6 uu_x$ and $u_{xxx}$')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.savefig(f'{results_dir}/check_sol.png')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    
    main(system=args.system)