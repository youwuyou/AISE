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
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import shutil

from visualization import (
plot_derivatives,
plot_pde_comparison
)

from train import Net
from optimizers import (
generalized_condition_number,
STRidge,
TrainSTRidge
)

from feature_library import (
build_theta,
build_u_t,
generate_candidate_symbols,
print_discovered_equation
)


def compute_derivatives(model, x, t, symbols, include_constant=True, include_u=True):
    """
    Compute derivatives based on NN model that approximates 1D spatiotemporal u(x,t)
    - symbols: symbolic expressions are derivatives to be computed need to be specified
    """
    results = {}
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    # First compute all basic derivatives we'll need
    basic_derivatives = {}
    max_x_order = max(str(expr).count('x') for expr in symbols)
    max_t_order = max(str(expr).count('t') for expr in symbols)
    
    # Add constant term if requested
    if include_constant:
        results['1'] = torch.ones((u.shape[0] * u.shape[1], 1))
    
    # Add base function u if requested
    if include_u:
        results['u'] = u
    
    # Compute x derivatives
    current = u
    basic_derivatives['u'] = u
    for order in range(1, max_x_order + 1):
        grad = torch.ones_like(current)
        current = torch.autograd.grad(current, x, grad_outputs=grad, create_graph=True)[0]
        basic_derivatives[f'u_{"x" * order}'] = current
    
    # Compute t derivatives
    current = u
    for order in range(1, max_t_order + 1):
        grad = torch.ones_like(current)
        current = torch.autograd.grad(current, t, grad_outputs=grad, create_graph=True)[0]
        basic_derivatives[f'u_{"t" * order}'] = current
    
    # Now construct each candidate expression
    for expr in symbols:
        str_expr = str(expr)
                
        if '*' not in str_expr:
            # It's a basic derivative, just copy it
            if str_expr != 'u' or include_u:  # Only include u if requested
                results[str_expr] = basic_derivatives[str_expr]
        else:
            # It's a product
            terms = str_expr.split('*')
            # Initialize with first term
            result = basic_derivatives[terms[0]]
            # Multiply by subsequent terms
            for term in terms[1:]:
                result = result * basic_derivatives[term]
            results[str_expr] = result
                
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
        activation_fun = nn.Tanh()
        width = 64 # 3.88%
    else:
        path = 'data/2.npz'
        name = "KdV Equation"
        activation_fun = nn.GELU()
        width = 64 # 4.79%

    data = np.load(path)

    # Shape: (256, 101)
    u = data['u']        
    x = data['x']
    t = data['t']

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
    model = Net(width=width, activation_fun=activation_fun).to(device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Model loaded from {model_path}")

    #==================================================
    # Computing derivatives
    #==================================================
    # Select different candidate symbols for different systems
    if system == 1:
        symbols = generate_candidate_symbols(
            max_x_order=3,     # Up to u_xxx
            max_t_order=0,
            binary_ops=['mul'],
            power_orders=[1],
            allowed_mul_orders=[(0,1), (0,2)],
            exclude_u_t=True
        )
    else:
        symbols = generate_candidate_symbols(
            max_x_order=3,     # Up to u_xxx
            max_t_order=0,
            binary_ops=['mul'],
            power_orders=[1],
            allowed_mul_orders=[(0,1), (0,2)],
            exclude_u_t=True
        )

    derivatives = compute_derivatives(model, x_tensor, t_tensor, symbols, 
                               include_constant=True,
                               include_u=True)

    # Manually filter out some entries
    # derivatives.pop('u')
    # derivatives.pop('u_t')
    derivatives.pop('u_x')

    print(f"{len(list(derivatives.keys()))} derivatives keys: {list(derivatives.keys())}")
    candidates = list(derivatives.keys())

    #==================================================
    # Assemble LSE
    #==================================================
    Theta = build_theta(u_tensor, derivatives)
    u_t   = build_u_t(model, x_tensor, t_tensor)

    # c_theta = np.linalg.cond(Theta.cpu().detach().numpy())
    # print(f"condition number of Theta is {c_theta}")
    # condition number of Theta is 4.343603610992432
    # condition number of Theta is 68.19337463378906

    c_theta = generalized_condition_number(Theta.cpu().detach().numpy())
    print(f"condition number of Theta is {c_theta}")
    # condition number of Theta is 4.34360408782959
    # condition number of Theta is 68.19335174560547

    #==================================================
    # Sparse regression for LSE
    #==================================================
    # Assembling Theta and u_t
    Theta_np = Theta.cpu().detach().numpy()
    u_t_np = u_t.cpu().detach().numpy()

    ξ_best = TrainSTRidge(
        Theta_np, u_t_np,
        lam=1e-6,
        d_tol=5e-3,
        maxit=10,
        STR_iters=10,
        l0_penalty=None,
        split=0.7,
        print_best_tol=True
    )
    print(f"Found coefficients {ξ_best}")

    # After running ridge regression:
    print_discovered_equation(candidates, ξ_best)

    #==================================================
    # Prepare data for plotting
    #==================================================
    # Store all functions (u, u_t, and derivatives) in a single dictionary
    functions = {}

    # Add u and u_t
    functions['u'] = u
    functions['u_t'] = u_t.detach().cpu().numpy().reshape(u.shape)

    # Add all derivatives
    for key, value in derivatives.items():
        if key != 'constant':  # Skip constant term
            functions[key] = value.detach().cpu().numpy().reshape(u.shape)

    # Call the function with the functions dictionary
    plot_derivatives(
        model=model,
        x_tensor=x_tensor,
        t_tensor=t_tensor,
        X=X,
        t=t,
        functions=functions,
        system=system,
        results_dir=results_dir
    )

    # Plot 
    # Plot a specific time snapshot and its derivatives
    snapshot = u.shape[1] // 5  # Choose a specific time snapshot, e.g., 1/4th of the total time
    if system == 1:
        # For u_t + u*u_x = 0.1*u_xx
        plot_pde_comparison(
            X=X,
            functions=functions,
            lhs_terms=[('u_t', 1), ('u*u_x', 1)],
            rhs_terms=[('u_xx', 0.1)],
            snapshot=snapshot,
            results_dir=results_dir
        )
    else:
        # For u_t + 6*u*u_x = - u_xxx
        plot_pde_comparison(
            X=X,
            functions=functions,
            lhs_terms=[('u_t', 1), ('u*u_x', 6)],
            rhs_terms=[('u_xxx', -1)],  # Empty list since everything is on LHS
            snapshot=snapshot,
            results_dir=results_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    
    main(system=args.system)