"""
Main module that uses PDE-Find to do symbolic regression on the governing equation from underlying dataset
- Finishes predictions of governing equations for system 1 & 2
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from visualization import (
plot_derivatives,
plot_pde_comparison
)

from optimizers import (
generalized_condition_number,
STRidge,
TrainSTRidge
)

from feature_library import (
build_theta,
build_u_t,
generate_candidate_symbols,
print_discovered_equation,
compute_derivatives_autodiff
)

from fnn import (
Net,
load_model
)

def main(system=1):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory for the current system
    results_dir = f"results/system_{system}"
    os.makedirs(results_dir, exist_ok=True)

    # Get the latest experiment directory
    checkpoint_dir = sorted(Path(f'checkpoints/system_{system}').glob('pde_sol_*'), 
                        key=lambda d: d.stat().st_mtime)

    # Load the model using utility function
    model = load_model(checkpoint_dir[-1])
    print(f"Loading FNN that approximates system {system} from: {checkpoint_dir[-1]}")

    # Load and prepare dataset
    if system == 1:
        path = 'data/1.npz'
        name = "Burgers' Equation"
    else:
        path = 'data/2.npz'
        name = "KdV Equation"

    try:
        print(f"Testing on dataset loaded from {path}")
        data = np.load(path)
    except FileNotFoundError:
        print(f"File not found: {path}, please ensure to download the zipped file `systems.zip` with datasets from here https://polybox.ethz.ch/index.php/f/3927719498, unzip it and ensure `data/X.npz` exist.")
        raise SystemExit

    u = data['u']
    x = data['x']
    t = data['t']

    # Data preprocessing
    # Prepare meshgrid
    if x.ndim == 1 and t.ndim == 1:
        X, T = np.meshgrid(x, t, indexing='ij')
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

    #==================================================
    # Computing derivatives
    #==================================================
    # Select different candidate symbols for different systems
    if system == 1:
        symbols = generate_candidate_symbols(
            max_x_order=3,     # Up to u_xxx
            max_t_order=0,
            binary_ops=['mul'],
            power_orders=[1, 2, 3], # base power orders
            allowed_mul_orders=[(0,1), (0,2)],
            exclude_u_t=True
        )
        derivatives = compute_derivatives_autodiff(model, x_tensor, t_tensor, symbols, 
                                include_constant=True,
                                include_u=True)
    else:
        symbols = generate_candidate_symbols(
            max_x_order=3,     # Up to u_xxx
            max_t_order=2,     # Up to u_tt
            binary_ops=['mul'],
            power_orders=[1],
            allowed_mul_orders=[(0,1), (0,2), (0,3), (1,1), (2,2), (3,3)],
            exclude_u_t=False
        )

        derivatives = compute_derivatives_autodiff(model, x_tensor, t_tensor, symbols, 
                                include_constant=True,
                                include_u=True)

        # Manually filter out some entries
        derivatives.pop('u_t') # this is already lhs, we want just u_tt and co

    print(f"{len(list(derivatives.keys()))} candidates: {list(derivatives.keys())}")
    candidates = list(derivatives.keys())

    #==================================================
    # Assemble LSE
    #==================================================
    Theta = build_theta(u_tensor, derivatives)
    u_t   = build_u_t(model, x_tensor, t_tensor)

    c_theta = generalized_condition_number(Theta.cpu().detach().numpy())
    print(f"condition number of Theta is {c_theta}")

    #==================================================
    # Sparse regression for LSE
    #==================================================
    # Assembling Theta and u_t
    Theta_np = Theta.cpu().detach().numpy()
    u_t_np = u_t.cpu().detach().numpy()

    print(f"shape of Theta_np is {Theta_np.shape}")
    print(f"shape of u_t_np is {u_t_np.shape}")

    if system == 1:
        λ = 1e-6
        maxiter = 50
        split = 0.5
        η = 1e-3
        d_tol = 5e-3
        STR_iters = 10
    else:
        λ = 1e-6
        maxiter = 50
        split = 0.7
        η = 1e-3
        d_tol = 1e-2
        STR_iters = 10

    # Find best coefficients
    ξ_best = TrainSTRidge(
        Theta_np,
        u_t_np,
        λ=λ,
        d_tol=d_tol,
        maxiter=maxiter,
        STR_iters=STR_iters,
        η=η,
        split=split,
        print_best_tol=True
    )
    # After running ridge regression:
    print_discovered_equation(candidates, ξ_best, f_symbol="u_t")

    #==================================================
    # Prepare data for plotting
    #==================================================
    # Store all functions (u, u_t, and derivatives) in a single dictionary
    functions = {}

    # Add u and u_t
    functions['u'] = u
    functions['u_t'] = u_t_np.reshape(u.shape)

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
    snapshot = u.shape[1] // 3  # Choose a specific time snapshot, e.g., 1/3th of the total time
    if system == 1:
        # For u_t = - u*u_x + 0.1*u_xx
        # compare results we got
        plot_pde_comparison(
            X=X,
            functions=functions,
            lhs_terms=[('u_t', 1.0)],
            rhs_terms=[('u*u_x', -0.997335), ('u_xx', 0.099140)],
            snapshot=snapshot,
            results_dir=results_dir
        )
    else:
        # For u_t = - 6*u*u_x - u_xxx
        # compare results we got
        plot_pde_comparison(
            X=X,
            functions=functions,
            lhs_terms=[('u_t', 1.0)],
            rhs_terms=[('u*u_x', -5.964117), ('u_xxx', -0.987785)],  # Empty list since everything is on LHS
            snapshot=snapshot,
            results_dir=results_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    
    main(system=args.system)