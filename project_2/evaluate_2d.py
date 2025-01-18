"""
Main module that uses PDE-Find for 2D coupled PDE system
- Finishes prediction of the governing equations of the PDE system
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from visualization import plot_2d_heatmap_anim

from optimizers import (
generalized_condition_number,
STRidge,
TrainSTRidge
)

from feature_library import print_discovered_equation

def prepare_tensors(U, V, X, Y, T):
    # Convert to torch tensors and enable gradient tracking
    u = torch.from_numpy(U).float().requires_grad_(True)
    v = torch.from_numpy(V).float().requires_grad_(True)
    x = torch.from_numpy(X).float().requires_grad_(True)
    y = torch.from_numpy(Y).float().requires_grad_(True)
    t = torch.from_numpy(T).float().requires_grad_(True)
    return u, v, x, y, t

def collect_candidates_torch_grad(f_array, f_symbols, dx, dy):
    candidates = {}
    for f, f_symbol in zip(f_array, f_symbols):

        # Original function
        candidates[f_symbol] = f

        # print(f"Currently computing derivatives for function {f_symbol}")
        # First derivatives
        f_x = torch.gradient(f, spacing=[dx], dim=0)[0]
        f_y = torch.gradient(f, spacing=[dy], dim=1)[0]
        
        candidates[f_symbol + '_x'] = f_x
        candidates[f_symbol + '_y'] = f_y
        
        # Second derivatives
        f_xx = torch.gradient(f_x, spacing=[dx], dim=0)[0]
        f_yy = torch.gradient(f_y, spacing=[dy], dim=1)[0]
        f_xy = torch.gradient(f_x, spacing=[dy], dim=1)[0]
        
        candidates[f_symbol + '_xx'] = f_xx
        candidates[f_symbol + '_yy'] = f_yy
        candidates[f_symbol + '_xy'] = f_xy
        
        # Powers
        candidates[f_symbol + '**4'] = f**4
        candidates[f_symbol + '**3'] = f**3
        candidates[f_symbol + '**2'] = f**2

    return candidates

def main(create_gif=False):
    system = 3
    path = f'data/{system}.npz'
    name = "Reaction-Diffusion Equation"

    # Create results directory for the current system
    results_dir = f"results/system_{system}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Testing on dataset loaded from {path}")
    vectorial_data = np.load(path)

    # Load the arrays from the file
    U = vectorial_data['u']
    V = vectorial_data['v']
    X = vectorial_data['x']
    Y = vectorial_data['y']
    T = vectorial_data['t']

    # Get domain information
    nx, ny, nt = U.shape
    t_tot = T[0, -1, -1]
    dt = 0.05
    x_end = X.max(); x_start = X.min(); lx = x_end - x_start; dx = lx / nx
    y_end = Y.max(); y_start = Y.min(); ly = y_end - y_start; dy = ly / ny

    # (256, 256) 2D spatial points, 201 time nt in total
    print(f"({nx}, {ny}) 2D spatial points, {nt} time nt in total")

    # Convert to tensors
    u, v, x, y, t = prepare_tensors(U, V, X, Y, T)

    #=================================================
    # Computing derivatives
    #==================================================    
    candidates = collect_candidates_torch_grad([u, v], ["u", "v"], dx, dy)

    # Add products up to power of 4 in sum
    candidates['u*v'] = u * v

    candidates['u*v**2'] = u * candidates['v**2']
    candidates['u**2*v'] = candidates['u**2'] * v

    candidates['u*v**3'] = u * candidates['v**3']
    candidates['u**3*v'] = candidates['u**3'] * v

    # Flatten derivatives into shape (nx*ny*nt)x1 column vectors
    candidates_flat = {symbol: derivative.flatten() for symbol, derivative in candidates.items()}
    print(f"candidates {list(candidates.keys())}")

    #==================================================
    # Assemble LSE
    #==================================================

    D = len(list(candidates.keys()))
    Theta = torch.zeros((nx*ny*nt, D))
    print(f"{D} candidates are used, we built Theta matrix of shape {Theta.shape}")

    for j in range(D):
        symbol, vec = list(candidates_flat.items())[j]
        Theta[:, j] = vec

    u_t = torch.gradient(u, spacing=[dt], dim=2)[0]
    v_t = torch.gradient(v, spacing=[dt], dim=2)[0]

    c_theta = generalized_condition_number(Theta.cpu().detach().numpy())
    print(f"condition number of Theta is {c_theta}")
    print(f"shape of u_t is {u_t.shape}")

    # Sample only certain rows of Theta and u_t, v_t
    torch.manual_seed(0)
    num_samples = 10000
    indices = torch.randperm(nx*ny*nt)[:num_samples]

    Theta = Theta[indices]
    u_t = u_t.flatten()[indices]
    v_t = v_t.flatten()[indices]

    #==================================================
    # Sparse regression for LSE
    #==================================================
    # Assembling Theta and u_t
    Theta_np = Theta.cpu().detach().numpy()
    u_t_np = u_t.cpu().detach().numpy().reshape(-1, 1)
    v_t_np = v_t.cpu().detach().numpy().reshape(-1, 1)

    print(f"shape of Theta_np is {Theta_np.shape}")
    print(f"shape of u_t_np is {u_t_np.shape}")
    print(f"shape of v_t_np is {v_t_np.shape}")

    λ = 1e-6
    maxiter = 60
    split = 0.8
    η = 1e-3
    d_tol = 5e-3
    STR_iters = 10

    # Find best coefficients for both equations
    for rhs, symbol in [(u_t_np, "u_t"), (v_t_np, "v_t")]:
        ξ_best = TrainSTRidge(
            Theta_np,
            rhs,
            λ=λ,
            d_tol=d_tol,
            maxiter=maxiter,
            STR_iters=STR_iters,
            η=η,
            split=split,
            print_best_tol=False
        )
        # After running ridge regression:
        print_discovered_equation(candidates, ξ_best, f_symbol=symbol)

        # Print error
        error = np.linalg.norm(rhs - Theta_np @ ξ_best, 2) / np.linalg.norm(rhs, 2)

        print(f"Relative L2 error {error * 100}% ")

    # Generate animations for both U and V (original data)
    if create_gif:
        print("\nProcessing U variable...")
        plot_2d_heatmap_anim(U, T, X, Y, "u", results_dir)
        
        print("\nProcessing V variable...")
        plot_2d_heatmap_anim(V, T, X, Y, "v", results_dir)

    print(f"\nAll results saved in {results_dir}")

if __name__ == "__main__":
    main()