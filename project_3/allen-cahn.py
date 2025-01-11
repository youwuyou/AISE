"""
Code reference and inspirations taken from:

- Fourier
    - General: https://databookuw.com/

- Gaussian
    - General: https://katbailey.github.io/post/gaussian-processes-for-dummies/
    - component normalization to [-1, 1]: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
"""

import numpy as np

import scipy
import torch
import torch.nn.functional as F
import os
import torch.nn as nn

import matplotlib.pyplot as plt

def enforce_normalization(f):
    """Enforce normalization to domain [-1, 1]"""
    return 2 * (f - min(f)) / ( max(f) - min(f) ) - 1

def generate_fourier_ic(x, n_modes=10, seed=None, L:float = 2.0):
    """Generate random L-periodic Fourier series initial condition.
    Hints:
    1. Use random coefficients for sin and cos terms
    2. Ensure the result is normalized to [-1, 1]
    3. Consider using np.random.normal for coefficients
    """
    if seed is not None:
        np.random.seed(seed)

    u0 = np.zeros_like(x)

    # Generate coefficients for Fourier series
    a0 = np.random.normal()
    a  = np.random.normal(size = n_modes-1)
    b  = np.random.normal(size = n_modes-1)

    # Compute the Fourier series up to specified mode
    for k in range(1, n_modes - 1):
        u0 += a[k] * np.cos((2*np.pi*k*x) / L) + b[k] * np.sin((2*np.pi*k*x)/L)

    u0 += 0.5 * a0
    
    # Normalize to [-1, 1]
    return enforce_normalization(u0)

def gaussian(x, mean, variance):
    """
    N(µ, σ²) = 1/√(2π σ²) · exp(-(x-µ)²/(2σ²))
    """
    return 1.0 / np.sqrt(2*np.pi * variance) * np.exp(-(x-mean)**2 / (2 * variance))

def generate_gmm_ic(x, n_components=None, seed=None):
    """Generate Gaussian mixture model initial condition.
    Hints:
    1. Random number of components if n_components is None
    2. Use random means, variances, and weights
    3. Ensure result is normalized to [-1, 1]

    Returns:
        u0: Normalized GMM values in range [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_components is None:
        n_components = np.random.randint(2, 5)
    
    # Generate means
    # Separate the interval [-0.6, 0.6] into sub-intervals for more evenly-spread Gaussians
    start = -0.60; end = 0.60
    width = (end - start) / n_components
    means = np.zeros(n_components)
    for i in range(n_components):
        sub_start = start + i * width
        sub_end = sub_start + width
        means[i] = np.random.normal(loc=(sub_start + sub_end) / 2, scale=width / 4)
        
    # Generate variance
    sigmas = np.random.uniform(0.1, 0.3, n_components)  # Adjusted sigma range
    variances = sigmas**2

    # Generate weights
    weights = scipy.special.softmax(np.random.rand(n_components))
    assert np.isclose(sum(weights), 1.0), "Sum of weights for Gaussian components must equal to 1"

    # Compute GMM component
    u0 = np.zeros_like(x)
    for i in range(n_components):
        f_i = gaussian(x, means[i], variances[i])
        u0 += weights[i] * f_i # Add with weight

    # Enforce periodic BC
    bc_error = np.abs(u0[-1] - u0[0])
    if bc_error > 1e-4:
        u0 -= (u0[-1] - u0[0]) * np.linspace(0, 1, len(x))

    # Normalize the final values of u0 to [-1, 1]
    return enforce_normalization(u0)

def generate_piecewise_ic(x, n_pieces=None, seed=None, y_scale: float = 1.0):
    """Generate piecewise linear initial condition.
    Hints:
    1. Generate random breakpoints
    2. Create piecewise linear function
    3. Add occasional discontinuities
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_pieces is None:
        n_pieces = np.random.randint(3, 7)

    # Generate random 1D breakpoints
    breakpoints = np.sort(np.random.uniform(x.min(), x.max(), n_pieces))
    
    # Generate random nodal values at breakpoints, default between [-1.0, 1.0]
    y_values = np.random.uniform(-y_scale, y_scale, n_pieces)

    # Enforcing periodic boundary condition
    y_values[-1] = y_values[0]
    
    # Create piecewise linear function by interpolation
    return np.interp(x, breakpoints, y_values)

def allen_cahn_rhs(t, u, epsilon, x_grid):
    """Implement Allen-Cahn equation RHS:
        ∂u/∂t = Δu - (1/ε²)(u³ - u)
    """
    dx = x_grid[1] - x_grid[0]

    u = torch.from_numpy(u)
    
    # Compute 1D Laplacian (Δu) with periodic boundary conditions
    u_x  = torch.gradient(u, spacing=[dx], dim=0, edge_order=2)[0]
    u_xx = torch.gradient(u_x, spacing=[dx], dim=0, edge_order=2)[0]

    # Compute nonlinear term (1/ε²)(u³ - u)
    nonlinear_term = (1.0 / epsilon**2) * (u**3 - u)

    # Return full RHS
    return u_xx - nonlinear_term

def generate_dataset(n_samples, epsilon: float, x_grid, t_eval, ic_type='fourier', seed=None):
    """Generate dataset for Allen-Cahn equation."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize dataset array
    dataset = np.zeros((n_samples, len(t_eval), len(x_grid)))
    
    # Generate samples
    for i in range(n_samples):
        # Generate initial condition based on type
        print(f"Generating {i}-th sample for {ic_type}")
        if ic_type == 'fourier':
            u0 = generate_fourier_ic(x_grid, seed=seed+i if seed else None)
        elif ic_type == 'gmm':
            u0 = generate_gmm_ic(x_grid, seed=seed+i if seed else None)
        elif ic_type == 'piecewise':
            u0 = generate_piecewise_ic(x_grid, seed=seed+i if seed else None)
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")
        
        # Solve PDE using solve_ivp
        sol = scipy.integrate.solve_ivp(
            allen_cahn_rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=t_eval,
            args=(epsilon, x_grid),
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        dataset[i] = sol.y.T
    
    return dataset

def main():
    """Generate all datasets."""
    # Set up spatial grid
    nx = 128
    x_grid = np.linspace(-1, 1, nx) # (128, )
    print(f"shape of x_grid: {x_grid.shape}")

    # Set up temporal grid
    dt = 0.25
    t_eval = np.arange(0, 5*dt, dt) # (5, ) [0.0, 0.25, 0.50, 0.75, 1.0]
    print(f"t_eval = {t_eval}")
    
    # Parameters for datasets
    epsilons = [0.1, 0.05, 0.02]  # Different epsilon values
    n_train = 1000  # Number of training samples per configuration
    n_test = 200    # Number of test samples
    base_seed = 1  # For reproducibility

    # TODO: scale the time depending on values of epsilon
    ic_types = ['fourier', 'gmm', 'piecewise']
    for ic_type in ic_types:
        # Create data directory for storing different types of IC
        data_dir = f"data/{ic_type}"
        os.makedirs(data_dir, exist_ok=True)

        # Start from base seed for current datatype
        seed = base_seed
        for epsilon in epsilons:
            data_dir = f"data/{ic_type}/dt_{dt}_ep_{epsilon}"
            os.makedirs(data_dir, exist_ok=True)

            # Generate training datasets for each epsilon and IC type
            train_dataset = generate_dataset(n_train, epsilon, x_grid, t_eval, ic_type=ic_type, seed=seed)
            np.save(f"{data_dir}/train_sol.npy", train_dataset)

            # Generate standard test dataset
            test_dataset = generate_dataset(n_test, epsilon, x_grid, t_eval, ic_type=ic_type, seed=seed+n_train)
            np.save(f"{data_dir}/test_sol.npy", test_dataset)


    # TODO: Generate OOD test datasets (high frequency, sharp transitions)
    # TODO: Save all datasets using np.save

    # Testing each data generation code
    # Create a figure with 3 subplots arranged horizontally
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plot piecewise
    u0_pw = generate_piecewise_ic(x_grid, seed=None)
    ax1.plot(x_grid, u0_pw)
    ax1.set_title('Final PW')

    # Plot GMM
    u0_gmm = generate_gmm_ic(x_grid, n_components=None, seed=None)
    ax2.plot(x_grid, u0_gmm)
    ax2.set_title('Final GMM')

    # Plot Fourier
    u0_fourier = generate_fourier_ic(x_grid, seed=None)
    ax3.plot(x_grid, u0_fourier)
    ax3.set_title('Final Fourier')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig("combined_plots.png")
    plt.close()

if __name__ == '__main__':
    main()
