"""
This module generates all datasets used in the training of the "Foundation Model for Phase-Field Dynamics" project.

For all sampled initial conditions, we ensure the periodic boundary condition is fulfilled for u(x, 0).

Code reference and inspirations taken from:

- Fourier
    - General: https://databookuw.com/

- Gaussian
    - component normalization to [-1, 1]: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
"""

import numpy as np

import scipy
import torch
import torch.nn.functional as F
import os
import torch.nn as nn

import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Dict
from dataclasses import dataclass

from utils import (
print_bold,
get_dataset_folder_name
)
import json
import argparse
from pathlib import Path
from matplotlib import animation

@dataclass
class DomainConfig:
    """Configuration for the 1D spatiotemporal domain"""
    # Spatial domain
    x_min: float = -1.0
    x_max: float = 1.0
    nx: int = 128

    # Temporal domain
    nt: int   = 5
    dt: float = 0.005
 
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)
    
    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.nx) # (128, )

    @property
    def time_points(self) -> np.ndarray:
        return np.arange(0, self.nt * self.dt, self.dt) # (5, ) [0.0, 0.25, 0.50, 0.75, 1.0]


def enforce_normalization(f):
    """Enforce normalization to domain [-1, 1]"""
    return 2 * (f - min(f)) / ( max(f) - min(f) ) - 1

class FunctionSampler:
    """Base class for function samplers"""
    def sample(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class PiecewiseLinearSampler(FunctionSampler):
    """Samples piecewise linear functions"""
    def __init__(self, y_scale: float = 1.0, n_pieces: int = 4):
        self.y_scale = y_scale
        self.n_pieces = n_pieces # could be None, initialized at sampling time
            
    def sample(self, x: np.ndarray, seed = None) -> np.ndarray:
        """Sample a piecewise linear function"""
        if seed is not None:
            seed = np.random.seed(seed)

        if self.n_pieces is None:
            self.n_pieces = np.random.randint(4, 6)

        # Generate random 1D breakpoints
        breakpoints = np.sort(np.random.uniform(x.min(), x.max(), self.n_pieces))
        
        # Generate random y values at breakpoints, ensuring -1 and 1 appear
        y_values = np.random.uniform(-self.y_scale, self.y_scale, self.n_pieces)
        
        # Randomly select two different positions to place -1 and 1
        positions = np.random.choice(range(self.n_pieces - 1), size=2, replace=False)
        y_values[positions[0]] = -self.y_scale  # Place -1
        y_values[positions[1]] = self.y_scale   # Place 1
        
        # Enforcing periodic boundary condition
        y_values[-1] = y_values[0]

        # Interpolate
        return np.interp(x, breakpoints, y_values)

class GaussianMixtureSampler(FunctionSampler):
    """Samples Gaussian mixture"""
    def __init__(self, y_scale: float = 1.0, n_components: int = None):
        self.n_components = n_components # could be None, initialized at sampling time

    def sample(self, x: np.ndarray, seed = None) -> np.ndarray:
        if seed is not None:
            seed = np.random.seed(seed)
    
        if self.n_components is None:
            self.n_components = np.random.randint(2, 5)

        # Generate means
        # Separate the interval [-0.6, 0.6] into sub-intervals for more evenly-spread Gaussians
        start = -0.60; end = 0.60
        width = (end - start) / self.n_components
        means = np.zeros(self.n_components)
        for i in range(self.n_components):
            sub_start = start + i * width
            sub_end = sub_start + width
            means[i] = np.random.normal(loc=(sub_start + sub_end) / 2, scale=width / 4)
            
        # Generate variance
        sigmas = np.random.uniform(0.1, 0.3, self.n_components)  # Adjusted sigma range
        variances = sigmas**2

        # Generate weights
        weights = scipy.special.softmax(np.random.rand(self.n_components))
        assert np.isclose(sum(weights), 1.0), "Sum of weights for Gaussian components must equal to 1"

        # Compute GMM component
        u0 = np.zeros_like(x)
        for i in range(self.n_components):
            f_i = self.gaussian(x, means[i], variances[i])
            u0 += weights[i] * f_i # Add with weight

        # Enforce periodic BC
        bc_error = np.abs(u0[-1] - u0[0])
        if bc_error > 1e-4:
            u0 -= (u0[-1] - u0[0]) * np.linspace(0, 1, len(x))

        # Normalize the final values of u0 to [-1, 1]
        return enforce_normalization(u0)

    def gaussian(self, x, mean, variance):
        """
        N(µ, σ²) = 1/√(2π σ²) · exp(-(x-µ)²/(2σ²))
        """
        return 1.0 / np.sqrt(2*np.pi * variance) * np.exp(-(x-mean)**2 / (2 * variance))


class FourierSeriesSampler(FunctionSampler):
    """
    Samples a truncated Fourier series up to `n_modes`.
        - periodic boundary condition embedded by specifying domain length L
    """

    def __init__(self, n_modes: int = 10, L: float = 2.0):
        self.n_modes = n_modes
        self.L       = L # domain length

    def sample(self, x: np.ndarray, seed = None) -> np.ndarray:    
        if seed is not None:
            seed = np.random.seed(seed)

        u0 = np.zeros_like(x)

        # Generate coefficients for Fourier series
        a0 = np.random.normal()
        a  = np.random.normal(size = self.n_modes-1)
        b  = np.random.normal(size = self.n_modes-1)

        # Compute the Fourier series up to specified mode
        for k in range(1, self.n_modes - 1):
            u0 += a[k] * np.cos((2*np.pi*k*x) / self.L) + b[k] * np.sin((2*np.pi*k*x)/self.L)

        u0 += 0.5 * a0
        
        # Normalize to [-1, 1]
        return enforce_normalization(u0)

def allen_cahn_rhs(t, u, epsilon, x):
    """Implement Allen-Cahn equation RHS:
        ∂u/∂t = Δu - (1/ε²)(u³ - u)
    """
    dx = x[1] - x[0]

    u = torch.from_numpy(u)
    
    # Compute 1D Laplacian (Δu) with periodic boundary conditions
    u_x  = torch.gradient(u, spacing=[dx], dim=0, edge_order=2)[0]
    u_xx = torch.gradient(u_x, spacing=[dx], dim=0, edge_order=2)[0]

    # Compute nonlinear term (1/ε²)(u³ - u)
    nonlinear_term = (1.0 / epsilon**2) * (u**3 - u)

    # Return full RHS
    return u_xx - nonlinear_term

class DatasetGenerator:
    """Main class for generating the Allen-Cahn equation dataset"""
    def __init__(self, domain: DomainConfig, epsilon_values: list, samplers: dict = None):
        self.domain         = domain
        self.epsilon_values = epsilon_values

        if samplers == None:
            self.samplers = {
            'PL': PiecewiseLinearSampler(),
            'GM': GaussianMixtureSampler(),
            'FS': FourierSeriesSampler()
            }
        else:
            self.samplers = samplers
    
    def generate_dataset(self, n_samples, base_seed=None):
        """Generate dataset according to Algorithm 1"""
        dataset = {}
        
        for sampler_name, sampler in self.samplers.items():
            print_bold(f"Generating {sampler_name} samples for ɛ in {self.epsilon_values}...")
            dataset[sampler_name] = {}

            epsilon_dict = {}
            for eps in self.epsilon_values:
                print(f"For ɛ = {eps}:")
                trajectories = np.zeros((n_samples, len(self.domain.time_points), len(self.domain.x)))
                for i in range(0, n_samples-1):
                    # Sample initial condition
                    if (i + 1) % 100 == 0 or i == 0:
                        print(f"Generating sample {i + 1}/{n_samples}")
                    u0 = sampler.sample(self.domain.x, seed=base_seed+i if base_seed else None)
                    time_points = self.domain.time_points

                    # Solve PDE using solve_ivp
                    sol = scipy.integrate.solve_ivp(
                        allen_cahn_rhs,
                        t_span=(time_points[0], time_points[-1]),
                        y0=u0,
                        t_eval=time_points,
                        args=(eps, self.domain.x),
                        method='Radau',
                        rtol=1e-6,
                        atol=1e-6
                    )

                    # Store current trajectory
                    trajectories[i] = sol.y.T

                # Store trajectories for current epsilon value                
                epsilon_dict[eps] = trajectories

            # Store solution trajectory
            dataset[sampler_name] = epsilon_dict

            print(f"Completed {sampler_name} dataset generation")
        
        return dataset

    def plot_samples(self, dataset, n_samples: int = 2, snapshot: int = 0):
        """Plot some samples from each function class at initial time"""
        rows = len(self.samplers)
        cols = 1
        
        # Adjust figure size for one column
        fig, axes = plt.subplots(rows, cols, figsize=(8, 4*rows), squeeze=False)
        
        for i, (sampler_name, samples) in enumerate(dataset.items()):
            if sampler_name == "PL":
                ic_type = "Piecewise Linear Function"
            elif sampler_name == "GM":
                ic_type = "Gaussian Mixture"
            elif sampler_name == "FS":
                ic_type = "Fourier Series"                

            # Only use the first epsilon value since they're all the same at t=0
            eps = list(samples.keys())[0]
            trajectories = dataset[sampler_name][eps]

            # Plot samples of different idx
            for k in range(n_samples):
                u = trajectories[k]
                axes[i,0].plot(self.domain.x, u[0,:], label=f'Sample {k+1}')

            # Update title to show both u(x,0) and sampler name
            axes[i,0].set_title(f'{ic_type}: u(x, 0)', fontsize=12, pad=10)
            axes[i,0].set_ylim([-1.5, 1.5])
            axes[i,0].grid(True, linestyle='--', alpha=0.7)
            axes[i,0].legend(loc='upper right', framealpha=0.9, fontsize=10)

        # Adjust spacing between subplots
        plt.tight_layout()
        
        return fig

    def plot_sample_animation(self, dataset, n_samples: int = 3):
        """Create an animation of trajectory evolution over time"""
        rows = len(self.samplers)
        cols = len(self.epsilon_values)
        
        # Create figure with additional space on the right for sampler names
        fig = plt.figure(figsize=(3*cols + 1, 3*rows))
        
        # Create a grid specification with space for sampler names
        gs = plt.GridSpec(rows, cols + 1, width_ratios=[1]*cols + [0.2])
        
        # Store all axes objects
        axes = [[plt.subplot(gs[i, j]) for j in range(cols)] for i in range(rows)]
        axes = np.array(axes)
        
        # Store line objects for animation
        lines = []
        
        # Set simple, distinct colors
        basic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Standard matplotlib colors
        
        # Initialize plots
        for i, (sampler_name, samples) in enumerate(dataset.items()):
            epsilons = list(samples.keys())

            if sampler_name == "PL":
                ic_type = "Piecewise\nLinear\nFunction"
            elif sampler_name == "GM":
                ic_type = "Gaussian\nMixture"
            elif sampler_name == "FS":
                ic_type = "Fourier\nSeries"

            # Add sampler name on the right
            ax_text = plt.subplot(gs[i, -1])
            ax_text.text(0, 0.5, ic_type, 
                        rotation=0, 
                        verticalalignment='center',
                        fontsize=12,
                        fontweight='bold')
            ax_text.axis('off')
            
            for j in range(cols):
                eps = epsilons[j]
                trajectories = dataset[sampler_name][eps]
                
                # Initialize with empty data using basic colors
                for k in range(n_samples):
                    line, = axes[i,j].plot([], [], 
                                        label=f'Sample {k+1}',
                                        color=basic_colors[k % len(basic_colors)],
                                        linewidth=2)
                    lines.append(line)
                
                # Customize subplot appearance
                axes[i,j].set_title(f'ε = {eps}', pad=10, fontsize=10)
                axes[i,j].set_xlim(self.domain.x.min(), self.domain.x.max())
                axes[i,j].set_ylim(-1, 1)
                axes[i,j].legend(loc='upper right', fontsize=8)
                axes[i,j].grid(True, alpha=0.3)
                
                # Only show y-axis labels for leftmost plots
                if j != 0:
                    axes[i,j].set_yticklabels([])
                
                # Only show x-axis labels for bottom plots
                if i != rows-1:
                    axes[i,j].set_xticklabels([])
                    
                # Add some padding to the axes
                axes[i,j].tick_params(pad=8)
        
        def init():
            """Initialize animation"""
            for line in lines:
                line.set_data([], [])
            return lines
        
        def animate(frame):
            """Update animation at each frame"""
            line_idx = 0
            for sampler_name, samples in dataset.items():
                epsilons = list(samples.keys())
                for j in range(cols):
                    eps = epsilons[j]
                    trajectories = dataset[sampler_name][eps]
                    
                    for k in range(n_samples):
                        u = trajectories[k]
                        lines[line_idx].set_data(self.domain.x, u[frame,:])
                        line_idx += 1
            return lines
        
        # Create animation
        first_sampler = next(iter(dataset.values()))
        first_epsilon = next(iter(first_sampler.keys()))
        first_trajectory = first_sampler[first_epsilon][0]
        n_frames = first_trajectory.shape[0]
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=n_frames, 
                                    interval=1000,
                                    blit=True,
                                    repeat=True,
                                    repeat_delay=2000)
        plt.tight_layout()
        return fig, anim


def plot(domain: DomainConfig, added_epsilon_values:list, n_samples: int = 5):
    # Load the latest created dataset
    data_folders = sorted(Path(f'data').glob('dt_*'), key=lambda d: d.stat().st_mtime)
    data_folder  = data_folders[-1]
    print_bold(f"Loading dataset from {data_folder}")

    with open(f'{data_folder}/config.json', 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    time_points = np.array(config['temporal_grid']['time_points'])
    epsilon_values = config['dataset_params']['epsilon_values']
    generator = DatasetGenerator(domain, epsilon_values)
    eps_generator = DatasetGenerator(domain, added_epsilon_values)

    # Plot training samples
    print_bold(f"Plotting {n_samples} training samples")
    train_data_dict = np.load(f"{data_folder}/train_sol.npy", allow_pickle=True).item()
    fig = generator.plot_samples(train_data_dict, n_samples=n_samples)
    plt.savefig(f"{data_folder}/sample_comparison_train")
    plt.close()

    fig, anim = generator.plot_sample_animation(train_data_dict)
    anim.save(f'{data_folder}/sol_dt_{domain.dt}_train.gif', writer='pillow')
    plt.close()

    # Plot testing samples
    print_bold(f"Plotting {n_samples} testing samples")
    test_data_dict = np.load(f"{data_folder}/test_sol.npy", allow_pickle=True).item()
    fig = generator.plot_samples(test_data_dict, n_samples=n_samples)
    plt.savefig(f"{data_folder}/sample_comparison_test")
    plt.close()

    fig, anim = generator.plot_sample_animation(test_data_dict)
    anim.save(f'{data_folder}/sol_dt_{domain.dt}_test.gif', writer='pillow')
    plt.close()

    # Plot OOD testing samples
    print_bold(f"Plotting {n_samples} OOD testing samples")
    ood_test_data_dict = np.load(f"{data_folder}/test_sol_OOD.npy", allow_pickle=True).item()
    fig = generator.plot_samples(ood_test_data_dict, n_samples=n_samples)
    plt.savefig(f"{data_folder}/sample_comparison_ood")
    plt.close()

    fig, anim = generator.plot_sample_animation(ood_test_data_dict)
    anim.save(f'{data_folder}/sol_dt_{domain.dt}_test_ood.gif', writer='pillow')
    plt.close()

    # Plot testing samples with different ɛ
    print_bold(f"Plotting {n_samples} testing samples with different ɛ")
    ood_test_data_dict = np.load(f"{data_folder}/test_sol_eps.npy", allow_pickle=True).item()
    fig = eps_generator.plot_samples(ood_test_data_dict, n_samples=n_samples)
    plt.savefig(f"{data_folder}/sample_comparison_eps")
    plt.close()

    fig, anim = eps_generator.plot_sample_animation(ood_test_data_dict)
    anim.save(f'{data_folder}/sol_dt_{domain.dt}_test_eps.gif', writer='pillow')
    plt.close()


def generate(domain: DomainConfig, 
            epsilon_values: list,
            added_epsilon_values: list,
            base_seed = 1,
            n_train: int = 1000,
            n_test: int = 200,
            n_pieces: int = 6,
            n_components: int = 10,
            n_modes: int = 20
            ):
    #==================================================
    # Choose initial conditions
    #==================================================
    samplers = {
    'PL': PiecewiseLinearSampler(),
    'GM': GaussianMixtureSampler(),
    'FS': FourierSeriesSampler()
    }

    ood_samplers = {
    'PL': PiecewiseLinearSampler(n_pieces = n_pieces),
    'GM': GaussianMixtureSampler(n_components = n_components),
    'FS': FourierSeriesSampler(n_modes = n_modes)
    }

    # Create a configuration dictionary
    config = {
        'samplers': str(samplers.keys()),
        'params': {
            'n_pieces':     samplers['PL'].n_pieces,
            'n_components': samplers['GM'].n_components,
            'n_modes':      samplers['FS'].n_modes,
        },
        'ood_params': {
            'n_pieces':     ood_samplers['PL'].n_pieces, 
            'n_components': ood_samplers['GM'].n_components,
            'n_modes':      ood_samplers['FS'].n_modes,
        },
        'spatial_grid': {
            'nx': domain.nx,
            'x_min': domain.x_min,
            'x_max': domain.x_max,
            'x_grid': domain.x.tolist()  # Convert numpy array to list for JSON serialization
        },
        'temporal_grid': {
            'dt': domain.dt,
            'time_points': domain.time_points.tolist()  # Convert numpy array to list
        },
        'dataset_params': {
            'epsilon_values': epsilon_values,
            'added_epsilon_values': added_epsilon_values,
            'n_train': n_train,
            'n_test': n_test,
            'base_seed': base_seed
        }
    }
    data_dir = f"data/{get_dataset_folder_name(domain.dt)}"
    os.makedirs(data_dir, exist_ok=True)
    with open(f"{data_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    #==================================================
    # Generate dataset
    #==================================================
    # Create generator
    generator     = DatasetGenerator(domain, epsilon_values, samplers)
    ood_generator = DatasetGenerator(domain, epsilon_values, ood_samplers)
    eps_generator = DatasetGenerator(domain, added_epsilon_values, samplers)

    # Generate training & test datasets for each epsilon and IC type
    print_bold(f"Generating Training dataset with {n_train} samples")
    train_dataset = generator.generate_dataset(n_samples=n_train, base_seed=base_seed)
    np.save(f"{data_dir}/train_sol.npy", train_dataset)

    print_bold(f"Generating Testing dataset with {n_test} samples")
    test_dataset = generator.generate_dataset(n_samples=n_test, base_seed=base_seed+n_train)
    np.save(f"{data_dir}/test_sol.npy", test_dataset)

    # Generate OOD test datasets (high frequency, sharp transitions)
    # - Use same epsilon values, different sampler setups!
    print_bold(f"Generating OOD Testing dataset with {n_test} samples (Varying samplers)")
    ood_test_dataset = ood_generator.generate_dataset(n_samples=n_test, base_seed=base_seed+n_train+n_test)
    np.save(f"{data_dir}/test_sol_OOD.npy", ood_test_dataset)

    # Generate test datasets (ɛ value interpolation, extrapolation)
    # - Use different epsilon values, same sampler setups!
    print_bold(f"Generating Testing dataset with {n_test} samples (Extra-, interpolation of ɛ)")
    epsilon_test_dataset = eps_generator.generate_dataset(n_samples=n_test, base_seed=base_seed+n_train+n_test+n_test)
    np.save(f"{data_dir}/test_sol_eps.npy", epsilon_test_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generator and plotter')
    parser.add_argument('--plot', action='store_true', help='Only plot the data')
    parser.add_argument('--generate', action='store_true', help='Only generate data without plotting')
    
    args = parser.parse_args()
    
    # Specify domain config
    time_scale = 1e-2
    # dt = 0.50 * time_scale
    dt = 0.25 * time_scale
    domain = DomainConfig(x_min=-1.0, x_max=1.0, nx=128, nt = 5, dt = dt)

    # Specify parameters
    epsilon_values = [10.0, 0.5, 0.1, 0.05, 0.01]

    # Additional epsilon value (Extrapolation and interpolation)
    added_epsilon_values = [1000.0, 100.0, 5.0, 1.0, 0.75, 0.008, 0.006]

    # In all
    # [1000.0, 100.0, 10.0, 5.0, 1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.008, 0.006]
    # Original
    # [      ,      , 10.0,    ,    ,     , 0.5, 0.25, 0.1, 0.05, 0.01,      ,       ]
    # Interpolation
    # [      ,      ,     , 5.0, 1.0, 0.75,     ,    ,    ,     ,     ,     ,       ]
    # Extrapolation
    # [1000.0, 100.0,     ,    ,    ,     ,     ,     ,    ,     ,     , 0.008, 0.006]

    if not args.plot and not args.generate:
        generate(domain, epsilon_values, added_epsilon_values)  # If no flags are provided, do both operations
        plot(domain, added_epsilon_values)
    elif args.plot:
        plot(domain, added_epsilon_values) # plot only
    elif args.generate:
        generate(domain, epsilon_values, added_epsilon_values) # generate only