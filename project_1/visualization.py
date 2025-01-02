"""
Utility functions for plotting results used in report of project 1
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats

#============================================
#  General & Time-independent Visualization
#============================================

def plot_training_history(experiment_dir: Path, save_dir: Optional[Path] = None) -> None:
    experiment_dir = Path(experiment_dir)
    save_dir = Path(save_dir) if save_dir else experiment_dir
    save_dir.mkdir(exist_ok=True)
    
    with open(experiment_dir / 'training_config.json', 'r') as f:
        config = json.load(f)
        data_mode = config['training_config']['data_mode']
        history = config['training_history']

    if data_mode == "onetoone":
        data_mode = "One-to-One"
    elif data_mode == "all2all":
        data_mode = "All2All"
    elif data_mode == "onetoall":
        data_mode = "One-to-All"        
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
        
    plt.title(f'Training History - FNO ({data_mode})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend()
    
    plt.savefig(save_dir / 'training_history.png',
               bbox_inches='tight', dpi=300)
    plt.close()

def plot_resolution_comparison(model: torch.nn.Module, 
                             data_dict: Dict,
                             results_dict: Dict,
                             save_dir: Path,
                             title: Optional[str] = None) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    resolutions = sorted(data_dict.keys())
    colors = {
        'True': '#000000',
        'Custom FNO': 'tab:orange'
    }
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    axes = np.array(axes)
    
    fig.suptitle(title if title else 'Model Prediction Across Resolutions', 
                fontsize=16, y=0.95, weight='bold')
    
    global_min = float('inf')
    global_max = float('-inf')
    
    for res in resolutions:
        sample_idx = 0
        true_sol = data_dict[res]['custom'][1][sample_idx].cpu().detach()
        global_min = min(global_min, true_sol.min().item())
        global_max = max(global_max, true_sol.max().item())
        
        pred = results_dict['Custom FNO']['predictions'][res][sample_idx].cpu().detach()
        global_min = min(global_min, pred.min().item())
        global_max = max(global_max, pred.max().item())
    
    y_padding = (global_max - global_min) * 0.15
    y_min = global_min - y_padding
    y_max = global_max + y_padding
    
    for idx, res in enumerate(resolutions):
        i, j = idx // 2, idx % 2
        sample_idx = 0
        x_grid = torch.linspace(0, 1, res).cpu()
        true_sol = data_dict[res]['custom'][1][sample_idx].cpu().detach()
        
        axes[i, j].plot(x_grid, true_sol,
                       '-', color=colors['True'], 
                       label='Ground Truth', 
                       linewidth=2, 
                       alpha=0.8)
        
        error = results_dict['Custom FNO']['errors'][idx]
        pred = results_dict['Custom FNO']['predictions'][res][sample_idx].cpu().detach()
        
        axes[i, j].plot(x_grid, 
                      pred,
                      '--', 
                      color=colors['Custom FNO'],
                      label='Custom FNO',
                      linewidth=1.5,
                      alpha=0.8)
        
        marker_size = max(3, 15 // (res/32))
        marker_alpha = max(0.2, 0.5 // (res/32))

        axes[i, j].scatter(x_grid, 
                         pred,
                         marker='o',
                         color=colors['Custom FNO'],
                         s=marker_size,
                         alpha=marker_alpha)
    
        axes[i, j].set_title(f'Resolution: {res} points', pad=10, fontsize=12, weight='bold')
        axes[i, j].grid(True, linestyle='--', alpha=0.3)
        axes[i, j].legend(loc='upper right', framealpha=0.9)
        axes[i, j].set_xlabel('x', fontsize=10)
        axes[i, j].set_ylabel('u(x, t = 1)', fontsize=10)
        
        axes[i, j].set_ylim(y_min, y_max)
        axes[i, j].set_xlim(-0.02, 1.02)
        
        axes[i, j].grid(True, which='major', linestyle='-', alpha=0.2)
        axes[i, j].grid(True, which='minor', linestyle=':', alpha=0.1)
        axes[i, j].minorticks_on()
        
        for spine in axes[i, j].spines.values():
            spine.set_linewidth(0.8)
            
        error_text = f'Custom FNO: {error:.2f}%'
        axes[i, j].set_title(f'Resolution: {res} points\n{error_text}', 
                           pad=10, fontsize=12, weight='bold')

    plt.savefig(save_dir / 'resolution_comparison.png', 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

def plot_l2_error_by_resolution(results: Dict[str, Dict[str, List[float]]], 
                              resolutions: List[int],
                              save_dir: Path,
                              training_resolution: Optional[int] = 64,
                              title: Optional[str] = None) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {'Custom FNO': '#1f77b4'}
    ax.grid(True, linestyle='--', alpha=0.2, which='both')
    ax.set_axisbelow(True)
    
    model_results = results['Custom FNO']
    line = ax.plot(resolutions, 
                  model_results['errors'],
                  '-',
                  label='Custom FNO',
                  color=colors['Custom FNO'],
                  linewidth=2.5,
                  zorder=3)
    
    ax.plot(resolutions,
            model_results['errors'],
            'o',
            color=colors['Custom FNO'],
            markersize=10,
            markeredgewidth=2,
            markeredgecolor='white',
            zorder=4)
    
    ax.plot(resolutions,
            model_results['errors'],
            'o',
            color=colors['Custom FNO'],
            markersize=6,
            zorder=5)
    
    for j, (x, y) in enumerate(zip(resolutions, model_results['errors'])):
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        ax.annotate(f'{y:.2f}%', 
                   (x, y),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   va='bottom',
                   fontsize=10,
                   bbox=bbox_props,
                   zorder=6)
    
    if training_resolution is not None:
        ymin = min(model_results['errors'])
        ymax = max(model_results['errors'])
        padding = (ymax - ymin) * 0.1
        y_limits = (ymin - padding * 2, ymax + padding * 3)
        
        ax.vlines(x=training_resolution, ymin=y_limits[0], ymax=y_limits[1], 
                  colors='grey', linestyles='--', label='Training Resolution',
                  linewidth=1.5, zorder=2, alpha=0.7)
    
    ax.set_xlabel('No. Nodal Points', fontsize=12, labelpad=10)
    ax.set_ylabel('Average Relative L2 Error (%)', fontsize=12, labelpad=10)
    ax.set_title(title if title else 'Test Error Across Resolutions', 
                 fontsize=14, 
                 pad=20,
                 weight='bold')
    
    ax.set_xticks(resolutions)
    ax.set_xticklabels(resolutions, fontsize=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    ymin = min(model_results['errors'])
    ymax = max(model_results['errors'])
    padding = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - padding * 2, ymax + padding * 3)
    
    legend = ax.legend(loc='upper right',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='gray',
                      fancybox=True,
                      fontsize=10)
    legend.get_frame().set_linewidth(0.5)
    
    plt.savefig(save_dir / 'l2_error_resolution_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_error_distributions(in_dist_results, ood_results, save_path):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    
    model_colors = {'Custom FNO': '#1f77b4'}
    stat_styles = {
        'mode': '--',
        'mean': '-', 
        'median': ':',
    }

    all_errors = []
    max_density = 0
    for results in [in_dist_results, ood_results]:
        errors = np.array(results['Custom FNO']['individual_errors'])
        all_errors.extend(errors)
        counts, _ = np.histogram(errors, bins=30, density=True)
        max_density = max(max_density, np.max(counts))

    global_xlim = (0, np.ceil(max(all_errors)))
    global_ylim = (0, max_density * 1.2)

    plt.figtext(0.25, 0.95, 'In-Distribution Data', fontsize=15, weight='bold', ha='center')
    plt.figtext(0.75, 0.95, 'Out-of-Distribution Data', fontsize=15, weight='bold', ha='center')

    stat_lines = []
    stat_labels = []
    for dist_idx, results in enumerate([in_dist_results, ood_results]):
        ax = fig.add_subplot(gs[0, dist_idx])
        
        errors = np.array(results['Custom FNO']['individual_errors'])
        plt.hist(errors, bins=30, density=True, alpha=0.5,
                color=model_colors['Custom FNO'],
                edgecolor='white', linewidth=1)
        
        counts, bin_edges = np.histogram(errors, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        stats = {
            'mode': bin_centers[np.argmax(counts)],
            'mean': np.mean(errors),
            'median': np.median(errors)
        }
        
        for stat, style in stat_styles.items():
            stat_value = stats[stat]
            line = plt.axvline(stat_value, color=model_colors['Custom FNO'],
                            linestyle=style, alpha=1.0, linewidth=2)
            if dist_idx == 0:
                stat_lines.append(line)
                stat_labels.append(stat.capitalize())
            
            if stat == 'mean':
                label = f'Mean: {stat_value:.2f}%'
                bbox_props = dict(
                    boxstyle='round,pad=0.5',
                    fc='white',
                    ec=model_colors['Custom FNO'],
                    alpha=0.8
                )
                plt.text(stat_value + 0.5, global_ylim[1] * 0.8,
                        label,
                        color=model_colors['Custom FNO'],
                        fontsize=12,
                        fontweight='bold',
                        bbox=bbox_props,
                        horizontalalignment='left')
        
        plt.xlabel('Test Error (%)', fontsize=11)
        plt.ylabel('Density', fontsize=11)
        plt.grid(True, alpha=0.2)
        plt.xlim(global_xlim)
        plt.ylim(global_ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.legend(stat_lines, stat_labels,
                title='Statistics',
                loc='upper right',
                bbox_to_anchor=(0.93, 0.93),
                labelcolor='#404040')

    plt.subplots_adjust(right=0.9, top=0.85)
    plt.savefig(save_path.parent / 'ood_error_distribution_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

#============================================
#  Time-dependent Visualization
#============================================
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator

def plot_l2_error_by_timestep(results: Dict[str, Union[float, Dict[float, float]]], 
                             t_i: List[float],
                             save_dir: Path,
                             title: Optional[str] = None) -> None:
    """
    Modified to work with both single error values and error curves.
    
    Args:
        results: Dictionary with model names and their error values/curves
        t_i: List of timesteps
        save_dir: Directory to save the plot
        title: Optional title for the plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors for different models
    colors = {
        'Direct': 'red',
        'AR': 'dodgerblue'
    }
    
    ax.grid(True, linestyle='--', alpha=0.2, which='both')
    ax.set_axisbelow(True)
    
    max_error = 0
    
    for model_name, errors in results.items():
        if model_name == 'error':
            continue  # Skip the final error entry
            
        if isinstance(errors, dict):
            # Plot error curve
            error_values = [errors[t] for t in t_i]
            max_error = max(max_error, max(error_values))
            
            # Plot line
            ax.plot(t_i, error_values,
                   '-',
                   label=model_name,
                   color=colors.get(model_name, 'dodgerblue'),
                   linewidth=2.5,
                   zorder=3)
            
            # Add markers
            ax.plot(t_i, error_values,
                   'o',
                   color=colors.get(model_name, 'dodgerblue'),
                   markersize=10,
                   markeredgewidth=2,
                   markeredgecolor='white',
                   zorder=4)
            
            # Add error value annotations
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            for t, err in zip(t_i, error_values):
                ax.annotate(f"{err:.2f}%", 
                          (t, err),
                          textcoords="offset points",
                          xytext=(0, 10),
                          ha='center',
                          bbox=bbox_props,
                          zorder=6)
    
    ax.set_xlabel('Time', fontsize=12, labelpad=10)
    ax.set_ylabel('Average Relative L2 Error (%)', fontsize=12, labelpad=10)
    ax.set_title(title if title else 'Test Error', 
                 fontsize=14, 
                 pad=20,
                 weight='bold')
    
    ax.set_xticks(t_i)
    ax.set_xticklabels([f'{t:.2f}' for t in t_i], fontsize=10)
    
    # Add some padding to y-axis
    ax.set_ylim(0, max_error * 1.2)
    
    legend = ax.legend(loc='upper right',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='gray',
                      fancybox=True,
                      fontsize=10)
    legend.get_frame().set_linewidth(0.5)
    
    plt.savefig(save_dir / 'l2_error_all2all.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def plot_trajectory_at_time(predictions, data, test_idx=0, res_dir=None, figsize=(15, 10), filename="predictions_over_time.png"):
    """
    Modified to work with the new all2all prediction format.
    
    Args:
        predictions: Predictions array from all2all evaluation
        data: Tuple of (u0, uT) where u0 is initial state and uT is final state
        test_idx: Index of the test sample to visualize
        res_dir: Directory to save the plots
        figsize: Figure size tuple
        filename: Name of the file to save the figure
    """
    res_dir = Path(res_dir) if res_dir else Path('.')
    res_dir.mkdir(parents=True, exist_ok=True)

    # For all2all, we'll plot initial and final states
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Unpack data tuple
    u0, uT = data
    x = np.linspace(0, 1, u0.shape[-1])
    
    # Plot initial state
    ax1.plot(x, u0[test_idx], 'b-', label='Initial State', linewidth=2)
    ax1.set_title('t = 0.0')
    ax1.grid(True)
    ax1.legend()
    
    # Plot final state comparison
    ax2.plot(x, uT[test_idx], 'b-', label='True Solution', linewidth=2)
    ax2.plot(x, predictions[test_idx], 'r--', label='Prediction', linewidth=2)
    ax2.set_title('t = 1.0')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    save_path = res_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_ibvp_sol_heatmap(data_path, model, predictions, trajectory_indices, res_dir, figsize=(24, 5), n_interp_points=200):
    """
    Modified to work with the new all2all prediction format and linear interpolation.
    Calculates errors at original data points before interpolation.
    """
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(data_path)
    
    # Create predictions dictionary for compatibility
    predictions_dict = {
        0.0: data[:, 0],  # Initial condition
        1.0: predictions   # Final predictions
    }

    # Original and interpolation grids
    x_orig = np.linspace(0, 1, data.shape[2])
    t_orig = np.array([0.0, 1.0])  # Only initial and final times
    x_fine = np.linspace(0, 1, n_interp_points)
    t_fine = np.linspace(0, 1, n_interp_points)
    
    # Create figure
    n_trajectories = len(trajectory_indices)
    height_per_row = figsize[1]
    fig = plt.figure(figsize=(figsize[0], height_per_row * n_trajectories))
    
    # Create grid with adjusted spacing
    outer_grid = GridSpec(n_trajectories, 1, figure=fig, hspace=0.5)

    # Calculate errors at original points and track max error
    max_error = 0
    original_errors = []
    Z_fine_exact_list = []
    Z_fine_pred_list = []

    # First pass to calculate errors and get max error
    for traj_idx in trajectory_indices:
        # Calculate error at original points
        exact_values = np.stack([data[traj_idx, 0], data[traj_idx, -1]])  # Initial and final states
        pred_values = np.stack([predictions_dict[0.0][traj_idx], predictions_dict[1.0][traj_idx]])
        
        # Calculate error at original points
        original_error = np.abs(exact_values - pred_values)
        original_errors.append(original_error)
        max_error = max(max_error, original_error.max())

        # Now interpolate both exact and predicted solutions
        interp_func_exact = RegularGridInterpolator(
            (t_orig, x_orig),
            exact_values,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        interp_func_pred = RegularGridInterpolator(
            (t_orig, x_orig),
            pred_values,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # Create fine grid points
        X_fine, T_fine = np.meshgrid(x_fine, t_fine)
        pts = np.array([T_fine.flatten(), X_fine.flatten()]).T
        
        # Interpolate both solutions
        Z_fine_exact = interp_func_exact(pts).reshape(n_interp_points, n_interp_points)
        Z_fine_pred = interp_func_pred(pts).reshape(n_interp_points, n_interp_points)
        
        Z_fine_exact_list.append(Z_fine_exact)
        Z_fine_pred_list.append(Z_fine_pred)

    # Create colormap for line plots
    cmap = plt.cm.coolwarm

    # Plot trajectories
    for idx_pos, (traj_idx, original_error, Z_fine_exact, Z_fine_pred) in enumerate(zip(
        trajectory_indices, original_errors, Z_fine_exact_list, Z_fine_pred_list)):
        
        # Create row with adjusted width ratios
        width = 0.5
        inner_grid = GridSpecFromSubplotSpec(1, 4,
                                           subplot_spec=outer_grid[idx_pos],
                                           width_ratios=[width, width, width, width],
                                           wspace=0.1)

        # 1. Initial condition with colored line
        ax1 = fig.add_subplot(inner_grid[0])
        y = data[traj_idx, 0]
        norm = plt.Normalize(data[traj_idx].min(), data[traj_idx].max())
        points = np.array([x_orig, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(y)
        lc.set_linewidth(2)
        line = ax1.add_collection(lc)
        plt.colorbar(line, ax=ax1)
        ax1.set_xlim(x_orig.min(), x_orig.max())
        ax1.set_ylim(y.min(), y.max())
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x, t = 0)')
        ax1.set_title(f'Initial Condition\n(Trajectory {traj_idx})')
        ax1.grid(True, which="both", ls=":")
        ax1.set_box_aspect(1)

        # 2. Exact solution
        ax2 = fig.add_subplot(inner_grid[1])
        im2 = ax2.pcolormesh(X_fine, T_fine, Z_fine_exact,
                            shading='auto',
                            cmap=cmap,
                            vmin=data[traj_idx].min(),
                            vmax=data[traj_idx].max())
        plt.colorbar(im2, ax=ax2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title('Exact u(x, t)')
        ax2.set_box_aspect(1)

        # 3. Predicted solution
        ax3 = fig.add_subplot(inner_grid[2])
        im3 = ax3.pcolormesh(X_fine, T_fine, Z_fine_pred,
                            shading='auto',
                            cmap=cmap,
                            vmin=data[traj_idx].min(),
                            vmax=data[traj_idx].max())
        plt.colorbar(im3, ax=ax3)
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_title('Predict u(x, t)')
        ax3.set_box_aspect(1)

        # 4. Error plot using original error data
        ax4 = fig.add_subplot(inner_grid[3])
        # Interpolate the error data for visualization
        error_interp = RegularGridInterpolator(
            (t_orig, x_orig),
            original_error,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        Z_fine_error = error_interp(pts).reshape(n_interp_points, n_interp_points)
        
        im4 = ax4.pcolormesh(X_fine, T_fine, Z_fine_error,
                            shading='auto',
                            cmap=cmap,
                            vmin=0,
                            vmax=max_error)
        plt.colorbar(im4, ax=ax4)
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        ax4.set_title(f'Absolute Error\n(Max Error: {original_error.max():.2e})')
        ax4.set_box_aspect(1)    
    
    plt.tight_layout()
    save_path = res_dir / "ibvp_sol_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()