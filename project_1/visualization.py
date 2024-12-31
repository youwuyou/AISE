"""
Utility functions for plotting results used in report of project 1
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
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
        history = config['training_history']
    
    model_name = "Custom FNO" if "custom_fno" in str(experiment_dir) else "Library FNO"
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
        
    plt.title(f'Training History - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend()
    
    plt.savefig(save_dir / f'training_history_{model_name.lower().replace(" ", "_")}.png',
               bbox_inches='tight', dpi=300)
    plt.close()

def plot_combined_training_history(custom_dir: Path, library_dir: Path, save_dir: Optional[Path] = None) -> None:
    custom_dir, library_dir = Path(custom_dir), Path(library_dir)
    save_dir = Path(save_dir) if save_dir else custom_dir.parent
    save_dir.mkdir(exist_ok=True)
    
    with open(custom_dir / 'training_config.json', 'r') as f:
        custom_history = json.load(f)['training_history']
    
    with open(library_dir / 'training_config.json', 'r') as f:
        library_history = json.load(f)['training_history']
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(custom_history['train_loss'], 
            label='Custom FNO - Training', 
            color='darkgreen', 
            alpha=0.7)
    plt.plot(custom_history['val_loss'], 
            label='Custom FNO - Validation', 
            color='lightgreen', 
            linestyle='--', 
            alpha=0.7)
    
    plt.plot(library_history['train_loss'], 
            label='Library FNO - Training', 
            color='darkviolet', 
            alpha=0.7)
    plt.plot(library_history['val_loss'], 
            label='Library FNO - Validation', 
            color='plum', 
            linestyle='--', 
            alpha=0.7)
    
    custom_stop = len(custom_history['train_loss']) - 1
    library_stop = len(library_history['train_loss']) - 1
    
    plt.axvline(x=custom_stop, 
                color='seagreen', 
                linestyle=':', 
                alpha=0.5,
                label=f'Custom Stop (epoch {custom_stop})')
    plt.axvline(x=library_stop, 
                color='purple', 
                linestyle=':', 
                alpha=0.5,
                label=f'Library Stop (epoch {library_stop})')
    
    plt.title('Training History Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # Calculate position to avoid vertical lines and place legend
    plt.legend(loc='upper right')  # Simple solution - let matplotlib choose optimal position

    plt.savefig(save_dir / 'training_history_comparison.png',
               bbox_inches='tight', dpi=300)
    plt.close()


def plot_resolution_comparison(models: Dict[str, torch.nn.Module], 
                             data_dict: Dict,
                             results_dict: Dict,
                             save_dir: Path,
                             title: Optional[str] = None) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    resolutions = sorted(data_dict.keys())
    model_names = list(models.keys())
    colors = {
        'True': '#000000',
        'Custom FNO': '#1f77b4',
        'Library FNO': '#ff7f0e'
    }
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    axes = np.array(axes)
    
    fig.suptitle(title if title else 'Model Prediction Across Resolutions', 
                fontsize=16, y=0.95, weight='bold')
    
    # First pass for global min/max
    global_min = float('inf')
    global_max = float('-inf')
    
    for res in resolutions:
        sample_idx = 0
        true_sol = data_dict[res]['custom'][1][sample_idx].cpu().detach()
        global_min = min(global_min, true_sol.min().item())
        global_max = max(global_max, true_sol.max().item())
        
        for name in model_names:
            pred = results_dict[name]['predictions'][res][sample_idx].cpu().detach()
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
        
        for name in model_names:
            error = results_dict[name]['errors'][idx]
            pred = results_dict[name]['predictions'][res][sample_idx].cpu().detach()
            
            axes[i, j].plot(x_grid, 
                          pred,
                          '--', 
                          color=colors[name],
                          label=f'{name}',
                          linewidth=1.5,
                          alpha=0.8)
            
            marker_size = max(3, 15 // (res/32))
            marker_alpha = max(0.2, 0.5 // (res/32))

            axes[i, j].scatter(x_grid, 
                             pred,
                             marker='o' if 'Custom' in name else 's',
                             color=colors[name],
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
            
        error_text = ' '.join([f'{name}: {results_dict[name]["errors"][idx]:.2f}%' 
                             for name in model_names])
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
    
    colors = {
        'Custom FNO': '#1f77b4',
        'Library FNO': '#ff7f0e'
    }
    
    ax.grid(True, linestyle='--', alpha=0.2, which='both')
    ax.set_axisbelow(True)
    
    # Plot lines and markers for each model
    for i, (model_name, model_results) in enumerate(results.items()):
        line = ax.plot(resolutions, 
                      model_results['errors'],
                      '-',
                      label=model_name,
                      color=colors[model_name],
                      linewidth=2.5,
                      zorder=3)
        
        ax.plot(resolutions,
                model_results['errors'],
                'o',
                color=colors[model_name],
                markersize=10,
                markeredgewidth=2,
                markeredgecolor='white',
                zorder=4)
        
        ax.plot(resolutions,
                model_results['errors'],
                'o',
                color=colors[model_name],
                markersize=6,
                zorder=5)
        
        for j, (x, y) in enumerate(zip(resolutions, model_results['errors'])):
            y_offset = 10 if i == 0 or j != len(resolutions) - 1 else -20
            
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            ax.annotate(f'{y:.2f}%', 
                       (x, y),
                       textcoords="offset points",
                       xytext=(0, y_offset),
                       ha='center',
                       va='bottom' if y_offset > 0 else 'top',
                       fontsize=10,
                       bbox=bbox_props,
                       zorder=6)
    
    # Add vertical line at training resolution if specified
    if training_resolution is not None:
        ymin = min(min(r['errors']) for r in results.values())
        ymax = max(max(r['errors']) for r in results.values())
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
    
    ymin = min(min(r['errors']) for r in results.values())
    ymax = max(max(r['errors']) for r in results.values())
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
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4)

    model_colors = {
        'Custom FNO': '#1f77b4',
        'Library FNO': '#ff7f0e'
    }

    stat_styles = {
        'mode': '--',
        'mean': '-', 
        'median': ':',
    }

    # First pass to determine global axis limits
    all_errors = []
    max_density = 0
    #    for model in ['Custom FNO', 'Library FNO']:
    for model in ['Custom FNO']:

       for results in [in_dist_results, ood_results]:
           errors = np.array(results[model]['individual_errors'])
           all_errors.extend(errors)
           counts, _ = np.histogram(errors, bins=30, density=True)
           max_density = max(max_density, np.max(counts))

    global_xlim = (0, np.ceil(max(all_errors)))
    global_ylim = (0, max_density * 1.2)  # Add 20% padding for labels

    # Add model labels on right side
    plt.figtext(0.95, 0.75, 'Custom FNO', color=model_colors['Custom FNO'], rotation=270, fontsize=14, weight='bold')
    plt.figtext(0.95, 0.25, 'Library FNO', color=model_colors['Library FNO'], rotation=270, fontsize=14, weight='bold')

    # Add distribution type labels at top
    plt.figtext(0.25, 0.95, 'In-Distribution Data', fontsize=15, weight='bold', ha='center')
    plt.figtext(0.75, 0.95, 'Out-of-Distribution Data', fontsize=15, weight='bold', ha='center')

    # FIXME: enumerate(['Custom FNO', 'Library FNO'])
    for model_idx, model_name in enumerate(['Custom FNO']):
        stat_lines = []
        stat_labels = []
        for dist_idx, results in enumerate([in_dist_results, ood_results]):
            ax = fig.add_subplot(gs[model_idx, dist_idx])
            
            errors = np.array(results[model_name]['individual_errors'])
            plt.hist(errors, bins=30, density=True, alpha=0.5,
                    color=model_colors[model_name],
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
                line = plt.axvline(stat_value, color=model_colors[model_name],
                                linestyle=style, alpha=1.0, linewidth=2)
                if dist_idx == 0:
                    stat_lines.append(line)
                    stat_labels.append(stat.capitalize())
                
                # Add mean value label with background
                if stat == 'mean':
                    label = f'Mean: {stat_value:.2f}%'
                    bbox_props = dict(
                        boxstyle='round,pad=0.5',
                        fc='white',
                        ec=model_colors[model_name],
                        alpha=0.8
                    )
                    plt.text(stat_value + 0.5, global_ylim[1] * 0.8,
                            label,
                            color=model_colors[model_name],
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
                    bbox_to_anchor=(0.93, 0.93 - 0.5 * model_idx),
                    labelcolor='#404040')

    plt.subplots_adjust(right=0.9, top=0.9)
    plt.savefig(save_path.parent / 'ood_error_distribution_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

#============================================
#  Time-dependent Visualization
#============================================
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator

def plot_l2_error_by_timestep(results: Dict[str, Dict[str, List[float]]], 
                             timesteps: List[float],
                             save_dir: Path,
                             title: Optional[str] = None) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        'AR': 'dodgerblue',
        'Direct': 'red'
    }
    
    ax.grid(True, linestyle='--', alpha=0.2, which='both')
    ax.set_axisbelow(True)
    
    # Plot lines and markers for each model
    for i, (model_name, model_results) in enumerate(results.items()):
        line = ax.plot(timesteps, 
                      model_results['errors'],
                      '-',
                      label=model_name,
                      color=colors[model_name],
                      linewidth=2.5,
                      zorder=3)
        
        ax.plot(timesteps,
                model_results['errors'],
                'o',
                color=colors[model_name],
                markersize=10,
                markeredgewidth=2,
                markeredgecolor='white',
                zorder=4)
        
        ax.plot(timesteps,
                model_results['errors'],
                'o',
                color=colors[model_name],
                markersize=6,
                zorder=5)
        
        for j, (x, y) in enumerate(zip(timesteps, model_results['errors'])):
            y_offset = 10 if i == 0 or j != len(timesteps) - 1 else -20
            
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            ax.annotate(f'{y:.2f}%', 
                       (x, y),
                       textcoords="offset points",
                       xytext=(0, y_offset),
                       ha='center',
                       va='bottom' if y_offset > 0 else 'top',
                       fontsize=10,
                       bbox=bbox_props,
                       zorder=6)
    
    ax.set_xlabel('Time', fontsize=12, labelpad=10)
    ax.set_ylabel('Average Relative L2 Error (%)', fontsize=12, labelpad=10)
    ax.set_title(title if title else 'Test Error Across Time (s = 64)', 
                 fontsize=14, 
                 pad=20,
                 weight='bold')
    
    ax.set_xticks(timesteps)
    ax.set_xticklabels([f'{t:.2f}' for t in timesteps], fontsize=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    ymin = min(min(r['errors']) for r in results.values())
    ymax = max(max(r['errors']) for r in results.values())
    padding = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - padding * 2, ymax + padding * 3)
    
    legend = ax.legend(loc='upper right',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='gray',
                      fancybox=True,
                      fontsize=10)
    legend.get_frame().set_linewidth(0.5)
    
    plt.savefig(save_dir / 'l2_error_timestep_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_ibvp_sol_heatmap(data_path, model, predictions_dict, trajectory_indices, res_dir, figsize=(24, 5), n_interp_points=200):
    """
    Visualize multiple trajectories with their predictions in a grid and save the figure.
    Initial condition line plot is now colored using the same colormap as heatmaps.
    """
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(data_path)

    # Validate indices
    trajectory_indices = np.array(trajectory_indices)
    if not np.all((trajectory_indices >= 0) & (trajectory_indices < data.shape[0])):
        raise ValueError(f"All indices must be in range [0, {data.shape[0]-1}]")

    # Create figure with adjusted size
    n_trajectories = len(trajectory_indices)
    height_per_row = figsize[1]
    fig = plt.figure(figsize=(figsize[0], height_per_row * n_trajectories))

    # Create grid with adjusted spacing
    outer_grid = GridSpec(n_trajectories, 1, figure=fig, hspace=0.5)

    # Original and interpolation grids
    x_orig = np.linspace(0, 1, data.shape[2])
    t_orig = np.linspace(0, 1, data.shape[1])
    x_fine = np.linspace(0, 1, n_interp_points)
    t_fine = np.linspace(0, 1, n_interp_points)

    # Track max error for consistent colorbar
    max_error = 0
    Z_fine_errors = []

    # First pass to get max error
    for traj_idx in trajectory_indices:
        interp_func = RegularGridInterpolator((t_orig, x_orig), data[traj_idx],
                                            method='cubic',
                                            bounds_error=False,
                                            fill_value=None)

        X_fine, T_fine = np.meshgrid(x_fine, t_fine)
        pts = np.array([T_fine.flatten(), X_fine.flatten()]).T
        Z_fine_exact = interp_func(pts).reshape(n_interp_points, n_interp_points)

        times = sorted(predictions_dict.keys())
        pred_values = np.array([predictions_dict[t][traj_idx] for t in times])

        interp_func_pred = RegularGridInterpolator((times, x_orig), pred_values,
                                                 method='cubic',
                                                 bounds_error=False,
                                                 fill_value=None)

        Z_fine_pred = interp_func_pred(pts).reshape(n_interp_points, n_interp_points)
        Z_fine_error = np.abs(Z_fine_pred - Z_fine_exact)
        Z_fine_errors.append(Z_fine_error)
        max_error = max(max_error, Z_fine_error.max())

    # Create colormap for line plots
    cmap = plt.cm.coolwarm

    # Plot trajectories
    for idx_pos, (traj_idx, Z_fine_error) in enumerate(zip(trajectory_indices, Z_fine_errors)):
        # Create row with adjusted width ratios
        width = 0.5
        inner_grid = GridSpecFromSubplotSpec(1, 4,
                                           subplot_spec=outer_grid[idx_pos],
                                           width_ratios=[width, width, width, width],
                                           wspace=0.1)

        # 1. Initial condition with colored line
        ax1 = fig.add_subplot(inner_grid[0])
        
        # Get initial condition values
        y = data[traj_idx, 0]
        
        # Normalize values to [0, 1] for colormap
        norm = plt.Normalize(data[traj_idx].min(), data[traj_idx].max())
        
        # Create colored line segments
        points = np.array([x_orig, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(y)
        lc.set_linewidth(2)
        
        # Add colored line to plot
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
        interp_func = RegularGridInterpolator((t_orig, x_orig), data[traj_idx],
                                            method='cubic',
                                            bounds_error=False,
                                            fill_value=None)

        X_fine, T_fine = np.meshgrid(x_fine, t_fine)
        pts = np.array([T_fine.flatten(), X_fine.flatten()]).T
        Z_fine_exact = interp_func(pts).reshape(n_interp_points, n_interp_points)

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
        times = sorted(predictions_dict.keys())
        pred_values = np.array([predictions_dict[t][traj_idx] for t in times])

        interp_func_pred = RegularGridInterpolator((times, x_orig), pred_values,
                                                 method='cubic',
                                                 bounds_error=False,
                                                 fill_value=None)

        Z_fine_pred = interp_func_pred(pts).reshape(n_interp_points, n_interp_points)

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

        # 4. Error with consistent colorbar
        ax4 = fig.add_subplot(inner_grid[3])
        im4 = ax4.pcolormesh(X_fine, T_fine, Z_fine_error,
                            shading='auto',
                            cmap=cmap,
                            vmin=0,
                            vmax=max_error)
        plt.colorbar(im4, ax=ax4)
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        ax4.set_title('Absolute Error')
        ax4.set_box_aspect(1)

    plt.tight_layout()
    save_path = res_dir / "ibvp_sol_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_trajectory_at_time(predictions_dict, data, test_idx=0, res_dir=None, figsize=(15, 10), filename="predictions_over_time.png"):
    """
    Visualize predictions at different timesteps and save the figure.

    Args:
        predictions_dict: Dictionary of predictions at different times
        data: Ground truth data array
        test_idx: Index of the test sample to visualize
        res_dir: Directory to save the plots
        figsize: Figure size tuple
        filename: Name of the file to save the figure
    """
    res_dir = Path(res_dir) if res_dir else Path('.')
    res_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(predictions_dict), 1, figsize=figsize)
    if len(predictions_dict) == 1:
        axes = [axes]

    x = np.linspace(0, 1, predictions_dict[list(predictions_dict.keys())[0]].shape[1])

    for i, (t_val, preds) in enumerate(predictions_dict.items()):
        true_idx = int(t_val * (data.shape[1] - 1))

        axes[i].plot(x, data[test_idx, true_idx], 'b-', label='True Solution', linewidth=2)
        axes[i].plot(x, preds[test_idx], 'r--', label='Prediction', linewidth=2)
        axes[i].set_title(f't = {t_val:.2f}')
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    save_path = res_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
