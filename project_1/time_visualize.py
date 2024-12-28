import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

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
