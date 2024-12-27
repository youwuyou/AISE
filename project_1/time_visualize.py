import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import numpy as np



def visualize_predictions_heatmap(data_path, model, predictions_dict, trajectory_indices, res_dir, figsize=(24, 5), n_interp_points=200):
    """
    Visualize multiple trajectories with their predictions in a grid and save the figure.

    Args:
        data_path: Path to the data file
        model: Trained model
        predictions_dict: Dictionary of predictions at different times
        trajectory_indices: List/array of trajectory indices (valid range: 0-127)
        res_dir: Directory to save the plots
        figsize: Figure size (default adjusted for better aspect ratio)
        n_interp_points: Number of interpolation points for smooth visualization
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

    # Plot trajectories
    for idx_pos, (traj_idx, Z_fine_error) in enumerate(zip(trajectory_indices, Z_fine_errors)):
        # Create row with adjusted width ratios
        inner_grid = GridSpecFromSubplotSpec(1, 4,
                                           subplot_spec=outer_grid[idx_pos],
                                           width_ratios=[0.8, 1, 1, 1],
                                           wspace=0.3)

        # 1. Initial condition with adjusted aspect ratio
        ax1 = fig.add_subplot(inner_grid[0])
        ax1.plot(x_orig, data[traj_idx, 0], 'b-', lw=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u')
        ax1.set_title(f'Initial Condition\n(Trajectory {traj_idx})')
        ax1.grid(True, which="both", ls=":")

        # Set aspect ratio to be more square
        ax1.set_box_aspect(1)

        # 2. Exact solution with adjusted colormap
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
                            cmap='RdYlBu',
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
                            cmap='RdYlBu',
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
                            cmap='RdYlBu',
                            vmin=0,
                            vmax=max_error)
        plt.colorbar(im4, ax=ax4)
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        ax4.set_title('Absolute Error')
        ax4.set_box_aspect(1)

    plt.tight_layout()
    save_path = res_dir / "predictions_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_predictions(predictions_dict, data, test_idx=0, res_dir=None, figsize=(15, 10), filename="predictions_over_time.png"):
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
