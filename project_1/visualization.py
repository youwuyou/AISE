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

def plot_trajectory_grid(input_test: torch.Tensor, 
                        output_test: torch.Tensor, 
                        model: torch.nn.Module,
                        individual_errors: List[float],  # Now in percentage form
                        predictions: torch.Tensor,
                        save_path: Path,
                        model_type: str = 'custom',
                        num_plots: int = 128,
                        grid_size: tuple = (8, 16),
                        title: Optional[str] = None) -> None:
    with torch.no_grad():
        fig = plt.figure(figsize=(40, 22))
        gs = fig.add_gridspec(grid_size[0]+1, grid_size[1], height_ratios=[0.2] + [1]*grid_size[0])
        
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.set_visible(False)
        fig.suptitle(title if title else 'Wave Solutions: True vs Predicted', fontsize=20, y=0.98, weight='bold')
        
        for idx in range(num_plots):
            i, j = idx // grid_size[1], idx % grid_size[1]
            ax = fig.add_subplot(gs[i+1, j])
            
            if model_type == 'library':
                x_coords = input_test[idx, 1, :, 0].detach()
                true_sol = output_test[idx, 0, :, 0].detach()
                pred_sol = predictions[idx].detach()
            else:
                x_coords = input_test[idx, :, 1].detach()
                true_sol = output_test[idx].detach()
                pred_sol = predictions[idx].detach()
            
            ax.grid(True, which="both", ls=":", alpha=0.3)
            ax.plot(x_coords, true_sol, label="True", c="#1f77b4", lw=1.5)
            ax.plot(x_coords, pred_sol, label="Predicted", c="#ff7f0e", lw=1.5, ls='--')
            
            ax.set_title(f'Trajectory {idx+1}\nError: {individual_errors[idx]:.2f}%', 
                        fontsize=10, 
                        pad=5)
            
            if i != grid_size[0]-1:
                ax.set_xticks([])
            if j != 0:
                ax.set_yticks([])
            
            if idx == 0:
                ax.legend(fontsize=8, loc='upper right')
        
        # Average of individual percentage errors (they are already properly computed)
        avg_error = sum(individual_errors[:num_plots]) / num_plots
        fig.text(0.5, 0.02, f'Average Relative L2 Error: {avg_error:.2f}%', 
                ha='center', fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
    
    # Increase figure size and adjust spacing
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    axes = np.array(axes)
    
    fig.suptitle(title if title else 'Model Prediction Across Resolutions', 
                fontsize=16, y=0.95, weight='bold')
    
    # First pass to determine global min/max for consistent y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    
    for res in resolutions:
        sample_idx = 0
        true_sol = data_dict[res]['custom'][1][sample_idx].detach()
        global_min = min(global_min, true_sol.min().item())
        global_max = max(global_max, true_sol.max().item())
        
        for name in model_names:
            pred = results_dict[name]['predictions'][res][sample_idx].detach()
            global_min = min(global_min, pred.min().item())
            global_max = max(global_max, pred.max().item())
    
    # Add padding to y-axis limits
    y_padding = (global_max - global_min) * 0.15
    y_min = global_min - y_padding
    y_max = global_max + y_padding
    
    for idx, res in enumerate(resolutions):
        i, j = idx // 2, idx % 2
        sample_idx = 0
        x_grid = torch.linspace(0, 1, res)
        true_sol = data_dict[res]['custom'][1][sample_idx].detach()
        
        # Plot ground truth
        axes[i, j].plot(x_grid, true_sol,
                       '-', color=colors['True'], 
                       label='Ground Truth', 
                       linewidth=2, 
                       alpha=0.8)
        
        # Plot model predictions
        for name in model_names:
            error = results_dict[name]['errors'][idx]
            pred = results_dict[name]['predictions'][res][sample_idx]
            
            axes[i, j].plot(x_grid, 
                          pred.detach(),
                          '--', 
                          color=colors[name],
                          label=f'{name}',
                          linewidth=1.5,
                          alpha=0.8)
            
            # Adjust marker size and alpha based on resolution
            marker_size = max(3, 15 // (res/32))
            marker_alpha = max(0.2, 0.5 // (res/32))

            axes[i, j].scatter(x_grid, 
                             pred.detach(),
                             marker='o' if 'Custom' in name else 's',
                             color=colors[name],
                             s=marker_size,
                             alpha=marker_alpha)
        
        # Enhance subplot styling
        axes[i, j].set_title(f'Resolution: {res} points', pad=10, fontsize=12, weight='bold')
        axes[i, j].grid(True, linestyle='--', alpha=0.3)
        axes[i, j].legend(loc='upper right', framealpha=0.9)
        axes[i, j].set_xlabel('x', fontsize=10)
        axes[i, j].set_ylabel('u(x, t = 1)', fontsize=10)
        
        # Set consistent y-axis limits across all subplots
        axes[i, j].set_ylim(y_min, y_max)
        
        # Set x-axis limits
        axes[i, j].set_xlim(-0.02, 1.02)
        
        # Enhance grid and spines
        axes[i, j].grid(True, which='major', linestyle='-', alpha=0.2)
        axes[i, j].grid(True, which='minor', linestyle=':', alpha=0.1)
        axes[i, j].minorticks_on()
        
        for spine in axes[i, j].spines.values():
            spine.set_linewidth(0.8)
            
        # Add error percentage to title
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
   for model in ['Custom FNO', 'Library FNO']:
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

   for model_idx, model_name in enumerate(['Custom FNO', 'Library FNO']):
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