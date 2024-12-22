# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple

def plot_trajectory_grid(input_test, output_test, model, model_type='custom', num_plots=64, grid_size=(8,8), title=None):
    """
    Create a grid of trajectory plots comparing true vs predicted solutions.
    Returns the figure object for saving.
    """
    with torch.no_grad():
        fig = plt.figure(figsize=(20, 22))
        gs = fig.add_gridspec(grid_size[0]+1, grid_size[1], height_ratios=[0.2] + [1]*grid_size[0])
        
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.set_visible(False)
        fig.suptitle(title if title else 'Wave Solutions: True vs Predicted', fontsize=20, y=0.98)
        
        total_l2_error = 0
        
        for idx in range(num_plots):
            i, j = idx // grid_size[1], idx % grid_size[1]
            ax = fig.add_subplot(gs[i+1, j])
            
            # Get predictions based on model type
            if model_type == 'library':
                # Library model: [batch, 2, spatial, 1]
                input_n = input_test[idx:idx+1]
                output_true_n = output_test[idx:idx+1]
                output_pred_n = model(input_n)
                
                # Extract plotting data
                x_coords = input_n[0, 1, :, 0].detach()
                true_sol = output_true_n[0, 0, :, 0].detach()
                pred_sol = output_pred_n[0, 0, :, 0].detach()
            else:
                # Custom model: [batch, spatial, 2]
                input_n = input_test[idx:idx+1]
                output_true_n = output_test[idx:idx+1]
                output_pred_n = model(input_n)
                
                # Extract plotting data
                x_coords = input_n[0, :, 1].detach()
                true_sol = output_true_n[0].detach()
                pred_sol = output_pred_n[0].detach()
            
            # Calculate error on squeezed tensors
            err = torch.norm(pred_sol - true_sol, p=2) / torch.norm(true_sol, p=2) * 100
            total_l2_error += err.item()
            
            # Plot
            ax.grid(True, which="both", ls=":", alpha=0.3)
            ax.plot(x_coords, true_sol, label="True", c="#1f77b4", lw=1.5)
            ax.plot(x_coords, pred_sol, label="Predicted", c="#ff7f0e", lw=1.5, ls='--')
            
            ax.set_title(f'Trajectory {idx+1}\nError: {err.item():.2f}%', 
                        fontsize=10, pad=5)
            
            if i != grid_size[0]-1:
                ax.set_xticks([])
            if j != 0:
                ax.set_yticks([])
            
            if idx == 0:
                ax.legend(fontsize=8, loc='upper right')
        
        avg_error = total_l2_error / num_plots
        fig.text(0.5, 0.02, f'Average Relative L2 Error: {avg_error:.2f}%', 
                ha='center', fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05)
        return fig

def plot_resolution_comparison(models: Dict[str, torch.nn.Module], data_dict: Dict, title=None):
    """Compare model performance across different resolutions."""
    resolutions = [32, 64, 96, 128]
    results = {}
    
    # Define a consistent color scheme and model names at the start
    model_names = list(models.keys())
    colors = {
        'True': '#000000',
        'Custom FNO': '#1f77b4',
        'Library FNO': '#ff7f0e'
    }
    
    # Increase figure size and adjust subplot layout
    fig = plt.figure(figsize=(12, 4*len(resolutions)))
    gs = fig.add_gridspec(len(resolutions), 2, height_ratios=[1]*len(resolutions), 
                         width_ratios=[2.5, 1])  # Adjusted width ratio for better balance
    axes = []
    for i in range(len(resolutions)):
        axes.append([fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])])
    axes = np.array(axes)
    
    fig.suptitle(title if title else 'Resolution Comparison Across Models', 
                fontsize=16, y=0.98)
    
    # Add spacing between subplots
    plt.subplots_adjust(hspace=0.4)
    
    for i, res in enumerate(resolutions):
        sample_idx = 0
        x_grid = torch.linspace(0, 1, res)
        true_sol = data_dict[res]['custom'][1][sample_idx].detach()
        
        # Plot ground truth
        axes[i, 0].plot(x_grid, true_sol,
                       '-', color=colors['True'], 
                       label='True Solution', 
                       linewidth=2, 
                       alpha=0.8)
        
        model_errors = []
        for name, model in models.items():
            with torch.no_grad():
                model_type = 'library' if 'Library' in name else 'custom'
                input_data, output_data = data_dict[res][model_type.lower()]
                
                pred = model(input_data[sample_idx:sample_idx+1])
                if model_type == 'library':
                    pred = pred.squeeze(-1).squeeze(1)
                else:
                    pred = pred.squeeze(-1)
                
                if model_type == 'library':
                    true_output = output_data.squeeze(-1).squeeze(1)
                else:
                    true_output = output_data
                    
                error = torch.norm(pred - true_output[sample_idx:sample_idx+1], p=2) / \
                        torch.norm(true_output[sample_idx:sample_idx+1], p=2) * 100
                
                if name not in results:
                    results[name] = {'errors': []}
                results[name]['errors'].append(error.item())
                
                # Plot prediction with improved styling
                axes[i, 0].plot(x_grid, 
                              pred[0].detach(),
                              '--', 
                              color=colors[name],
                              label=f'{name}\nError: {error:.2f}%',
                              linewidth=1.5,
                              alpha=0.8)
                
                # Adjusted marker size and alpha for better visibility
                marker_size = max(3, 20 // (res/32))
                marker_alpha = max(0.2, 0.6 // (res/32))

                axes[i, 0].scatter(x_grid, 
                                pred[0].detach(),
                                marker='o' if 'Custom' in name else 's',
                                color=colors[name],
                                s=marker_size,
                                alpha=marker_alpha)
                model_errors.append(error.item())
        
        # Improve left subplot styling
        axes[i, 0].set_title(f'Resolution: {res} points', pad=10, fontsize=12)
        axes[i, 0].grid(True, linestyle='--', alpha=0.2)
        axes[i, 0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        axes[i, 0].set_xlabel('x', fontsize=10)
        axes[i, 0].set_ylabel('u(x, t = 1)', fontsize=10)
        axes[i, 0].set_ylim(min(true_sol)*1.1, max(true_sol)*1.1)
        
        # Improve error bar plot
        bar_colors = [colors[name] for name in model_names]
        
        # Adjust bar width to make them thinner
        bar_width = 0.2  # Reduced bar width
        spacing = 0.05   # Minimal spacing between bars

        # Calculate positions for centered and thinner bars
        positions = np.arange(len(model_names)) * (bar_width + spacing)  # Spread with small spacing
        positions = positions - (np.mean(positions) / 2)  # Center around 0

        
        bars = axes[i, 1].bar(positions, model_errors, color=bar_colors, 
                            alpha=0.7, width=bar_width)



        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}%',
                          ha='center', va='bottom',
                          fontsize=9)
        
        # Improve right subplot styling
        axes[i, 1].set_title(f'Model Errors at Resolution {res}', fontsize=12)
        axes[i, 1].set_ylabel('Relative L2 Error (%)', fontsize=10)
        axes[i, 1].grid(True, linestyle='--', alpha=0.2)
        
        # Fix the tick positions to match the number of labels
        axes[i, 1].set_xticks(positions)
        axes[i, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Set y-axis limits for error bars with some padding
        max_error = max(model_errors) * 1.15
        axes[i, 1].set_ylim(0, max_error)
        
        # Add subtle spines
        for spine in axes[i, 1].spines.values():
            spine.set_linewidth(0.5)

    plt.tight_layout()
    return fig, results

def plot_l2_error_by_resolution(results: Dict[str, Dict[str, List[float]]], resolutions: List[int], title=None):
    """
    Create an enhanced line plot showing L2 error trends across different resolutions.
    
    Args:
        results: Dictionary containing model results with structure:
                {model_name: {'errors': [error_res1, error_res2, ...]}}
        resolutions: List of resolution values
        title: Optional plot title
    
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define consistent colors and styles
    colors = {
        'Custom FNO': '#1f77b4',
        'Library FNO': '#ff7f0e'
    }
    
    # Add subtle grid with lower alpha
    ax.grid(True, linestyle='--', alpha=0.2, which='both')
    ax.set_axisbelow(True)
    
    # Plot error trends for each model with enhanced styling
    for i, (model_name, model_results) in enumerate(results.items()):
        # Main line
        line = ax.plot(resolutions, 
                      model_results['errors'],
                      '-',
                      label=model_name,
                      color=colors[model_name],
                      linewidth=2.5,
                      zorder=3)
        
        # Add points with white edge
        ax.plot(resolutions,
                model_results['errors'],
                'o',
                color=colors[model_name],
                markersize=10,
                markeredgewidth=2,
                markeredgecolor='white',
                zorder=4)
        
        # Add smaller solid points
        ax.plot(resolutions,
                model_results['errors'],
                'o',
                color=colors[model_name],
                markersize=6,
                zorder=5)
        
        # Add value annotations with different offsets for last points
        for j, (x, y) in enumerate(zip(resolutions, model_results['errors'])):
            # Determine y-offset based on position and model
            if j == len(resolutions) - 1:  # Last points
                y_offset = 10 if i == 0 else -20  # Different offsets for different models
            else:
                y_offset = 10
            
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
    
    # Customize plot
    ax.set_xlabel('No. Nodal Points', fontsize=12, labelpad=10)
    ax.set_ylabel('Relative L2 Error (%)', fontsize=12, labelpad=10)
    ax.set_title(title if title else 'L2 Error vs Resolution', 
                 fontsize=14, 
                 pad=20,
                 weight='bold')
    
    # Set x-axis ticks and labels
    ax.set_xticks(resolutions)
    ax.set_xticklabels(resolutions, fontsize=10)
    
    # Add subtle spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Adjust y-axis limits to add some padding
    ymin = min(min(r['errors']) for r in results.values())
    ymax = max(max(r['errors']) for r in results.values())
    padding = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - padding * 2, ymax + padding * 3)  # Extra padding for labels
    
    # Enhanced legend
    legend = ax.legend(loc='upper right',
                      frameon=True,
                      framealpha=0.95,
                      edgecolor='gray',
                      fancybox=True,
                      fontsize=10)
    legend.get_frame().set_linewidth(0.5)
    
    plt.tight_layout()
    return fig