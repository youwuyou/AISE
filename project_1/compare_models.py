from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json  # Add this import

from neuralop.models import FNO as LibraryFNO
from custom_fno_1d import FNO1d
from visualization import plot_combined_training_history, plot_training_history, plot_trajectory_grid, plot_resolution_comparison, plot_l2_error_by_resolution

def load_model(checkpoint_dir: str, model_type: str) -> torch.nn.Module:
    """
    Load a model from checkpoint directory.
    
    Args:
        checkpoint_dir (str): Directory containing model files
        model_type (str): Type of model ('custom' or 'library')
    
    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load configuration
    with open(checkpoint_dir / 'training_config.json', 'r') as f:
        config_dict = json.load(f)
    
    model_config = config_dict['model_config']
    
    # Initialize model based on type
    if model_type == 'custom':
        # Remove model_type if it exists in model_config
        model_args = {k: v for k, v in model_config.items() if k != 'model_type'}
        model = FNO1d(**model_args)
    else:  # library
        # Remove model_type if it exists in model_config
        model_args = {k: v for k, v in model_config.items() if k != 'model_type'}
        model = LibraryFNO(**model_args)
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))
    model.eval()
    return model

def evaluate_models(models, data_path):
    """Evaluate multiple models on test data."""
    data = np.load(data_path)
    n_samples, _, resolution = data.shape  # Automatically get dimensions from data
    
    u0 = torch.from_numpy(data[:, 0, :]).float()
    uT = torch.from_numpy(data[:, -1, :]).float()
    
    print(f"Data shape: {data.shape} (trajectories, timestamps, resolution)")
    print(f"Testing on {n_samples} trajectories with resolution {resolution}")
    
    results = {}
    
    for name, model in models.items():
        with torch.no_grad():
            model_type = 'library' if 'Library' in name else 'custom'
            
            if model_type == 'library':
                x_grid = torch.linspace(0, 1, resolution).float()
                x_grid_expanded = x_grid.expand(u0.shape[0], -1)
                model_input = torch.stack((u0, x_grid_expanded), dim=1).unsqueeze(-1)
                predictions = model(model_input)
                predictions = predictions.squeeze(-1).squeeze(1)
            else:
                x_grid = torch.linspace(0, 1, resolution).float()
                x_grid_expanded = x_grid.expand(u0.shape[0], -1)
                model_input = torch.stack((u0, x_grid_expanded), dim=-1)
                predictions = model(model_input)
                predictions = predictions.squeeze(-1)
                        
            # Calculate individual relative L2 errors for each sample
            individual_errors = torch.norm(predictions - uT, p=2, dim=1) / torch.norm(uT, p=2, dim=1)
            
            # Calculate the average error across all samples
            average_error = individual_errors.mean()
            
            results[name] = {
                'predictions': predictions,
                'error': average_error.item() * 100, # Average L2 Error (in %)
                'individual_errors': individual_errors.tolist()
            }
    
    return results, (u0, uT)

def prepare_resolution_data(resolutions, n_samples=64):
    """Prepare data dictionary for resolution comparison."""
    data_dict = {}
    for res in resolutions:
        data = np.load(f"data/test_sol_res_{res}.npy")
        u0 = torch.from_numpy(data[:n_samples, 0, :]).float()
        uT = torch.from_numpy(data[:n_samples, -1, :]).float()
        
        # Create grid for this resolution
        x_grid = torch.linspace(0, 1, res).float()
        
        # Prepare input for custom model
        x_grid_expanded = x_grid.expand(u0.shape[0], -1)
        custom_input = torch.stack((u0, x_grid_expanded), dim=-1)
        
        # Prepare input for library model
        library_input = torch.stack((u0, x_grid_expanded), dim=1).unsqueeze(-1)
        
        data_dict[res] = {
            'custom': (custom_input, uT),
            'library': (library_input, uT.unsqueeze(1).unsqueeze(-1))
        }
    return data_dict


def main():
    # Create res directory if it doesn't exist
    res_dir = Path('results')
    res_dir.mkdir(exist_ok=True)
    
    # Find latest experiment directories
    custom_experiments = sorted(Path('checkpoints/custom_fno').glob('fno_*'))
    library_experiments = sorted(Path('checkpoints/library_fno').glob('fno_*'))
    
    if not custom_experiments or not library_experiments:
        raise ValueError("No experiment directories found. Please run training first.")
    
    # Load models using latest experiments
    models = {
        'Custom FNO': load_model(custom_experiments[-1], 'custom'),
        'Library FNO': load_model(library_experiments[-1], 'library')
    }
    
    print(f"Loading Custom FNO from: {custom_experiments[-1]}")
    print(f"Loading Library FNO from: {library_experiments[-1]}")
    
    print("\033[1mTask 1: Evaluating FNO models from one-to-one training on standard test set...\033[0m")    

    # Add training history plots
    print("Plotting training histories...")
    for exp_dir in [custom_experiments[-1], library_experiments[-1]]:
        plot_training_history(exp_dir, save_dir=res_dir)

    # Add combined training history plot
    plot_combined_training_history(
        custom_experiments[-1],
        library_experiments[-1],
        save_dir=res_dir
    )

    results, test_data = evaluate_models(models, "data/test_sol.npy")
    print(f"\nAverage Relative L2 Error Over {test_data[0].shape[0]} Testing Trajectories (resolution {test_data[0].shape[1]}):")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name}: {result['error']:.2f}%")
    
    # Resolution tests
    print("\n\033[1mTask 2: Testing on different resolutions:\033[0m")
    resolutions = [32, 64, 96, 128]
    resolution_results = {model_name: {'errors': []} for model_name in models.keys()}
    
    for res in resolutions:
        print(f"\nResolution: {res}")
        results, _ = evaluate_models(models, f"data/test_sol_res_{res}.npy")
        print(f"\nAverage Relative L2 Error Over {_[0].shape[0]} Testing Trajectories (resolution {_[0].shape[1]}):")
        print("-" * 50)
        for name, result in results.items():
            print(f"{name}: {result['error']:.2f}%")
            resolution_results[name]['errors'].append(result['error'])
    
    print("\n\033[1mTask 3: Testing on OOD dataset:\033[0m")
    print("\n\033[1mTODO!\033[0m")


    print("\n\033[1mTask 4: Testing on All2All Training:\033[0m")
    print("\n\033[1mTODO!\033[0m")


    # Visualization
    print("\nGenerating visualizations...")
    
    # Prepare data for visualizations
    u0, uT = test_data
    custom_input = torch.stack((u0, torch.linspace(0, 1, u0.shape[1]).expand(u0.shape)), dim=-1)
    library_input = torch.stack((u0, torch.linspace(0, 1, u0.shape[1]).expand(u0.shape)), dim=1).unsqueeze(-1)
    
    # Plot trajectory grids
    for name, model in models.items():
        model_type = 'library' if 'Library' in name else 'custom'
        input_data = library_input if model_type == 'library' else custom_input
        output_format = uT.unsqueeze(1).unsqueeze(-1) if model_type == 'library' else uT.unsqueeze(-1)
        
        fig = plot_trajectory_grid(input_data, output_format, model, 
                                 model_type=model_type,
                                 title=f"Wave Solutions: {name}")
        fig.savefig(res_dir / f'trajectory_grid_{name.lower().replace(" ", "_")}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Plot resolution comparison
    resolution_data = prepare_resolution_data(resolutions)
    fig, res_results = plot_resolution_comparison(
        models, resolution_data,
        title="Resolution Comparison Across Models"
    )
    fig.savefig(res_dir / 'resolution_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Add the new L2 error resolution plot
    fig_l2 = plot_l2_error_by_resolution(
        resolution_results,
        resolutions,
        title="Test Error Across Resolutions"
    )
    fig_l2.savefig(res_dir / 'l2_error_resolution_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig_l2)
    
    print(f"\nAll plots have been saved in the '{res_dir}' directory.")

if __name__ == "__main__":
    main()