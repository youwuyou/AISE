"""
Main module that evaluates the already trained FNO models on various dataset.

- finishes task 1 - 3
- evaluates models without time dependency
- compares custom implementation defined in fno.py and library-based model imported from neuralop
"""

import torch
import numpy as np
from pathlib import Path
import json

from neuralop.models import FNO as LibraryFNO
from fno import FNO1d
from visualization import (
    plot_combined_training_history,
    plot_training_history, 
    plot_trajectory_grid, 
    plot_resolution_comparison,
    plot_l2_error_by_resolution,
    plot_error_distributions
)

def load_model(checkpoint_dir: str, model_type: str) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(checkpoint_dir)
    
    with open(checkpoint_dir / 'training_config.json', 'r') as f:
        config_dict = json.load(f)
    
    model_config = config_dict['model_config']
    
    if model_type == 'custom':
        model_args = {k: v for k, v in model_config.items() if k != 'model_type'}
        model = FNO1d(**model_args)
    else:  # library
        model_args = {k: v for k, v in model_config.items() if k != 'model_type'}
        model = LibraryFNO(**model_args)
    
    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth', weights_only=True))    
    model = model.to(device)  # Move model to GPU
    model.eval()
    return model

def evaluate_models(models, data_path):
    device = next(iter(models.values())).parameters().__next__().device  # Get device from model
    
    data = np.load(data_path)
    n_samples, _, resolution = data.shape
    
    u0 = torch.from_numpy(data[:, 0, :]).float().to(device)
    uT = torch.from_numpy(data[:, -1, :]).float().to(device)
    
    print(f"Data shape: {data.shape} (trajectories, timestamps, resolution)")
    print(f"Testing on {n_samples} trajectories with resolution {resolution}")
    
    results = {}
    
    for name, model in models.items():
        with torch.no_grad():
            model_type = 'library' if 'Library' in name else 'custom'
            
            if model_type == 'library':
                x_grid = torch.linspace(0, 1, resolution).float().to(device)
                x_grid_expanded = x_grid.expand(u0.shape[0], -1)
                model_input = torch.stack((u0, x_grid_expanded), dim=1).unsqueeze(-1)
                predictions = model(model_input)
                predictions = predictions.squeeze(-1).squeeze(1)
            else:
                x_grid = torch.linspace(0, 1, resolution).float().to(device)
                x_grid_expanded = x_grid.expand(u0.shape[0], -1)
                model_input = torch.stack((u0, x_grid_expanded), dim=-1)
                predictions = model(model_input)
                predictions = predictions.squeeze(-1)
            
            individual_abs_errors = torch.norm(predictions - uT, p=2, dim=1) / torch.norm(uT, p=2, dim=1)
            average_error = individual_abs_errors.mean().item() * 100
            individual_errors_percent = individual_abs_errors.mul(100).tolist()
            
            results[name] = {
                'predictions': predictions.cpu(),  # Move back to CPU for plotting
                'error': average_error,
                'individual_errors': individual_errors_percent
            }
    
    return results, (u0.cpu(), uT.cpu())  # Return CPU tensors for plotting

def prepare_resolution_data(resolutions, n_samples=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = {}
    for res in resolutions:
        data = np.load(f"data/test_sol_res_{res}.npy")
        u0 = torch.from_numpy(data[:n_samples, 0, :]).float().to(device)
        uT = torch.from_numpy(data[:n_samples, -1, :]).float().to(device)
        
        x_grid = torch.linspace(0, 1, res).float().to(device)
        x_grid_expanded = x_grid.expand(u0.shape[0], -1)
        
        data_dict[res] = {
            'custom': (torch.stack((u0, x_grid_expanded), dim=-1), uT),
            'library': (torch.stack((u0, x_grid_expanded), dim=1).unsqueeze(-1), 
                       uT.unsqueeze(1).unsqueeze(-1))
        }
    return data_dict

def task1_evaluation(models, res_dir):
    print("\033[1mTask 1: Evaluating FNO models from one-to-one training on standard test set...\033[0m")    
    
    results, test_data = evaluate_models(models, "data/test_sol.npy")
    print(f"\nAverage Relative L2 Error Over {test_data[0].shape[0]} Testing Trajectories (resolution {test_data[0].shape[1]}):")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name}: {result['error']:.2f}%")
    
    device = next(iter(models.values())).parameters().__next__().device
    u0, uT = test_data
    # Move to device for computation
    u0, uT = u0.to(device), uT.to(device)
    x_grid = torch.linspace(0, 1, u0.shape[1]).to(device)
    
    custom_input = torch.stack((u0, x_grid.expand(u0.shape)), dim=-1)
    library_input = torch.stack((u0, x_grid.expand(u0.shape)), dim=1).unsqueeze(-1)
    
    for name, model in models.items():
        model_type = 'library' if 'Library' in name else 'custom'
        input_data = library_input if model_type == 'library' else custom_input
        output_format = uT.unsqueeze(1).unsqueeze(-1) if model_type == 'library' else uT
        
        # Move predictions back to CPU before plotting
        plot_trajectory_grid(
            input_data.cpu(), 
            output_format.cpu(), 
            model,
            individual_errors=results[name]['individual_errors'],
            predictions=results[name]['predictions'],
            save_path=res_dir / f'trajectory_grid_{name.lower().replace(" ", "_")}.png',
            model_type=model_type,
            title=f"Wave Solutions: {name}"
        )
    
    return results

def task2_evaluation(models, res_dir):
    print("\n\033[1mTask 2: Testing on different resolutions:\033[0m")
    resolutions = [32, 64, 96, 128]
    resolution_results = {name: {'errors': [], 'predictions': {}, 'abs_errors': []} for name in models.keys()}
    
    for res in resolutions:
        print(f"\nResolution: {res}")
        results, _ = evaluate_models(models, f"data/test_sol_res_{res}.npy")
        print(f"\nAverage Relative L2 Error Over {_[0].shape[0]} Testing Trajectories (resolution {_[0].shape[1]}):")
        print("-" * 50)
        for name, result in results.items():
            print(f"{name}: {result['error']:.2f}%")
            resolution_results[name]['errors'].append(result['error'])
            resolution_results[name]['predictions'][res] = result['predictions']
    
    resolution_data = prepare_resolution_data(resolutions)
    
    plot_resolution_comparison(
        models, 
        resolution_data,
        resolution_results,
        save_dir=res_dir,
    )
    
    plot_l2_error_by_resolution(
        resolution_results,
        resolutions,
        save_dir=res_dir,
    )
    
    return resolution_results


def task3_evaluation(models, res_dir):    
    print("\033[1mTask 3: Testing on OOD dataset:\033[0m")

    # Get in-distribution results first
    in_dist_results, in_dist_data = evaluate_models(models, "data/test_sol.npy")
    
    # Get OOD results
    ood_results, ood_data = evaluate_models(models, "data/test_sol_OOD.npy")
    
    # Print in-distribution results
    print(f"\nIn-Distribution - Average Relative L2 Error Over {in_dist_data[0].shape[0]} Testing Trajectories:")
    print("-" * 50)
    for name, result in in_dist_results.items():
        print(f"{name}: {result['error']:.2f}%")
        
    # Print OOD results
    print(f"\nOut-of-Distribution - Average Relative L2 Error Over {ood_data[0].shape[0]} Testing Trajectories:")
    print("-" * 50)
    for name, result in ood_results.items():
        print(f"{name}: {result['error']:.2f}%")
    
    # Plot error distributions
    plot_error_distributions(
        in_dist_results,
        ood_results, 
        save_path=res_dir / 'error_distributions.png'
    )
    
    # Plot trajectory grids for OOD data
    u0, uT = ood_data
    custom_input = torch.stack((u0, torch.linspace(0, 1, u0.shape[1]).expand(u0.shape)), dim=-1)
    library_input = torch.stack((u0, torch.linspace(0, 1, u0.shape[1]).expand(u0.shape)), dim=1).unsqueeze(-1)
    
    for name, model in models.items():
        model_type = 'library' if 'Library' in name else 'custom'
        input_data = library_input if model_type == 'library' else custom_input
        output_format = uT.unsqueeze(1).unsqueeze(-1) if model_type == 'library' else uT
        
        plot_trajectory_grid(
            input_data,
            output_format,
            model,
            individual_errors=ood_results[name]['individual_errors'],
            predictions=ood_results[name]['predictions'],
            save_path=res_dir / f'ood_trajectory_grid_{name.lower().replace(" ", "_")}.png',
            model_type=model_type,
            title=f"Wave Solutions: {name} (OOD)"
        )
    
    return in_dist_results, ood_results


def main():
    res_dir = Path('results')
    res_dir.mkdir(exist_ok=True)
    
    custom_experiments = sorted(Path('checkpoints/custom_fno').glob('fno_*'))
    library_experiments = sorted(Path('checkpoints/library_fno').glob('fno_*'))
    
    if not custom_experiments or not library_experiments:
        raise ValueError("No experiment directories found. Please run training first.")
    
    models = {
        'Custom FNO': load_model(custom_experiments[-1], 'custom'),
        'Library FNO': load_model(library_experiments[-1], 'library')
    }
    
    print(f"Loading Custom FNO from: {custom_experiments[-1]}")
    print(f"Loading Library FNO from: {library_experiments[-1]}")
    
    print("Plotting training histories...")
    for exp_dir in [custom_experiments[-1], library_experiments[-1]]:
        plot_training_history(exp_dir)
    
    plot_combined_training_history(
        custom_experiments[-1],
        library_experiments[-1],
        save_dir=res_dir
    )
    
    task1_results = task1_evaluation(models, res_dir)
    task2_results = task2_evaluation(models, res_dir)
    task3_results = task3_evaluation(models, res_dir)

    print("\n\033[1mTask 4: Testing on All2All Training:\033[0m")
    print("\n\033[1mTODO!\033[0m")
    
    print("\n\033[1mBonus Task: Evaluate All2All Training on Different Timesteps:\033[0m")
    print("\n\033[1mTODO!\033[0m")


    print(f"\nAll plots have been saved in the '{res_dir}' directory.")

if __name__ == "__main__":
    main()