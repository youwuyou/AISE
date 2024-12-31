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
    plot_resolution_comparison,
    plot_l2_error_by_resolution,
    plot_error_distributions
)
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset import OneToOne

def evaluate_models(models, 
                    data_path: str, 
                    batch_size=5,
                    start_idx=0,
                    end_idx=4,
                    ):

    strategy = OneToOne("validation", data_path=data_path, start_idx=start_idx, end_idx=end_idx)
    test_loader = DataLoader(strategy, batch_size=batch_size, shuffle=False)

    device = next(iter(models.values())).parameters().__next__().device
    results = {}
    
    # Store initial and final states for plotting
    all_u0 = []
    all_uT = []
    
    for name, model in models.items():
        model.eval()
        all_predictions = []
        all_errors = []
        
        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                # Input is already properly formatted from DataLoader
                predictions = model(batch_input)
                predictions = predictions.squeeze(-1)
                targets = batch_target.squeeze(-1)
                
                # Store initial conditions for plotting
                if len(all_u0) < len(test_loader.dataset):
                    all_u0.append(batch_input[:, 0].cpu())  # First channel is u0
                    all_uT.append(targets.cpu())
                
                # Calculate errors
                individual_abs_errors = torch.norm(predictions - targets, p=2, dim=1) / torch.norm(targets, p=2, dim=1)
                all_predictions.append(predictions.cpu())
                all_errors.extend(individual_abs_errors.mul(100).tolist())
            
            # Combine results
            predictions = torch.cat(all_predictions, dim=0)
            average_error = sum(all_errors) / len(all_errors)
            
            results[name] = {
                'predictions': predictions,
                'error': average_error,
                'individual_errors': all_errors
            }
    
    # Combine all initial and target states
    u0 = torch.cat(all_u0, dim=0)
    uT = torch.cat(all_uT, dim=0)
    
    return results, (u0, uT)

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


def task1_evaluation(models, res_dir):
    print("\033[1mTask 1: Evaluating FNO models from one-to-one training on standard test set...\033[0m")    
    
    results, test_data = evaluate_models(models, "data/test_sol.npy")
    print(f"\nAverage Relative L2 Error Over {test_data[0].shape[0]} Testing Trajectories (resolution {test_data[0].shape[1]}):")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name}: {result['error']:.2f}%")
    return results

def task2_evaluation(models, res_dir):
    print("\n\033[1mTask 2: Testing on different resolutions:\033[0m")
    resolutions = [32, 64, 96, 128]
    resolution_results = {name: {'errors': [], 'predictions': {}, 'abs_errors': []} for name in models.keys()}
    
    for res in resolutions:
        print(f"\nResolution: {res}")
        results, _ = evaluate_models(models, f"data/test_sol_res_{res}.npy", end_idx=1)
        print(f"\nAverage Relative L2 Error Over {_[0].shape[0]} Testing Trajectories (resolution {_[0].shape[1]}):")
        print("-" * 50)
        for name, result in results.items():
            print(f"{name}: {result['error']:.2f}%")
            resolution_results[name]['errors'].append(result['error'])
            resolution_results[name]['predictions'][res] = result['predictions']
    
    resolution_data = {}
    for res in resolutions:
        dataset = OneToOne(
            which="validation", 
            data_path=f"data/test_sol_res_{res}.npy",
            start_idx=0,
            end_idx=1
        )
        
        input_data = torch.stack((
            dataset.u_start,
            dataset.v_start,
            dataset.x_grid.repeat(len(dataset.u_start), 1),
            torch.full_like(dataset.u_start, dataset.dt)
        ), dim=-1)
        
        resolution_data[res] = {
            'custom': (input_data, dataset.u_end)
        }    
    
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
    ood_results, ood_data = evaluate_models(models, "data/test_sol_OOD.npy", end_idx=1)
    
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
        # 'Library FNO': load_model(library_experiments[-1], 'library')
    }

    results, (u0, uT) = evaluate_models(models, "data/test_sol.npy")

    print(f"\nAverage Relative L2 Error Over {u0.shape[0]} Testing Trajectories (resolution {u0.shape[1]}):")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name}: {result['error']:.2f}%")

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