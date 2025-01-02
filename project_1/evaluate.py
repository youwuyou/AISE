"""
Main module that evaluates the already trained FNO model on various dataset.

- finishes task 1 - 3
- evaluates FNO model without time dependency
"""

import torch
from pathlib import Path
import json

from visualization import (
    plot_training_history, 
    plot_resolution_comparison,
    plot_l2_error_by_resolution,
    plot_error_distributions
)
from utils import (
load_model,
print_bold
)
from torch.utils.data import DataLoader
from dataset import OneToOne, All2All

def evaluate_direct(model, 
                  data_path: str,
                  batch_size=5, 
                  start_idx=0, 
                  end_idx=4,
                  dt=0.25,
                  strategy="onetoone",
                  time_pairs=None,
                  device=None):
    """Evaluate the performance of a given model on a validation dataset."""
    
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()

    if strategy == "onetoone":
        strategy = OneToOne("validation", data_path=data_path, start_idx=start_idx, end_idx=end_idx, device=device)
    elif strategy == "all2all" and not time_pairs:
        strategy = All2All("validation", data_path=data_path, device=device, dt=dt)
    elif strategy == "all2all" and time_pairs:
        strategy = All2All("validation", data_path=data_path, time_pairs =time_pairs, device=device)
    else:
        raise ValueError("Invalid strategy. Please choose either 'onetoone' or 'all2all'.")

    test_loader = DataLoader(strategy, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_errors = torch.tensor([], device=device)
    u0, uT = [], []
    
    with torch.no_grad():
        for batch_input, batch_target in test_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            predictions = model(batch_input)
            predictions = predictions.squeeze(-1)
            targets = batch_target.squeeze(-1)
            
            u0.append(batch_input[:, 0].cpu())
            uT.append(targets.cpu())
            
            # Relative L2 norm
            individual_abs_errors = torch.norm(predictions - targets, p=2, dim=1) / torch.norm(targets, p=2, dim=1)

            # Record current prediction and error
            all_predictions.append(predictions.cpu())
            all_errors = torch.cat([all_errors, individual_abs_errors])

        # Concatenate all predictions in order
        predictions = torch.cat(all_predictions, dim=0)

        # Calculate average relative L2 error
        average_error = all_errors.mean().item() * 100  # (in %)
        
        results = {
            'predictions': predictions,
            'error': average_error,
            'individual_errors': (all_errors * 100).cpu().tolist()  # (in %)
        }
    
    u0 = torch.cat(u0, dim=0)
    uT = torch.cat(uT, dim=0)
    
    return results, (u0, uT)

def task1_evaluation(model, res_dir):
    print_bold("Task 1: Evaluating FNO model from One-to-One training on standard test set...")    
    result, test_data = evaluate_direct(model, "data/test_sol.npy")
    print(f"Resolution: {test_data[0].shape[0]}")    
    print(f"Average Relative L2 Error Over {test_data[0].shape[0]} Testing Trajectories (resolution {test_data[0].shape[1]}):")
    print(f"Custom FNO: {result['error']:.2f}%")
    return result

def task2_evaluation(model, res_dir):
    print_bold("Task 2: Testing on different resolutions:")
    resolutions = [32, 64, 96, 128]
    resolution_results = {'Custom FNO': {'errors': [], 'predictions': {}, 'abs_errors': []}}
    
    for res in resolutions:
        print(f"Resolution: {res}")
        result, test_data = evaluate_direct(model, f"data/test_sol_res_{res}.npy", end_idx=1)
        
        print(f"Average Relative L2 Error Over {test_data[0].shape[0]} Testing Trajectories (resolution {test_data[0].shape[1]}):")
        print(f"Custom FNO: {result['error']:.2f}%")
        print("-" * 50)
        resolution_results['Custom FNO']['errors'].append(result['error'])
        resolution_results['Custom FNO']['predictions'][res] = result['predictions']
    
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
    
    plot_resolution_comparison(model, resolution_data, resolution_results, save_dir=res_dir)
    plot_l2_error_by_resolution(resolution_results, resolutions, save_dir=res_dir)
    
    return resolution_results

def task3_evaluation(model, res_dir):    
    print_bold("Task 3: Testing on OOD dataset:")
    
    in_result, in_data = evaluate_direct(model, "data/test_sol.npy")
    ood_result, ood_data = evaluate_direct(model, "data/test_sol_OOD.npy", end_idx=1)
    
    print(f"In-Distribution - Average Relative L2 Error Over {in_data[0].shape[0]} Testing Trajectories:")
    print(f"Custom FNO: {in_result['error']:.2f}%")
    print("-" * 50)
        
    print(f"Out-of-Distribution - Average Relative L2 Error Over {ood_data[0].shape[0]} Testing Trajectories:")
    print(f"Custom FNO: {ood_result['error']:.2f}%")
    
    in_dist_results = {'Custom FNO': in_result}
    ood_results = {'Custom FNO': ood_result}
    plot_error_distributions(in_dist_results, ood_results, save_path=res_dir / 'error_distributions.png')
    
    return in_result, ood_result

def main():
    res_dir = Path('results')
    res_dir.mkdir(exist_ok=True)

    data_mode = "onetoone"    
    fno_folders = sorted(Path(f'checkpoints/{data_mode}').glob('fno_*'), key=lambda d: d.stat().st_mtime)

    if not fno_folders:
        raise ValueError("No experiment directories found. Please run training first.")
    
    model = load_model(fno_folders[-1])
    print(f"Loading Custom FNO from: {fno_folders[-1]}")
    
    print("Plotting training history...")
    plot_training_history(fno_folders[-1])
    
    task1_results = task1_evaluation(model, res_dir)
    task2_results = task2_evaluation(model, res_dir)
    task3_results = task3_evaluation(model, res_dir)
    
    print_bold("Task 4: Testing on All2All Training:")
    print_bold("TODO!")
    
    print_bold("Bonus Task: Evaluate All2All Training on Different Timesteps:")
    print_bold("TODO!")
    
    print(f"All plots have been saved in the '{res_dir}' directory.")

if __name__ == "__main__":
    main()