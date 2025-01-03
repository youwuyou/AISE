"""
This module evaluates the already trained custom FNO model.

- finishes task 4 & bonus task
- evaluates custom time-dependent with time dependency
"""

import numpy as np
import torch
from pathlib import Path
import json

from fno import FNO1d
from visualization import (
    plot_training_history,
    plot_ibvp_sol_heatmap,
    plot_trajectory_at_time,
    plot_error_distributions,
    plot_model_errors_temporal
)

from utils import (
load_model,
print_bold
)
from torch.utils.data import DataLoader

from typing import Dict, List
from enum import Enum, auto
from dataset import All2All

# FIXME: move it else-where
from evaluate import evaluate_direct

def evaluate_autoregressive(model, 
                            data_path, 
                            timesteps, 
                            batch_size=5, 
                            base_dt = 0.25,
                            start_idx = 0,
                            end_idx = 4,
                            device="cuda"):
    model.eval()
    strategy = All2All("testing", 
                       data_path=data_path,
                       time_pairs=[(start_idx, end_idx)],
                       dt=base_dt,
                       device=device)

    test_loader = DataLoader(strategy, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_errors = torch.tensor([], device=device)
    u0, uT = [], []
    
    with torch.no_grad():
        for batch_input, batch_target in test_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            u_start = batch_input[..., 0]
            v_start = batch_input[..., 1]  # This is already central-differenced from All2All
            x_grid = batch_input[..., 2]
            
            u_current = u_start
            v_current = v_start
            
            for step_size in timesteps:
                dt = torch.full_like(u_start, base_dt * step_size, device=device)
                
                current_input = torch.stack(
                    (u_current, v_current, x_grid, dt), dim=-1
                )
                
                u_next = model(current_input).squeeze(-1)
                # Forward difference for velocity (matching All2All's treatment of initial velocity)
                v_next = (u_next - u_current) / dt
                u_current = u_next
                v_current = v_next
            
            predictions = u_current
            targets = batch_target.squeeze(-1)
            
            u0.append(batch_input[..., 0].cpu())
            uT.append(targets.cpu())
            
            individual_abs_errors = torch.norm(predictions - targets, p=2, dim=1) / torch.norm(targets, p=2, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_errors = torch.cat([all_errors, individual_abs_errors])

    predictions = torch.cat(all_predictions, dim=0)
    average_error = all_errors.mean().item() * 100
    
    results = {
        'predictions': predictions,
        'error': average_error,
        'individual_errors': (all_errors * 100).cpu().tolist()
    }
    
    u0 = torch.cat(u0, dim=0)
    uT = torch.cat(uT, dim=0)
    
    return results, (u0, uT)


def task4_evaluation(model, res_dir):
    print_bold("1. Direct Evaluation")    
    result_a2a, data_a2a = evaluate_direct(model, "data/test_sol.npy", 
                                        time_pairs=[(0, 4)],
                                        strategy="all2all")
    print(f"FNO Evaluation (t=1.0)  Error: {result_a2a['error']:.2f}%")
    
    # 3. Autoregressive evaluation with different timestep combinations for nt = 4 in total
    print_bold("2. Autoregressive Evaluation (t=1.0):")
    combinations = [
        [1,1,1,1],
        [1,1,2],
        [1,2,1],
        [2,1,1],
        [2,2],
        [1,3],
        [3,1],
        [4]
    ]
    
    ar_results = {}
    print("\nAutoregressive Results:")
    print(f"{'Timesteps':<20} {'Error %':<10}")
    print("-" * 25)
    
    for timesteps in combinations:
        result, _ = evaluate_autoregressive(model, "data/test_sol.npy", timesteps=timesteps)
        timestep_str = '+'.join(map(str, timesteps))
        ar_results[timestep_str] = result['error']
        if timestep_str == '4':
            if timestep_str == '4':
                print(f"{timestep_str} (equi. to direct)  {result['error']:.2f}%")
            else:
                print(f"{timestep_str:<20} {result['error']:.2f}%")
        else:
            print(f"{timestep_str:<20} {result['error']:.2f}%")
        
    return result_a2a['predictions']

def bonus_task_evaluation(model, res_dir):
    """
    Bonus Task evaluation using the new evaluate_direct with all2all strategy
    """
    
    # Load initial data for t=0
    data = np.load("data/test_sol.npy")
    all_predictions = {
        0: data[:, 0]  # Initial condition from data
    }
    
    print_bold("In-distribution Data Results Across Time:")
    avg_errors_across_time = []
    for end_idx in [1, 2, 3, 4]:
        print(f"End time: t = {end_idx * 0.25}")
        result, test_data = evaluate_direct(
            model,
            data_path="data/test_sol.npy",
            strategy="all2all",
            time_pairs=[(0, end_idx)]
        )
        # Store predictions for each timestep
        all_predictions[end_idx] = result['predictions']
        error = result['error']
        avg_errors_across_time.append(error)

        print(f"shape of all_predictions[end_idx]: {all_predictions[end_idx].shape}")

        print(f"Average Relative L2 Error: {result['error']:.2f}%")
        print("-" * 50)
    
    plot_ibvp_sol_heatmap(
        "data/test_sol.npy",
        model,
        all_predictions,
        trajectory_indices=[0, 63, 127],  # choose between {0,...,127}
        res_dir=res_dir
    )

    # Testing on OOD data
    # use autoregressive eval with specifying to take 4 timesteps of size dt=0.25 (equiv to direct eval)
    ood_result, ood_data = evaluate_autoregressive(model, 
                                        "data/test_sol_OOD.npy",
                                        end_idx=1,
                                        timesteps=[4])
    
    print_bold("OOD Data Results at t = 1.0:")
    print(f"Resolution: {ood_data[0].shape}, {ood_data[1].shape}")
    print(f"Average Relative L2 Error: {ood_result['error']:.2f}%\n")

    return avg_errors_across_time


def main():
    res_dir = Path("results/time")
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load time-dependent FNO models trained via one-to-all (vanilla) training
    print_bold("Task 4: Evaluation of Time-dependent Training at End Time (t = 1.0)")
    models = {}
    for data_mode in ['onetoall', 'all2all']:
        fno_with_time_folders = sorted(Path(f"checkpoints/{data_mode}").glob("fno_*"), key=lambda d: d.stat().st_mtime)

        if not fno_with_time_folders:
            raise ValueError("No experiment directories found. Please run training first.")

        # Load model from checkpoint
        model = load_model(fno_with_time_folders[-1])
        print_bold(f"Loading FNO ({data_mode}) from: {fno_with_time_folders[-1]}, plotting training history...")

        plot_training_history(fno_with_time_folders[-1])

        # Run evaluate at t = 1.0 using both one-to-all and all2all model
        task4_results = task4_evaluation(model, res_dir)

        # Store the loaded model in the models dictionary
        models[data_mode] = model

    # Using only all2all trained model here
    print_bold(f"Bonus Task: Evaluate All2All Training Across Time:")

    model_errors = []
    for data_mode in ['onetoall', 'all2all']:
        print_bold(f"Using FNO ({data_mode})")
        errors_across_time = bonus_task_evaluation(models[data_mode], res_dir)
        print(f"all average errors: {errors_across_time}")
        model_errors.append(errors_across_time)

    # Plot model error across time
    plot_model_errors_temporal(model_errors, res_dir)
    print(f"\nAll plots have been saved in the '{res_dir}' directory.")

if __name__ == "__main__":
    main()