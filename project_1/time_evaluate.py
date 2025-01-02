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
    plot_l2_error_by_timestep
)

from utils import (
load_model,
print_bold
)
from torch.utils.data import DataLoader

from enum import Enum, auto
from dataset import All2All, OneToOne

# FIXME: move it else-where
from evaluate import evaluate_direct



def evaluate_autoregressive(model, data_path, timesteps, batch_size=5, base_dt = 0.25, device="cuda"):
    model.eval()
    strategy = All2All("validation", 
                       data_path=data_path,
                       time_pairs=[(0, 4)],
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
    print_bold("Task 4: Evaluation of Time-dependent Training at End Time (t = 1.0)")

    print_bold("1. Direct Evaluation")    
    result_a2a, data_a2a = evaluate_direct(model, "data/test_sol.npy", 
                                        time_pairs=[(0, 4)], 
                                        strategy="all2all")
    print(f"All2All Evaluation (t=1.0)  Error: {result_a2a['error']:.2f}%")
    
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
    print_bold(f"Bonus Task: Evaluate All2All Training Across Time:")
    
    print_bold("In-distribution Data Results:")
    for end_idx in [1,2,3,4]:
        print(f"End time: t = {end_idx * 0.25}")
        result, test_data = evaluate_direct(
            model,
            data_path="data/test_sol.npy",
            strategy="all2all",
            time_pairs=[(0, end_idx)]
        )

        # Print result for each end time in {0.25, 0.5, 0.75, 1.0}
        print(f"Average Relative L2 Error: {result['error']:.2f}%")
        print("-" * 50)

    # OOD evaluation (t = 1.0)
    ood_result, ood_data = evaluate_direct(
        model,
        data_path="data/test_sol_OOD.npy",
        strategy="all2all",
        time_pairs=[(0,1)]
    )
    
    print_bold("OOD Data Results:")
    print(f"Resolution: {ood_data[0].shape}")
    print(f"Average Relative L2 Error: {ood_result['error']:.2f}%")

    # Visualizations
    # plot_trajectory_at_time(
    #     result['predictions'], 
    #     test_data, 
    #     res_dir=res_dir, 
    #     filename="bonus_test_data.png"
    # )
    
    # plot_trajectory_at_time(
    #     ood_result['predictions'], 
    #     ood_data, 
    #     res_dir=res_dir, 
    #     filename="bonus_ood_data.png"
    # )

    # Space-time heatmap for test data
    print_bold("Visualize spatio-temporal evolution with heatmap")
    result, test_data = evaluate_direct(
        model,
        data_path="data/test_sol.npy",
        strategy="all2all"
    )

    plot_ibvp_sol_heatmap(
        "data/test_sol.npy",
        model,
        result['predictions'],
        trajectory_indices=[0, 127],
        res_dir=res_dir
    )

    return result['predictions'], ood_result['predictions']


def main():
    res_dir = Path("results/time")
    res_dir.mkdir(exist_ok=True)

    # Load time-dependent FNO models trained via one-to-all (vanilla) training
    models = {}
    for data_mode in ['onetoall', 'all2all']:
        fno_with_time_folders = sorted(Path(f"checkpoints/{data_mode}").glob("fno_*"), key=lambda d: d.stat().st_mtime)

        if not fno_with_time_folders:
            raise ValueError("No experiment directories found. Please run training first.")

        # Load model from checkpoint
        model = load_model(fno_with_time_folders[-1])
        print(f"Loading FNO ({data_mode}) from: {fno_with_time_folders[-1]}")

        models[data_mode] = model  # Store the loaded model in the models dictionary

        print("Plotting training history...")
        plot_training_history(fno_with_time_folders[-1])


    # Run evaluate at t = 1.0 using both one-to-
    task4_results = task4_evaluation(models['onetoall'], res_dir)
    task4_results = task4_evaluation(models['all2all'], res_dir)

    # Using only all2all trained model here
    bonus_results = bonus_task_evaluation(models['all2all'], res_dir)
    bonus_results = bonus_task_evaluation(models['onetoall'], res_dir)

    print(f"\nAll plots have been saved in the '{res_dir}' directory.")

if __name__ == "__main__":
    main()