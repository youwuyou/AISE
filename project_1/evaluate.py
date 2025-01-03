"""
Main module that evaluates FNO models on various datasets. Handles both time-dependent and time-independent evaluation tasks.
- finishes all tasks 1 - 4 & bonus
"""

import torch
import numpy as np
from pathlib import Path

from visualization import (
    plot_training_history, 
    plot_resolution_comparison,
    plot_l2_error_by_resolution,
    plot_error_distributions,
    plot_ibvp_sol_heatmap,
    plot_model_errors_temporal
)
from utils import (
    load_model,
    print_bold,
    evaluate_direct,
    evaluate_autoregressive
)
from dataset import OneToOne

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
            which="testing", 
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
        
        resolution_data[res] = {'custom': (input_data, dataset.u_end)}    
    
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

def task4_evaluation(model, res_dir):
    print_bold("1. Direct Evaluation")    
    result_a2a, data_a2a = evaluate_direct(model, "data/test_sol.npy", 
                                        time_pairs=[(0, 4)],
                                        strategy="all2all")
    print(f"FNO Evaluation (t=1.0)  Error: {result_a2a['error']:.2f}%")
    
    print_bold("2. Autoregressive Evaluation (t=1.0):")
    combinations = [
        [1,1,1,1], [1,1,2], [1,2,1], [2,1,1],
        [2,2], [1,3], [3,1], [4]
    ]
    
    ar_results = {}
    print("\nAutoregressive Results:")
    print(f"{'Timesteps':<20} {'Error %':<10}")
    print("-" * 25)
    
    for timesteps in combinations:
        result, _ = evaluate_autoregressive(model, "data/test_sol.npy", timesteps=timesteps)
        timestep_str = '+'.join(map(str, timesteps))
        ar_results[timestep_str] = result['error']
        print(f"{timestep_str:<20} {result['error']:.2f}%")
        
    return result_a2a['predictions']

def bonus_task_evaluation(model, res_dir):
    data = np.load("data/test_sol.npy")
    all_predictions = {0: data[:, 0]}
    
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
        all_predictions[end_idx] = result['predictions']
        avg_errors_across_time.append(result['error'])
        print(f"Average Relative L2 Error: {result['error']:.2f}%")
        print("-" * 50)
    
    plot_ibvp_sol_heatmap(
        "data/test_sol.npy",
        model,
        all_predictions,
        trajectory_indices=[0, 63, 127],
        res_dir=res_dir
    )

    ood_result, ood_data = evaluate_autoregressive(model, 
                                        "data/test_sol_OOD.npy",
                                        end_idx=1,
                                        timesteps=[4])
    
    print_bold("OOD Data Results at t = 1.0:")
    print(f"Resolution: {ood_data[0].shape}, {ood_data[1].shape}")
    print(f"Average Relative L2 Error: {ood_result['error']:.2f}%\n")

    return avg_errors_across_time

def main():
    res_dir = Path('results')
    res_dir.mkdir(exist_ok=True)
    time_res_dir = res_dir / 'time'
    time_res_dir.mkdir(exist_ok=True)
    
    # Time-independent evaluation
    data_mode = "onetoone"    
    fno_folders = sorted(Path(f'checkpoints/{data_mode}').glob('fno_*'), 
                        key=lambda d: d.stat().st_mtime)

    if fno_folders:
        model = load_model(fno_folders[-1])
        print(f"Loading Custom FNO from: {fno_folders[-1]}")
        
        print("Plotting training history...")
        plot_training_history(fno_folders[-1])
        
        task1_results = task1_evaluation(model, res_dir)
        task2_results = task2_evaluation(model, res_dir)
        task3_results = task3_evaluation(model, res_dir)
    
    # Time-dependent evaluation
    models = {}
    for data_mode in ['onetoall', 'all2all']:
        fno_time_folders = sorted(Path(f"checkpoints/{data_mode}").glob("fno_*"), 
                                key=lambda d: d.stat().st_mtime)

        if fno_time_folders:
            model = load_model(fno_time_folders[-1])
            print_bold(f"Loading FNO ({data_mode}) from: {fno_time_folders[-1]}")
            plot_training_history(fno_time_folders[-1])
            task4_results = task4_evaluation(model, time_res_dir)
            models[data_mode] = model

    if models:
        print_bold("Bonus Task: Evaluate All2All Training Across Time:")
        model_errors = []
        for data_mode in ['onetoall', 'all2all']:
            print_bold(f"Using FNO ({data_mode})")
            errors = bonus_task_evaluation(models[data_mode], time_res_dir)
            model_errors.append(errors)

        plot_model_errors_temporal(model_errors, time_res_dir)
        
    print(f"All plots have been saved in the results directory.")

if __name__ == "__main__":
    main()