import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json

from time_custom_fno_1d import FNO1d
from matplotlib.gridspec import GridSpec
from time_visualize import (
    visualize_predictions_heatmap,
    visualize_predictions
)

def evaluate_model(model,
                   data_path, 
                   time_strategy="single",
                   timesteps=[0.25, 0.50, 0.75, 1.0],
                   autoregressive=False,
                   mean=0, 
                   std=0.3835):
    """[Keep original docstring]"""
    # Load data and set model in eval mode
    device = next(model.parameters()).device  # Get model's device
    data = np.load(data_path)
    model.eval()
    
    # Get initial conditions
    initial_conditions = data[:, 0]
    n_test = len(initial_conditions)
    
    # Initialize storage
    predictions_dict = {}
    errors_dict = {}
    current_conditions = torch.from_numpy(initial_conditions).float().to(device)
    current_time = 0.0
    
    with torch.no_grad():
        if not autoregressive:
            # Direct prediction at each timestep
            for t_val in timesteps:
                predictions = np.zeros_like(initial_conditions)
                relative_l2_errors = np.zeros(n_test)
                
                for i in range(n_test):
                    # Prepare input
                    x = torch.from_numpy(initial_conditions[i]).float().reshape(1, 1, -1).to(device)
                    x = (x - mean) / std
                    
                    # Add time channel
                    t_channel = torch.ones_like(x) * t_val
                    x = torch.cat([x, t_channel], dim=1)
                    
                    # Get prediction
                    t = torch.tensor([t_val]).float().reshape(1).to(device)
                    pred = model(x, t)
                    
                    # Move to CPU and denormalize prediction
                    pred = pred.cpu().squeeze().numpy() * std + mean
                    predictions[i] = pred
                    
                    # Compute error
                    true_idx = int(t_val * (data.shape[1] - 1))
                    ground_truth = data[i, true_idx]
                    error = np.linalg.norm(pred - ground_truth) / np.linalg.norm(ground_truth)
                    relative_l2_errors[i] = error * 100
                
                # Store results
                predictions_dict[t_val] = predictions
                errors_dict[t_val] = np.mean(relative_l2_errors)
                print(f"Average relative L2 error at t={t_val:.2f}: {errors_dict[t_val]:.4f}%")
        
        else:
            # Autoregressive prediction
            predictions_dict[0.0] = current_conditions.cpu().numpy()
            
            for dt in timesteps:
                current_time += dt
                next_predictions = torch.zeros_like(current_conditions)
                
                for i in range(n_test):
                    # Prepare input
                    x = current_conditions[i].reshape(1, 1, -1)
                    x = (x - mean) / std
                    
                    # Add time channel
                    t_channel = torch.ones_like(x) * dt
                    x = torch.cat([x, t_channel], dim=1)
                    
                    # Get prediction
                    t = torch.tensor([dt]).float().reshape(1).to(device)
                    pred = model(x, t)
                    
                    # Move to CPU and denormalize prediction
                    pred = pred.cpu().squeeze() * std + mean
                    next_predictions[i] = pred
                
                # Store predictions
                predictions_dict[current_time] = next_predictions.cpu().numpy()
                
                # Compute errors if ground truth is available
                try:
                    true_idx = int(current_time * (data.shape[1] - 1))
                    ground_truth = data[:, true_idx]
                    
                    relative_l2_errors = np.zeros(n_test)
                    for i in range(n_test):
                        error = np.linalg.norm(
                            next_predictions[i].cpu().numpy() - ground_truth[i]
                        ) / np.linalg.norm(ground_truth[i])
                        relative_l2_errors[i] = error * 100
                    
                    errors_dict[current_time] = np.mean(relative_l2_errors)
                    print(f"Average relative L2 error at t={current_time:.2f}: {errors_dict[current_time]:.4f}%")
                except IndexError:
                    print(f"No ground truth available for t={current_time:.2f}")
                
                # Update current conditions for next step
                current_conditions = next_predictions
    
    return predictions_dict, errors_dict

def evaluate_ood(model, data_path="data/test_sol_OOD.npy", mean=0, std=0.3835):
    """[Keep original docstring]"""
    device = next(model.parameters()).device  # Get model's device
    ood_data = np.load(data_path)
    model.eval()
    
    initial_conditions = ood_data[:, 0]
    final_solutions = ood_data[:, -1]
    n_test = len(initial_conditions)
    
    predictions = np.zeros_like(final_solutions)
    relative_l2_errors = np.zeros(n_test)
    
    with torch.no_grad():
        for i in range(n_test):
            x = torch.from_numpy(initial_conditions[i]).float().reshape(1, 1, -1).to(device)
            x = (x - mean) / std
            t_channel = torch.ones_like(x)
            x = torch.cat([x, t_channel], dim=1)
            t = torch.tensor([1.0]).float().reshape(1).to(device)
            
            pred = model(x, t)
            pred = pred.cpu().squeeze().numpy() * std + mean
            predictions[i] = pred
            
            error = np.linalg.norm(pred - final_solutions[i]) / np.linalg.norm(final_solutions[i])
            relative_l2_errors[i] = error * 100
    
    avg_error = np.mean(relative_l2_errors)
    print(f"Average relative L2 error on OOD data: {avg_error:.4f}%")
    return avg_error, predictions, final_solutions


def task4_evaluation(model, res_dir):
    """
    # Task 4: All2All Training
    1. Use 64 trajectories from the training dataset.
    2. Use all provided time snapshots (t = 0.0, 0.25, 0.50, 0.75, 1.0) 
       to train a time-dependent FNO model.
    3. ...
    4. Test on the test_sol.npy dataset at t=1.0
    5. ...
    6. ...
    """
    print("\n\033[1mTask 4: Testing on All2All Training:\033[0m")

    # 4. Evaluate at t = 1.0
    predictions1, errors1 = evaluate_model(
        model,
        data_path="data/test_sol.npy",
        autoregressive=False,
        timesteps=[1.0]
    )
    # Visualize
    test_data = np.load("data/test_sol.npy")
    visualize_predictions(predictions1, test_data, res_dir=res_dir, filename="task4_trajectory.png")

    # Possibly print or compare errors
    print("\n\033[1mTODO: Compare error to the one from Task 1.\033[0m")

    return predictions1


def bonus_task_evaluation(model, res_dir):
    """
    # Bonus Task
    1. Direct prediction at multiple timesteps: t=0.25, 0.50, 0.75, 1.0
    2. Autoregressive predictions
    3. Compare performance
    4. Evaluate on OOD
    5. Visualize with heatmaps
    """
    print("\n\033[1mBonus Task: Evaluate All2All Training on Different Timesteps:\033[0m")
    
    # 1. Direct prediction at multiple timesteps
    print("\n\033[1mDirect Inference at multiple timesteps\033[0m")
    predictions1, errors1 = evaluate_model(
        model,
        data_path="data/test_sol.npy",
        autoregressive=False,
        timesteps=[0.25, 0.50, 0.75, 1.0]
    )

    # 2. Autoregressive with smaller steps (4 steps of 0.25 => 1.0 total)
    print("\n\033[1mAuto regressive at multiple timesteps\033[0m")
    predictions2, errors2 = evaluate_model(
        model,
        data_path="data/test_sol.npy",
        autoregressive=True,
        timesteps=[0.25, 0.25, 0.25, 0.25]
    )

    # 3. Visualize and compare
    test_data = np.load("data/test_sol.npy")
    fig1 = visualize_predictions(predictions1, test_data, res_dir=res_dir, filename="bonus_direct_inference.png")
    fig2 = visualize_predictions(predictions2, test_data, res_dir=res_dir, filename="bonus_ar_inference.png")

    # 4. Compare errors
    plt.figure(figsize=(10, 5))
    plt.plot(list(errors1.keys()), list(errors1.values()), 'b-o', label='Direct')
    plt.plot(list(errors2.keys()), list(errors2.values()), 'r-o', label='Autoregressive')
    plt.xlabel('Time')
    plt.ylabel('Relative L2 Error (%)')
    plt.title('Error Comparison: Direct vs. Autoregressive')
    plt.grid(True)
    plt.legend()
    plt.savefig(res_dir / 'error_comparison.png')
    plt.close()

    # 5. Evaluate on OOD
    print("\n\033[1mTesting OOD at t = 1.0\033[0m")
    ood_error, ood_predictions, ood_truth = evaluate_ood(model)

    # 6. Space-time heatmap
    print("\n\033[1mVisualize with heapmap for direct inference\033[0m")
    timesteps = [0.25, 0.50, 0.75, 1.0]
    predictions_dict, _ = evaluate_model(
        model, 
        data_path="data/test_sol.npy",
        autoregressive=False,
        timesteps=timesteps
    )
    visualize_predictions_heatmap(
        "data/test_sol.npy",
        model,
        predictions_dict,
        trajectory_indices=[0, 2, 3, 4],
        res_dir=res_dir,
        figsize=(24, 5)
    )
    
    return predictions1, predictions2, predictions_dict



def main():
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Directory for saving results
    res_dir = Path("results_time")
    res_dir.mkdir(exist_ok=True)
    
    # Find the newest experiment folder under 'checkpoints/custom_fno/time'
    experiment_dirs = sorted(
        Path("checkpoints/custom_fno/time").glob("fno_*"),
        key=lambda d: d.stat().st_mtime  # sort by last-modified time
    )

    if not experiment_dirs:
        print("No experiment directories found in 'checkpoints/custom_fno/time' matching 'fno_*'.")
        return

    checkpoint_dir = experiment_dirs[-1]  # The last one is the newest
    print(f"Evaluating the newest experiment: {checkpoint_dir}")

    # Load config from the newest experiment folder
    config_file = checkpoint_dir / "training_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"No config file found at: {config_file}")

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    model_config = config_dict["model_config"]
    training_config = config_dict["training_config"]

    # Initialize the model from config
    model_args = {k: v for k, v in model_config.items() if k != "model_type"}
    model = FNO1d(**model_args)

    # Load the model weights and move to device
    model_path = checkpoint_dir / "best_model.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"No model checkpoint file at: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluate tasks
    print(f"Running evaluations in {checkpoint_dir} ...")
    task4_results = task4_evaluation(model, res_dir)
    bonus_results = bonus_task_evaluation(model, res_dir)

    print(f"\nAll plots have been saved in the '{res_dir}' directory.")

if __name__ == "__main__":
    main()