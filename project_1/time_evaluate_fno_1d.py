import numpy as np
import torch
from pathlib import Path
import json

from time_custom_fno_1d import FNO1d
from matplotlib.gridspec import GridSpec
from time_visualize import (
    plot_ibvp_sol_heatmap,
    plot_trajectory_at_time,
    plot_l2_error_by_timestep
)

from enum import Enum, auto

class InferenceStrategy(Enum):
    DIRECT = "direct"
    AUTOREGRESSIVE = "autoregressive"

def evaluate_model(model,
                  data_path, 
                  strategy: InferenceStrategy,
                  t_i: list,
                  t_tot: float = 1.0):
    """
    Evaluate model using specified inference strategy.
    
    Args:
        model: The FNO model to evaluate
        data_path: Path to the data file
        strategy: InferenceStrategy.DIRECT or InferenceStrategy.AUTOREGRESSIVE
        t_i: List of time points [t_0, t_1, ..., t_n] where t_0 is initial time
    """
    device = next(model.parameters()).device
    data = np.load(data_path)
    model.eval()
    
    # Get initial conditions at t = 0
    u_0 = data[:, 0, :]
    n_traj = len(u_0)
    
    predictions_dict = {}
    errors_dict = {}
    
    with torch.no_grad():
        if strategy == InferenceStrategy.DIRECT:
            target_times = t_i[1:]  # Exclude t_0
            
            for t_target in target_times:
                predictions = np.zeros_like(u_0)
                relative_l2_errors = np.zeros(n_traj)

                for i in range(n_traj):
                    # Create input tensor with current state and time difference
                    state = torch.from_numpy(u_0[i]).float().unsqueeze(0)  # [1, 64]
                    time_diff = torch.full_like(state, t_target)  # [1, 64]
                    x = torch.stack([state, time_diff], dim=1)  # [1, 2, 64]
                    x = x.to(device)
                    
                    # Direct inference to target time
                    t = torch.tensor([t_target]).float().to(device)
                    u_pred = model(x, t)
                    u_pred = u_pred.cpu().squeeze().numpy()
                    predictions[i] = u_pred
                    
                    # Compute error
                    if t_target > t_tot:
                        raise ValueError("t_target {t_target} > t_tot {t_tot} not supported by dataset at {data_path}")

                    training_dt = t_tot / (data.shape[1] - 1)
                    time_idx = int(t_target // training_dt)

                    u_true = data[i, time_idx]
                    error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
                    relative_l2_errors[i] = error
                
                predictions_dict[t_target] = predictions
                errors_dict[t_target] = np.mean(relative_l2_errors) * 100
                print(f"Direct inference - Average relative L2 error at t_{t_i.index(t_target)} = {t_target:.2f}: {errors_dict[t_target]:.4f}%")
        
        else:  # AUTOREGRESSIVE
            current_states = torch.from_numpy(u_0).float()  # [n_traj, 64]
            predictions_dict[t_i[0]] = current_states.cpu().numpy()
            
            for i in range(len(t_i) - 1):
                t_current, t_next = t_i[i], t_i[i+1]
                dt = t_next - t_current
                next_predictions = torch.zeros_like(current_states)
                
                for j in range(n_traj):
                    # Create input tensor with current state and time difference
                    state = current_states[j].unsqueeze(0)  # [1, 64]
                    time_diff = torch.full_like(state, dt)  # [1, 64]
                    x = torch.stack([state, time_diff], dim=1)  # [1, 2, 64]
                    x = x.to(device)
                    
                    t = torch.tensor([dt]).float().to(device)

                    u_pred = model(x, t)
                    u_pred = u_pred.cpu().squeeze()
                    next_predictions[j] = u_pred
                
                predictions_dict[t_next] = next_predictions.cpu().numpy()
                
                time_idx = int(t_next * (data.shape[1] - 1))
                u_true = data[:, time_idx]
                
                relative_l2_errors = np.zeros(n_traj)
                for j in range(n_traj):
                    error = np.linalg.norm(
                        next_predictions[j].cpu().numpy() - u_true[j]
                    ) / np.linalg.norm(u_true[j])
                    relative_l2_errors[j] = error
                
                errors_dict[t_next] = np.mean(relative_l2_errors) * 100
                print(f"Autoregressive - Average relative L2 error at t_{i+1} = {t_next:.2f}: {errors_dict[t_next]:.4f}%")
                
                current_states = next_predictions
    
    return predictions_dict, errors_dict

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

    # Define time points for evaluation (only t=0 and t=1)
    t_i = [0.0, 0.25, 0.5, 0.75, 1.0]

    # 4.1 Evaluate at t = 1.0 using direct inference
    predictions1, errors1 = evaluate_model(
        model,
        data_path="data/test_sol.npy",
        strategy=InferenceStrategy.DIRECT,
        t_i=t_i
    )

    # 4.2 Evaluate at t = 1.0 using autoregressive inference
    predictions2, errors2 = evaluate_model(
        model,
        data_path="data/test_sol.npy",
        strategy=InferenceStrategy.AUTOREGRESSIVE,
        t_i=[0.0, 0.5, 1.0]
    )


    # Visualize
    test_data = np.load("data/test_sol.npy")
    plot_trajectory_at_time(predictions1, test_data, res_dir=res_dir, filename="task4_trajectory_direct.png")
    plot_trajectory_at_time(predictions2, test_data, res_dir=res_dir, filename="task4_trajectory_ar.png")

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
    
    # Define time points t_i
    t_i = [0.0, 0.25, 0.50, 0.75, 1.0]  # t_0 through t_4
    
    # 1. Direct prediction at multiple timesteps
    print("\n\033[1mDirect Inference at multiple timesteps\033[0m")
    predictions1, errors1 = evaluate_model(
        model,
        data_path="data/test_sol.npy",
        strategy=InferenceStrategy.DIRECT,
        t_i=t_i
    )

    # 2. Autoregressive prediction
    print("\n\033[1mAutoregressive Inference\033[0m")
    predictions2, errors2 = evaluate_model(
        model,
        data_path="data/test_sol.npy",
        strategy=InferenceStrategy.AUTOREGRESSIVE,
        t_i=t_i
    )

    # 3. Visualize and compare AR vs Direct
    print("\n\033[1mPlotting Average Relative L2 Error Across Time\033[0m")
    results = {
        'Direct': {'errors': list(errors1.values())},
        'AR': {'errors': list(errors2.values())}
    }
    plot_l2_error_by_timestep(results, t_i[1:], res_dir)  # Exclude t_0

    # 4. OOD Evaluation - Print results for both methods
    print("\n\033[1mTesting OOD Performance:\033[0m")
    # OOD Direct
    print("\nDirect Inference on OOD data:")
    ood_direct_pred, ood_direct_errors = evaluate_model(
        model,
        data_path="data/test_sol_OOD.npy",
        strategy=InferenceStrategy.DIRECT,
        t_i=[0.0, 1.0]  # Only initial and final time
    )
    
    # OOD Autoregressive
    print("\nAutoregressive Inference on OOD data:")
    ood_ar_pred, ood_ar_errors = evaluate_model(
        model,
        data_path="data/test_sol_OOD.npy",
        strategy=InferenceStrategy.AUTOREGRESSIVE,
        t_i=t_i  # Full sequence of time points
    )

    # Visualizations
    test_data = np.load("data/test_sol.npy")
    fig1 = plot_trajectory_at_time(predictions1, test_data, res_dir=res_dir, filename="bonus_direct_inference.png")
    fig2 = plot_trajectory_at_time(predictions2, test_data, res_dir=res_dir, filename="bonus_ar_inference.png")

    # Space-time heatmap
    print("\n\033[1mVisualize with heatmap for direct inference\033[0m")
    plot_ibvp_sol_heatmap(
        "data/test_sol.npy",
        model,
        predictions1,
        trajectory_indices=[0, 127],
        res_dir=res_dir,
        figsize=(24, 5)
    )
    return predictions1, predictions2


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

    # Load checkpoint dictionary
    checkpoint = torch.load(model_path, map_location=device)
    # Extract just the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])  # Changed this line
    model = model.to(device)
    model.eval()

    # Evaluate tasks
    print(f"Running evaluations in {checkpoint_dir} ...")
    task4_results = task4_evaluation(model, res_dir)
    bonus_results = bonus_task_evaluation(model, res_dir)

    print(f"\nAll plots have been saved in the '{res_dir}' directory.")

if __name__ == "__main__":
    main()