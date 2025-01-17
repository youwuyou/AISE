import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tabulate import tabulate

from utils import (
print_bold,
load_base_model
)

from training import (
AllenCahnDataset,
get_loss_func,
train_step,
validation_step
)

import copy
import json
from pathlib import Path

@torch.no_grad()
def evaluate(model, data_loader, device: str = "cuda") -> dict:
    """
    Evaluate the trained or fine-tuned ACE foundation model with data loader
    - returns the average relative L2 error across all passed trajectories
    """

    model.eval()
    total_loss = 0.0
    count = 0
    
    for item_dict in data_loader:
        u_pred = model(item_dict['initial'].to(device), 
                      item_dict['epsilon'].to(device), 
                      item_dict['times'].to(device))
        ut = item_dict['target'].to(device)
        
        # Convert to numpy
        u_pred_np = u_pred.cpu().numpy()
        ut_np = ut.cpu().numpy()
        
        # Compute error between ground truth and prediction for each snapshot
        for t in range(ut_np.shape[1]):
            norm = np.linalg.norm(ut_np[:, t], 2)
            # Quality of our samples could be improved by eliminating this edge case...
            if norm == 0:
                print(f"catch norm equals zero at time t = {t}, abandon this sample:")
            else:
                total_loss += np.linalg.norm(u_pred_np[:, t] - ut_np[:, t], 2) / norm
                count += 1
        return total_loss / count


def count_parameters(model, details = False):
    """Count trainable and frozen parameters with proportions"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    
    print("\nParameter counts:")
    print(f"Trainable parameters: {trainable:,} ({trainable/total:.1%})")
    print(f"Frozen parameters:    {frozen:,} ({frozen/total:.1%})")
    print(f"Total parameters:     {total:,}")

    if details:
        print("\nDetailed parameter breakdown:")
        print("\nTrainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name:<50} {param.numel():>10,} ({param.numel()/total:.1%})")
        
        print("\nFrozen parameters:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"{name:<50} {param.numel():>10,} ({param.numel()/total:.1%})")
    return trainable, frozen, total

def fine_tune(model, dataset, dataset_name="test_sol", checkpoint_dir="checkpoints/",
              fine_tuning_epochs=100, device="cuda"):

    # Split dataset 80/20 for train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Fixed batch size
    batch_size = int(0.1 * len(dataset))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Freeze parameters based on their names
    freeze_film = False
    freeze_fno  = True
    for name, param in model.named_parameters():
        if 'input_layer' in name:
            param.requires_grad = False
        if 'output_layer' in name:
            param.requires_grad = False
        if freeze_film and 'FILM_layers' in name:
            param.requires_grad = False
        if freeze_fno and 'fno_layers' in name:
            param.requires_grad = False
    
    # Print which parameters are trainable
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    
    print("\nFrozen parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name}: {param.shape}")
    
    count_parameters(model)
    # Only optimize parameters that require gradients
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-2
    )

    loss_fn = get_loss_func("mse")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, min_lr=1e-4
    )
    model = model.to(device)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(fine_tuning_epochs):
        model.train()
        train_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            train_loss += train_step(model, batch, optimizer, loss_fn)
        train_loss /= len(loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += validation_step(model, batch, loss_fn)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, Path(checkpoint_dir) / f'finetuned_model_{dataset_name}.pth')

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def run_experiment(model, 
                    test_data_dict,
                    dataset_name, 
                    epsilon_values, 
                    time_points,
                    checkpoint_dir="checkpoints/",
                    res_dir="results/",
                    fewshot_num=20,
                    batch_size=2,
                    device="cuda",
                    normalize=False):

    # Base model is used for zero-shot, prepare for fine-tuned model
    fine_tuned_model = copy.deepcopy(model)
    try:
        fine_tuned_model.load_state_dict(torch.load(checkpoint_dir / f'finetuned_model_{dataset_name}.pth', weights_only=True))    
    except FileNotFoundError:
        print_bold(f"Fine-tune the model with {fewshot_num} examples for each dataset before evaluating")
        fewshot_num = fewshot_num
        finetune_dataset = AllenCahnDataset("finetune", test_data_dict, epsilon_values, time_points, fewshot_num=fewshot_num, normalize=normalize)
        fine_tuned_model = fine_tune(fine_tuned_model, finetune_dataset, dataset_name=dataset_name, checkpoint_dir=checkpoint_dir, device=device)
    else:
        print_bold(f"Fine-tuned model for {dataset_name} found")
        fine_tuned_model = fine_tuned_model.to(device)

    # Stores absolute L2 loss ut - ut_pred
    res_dict = {}
    for ic_type in ["PL", "FS", "GM"]:
        print_bold(f"Retrieving results for I.C. type: {ic_type}")
        res = {}
        traj_dict = {}
        for eps in epsilon_values:
            eps_res = {}
            # Standard test set with default samplers
            test_dataset  = AllenCahnDataset("testing",  test_data_dict, [eps], time_points, normalize=normalize, ic_types=[ic_type])
            data_loader   = DataLoader(test_dataset, batch_size=test_dataset.traj_total, shuffle=False)

            # Evaluating all sample trajectories for current I.C. + epsilons
            eps_res["zero-shot"]  = evaluate(model, data_loader)
            eps_res["fine-tuned"] = evaluate(fine_tuned_model, data_loader)

            # Plot one sample trajectory for current I.C + epsilon category
            trajectory_data = get_single_trajectory(model, fine_tuned_model, data_loader, device)
            traj_dict[eps]  = trajectory_data

            # stores two-level nested dict
            print(f"Testing for ɛ={eps} over {test_dataset.traj_total} trajectories")
            res[eps] = eps_res

        # Plot all epsilon example trajectories for current ic_typ
        plot_single_trajectory_comparison(dataset_name, 
                                        ic_type,
                                        epsilon_values,
                                        res,
                                        traj_dict, 
                                        res_dir)

        # Store results for current I.C.
        res_dict[ic_type] = res

    return res_dict

@torch.no_grad()
def get_single_trajectory(model, fine_tuned_model, data_loader, device: str = "cuda"):
    model.eval()
    
    # Get first batch
    item_dict = next(iter(data_loader))
    
    # Get predictions
    u0 = item_dict['initial'].to(device)
    epsilon = item_dict['epsilon'].to(device)
    times = item_dict['times'].to(device)
    targets = item_dict['target']

    u_pred = model(u0, epsilon, times)
    u_pred_tuned = fine_tuned_model(u0, epsilon, times)

    # Convert everything to numpy arrays
    idx = 5 # choosing trajectory at index 5
    trajectory_data = {
        'initial': u0[idx].cpu().numpy(),
        'target': targets[idx].cpu().numpy(),
        'predictions': u_pred[idx].cpu().numpy(),
        'predictions_tuned': u_pred_tuned[idx].cpu().numpy(),
        'epsilon': epsilon[idx].item(),
        'times': times[idx].cpu().numpy()
    }
 
    return trajectory_data

def plot_single_trajectory_comparison(dataset_name, 
                                      ic_type, 
                                      epsilon_values, 
                                      res, 
                                      traj_dict, 
                                      res_dir):
    import numpy as np
    from mpl_toolkits.mplot3d import art3d
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    z_min, z_max = -1, 1
    fig = plt.figure(figsize=(14, 6 * len(epsilon_values)))
    gs = gridspec.GridSpec(len(epsilon_values), 3, width_ratios=[1, 1, 0.08])

    for row, eps in enumerate(epsilon_values):
        traj_data = traj_dict[eps]
        x = np.linspace(-1, 1, len(traj_data['initial']))
        t = traj_data['times']
        y_min = min(t.min(), 0)
        y_max = max(t.max(), 0)
        for key in ['initial','target','predictions','predictions_tuned']:
            traj_data[key] = np.clip(traj_data[key], -1, 1)
        err_zeroshot  = res[eps]["zero-shot"]
        err_finetuned = res[eps]["fine-tuned"]

        ax1 = fig.add_subplot(gs[row, 0], projection='3d')
        ax2 = fig.add_subplot(gs[row, 1], projection='3d')
        ax_text = fig.add_subplot(gs[row, 2])
        ax_text.text(0, 0.5, f"ε = {eps}",
                     rotation=90,
                     verticalalignment='center',
                     fontsize=12,
                     fontweight='bold')
        ax_text.axis('off')

        ax1.plot(x, [0]*len(x), traj_data['initial'], color='gray', linewidth=2)
        for i in range(len(traj_data['target'])):
            color = plt.cm.Set1(i)
            ax1.plot(x, [t[i]]*len(x), traj_data['target'][i], color=color, linewidth=2)
            min_val = traj_data['target'][i].min()
            shape = list(zip(x, [t[i]]*len(x), traj_data['target'][i])) \
                  + list(zip(x[::-1], [t[i]]*len(x), [min_val]*len(x)))
            poly = art3d.Poly3DCollection([shape], alpha=0.2, facecolors=color)
            ax1.add_collection3d(poly)
            ax1.plot(x, [t[i]]*len(x), traj_data['predictions'][i], color=color, linewidth=2, linestyle='--')
        ax1.set_xlim([-1, 1]); ax1.set_ylim([y_min, y_max]); ax1.set_zlim([z_min, z_max])
        ax1.set_title(f'Base Model | Avg. Relative L2 Error: {err_zeroshot*100:.2f}%')
        ax1.view_init(elev=20, azim=-15)

        ax2.plot(x, [0]*len(x), traj_data['initial'], color='gray', linewidth=2)
        for i in range(len(traj_data['target'])):
            color = plt.cm.Set1(i)
            ax2.plot(x, [t[i]]*len(x), traj_data['target'][i], color=color, linewidth=2)
            min_val = traj_data['target'][i].min()
            shape = list(zip(x, [t[i]]*len(x), traj_data['target'][i])) \
                  + list(zip(x[::-1], [t[i]]*len(x), [min_val]*len(x)))
            poly = art3d.Poly3DCollection([shape], alpha=0.2, facecolors=color)
            ax2.add_collection3d(poly)
            ax2.plot(x, [t[i]]*len(x), traj_data['predictions_tuned'][i], color=color, linewidth=2, linestyle='--')
        ax2.set_xlim([-1, 1]); ax2.set_ylim([y_min, y_max]); ax2.set_zlim([z_min, z_max])
        ax2.set_title(f'Fine-tuned Model | L2 Error: {err_finetuned*100:.2f}%')
        ax2.view_init(elev=20, azim=-15)
        ax1.set_xlabel('Spatial X'); ax1.set_ylabel('Time'); ax1.set_zlabel('Amplitude')
        ax2.set_xlabel('Spatial X'); ax2.set_ylabel('Time'); ax2.set_zlabel('Amplitude')

    # With a single suptitle:
    fig.suptitle(f"Solutions Comparison ({ic_type})\nε in {epsilon_values}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{res_dir}/{dataset_name}_ic_{ic_type}_combined.png", dpi=300)
    plt.close()

def main():
    res_dir = Path('results/')
    res_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    #==================================================
    # Model initialization (by default from latest one)
    #==================================================
    ace_fno_folders = sorted(Path(f'checkpoints/').glob('ace_fno_*'), 
                        key=lambda d: d.stat().st_mtime)
    ace_fno_folder  = ace_fno_folders[-1]
    model = load_base_model(ace_fno_folder)
    print(f"Loaded Custom FNO from: {ace_fno_folder}")

    #==================================================
    # Load testing data
    #==================================================
    data_folders = sorted(Path(f'data').glob('dt_*'), key=lambda d: d.stat().st_mtime)
    data_folder  = data_folders[-1]
    print(f"Loading dataset from {data_folder}")

    with open(f'{data_folder}/config.json', 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    time_points = np.array(config['temporal_grid']['time_points'])
    epsilon_values = config['dataset_params']['epsilon_values']
    added_epsilon_values = config['dataset_params']['added_epsilon_values']

    # Settings we used for each dataset
    eps_mapping = {
        "test_sol": epsilon_values,
        "test_sol_OOD": epsilon_values,
        "test_sol_eps": added_epsilon_values,
    }

    for name, eps_vals in eps_mapping.items():
        test_data_dict = np.load(f"{data_folder}/{name}.npy", allow_pickle=True).item()
        res_dict = run_experiment(model,
                                  test_data_dict,
                                  name,
                                  eps_vals,
                                  time_points,
                                  checkpoint_dir=ace_fno_folder,
                                  res_dir=res_dir,
                                  device=device,
                                  normalize=False)
        table_data = []
        headers = ["IC Type", "ε", "Zero-shot", "Fine-tuned"]
        
        for ic_type, res in res_dict.items():
            # Add a header row for each IC type
            table_data.append([f"\n{ic_type}", "", "", ""])
            for eps, eps_res in res.items():
                table_data.append([
                    "",
                    f"{eps:.3f}",
                    f"{eps_res['zero-shot'] * 100:.4f}%",
                    f"{eps_res['fine-tuned'] * 100:.4f}%"
                ])
        
        print(f"\nResults Summary for {name}:")
        print(tabulate(table_data,
                    headers=headers,
                    tablefmt="simple",
                    colalign=("left", "right", "right", "right")))

if __name__ == '__main__':
    main()