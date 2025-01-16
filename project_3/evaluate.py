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
import argparse
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
    
    print(f"final count: {count}, total_loss {total_loss}")
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
        lr=1e-4
    )

    loss_fn = get_loss_func("mse")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, min_lr=1e-5
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
                    fewshot_num=30,
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
        for eps in epsilon_values:
            eps_res = {}
            # Standard test set with default samplers
            test_dataset = AllenCahnDataset("testing",  test_data_dict, [eps], time_points, normalize=normalize, ic_types=[ic_type])
            data_loader   = DataLoader(test_dataset, batch_size=test_dataset.traj_total, shuffle=False)

            trajectory = get_single_trajectory(model, data_loader, device)
            plot_trajectory_comparison(dataset_name, ic_type, eps, trajectory, res_dir)

            eps_res["zero-shot"]  = evaluate(model, data_loader)
            eps_res["fine-tuned"] = evaluate(fine_tuned_model, data_loader)

            # stores two-level nested dict
            print(f"Testing for ɛ={eps} over {test_dataset.traj_total} trajectories")
            res[eps] = eps_res

        # Store results for current I.C.
        res_dict[ic_type] = res

    return res_dict

@torch.no_grad()
def get_single_trajectory(model, data_loader, device: str = "cuda"):
    model.eval()
    
    # Get first batch
    item_dict = next(iter(data_loader))
    
    # Get predictions
    u_pred = model(item_dict['initial'].to(device), 
                  item_dict['epsilon'].to(device), 
                  item_dict['times'].to(device))
    
    # Convert everything to numpy arrays
    trajectory_data = {
        'initial': item_dict['initial'][0].cpu().numpy(),
        'target': item_dict['target'][0].cpu().numpy(),
        'predictions': u_pred[0].cpu().numpy(),
        'epsilon': item_dict['epsilon'][0].item(),
        'times': item_dict['times'][0].cpu().numpy()
    }
    
    return trajectory_data

def plot_trajectory_comparison(dataset_name, ic_type, eps, trajectory_data, res_dir):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, art3d
    import numpy as np
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 10, len(trajectory_data['initial']))
    t = trajectory_data['times']
    
    ax.plot(x, [0]*len(x), trajectory_data['initial'],
            color='grey', linewidth=2, label='u0')
    
    assert ((eps - trajectory_data['epsilon']) < 1e-3), f"Error: epsilon value does not match. Expected {eps}, but got {trajectory_data['epsilon']}"

    for i in range(len(trajectory_data['target'])):
        print(f"current i {i}")
        color = plt.cm.Set1(i)
        
        # Plot true solution
        ax.plot(x, [t[i]]*len(x), trajectory_data['target'][i],
                color=color, linewidth=2, label=f'u{i+1}')

        # Fill below true solution
        shape = list(zip(x, [t[i]]*len(x), trajectory_data['target'][i])) \
              + list(zip(x[::-1], [t[i]]*len(x), [0]*len(x)))
        poly = art3d.Poly3DCollection([shape], alpha=0.2, facecolors=color)
        ax.add_collection3d(poly)
        
        # Plot predictions
        ax.plot(x, [t[i]]*len(x), trajectory_data['predictions'][i],
                color=color, linewidth=2, linestyle='--')
    
    ax.set_xlabel('Spatial X')
    ax.set_ylabel('Time')
    ax.set_zlabel('Amplitude')
    ax.view_init(elev=20, azim=-15)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.title(f"Solutions Comparison (ɛ = {eps:.2f})", fontsize=16)
    plt.savefig(f"{res_dir}/{dataset_name}_ic_{ic_type}_eps_{eps}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main(finetune: bool):
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
                                  normalize=True)
        table_data = []
        headers = ["IC Type", "ε", "Zero-shot", "Fine-tuned", "Improvement"]
        
        for ic_type, res in res_dict.items():
            # Add a header row for each IC type
            table_data.append([f"\n{ic_type}", "", "", ""])
            for eps, eps_res in res.items():
                table_data.append([
                    "",
                    f"{eps:.3f}",
                    f"{eps_res['zero-shot'] * 100:.4f}%",
                    f"{eps_res['fine-tuned'] * 100:.4f}%",
                    f"{(1 - eps_res['fine-tuned']/eps_res['zero-shot'])*100:.4f}%"
                ])
        
        print(f"\nResults Summary for {name}:")
        print(tabulate(table_data,
                    headers=headers,
                    tablefmt="simple",
                    colalign=("left", "right", "right", "right")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    main(finetune=args.finetune)