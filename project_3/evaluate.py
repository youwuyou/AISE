import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from training import AllenCahnDataset
from utils import (
print_bold,
load_base_model
)

import argparse
import json
from pathlib import Path

@torch.no_grad()
def evaluate(model, val_loader, device: str = "cuda") -> float:
    model.eval()
    total_loss = 0.0
    for item_dict in val_loader:
        u0 = item_dict['initial'].to(device)
        ut = item_dict['target'].to(device)
        epsilon = item_dict['epsilon'].to(device)
        times = item_dict['times'].to(device)

        u_pred = model(u0, epsilon, times)
        loss = nn.MSELoss()(u_pred, ut)
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    relative_loss = (avg_loss / torch.mean(torch.pow(ut, 2))) * 100

    return relative_loss

from training import (
get_loss_func,
train_step,
validation_step
)

def fine_tune(model, dataset, dataset_name="test_sol", checkpoint_dir="checkpoints/",
              fine_tuning_epochs=30, device="cuda"):
    from torch.utils.data import random_split

    # Split dataset 80/20 for train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Fix batch size
    batch_size = int(0.1 * len(dataset))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = get_loss_func("mse")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
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

def run_experiment(model, test_data_dict, dataset_name, epsilon_values, time_points, checkpoint_dir="checkpoints/", fewshot_num=10, batch_size=32, device="cuda"):
    # Standard test set with default samplers
    test_dataset = AllenCahnDataset("testing",  test_data_dict, epsilon_values, time_points)
    val_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    zero_shot_loss  = evaluate(model, val_loader)

    # Fine-tuning model
    fewshot_num = 20
    print_bold(f"Fine-tune the model with {fewshot_num} examples for each dataset before evaluating")
    finetune_dataset = AllenCahnDataset("finetune", test_data_dict, epsilon_values, time_points, fewshot_num=fewshot_num)
    fine_tuned_model = fine_tune(model, finetune_dataset, dataset_name=dataset_name, checkpoint_dir=checkpoint_dir, device=device)
    fine_tuned_loss = evaluate(fine_tuned_model, val_loader)
    return (zero_shot_loss, fine_tuned_loss)

def main(finetune: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    #==================================================
    # Model initialization
    #==================================================
    ace_fno_folders = sorted(Path(f'checkpoints/').glob('ace_fno_*'), 
                        key=lambda d: d.stat().st_mtime)
    ace_fno_folder  = ace_fno_folders[-1] # fetch the latest one
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

    # Standard test set with default samplers
    print_bold(f"In-distribution test set with ɛ = {epsilon_values} and default samplers")
    test_data_dict = np.load(f"{data_folder}/test_sol.npy", allow_pickle=True).item()
    zero_shot_loss, fine_tuned_loss = run_experiment(model, 
                                                test_data_dict, 
                                                "test_sol", 
                                                epsilon_values, 
                                                time_points,
                                                checkpoint_dir=ace_fno_folder, 
                                                device=device)

    print(f"Zero shot loss {zero_shot_loss}, Fine-tuned loss {fine_tuned_loss}")

    #                     anti-curriculum         curriculum (depth 2)     curriculum (depth 4)
    # test_dataset     # 31.397735595703125%      15.19435977935791%       14.052427291870117%
    # test_ood_dataset # 41.573387145996094%      26.54030990600586%       23.40335464477539%
    # test_eps_dataset # 20312358.0%              4481.77685546875%        75799912.0%
    #      eps[1:6]                               16.2211971282959%        23.40335464477539%


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    main(finetune=args.finetune)