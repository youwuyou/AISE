"""
Training the mini-foundation model for Allen-Cahn equation (ACE) dataset
    - datasets shall be generated already via running data_generator.py
    - we use curriculum training with all I.C. types, varying epsilon values
        - anti-curriculum with decreasing difficulty shows nice training pattern, but bad results on testing data
        - in-curriculum order shows good results
    - batch sizes vary as we gradually increase the sample amount during training
        - we pre-selected some batch sizes that shall be used for different range of training sample sizes

After training:
    - the model is stored under the checkpoints/ directory
    - training config is available for history tracking
    - training history can be plotted with plot() function in this script

Fine-tuning and evaluation:
    - see evaluate.py script
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from model import AllenCahnFNO
import math

import json
from pathlib import Path

from utils import (
get_experiment_name,
save_config,
plot_training_history,
print_bold
)

class AllenCahnDataset(Dataset):
    def __init__(self, which, data, epsilon_values, time_points, training_quotient=0.9, fewshot_num=10, ic_types = None):
        """
        data: dictionary mapping ic_types to epsilon values 
            - then each epsilon value is mapped to numpy arrays of shape (n_samples, n_timesteps, n_points)
        epsilon_values: list of epsilon values
        time_points: numpy array of time points
        """
        self.data = data
        self.epsilon_values = epsilon_values
        self.time_points = time_points
        if ic_types is None:
            self.ic_types = list(data.keys())
        else:
            self.ic_types = ic_types
        print(f"data {self.ic_types}")
        
        # Create index mapping
        self.indices = []
        for ic_type in self.ic_types:
            for eps in epsilon_values:
                n_samples = len(data[ic_type][eps])
                training_samples = math.ceil(training_quotient * n_samples)
                validation_samples = n_samples - training_samples

                if which == "training":
                    start_idx = 0
                    end_idx = training_samples
                elif which == "validation":
                    start_idx = training_samples
                    end_idx = n_samples
                elif which == "testing":
                    start_idx = 0
                    end_idx = n_samples
                elif which == "finetune":
                    start_idx = 0
                    end_idx = min(fewshot_num, int(n_samples * 0.1))

                # Extend indices with tuples of (epsilon, sample_index) for the specified range
                self.indices.extend([(ic_type, eps, i) for i in range(start_idx, end_idx)])

        self.traj_total = len(list(self.indices))
        print(f"Total number of {which} data across {len(self.time_points)} timepoints: {self.traj_total}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary contains 4 keys:
        1. 'initial': tensor of length 128 (u(x, 0))
        2. 'target': 4x128 tensor showing decay patterns
            - use target[i] to access u(x, t = iΔt)
        3. 'epsilon': single value tensor [10.0]
        4. 'times': 4 timesteps [0.0025, 0.0050, 0.0075, 0.0100]  
        """
        ic_type, eps, sample_idx = self.indices[idx]
        trajectory = self.data[ic_type][eps][sample_idx]
        
        return {
            'initial': torch.FloatTensor(trajectory[0]),
            'target': torch.FloatTensor(trajectory[1:]),
            'epsilon': torch.FloatTensor([eps]),
            'times': torch.FloatTensor(self.time_points[1:])
        }

def get_loss_func(name: str):
    """
    Define custom loss function(s) for training
    TODO: Consider:
    - L2 loss on predictions
    - Physical constraints (energy, boundaries)
    - Gradient-based penalties
    """
    loss_fun = None
    if name == "l1":
        loss_fun = nn.L1Loss()
    elif name == "mse":
        loss_fun = nn.MSELoss()
    elif name == "cross_entropy":
        loss_fun = nn.CrossEntropyLoss()

    return loss_fun

def get_optimizer(model, learning_rate, warmup=True, warmup_steps=None):
    """
    Configure optimizer and learning rate schedule
    Considering learning rate schedule for curriculum
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    if warmup:
        # Create a warmup scheduler that will be chained with ReduceLROnPlateau
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of base lr
            end_factor=1.0,    # End at base lr
            total_iters=warmup_steps
        )
        return optimizer, warmup_scheduler
    return optimizer, None

def train_step(model, batch, optimizer, l):
    """
    Implement single training step and returns loss value
    1. Forward pass
    2. Loss computation
    3. Backward pass
    4. Optimizer step
    """
    optimizer.zero_grad()
    pred = model(batch['initial'], batch['epsilon'], batch['times'])
    loss_func = l(pred, batch['target']) 
    loss_func.backward()
    optimizer.step()
    return loss_func.item()

def validation_step(model, batch, l):
    """
    Implement single validation step and returns loss value
    """
    # Similar to train_step but without gradient updates
    pred = model(batch['initial'], batch['epsilon'], batch['times'])
    return l(pred, batch['target']).item()

def train_model(model, 
                train_data_dict,
                epsilon_values,
                time_points,               
                checkpoint_dir="checkpoints/",
                epochs=100, 
                device='cuda',
                learning_rate=1e-3,
                patience = 10,
                curriculum_steps=None):
    """
    Training loop with curriculum learning on epsilon values.
    
    curriculum_steps: list of (epoch, epsilon_subset) tuples defining when to introduce each epsilon value
    """
    train_dataset = AllenCahnDataset("training", train_data_dict, epsilon_values, time_points)
    val_dataset   = AllenCahnDataset("validation",train_data_dict, epsilon_values, time_points)

    total = train_dataset.traj_total
    if total < 1000:
        batch_size = 32
    elif total >= 1000 and total < 3000:
        batch_size = 64
    elif total >= 3000 and total < 6000:
        batch_size = 128
    elif total >= 6000:
        batch_size = 512

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer with curriculum adjustment
    initial_lr = learning_rate
    min_lr = 1e-5     # Minimum learning rate
    optimizer, warmup_scheduler = get_optimizer(
        model, 
        learning_rate=initial_lr,
        warmup=True,
        warmup_steps=len(curriculum_steps)
    )
    # Use ReduceLROnPlateau with a higher patience for curriculum stages
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,        # Reduce LR by half when plateauing
        patience=patience,
        min_lr=min_lr
    )
   
    model = model.to(device)
    best_val_loss = float('inf')
    l = get_loss_func("mse")

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    for epoch in range(epochs):
        # Update curriculum if needed
        if curriculum_steps:
            for step_epoch, eps_subset in curriculum_steps:
                if epoch == step_epoch:
                    # Load dataset of current curriculum
                    train_dataset = AllenCahnDataset("training", train_data_dict, eps_subset, time_points)
                    val_dataset   = AllenCahnDataset("validation",train_data_dict, eps_subset, time_points)

                    # Adjust learning rate based on curriculum stage
                    current_lr = optimizer.param_groups[0]['lr']
                    new_lr = max(current_lr * 0.7, min_lr)  # Reduce by 30% each stage
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    print(f"Adjusting learning rate to {new_lr} for epsilon values {eps_subset}")

                    # gradually increase number of batch
                    total = train_dataset.traj_total
                    if total < 1000:
                        batch_size = 32
                    elif total >= 1000 and total < 3000:
                        batch_size = 64
                    elif total >= 3000 and total < 6000:
                        batch_size = 128
                    elif total >= 6000:
                        batch_size = 512
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)
                    print(f"Curriculum update: now training on epsilon values {eps_subset} with batch size {batch_size}")
        
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            train_loss += train_step(model, batch, optimizer, l)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += validation_step(model, batch, l)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)       
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Apply warmup schedule if in warmup phase
        if warmup_scheduler and epoch < len(curriculum_steps):
            warmup_scheduler.step()

        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_val_loss'] = val_loss
            training_history['best_epoch']    = epoch
            torch.save(model.state_dict(), checkpoint_dir / 'base_model.pth')

    try:
        with open(checkpoint_dir / 'training_config.json', 'r') as f:
            full_config = json.load(f)
    except FileNotFoundError:
        full_config = {'training_config': config}
    
    full_config['training_history'] = training_history
    
    with open(checkpoint_dir / 'training_config.json', 'w') as f:
        json.dump(full_config, f, indent=4)

    return model


def main():
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
    model_config = {
        "modes": 16,
        "width": 64,
        "depth": 4,
        "device": device
    }

    model = AllenCahnFNO(**{k: v for k, v in model_config.items()})

    #==================================================
    # Load training data
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

    # Set and store training config
    epochs = 800
    epoch_with_all = int(epochs * 0.8)
    print(f"specified epoch index at which we enforce training with all epsilon values {epoch_with_all}")
    device = device
    learning_rate = 1e-4
    patience = 50

    # Generate curriculum step by even splitting
    curriculum_steps: list[tuple] = []
    covered_eps = []
    num_eps = len(epsilon_values)
    base_epoch = epoch_with_all // num_eps
    print(f"Total number of epsilon values for training: {num_eps}")
    print(f"base_epoch to introduce new epsilon value: {base_epoch}")

    # Curriculum training
    for idx, eps in enumerate(epsilon_values):
        covered_eps.append(eps)
        t = (idx * base_epoch, covered_eps.copy())
        curriculum_steps.append(t)
    print(f"curriculum_steps {curriculum_steps}")

    training_config = {
        'n_train': config['dataset_params']['n_train'],
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience,
        'curriculum_steps': curriculum_steps,
        'device': device
    }

    # Create experiment directory with model config
    experiment_name = get_experiment_name(model_config)
    checkpoint_dir = Path(f"checkpoints/") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Trained model will be stored under {checkpoint_dir}")

    # Save full config
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)

    #==================================================
    # Training
    #==================================================
    print_bold(f"Training for all IC data at once")
    train_data_dict = np.load(f"{data_folder}/train_sol.npy", allow_pickle=True).item()
    model = train_model(model,
                    train_data_dict,
                    epsilon_values,
                    time_points,
                    checkpoint_dir=checkpoint_dir,
                    epochs=epochs,
                    device=device,
                    learning_rate=learning_rate,
                    patience=patience,
                    curriculum_steps=curriculum_steps)

def plot():
    ace_fno_folders = sorted(Path(f'checkpoints/').glob('ace_fno_*'), 
                        key=lambda d: d.stat().st_mtime)
    print("Plotting training history...")
    plot_training_history(ace_fno_folders[-1])

if __name__ == '__main__':
    main()
    plot()