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
print_bold
)

class AllenCahnDataset(Dataset):
    def __init__(self, which, data, epsilon_values, time_points, training_quotient=0.8):
        """
        data: dictionary mapping epsilon values to numpy arrays of shape (n_samples, n_timesteps, n_points)
        epsilon_values: list of epsilon values
        time_points: numpy array of time points
        """
        self.data = data
        self.epsilon_values = epsilon_values
        self.time_points = time_points
        
        # Create index mapping
        self.indices = []
        for eps in epsilon_values:
            n_samples = len(data[eps])
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

            # Extend indices with tuples of (epsilon, sample_index) for the specified range
            self.indices.extend([(eps, i) for i in range(start_idx, end_idx)])
        print(f"Total number of {which} data across {len(self.time_points)} timepoints: {len(list(self.indices))}")
    
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
        eps, sample_idx = self.indices[idx]
        trajectory = self.data[eps][sample_idx]
        
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

def get_optimizer(model, learning_rate):
    """
    Configure optimizer and learning rate schedule
    Consider:
    TODO: - Learning rate schedule for curriculum
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return optimizer

def train_step(model, batch, optimizer, l):
    """
    Implement single training step
    """
    optimizer.zero_grad()
    pred = model(batch['initial'], batch['epsilon'], batch['times']) # 1. Forward pass
    loss_func = l(pred, batch['target']) # 2. Loss computation
    loss_func.backward()    # 3. Backward pass
    optimizer.step()        # 4. Optimizer step
    return loss_func.item() # Return loss value

def validation_step(model, batch, l):
    """
    Implement single validation step    
    """
    # Similar to train_step but without gradient updates
    pred = model(batch['initial'], batch['epsilon'], batch['times'])
    return l(pred, batch['target']).item() # Return loss value

def train_model(model, 
                train_dataset,
                val_dataset,
                checkpoint_dir="checkpoints/",
                batch_size=32, 
                epochs=100, 
                device='cuda',
                learning_rate=1e-3,
                patience = 5,
                curriculum_steps=None):
    """
    Training loop with curriculum learning on epsilon values.
    
    curriculum_steps: list of (epoch, epsilon_subset) tuples defining when to introduce each epsilon value
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = get_optimizer(model, learning_rate=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)
    
    model = model.to(device)
    best_val_loss = float('inf')

    # TODO: change this
    l = get_loss_func("mse")
    for epoch in range(epochs):
        # Update curriculum if needed
        # if curriculum_steps:
        #     for step_epoch, eps_subset in curriculum_steps:
        #         if epoch == step_epoch:
        #             train_dataset = AllenCahnDataset(train_data, eps_subset, time_points)
        #             train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #             print(f"Curriculum update: now training on epsilon values {eps_subset}")
        
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
        
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'base_model.pth')
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
    batch_size = 16
    epochs = 100
    device = device
    learning_rate = 1e-3
    # TODO: sort curriculum steps by increasing system difficulty
    # use epsilon_values we get from config
    patience = 5
    curriculum_steps = [
        (0, [0.1]),           # Start with largest epsilon
        (20, [0.1, 0.05]),    # Add medium epsilon
        (40, [0.1, 0.05, 0.02])  # Add smallest epsilon
    ]
    training_config = {
        'n_train': config['dataset_params']['n_train'],
        'batch_size': batch_size,
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
    print_bold("1. Pre-training of the Foundation model")
    train_data_dict = np.load(f"{data_folder}/train_sol.npy", allow_pickle=True).item()
    for ic_type in train_data_dict.keys():
        print(f"Training with IC data of function class {ic_type}")
        train_dataset = AllenCahnDataset("training", train_data_dict[ic_type], epsilon_values, time_points)
        val_dataset   = AllenCahnDataset("validation",train_data_dict[ic_type], epsilon_values, time_points)

        # Training of the model
        model = train_model(model,
                        train_dataset,
                        val_dataset,
                        checkpoint_dir=checkpoint_dir,
                        batch_size=batch_size, 
                        epochs=epochs, 
                        device=device,
                        learning_rate=learning_rate,
                        patience=patience,
                        curriculum_steps=curriculum_steps)

if __name__ == '__main__':
    main()
