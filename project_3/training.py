import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import AllenCahnFNO

import json
from pathlib import Path

from utils import print_bold

class AllenCahnDataset(Dataset):
    def __init__(self, data, epsilon_values, time_points):
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
            self.indices.extend([(eps, i) for i in range(n_samples)])
    
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

def get_optimizer(model, learning_rate=1e-3):
    """
    TODO: Configure optimizer and learning rate schedule
    Consider:
    - Adam with appropriate learning rate
    - Learning rate schedule for curriculum
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

def train_model(model, train_data, val_data, epsilon_values, time_points, 
                batch_size=32, epochs=100, device='cuda',
                learning_rate=1e-3, curriculum_steps=None):
    """
    Training loop with curriculum learning on epsilon values.
    
    curriculum_steps: list of (epoch, epsilon_subset) tuples defining when to introduce each epsilon value
    """
    train_dataset = AllenCahnDataset(train_data, epsilon_values, time_points)
    val_dataset = AllenCahnDataset(val_data, epsilon_values, time_points)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    model = model.to(device)
    best_val_loss = float('inf')

    # TODO: change this
    l = torch.nn.MSELoss()
    for epoch in range(epochs):
        # Update curriculum if needed
        if curriculum_steps:
            for step_epoch, eps_subset in curriculum_steps:
                if epoch == step_epoch:
                    train_dataset = AllenCahnDataset(train_data, eps_subset, time_points)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    print(f"Curriculum update: now training on epsilon values {eps_subset}")
        
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
            torch.save(model.state_dict(), 'best_model.pt')
    
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
    model = AllenCahnFNO(modes=16, width=128, depth=4, device=device)

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

    # TODO: sort curriculum steps by increasing system difficulty
    # use epsilon_values we get from config
    # Example curriculum steps
    curriculum_steps = [
        (0, [0.1]),           # Start with largest epsilon
        (20, [0.1, 0.05]),    # Add medium epsilon
        (40, [0.1, 0.05, 0.02])  # Add smallest epsilon
    ]

    # Reporting some other information
    n_train = config['dataset_params']['n_train']
    n_test = config['dataset_params']['n_test']
    dt = config['temporal_grid']['dt']
    nx = config['spatial_grid']['nx']
    x_grid = np.array(config['spatial_grid']['x_grid'])

    #==================================================
    # Training
    #==================================================
    print_bold("1. Pre-training of the Foundation model")
    train_data_dict = np.load(f"{data_folder}/train_sol.npy", allow_pickle=True).item()
    for ic_type in train_data_dict.keys():
        print(f"Training with IC data of function class {ic_type}")
        train_dataset = AllenCahnDataset(train_data_dict[ic_type], epsilon_values, time_points)
        # TODO: use train_model here

    #==================================================
    # Evaluation
    #==================================================    
    print_bold("2. Evaluation of the Foundation model")

    # Standard test set with default samplers
    print_bold(f"2.1 In-distribution test set with ɛ = {epsilon_values} and default samplers")
    test_data_dict = np.load(f"{data_folder}/test_sol.npy", allow_pickle=True).item()
    for ic_type in test_data_dict.keys():
        print(f"Evaluating with IC data of function class {ic_type}")
        test_dataset = AllenCahnDataset(test_data_dict[ic_type], epsilon_values, time_points)

    # OOD test set with special samplers but standard eps
    print_bold(f"2.2 OOD test set with same ɛ = {epsilon_values}, but special parameters for samplers: {config['ood_params']}")
    test_ood_data_dict = np.load(f"{data_folder}/test_sol_OOD.npy", allow_pickle=True).item()
    for ic_type in test_ood_data_dict.keys():
        print(f"Evaluating with IC data of function class {ic_type}")
        test_ood_dataset = AllenCahnDataset(test_ood_data_dict[ic_type], epsilon_values, time_points)

    # Epsilon test set with extra-, interpolated eps
    print_bold(f"2.3 Epsilon test set with special ɛ = {added_epsilon_values} and default samplers")
    test_eps_data_dict = np.load(f"{data_folder}/test_sol_eps.npy", allow_pickle=True).item()
    for ic_type in test_eps_data_dict.keys():
        print(f"Evaluating with IC data of function class {ic_type}")
        test_eps_dataset = AllenCahnDataset(test_eps_data_dict[ic_type], added_epsilon_values, time_points)
        

if __name__ == '__main__':
    main()
