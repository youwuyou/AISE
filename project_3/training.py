import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

import json
from pathlib import Path

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
        eps, sample_idx = self.indices[idx]
        trajectory = self.data[eps][sample_idx]
        
        return {
            'initial': torch.FloatTensor(trajectory[0]),
            'target': torch.FloatTensor(trajectory[1:]),
            'epsilon': torch.FloatTensor([eps]),
            'times': torch.FloatTensor(self.time_points[1:])
        }

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
            
            optimizer.zero_grad()
            
            # Forward pass - implement your model to handle these inputs
            pred = model(batch['initial'], batch['epsilon'], batch['times'])
            
            # Compute loss - you might want to modify this
            loss = nn.MSELoss()(pred, batch['target'])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch['initial'], batch['epsilon'], batch['times'])
                val_loss += nn.MSELoss()(pred, batch['target']).item()
        
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
    # Example curriculum steps
    curriculum_steps = [
        (0, [0.1]),           # Start with largest epsilon
        (20, [0.1, 0.05]),    # Add medium epsilon
        (40, [0.1, 0.05, 0.02])  # Add smallest epsilon
    ]

    # Load the latest created dataset
    data_folders = sorted(Path(f'data').glob('dt_*'), key=lambda d: d.stat().st_mtime)
    data_folder  = data_folders[-1]
    print(f"Loading dataset from {data_folder}")

    with open(f'{data_folder}/config.json', 'r') as f:
        config = json.load(f)

    # Extract parameters from config        
    time_points = np.array(config['temporal_grid']['time_points'])
    epsilon_values = config['dataset_params']['epsilon_values']

    # Fetch data dictionary
    train_data_dict = np.load(f"{data_folder}/train_sol.npy", allow_pickle=True).item()
    # test_data_dict = np.load(f"{data_folder}/test_sol.npy", allow_pickle=True).item()

    # Initialize data dictionary
    print(f"Training dataset with epsilon values {train_data_dict.keys()}")
    # print(f"Testing dataset with epsilon values {test_data_dict.keys()}")

    for i, (sampler_name, samples) in enumerate(train_data_dict.items()):
        print(f"current initial condition {sampler_name}")
        print(f"trajectories shape {samples.keys()}")

        for eps in samples.keys():
            print(f"current eps {eps}")

            trajectories = train_data_dict[sampler_name][eps]
            print(f"trajectories shape {trajectories.shape}")

        # of shape (3, 5, 128)
        print(f"{train_data_dict['PL'].keys()}")

        print(f"IC PL at epsilon {0.1}: {train_data_dict['PL'][0.1].shape}")

        # for eps in epsilon_values:
        #     print(f"current ")

        # # TODO: use train_model here, but move AllenCahnDataset out!
        # train_dataset = AllenCahnDataset(train_data_dict, epsilon_values, time_points)
        # test_dataset = AllenCahnDataset(test_data_dict, epsilon_values, time_points)

        # Reporting some information
        n_train = config['dataset_params']['n_train']
        n_test = config['dataset_params']['n_test']
        dt = config['temporal_grid']['dt']
        nx = config['spatial_grid']['nx']
        x_grid = np.array(config['spatial_grid']['x_grid'])


if __name__ == '__main__':
    main()
