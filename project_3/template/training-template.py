import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

# Example curriculum steps
curriculum_steps = [
    (0, [0.1]),           # Start with largest epsilon
    (20, [0.1, 0.05]),    # Add medium epsilon
    (40, [0.1, 0.05, 0.02])  # Add smallest epsilon
]
