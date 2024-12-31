"""
This file contains classes derived from `Dataset` used for specific training dataset we have

- OneToOne:
- All2All:
- OneToAll (TODO):


Training dataset
0. train_sol.npy: (128, 5, 64)

Testing datasets
1. test_sol.npy:         (128, 5, 64)
2. test_sol_res_{s}.npy: (128, 2, s)  with s ∈ {32, 64, 96, 128}
3. test_sol_OOD.npy:     (128, 2, 64)
"""

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, TensorDataset

class OneToOne(Dataset):
    def __init__(self, 
                which,
                training_samples=64, 
                data_path="data/train_sol.npy",
                lx=1.0, 
                dt=0.25, 
                start_idx=0,
                end_idx=4, 
                device='cuda'):
        # dataset = torch.from_numpy(np.load(data_path))
        dataset = torch.from_numpy(np.load(data_path)).type(torch.float32)

        if device == 'cuda':
            dataset = dataset.type(torch.float32).to(device)
        
        if which == "training":
            self.data = dataset[:training_samples]
        elif which == "validation":
            self.data = dataset[training_samples:]
        else:
            raise ValueError("Dataset must be initialized with 'training' or 'validation'")
            
        self.length = len(self.data)
        self.dt = dt
        self.nt = self.data.shape[1]
        num_timesteps = self.nt - 1
        
        # Stack all available timesteps
        self.u = torch.stack([self.data[:, i, :] for i in range(self.nt)], dim=1)
        
        # Calculate time derivatives based on available timesteps
        # Multiple timesteps - use central differences where possible
        self.v = []
        for i in range(self.nt - 1):
            # print(f"current i: {i}")
            if i == 0:
                # Set initial velocity to zero
                deriv = torch.zeros_like(self.u[:, 0])               
            elif i == self.nt:
                # Backward difference for last point
                deriv = (self.u[:, -1] - self.u[:, -2]) / self.dt
            else:
                # Central difference for interior points
                deriv = (self.u[:, i+1] - self.u[:, i-1]) / (2 * self.dt)
            self.v.append(deriv)
        self.v = torch.stack(self.v, dim=1)

        # Domain setup
        self.lx = lx
        self.nx = self.data.shape[-1]
        self.x_grid = torch.linspace(0, self.lx, self.nx).to(device)
        
        # Adjust indices if they exceed available timesteps
        assert(end_idx >= start_idx)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.u_start = self.u[:, self.start_idx]
        self.v_start = self.v[:, self.start_idx]
        self.u_end = self.u[:, self.end_idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        u_start = self.u_start[idx]
        v_start = self.v_start[idx]
        dt = torch.full_like(u_start, self.dt * (self.end_idx - self.start_idx), device=u_start.device)
        x_grid = self.x_grid.to(u_start.device)
        input_data = torch.stack((u_start, v_start, x_grid, dt), dim=-1)
        output_data = self.u_end[idx].unsqueeze(-1)        
        return input_data, output_data

def main():
    # Testing dataset loading
    batch_size   = 5
    training_set = DataLoader(OneToOne("training"), batch_size=batch_size, shuffle=True)
    testing_set  = DataLoader(OneToOne("validation"), batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
   main()