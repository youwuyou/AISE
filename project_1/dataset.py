"""
This file contains classes derived from `Dataset` used for specific training dataset we have

- OneToOne: used throughout task 1-3 for both training and evaluations
- All2All: used for task 4 & bonus task

Training dataset
0. train_sol.npy: (128, 5, 64)

Testing datasets
1. test_sol.npy:         (128, 5, 64)
2. test_sol_res_{s}.npy: (128, 2, s)  with s ∈ {32, 64, 96, 128}
3. test_sol_OOD.npy:     (128, 2, 64)
"""

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

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
        elif which == "testing":
            self.data = dataset
        else:
            raise ValueError("Dataset must be 'training', 'validation' or `testing`")
            
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

class All2All(Dataset):
    def __init__(self, 
                which,
                training_samples=64,
                data_path="data/train_sol.npy",
                lx=1.0, 
                dt=0.25,
                data_mode="all2all",
                time_pairs=None,
                device='cuda'):
        # Load and type dataset consistently
        dataset = torch.from_numpy(np.load(data_path)).type(torch.float32)
        self.device = device
        
        if device == 'cuda':
            dataset = dataset.to(device)
        
        if which == "training":
            self.data = dataset[:training_samples]
        elif which == "validation":
            self.data = dataset[training_samples:]
        elif which == "testing":
            self.data = dataset
        else:
            raise ValueError("Dataset must be 'training', 'validation' or `testing`")
            
        self.length = len(self.data)
        self.dt = dt
        self.nt = self.data.shape[1]  # Add explicit number of timesteps
        
        # Stack all available timesteps
        self.u = torch.stack([self.data[:, i, :] for i in range(self.nt)], dim=1)
        
        # Calculate time derivatives based on available timesteps
        self.v = []
        for i in range(self.nt - 1):
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
        self.nx = self.data.shape[-1]
        self.lx = lx
        self.x_grid = torch.linspace(0, self.lx, self.nx, device=self.device)
        
        if not time_pairs:
            self.time_pairs = self._generate(data_mode)
        else:
            self.time_pairs = time_pairs

        self.len_times = len(self.time_pairs)

        # DEBUG: use this to verify time pairs are selected correctly
        # print(f"Using {len(self.time_pairs)} time pair(s) per trajectory")
        # print(f"{self.time_pairs}")
        
        # Pre-compute dt values for all time pairs
        self.dt_values = torch.tensor(
            [(j - i) * self.dt for i, j in self.time_pairs],
            device=self.device
        )

    def _generate(self, data_mode):
        # Generate all possible time pairs (ensuring t_end >= t_start)
        if data_mode == "all2all":
            print(f"Using all2all strategy")
            pairs = [(i, j) for i in range(self.nt)
                            for j in range(i, self.nt)]
        elif data_mode == "onetoall":
            print(f"Using one-to-all (vanilla) strategy")
            pairs = [(0, j) for j in range(self.nt)]
        else:
            raise ValueError("Data mode must be 'all2all' or 'onetoall'")
        
        return pairs

    def __len__(self):
        return self.length * self.len_times
    
    def __getitem__(self, idx):
        # Calculate indices
        traj_idx = idx // self.len_times
        pair_idx = idx % self.len_times
        t_start, t_end = self.time_pairs[pair_idx]
        
        # Ensure valid time indices
        assert t_end >= t_start, f"Invalid time indices: start={t_start}, end={t_end}"
        assert t_start < self.nt, f"Start index {t_start} exceeds available timesteps {self.nt}"
        assert t_end < self.nt, f"End index {t_end} exceeds available timesteps {self.nt}"
        
        # Get data for the specific timesteps
        u_start = self.u[traj_idx, t_start]
        v_start = self.v[traj_idx, min(t_start, self.v.shape[1]-1)]
        dt = torch.full_like(u_start, self.dt_values[pair_idx])
        u_end = self.u[traj_idx, t_end]
        
        # Ensure all tensors are on the same device
        u_start = u_start.to(self.device)
        v_start = v_start.to(self.device)
        dt = dt.to(self.device)
        u_end = u_end.to(self.device)
        
        # Stack input data consistently with OneToOne
        input_data = torch.stack((u_start, v_start, self.x_grid, dt), dim=-1)
        output_data = u_end.unsqueeze(-1)
        
        return input_data, output_data

def main():
    # Initializing dataset loading for One-to-One strategy
    batch_size   = 5
    training_set = DataLoader(OneToOne("training"), batch_size=batch_size, shuffle=True)
    validation_set  = DataLoader(OneToOne("validation"), batch_size=batch_size, shuffle=False)

    # Initializing all2all loading for all2all strategy
    training_set = DataLoader(All2All("training"), batch_size=batch_size, shuffle=True)
    validation_set = DataLoader(All2All("validation"), batch_size=batch_size, shuffle=False)

    # Using custom time pairs chosen for test data
    testing_set = DataLoader(All2All("testing", training_samples=0, time_pairs = [(0, 4)]), batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
   main()