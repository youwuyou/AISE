import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import numpy as np
import json

class TrajectorySubset(Dataset):
    """A subset of the PDEDataset that uses specific trajectories."""
    def __init__(self, dataset, trajectory_indices):
        self.dataset = dataset
        self.trajectory_indices = trajectory_indices
        self.len_times = dataset.len_times
    
    def __len__(self):
        return len(self.trajectory_indices) * self.len_times
    
    def __getitem__(self, index):
        trajectory_idx = index // self.len_times
        time_pair_idx = index % self.len_times
        actual_trajectory_idx = self.trajectory_indices[trajectory_idx]
        return self.dataset.get_time_pair_sample(actual_trajectory_idx, time_pair_idx)
        
    def get_trajectory(self, idx):
        """Get a complete trajectory from the subset."""
        actual_idx = self.trajectory_indices[idx]
        return self.dataset.get_trajectory(actual_idx)

class PDEDataset(Dataset):
    def __init__(self,
                 data_path="data/train_sol.npy",
                 total_time=1.0,
                 training_samples=64,
                 training_mode="all2all"):
        """
        PDE Dataset that maintains trajectory integrity and generates time pairs for training.
        
        Args:
            data_path: Path to the .npy file containing trajectories
            total_time: Total time span of the simulation
            training_samples: Number of trajectories for training
            training_mode: Strategy for generating time pairs ('all2all' or 'vanilla')
        """
        self.data = np.load(data_path).astype(np.float32)
        self.total_trajectories = self.data.shape[0]
        self.timesteps = self.data.shape[1]
        self.spatial_dim = self.data.shape[2]
        
        assert self.total_trajectories >= 2 * training_samples, \
            f"Not enough trajectories ({self.total_trajectories}) for requested training samples ({training_samples})"
        
        self.train_size = training_samples
        self.val_size = training_samples
        self.dt = total_time / (self.timesteps - 1)
        self.training_mode = training_mode
        
        # Generate time pairs based on training mode
        self.time_pairs = self._generate_time_pairs()
        self.len_times = len(self.time_pairs)
        
        # Initialize trajectory indices as None (will be set during train/val split)
        self.train_trajectories = None
        self.val_trajectories = None
        
    def _generate_time_pairs(self):
        """Generate time pairs based on training mode."""
        if self.training_mode == "all2all":
            pairs = [(i, j) for i in range(self.timesteps) for j in range(i, self.timesteps)]
        elif self.training_mode == "vanilla":
            pairs = [(0, i) for i in range(self.timesteps)]
        else:
            raise ValueError("Training mode must be 'all2all' or 'vanilla'")
        return pairs
    
    def get_trajectory(self, idx):
        """
        Get a complete trajectory without breaking it into time pairs.
        
        Args:
            idx: Index of the trajectory
            
        Returns:
            trajectory: Complete trajectory data of shape (timesteps, spatial_dim)
        """
        return self.data[idx]
    
    def prepare_splits(self, random_seed=None):
        """
        Prepare training and validation splits at trajectory level.
        
        Args:
            random_seed: Optional seed for reproducible splitting
        """
        # Create list of all trajectory indices and shuffle them
        all_indices = list(range(self.total_trajectories))
        rng = np.random.default_rng(random_seed)
        rng.shuffle(all_indices)
        
        # Split into train and validation trajectory indices
        self.train_trajectories = all_indices[:self.train_size]
        self.val_trajectories = all_indices[self.train_size:self.train_size + self.val_size]
        
        # Verify no overlap
        assert len(set(self.train_trajectories).intersection(set(self.val_trajectories))) == 0, \
            "Error: Found overlap between training and validation trajectories!"
            
        return self.train_trajectories, self.val_trajectories
    
    def get_time_pair_sample(self, trajectory_idx, time_pair_idx):
        """
        Get a single time pair sample from a trajectory.
        
        Args:
            trajectory_idx: Index of the trajectory
            time_pair_idx: Index of the time pair
            
        Returns:
            tuple: (Δt, inputs, outputs) for the specified time pair
        """
        t_inp, t_out = self.time_pairs[time_pair_idx]
        Δt = torch.tensor((t_out - t_inp) * self.dt, dtype=torch.float32)
        
        # Get input and output data
        inp_np = self.data[trajectory_idx, t_inp].reshape(1, self.spatial_dim)
        inputs = torch.from_numpy(inp_np)
        time_channel = torch.ones_like(inputs) * Δt
        inputs = torch.cat([inputs, time_channel], dim=0)
        
        out_np = self.data[trajectory_idx, t_out].reshape(1, self.spatial_dim)
        outputs = torch.from_numpy(out_np)
        
        return Δt, inputs, outputs
    
    def __len__(self):
        """
        Returns the total number of samples (trajectories × time pairs).
        """
        if self.train_trajectories is None:
            raise RuntimeError("Dataset splits not prepared. Call prepare_splits() first.")
        return len(self.train_trajectories) * self.len_times
    
    def __getitem__(self, index):
        """
        Get a sample by index, maintaining trajectory integrity.
        
        Args:
            index: Global sample index
            
        Returns:
            tuple: (Δt, inputs, outputs) for the specified sample
        """
        if self.train_trajectories is None:
            raise RuntimeError("Dataset splits not prepared. Call prepare_splits() first.")
            
        # Convert global index to trajectory and time pair indices
        trajectory_idx = index // self.len_times
        time_pair_idx = index % self.len_times
        
        # Get the actual trajectory index from our stored train trajectories
        actual_trajectory_idx = self.train_trajectories[trajectory_idx]
        
        return self.get_time_pair_sample(actual_trajectory_idx, time_pair_idx)
    
    def get_validation_sample(self, index):
        """
        Get a validation sample, similar to __getitem__ but for validation set.
        """
        if self.val_trajectories is None:
            raise RuntimeError("Dataset splits not prepared. Call prepare_splits() first.")
            
        trajectory_idx = index // self.len_times
        time_pair_idx = index % self.len_times
        
        actual_trajectory_idx = self.val_trajectories[trajectory_idx]
        
        return self.get_time_pair_sample(actual_trajectory_idx, time_pair_idx)

    def report_statistics(self):
        """
        Report detailed statistics about the dataset.
        """
        print("\nDataset Statistics:")
        print(f"Total trajectories available: {self.total_trajectories}")
        print(f"Timesteps per trajectory: {self.timesteps}")
        print(f"Spatial dimensions: {self.spatial_dim}")
        print(f"\nTraining mode: {self.training_mode}")
        print(f"Time pairs per trajectory: {self.len_times}")
        
        if self.train_trajectories is not None:
            print(f"\nTraining trajectories: {len(self.train_trajectories)}")
            print(f"Validation trajectories: {len(self.val_trajectories)}")
            print(f"Total training samples: {len(self.train_trajectories) * self.len_times}")
            print(f"Total validation samples: {len(self.val_trajectories) * self.len_times}")

def train_model(model, training_set, validation_set, config, checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Use OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=len(training_set),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25,   
        final_div_factor=1e4
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    criterion = nn.MSELoss()

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }

    # Load existing config at start
    try:
        with open(checkpoint_dir / 'training_config.json', 'r') as f:
            full_config = json.load(f)
    except FileNotFoundError:
        print("No existing config found, creating new one")
        full_config = {'training_config': config}

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss_accum = 0.0
        
        for time_batch, input_batch, output_batch in training_set:
            # print(f"Training Time_batch: {time_batch}")
            time_batch = time_batch.to(device)
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            
            optimizer.zero_grad()
            output_pred_batch = model(input_batch, time_batch)
            loss = criterion(output_pred_batch, output_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
            
            optimizer.step()
            scheduler.step()  # Step per iteration with OneCycleLR
            
            train_loss_accum += loss.item()
        
        train_loss = train_loss_accum / len(training_set)

        # Validation phase
        with torch.no_grad():
            model.eval()
            val_loss_accum = 0.0
            for time_batch, input_batch, output_batch in validation_set:
                time_batch = time_batch.to(device)
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                # print(f"Time_batch: {time_batch}")

                output_pred_batch = model(input_batch, time_batch)
                val_loss_batch = criterion(output_pred_batch, output_batch)
                val_loss_accum += val_loss_batch.item()

        val_loss = val_loss_accum / len(validation_set)
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['lr'].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_val_loss'] = val_loss
            training_history['best_epoch'] = epoch
            epochs_without_improvement = 0

            # Save model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, checkpoint_dir / 'best_model.pth')

            # Update and save config
            full_config['training_history'] = training_history
            with open(checkpoint_dir / 'training_config.json', 'w') as f:
                json.dump(full_config, f, indent=4)
                
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if epoch % config['freq_print'] == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

    return model, training_history