import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from pathlib import Path
import numpy as np
import json

class PDEDataset(Dataset):
    def __init__(self,
                 data_path="data/train_sol.npy",
                 timesteps=5,
                 which="training",
                 training_samples=64,
                 training_mode="all2all"):
        """
        PDE Dataset with support for both all2all and one-at-a-time training strategies
        
        Args:
            data_path: Path to the data file
            timesteps: Number of time steps in the data (5 for this dataset)
            which: One of "training", "validation", or "test"
            training_samples: Number of training trajectories to use
            training_mode: Either "all2all" or "one-at-a-time"
        """
        self.data = np.load(data_path)
        self.T = timesteps
        self.training_mode = training_mode
        
        # Time pairs based on training mode
        if training_mode == "all2all":
            self.time_pairs = [(i, j) for i in range(0, self.T) 
                              for j in range(i + 1, self.T)]
            print(f"Using all2all training with {len(self.time_pairs)} pairs per trajectory (O(k²))")
        else:  # one-at-a-time
            self.time_pairs = [(i, i+1) for i in range(0, self.T-1)]
            print(f"Using one-at-a-time training with {len(self.time_pairs)} pairs per trajectory (O(k))")
        
        self.len_times = len(self.time_pairs)
        
        # Dataset size constants adjusted for actual data size
        self.N_max = 128 * self.len_times
        self.n_val = 32 * self.len_times
        self.n_test = 32 * self.len_times
        
        # Set dataset specific parameters
        if which == "training":
            self.length = min(training_samples * self.len_times, 
                            (128 - 64) * self.len_times)
            self.start = 0
        elif which == "validation":
            self.length = self.n_val
            self.start = (self.N_max - self.n_val - self.n_test) // self.len_times * self.len_times
        elif which == "test":
            self.length = self.n_test
            self.start = (self.N_max - self.n_test) // self.len_times * self.len_times
            
        # Normalization constants
        self.mean = 0
        self.std = 0.3835

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Get a single sample from the dataset
        
        Returns:
            time: Normalized time difference
            inputs: Input tensor with spatial and time channels [2, 64]
            outputs: Target solution [1, 64]
        """
        # Get trajectory and time pair indices
        sample_idx = index // self.len_times + self.start // self.len_times
        time_pair_idx = index % self.len_times
        
        # Ensure we don't exceed data bounds
        sample_idx = min(sample_idx, self.data.shape[0] - 1)
        
        # Get input and output time points
        t_inp, t_out = self.time_pairs[time_pair_idx]
        assert t_out > t_inp, "Time ordering violated"
        
        # Compute normalized time difference
        time = (t_out - t_inp) * 0.2  # Scale to [0, 1]
        
        # Get and normalize input
        inputs = torch.from_numpy(self.data[sample_idx, t_inp]).type(torch.float32).reshape(1, 64)
        inputs = (inputs - self.mean)/self.std
        
        # Create time channel
        inputs_t = torch.ones(1, 64).type(torch.float32) * time
        
        # Combine spatial and time channels
        inputs = torch.cat((inputs, inputs_t), 0)  # [2, 64]
        
        # Get and normalize output
        outputs = torch.from_numpy(self.data[sample_idx, t_out]).type(torch.float32).reshape(1, 64)
        outputs = (outputs - self.mean)/self.std
        
        return float(time), inputs, outputs

def train_model(model, training_set, testing_set, config, checkpoint_dir):
    """
    Model-agnostic training function supporting both training strategies
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                               lr=config['learning_rate'],
                               weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['step_size'],
        gamma=config['gamma']
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    criterion = nn.MSELoss()

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss_accum = 0.0
        
        for time_batch, input_batch, output_batch in training_set:
            optimizer.zero_grad()
            output_pred_batch = model(input_batch, time_batch)
            loss = criterion(output_pred_batch, output_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_accum += loss.item()
        
        train_loss = train_loss_accum / len(training_set)
        scheduler.step()

        # Validation phase
        with torch.no_grad():
            model.eval()
            val_loss_accum = 0.0
            for time_batch, input_batch, output_batch in testing_set:
                output_pred_batch = model(input_batch, time_batch)
                val_loss_batch = criterion(output_pred_batch, output_batch)
                val_loss_accum += val_loss_batch.item()

        val_loss = val_loss_accum / len(testing_set)

        # Update history and save best model
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_val_loss'] = val_loss
            training_history['best_epoch'] = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')

            try:
                with open(checkpoint_dir / 'training_config.json', 'r') as f:
                    full_config = json.load(f)
            except FileNotFoundError:
                full_config = {'training_config': config}

            full_config['training_history'] = training_history
            with open(checkpoint_dir / 'training_config.json', 'w') as f:
                json.dump(full_config, f, indent=4)

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if epoch % config['freq_print'] == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

    return model, training_history