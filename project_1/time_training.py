import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import numpy as np
import json


class PDEDataset(Dataset):
    def __init__(self,
                 data_path="data/train_sol.npy",
                 total_time=1.0,
                 training_samples=64,
                 training_mode="all2all"):
        """
        PDE Dataset with enhanced debugging for time pairs.
        """
        self.data = np.load(data_path).astype(np.float32)
        self.total_trajectories = self.data.shape[0]
        
        assert self.total_trajectories >= 2 * training_samples, \
            f"Not enough trajectories ({self.total_trajectories}) for requested training samples ({training_samples})"
        
        self.train_size = training_samples
        self.val_size = training_samples
        
        self.timesteps = self.data.shape[1]
        self.dt = total_time / (self.timesteps - 1)
        self.training_mode = training_mode

        # Enhanced time pairs generation with debugging
        if training_mode == "all2all":
            self.time_pairs = [(i, j) for i in range(0, self.timesteps) for j in range(i, self.timesteps)]
            print(f"\nall2all mode: using {len(self.time_pairs)} pairs starting from t0")
            # time_pairs: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
                    
            # Print detailed time pair information
            print("\nTime pairs analysis:")
            print(f"Number of timesteps: {self.timesteps}")
            print(f"Total time: {total_time}")
            print(f"Time step size (dt): {self.dt}")
            print(f"Number of time pairs per trajectory: {len(self.time_pairs)}")
            print("\nExample time differences:")
            for i, (t_inp, t_out) in enumerate(self.time_pairs):
                time_diff = (t_out - t_inp) * self.dt
                print(f"Pair {i}: (t{t_inp}, t{t_out}) → Δt = {time_diff:.4f}")
                if i >= 9:  # Show first 10 pairs only
                    print("...")
                    break
                    
        elif training_mode == "vanilla":
            self.time_pairs = [(0, i) for i in range(self.timesteps)]
            print(f"\nVanilla mode: using {len(self.time_pairs)} pairs starting from t0")
            # time_pairs: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        else:
            raise ValueError("Training strategy not implemented. Available strategies are 'all2all' or 'vanilla'.")

        print(f"time_pairs: {self.time_pairs}")
        self.len_times = len(self.time_pairs)
        self.train_length = self.train_size * self.len_times
        self.val_length = self.val_size * self.len_times

    def __getitem__(self, index):
        """
        Enhanced __getitem__ with time pair debugging.
        """
        is_validation = index >= self.train_length
        
        if is_validation:
            adjusted_index = index - self.train_length
            sample_idx = (adjusted_index // self.len_times) + self.train_size
        else:
            sample_idx = index // self.len_times
            
        time_pair_idx = index % self.len_times
        t_inp, t_out = self.time_pairs[time_pair_idx]        
        time = torch.tensor((t_out - t_inp) * self.dt, dtype=torch.float32)
        
        # Get input and output data
        inp_np = self.data[sample_idx, t_inp].reshape(1, 64)
        inputs = torch.from_numpy(inp_np)
        time_channel = torch.ones_like(inputs) * time
        inputs = torch.cat([inputs, time_channel], dim=0)
        
        out_np = self.data[sample_idx, t_out].reshape(1, 64)
        outputs = torch.from_numpy(out_np)

        return time, inputs, outputs

    def debug_batch(self, batch_size=5):
        """
        Helper method to debug a sample batch.
        """
        indices = np.random.choice(self.train_length, batch_size, replace=False)
        print("\nDebugging sample batch:")
        for idx in indices:
            time, inputs, outputs = self.__getitem__(idx)
            sample_idx = idx // self.len_times
            time_pair_idx = idx % self.len_times
            t_inp, t_out = self.time_pairs[time_pair_idx]
            print(f"Index {idx}: trajectory {sample_idx}, (t{t_inp}, t{t_out}) → Δt = {time.item():.4f}")

    def get_train_val_samplers(self):
        """
        Returns samplers for training and validation splits.
        Useful for creating DataLoaders.
        """
        train_indices = range(0, self.train_length)
        val_indices = range(self.train_length, self.train_length + self.val_length)
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        return train_sampler, val_sampler

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