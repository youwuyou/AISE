import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from neuralop.models import FNO
from pathlib import Path
import os
from datetime import datetime
from training import prepare_data, train_model


def main():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Configuration
    # config = {
    #     'n_train': 64,
    #     'batch_size': 2,
    #     'modes': 32,
    #     'width': 64,
    #     'learning_rate': 0.001,
    #     'epochs': 500,
    #     'step_size': 50,
    #     'gamma': 0.5,
    #     'patience': 15,
    #     'freq_print': 1
    # }

    # 9.77%
    # config = {
    #     'n_train': 64,
    #     'batch_size': 5,
    #     'modes': 10,
    #     'width': 128,
    #     'learning_rate': 0.0001,
    #     'epochs': 500,
    #     'step_size': 100,
    #     'gamma': 0.1,
    #     'patience': 50,
    #     'freq_print': 1
    # }


    # 7.3%
    # config = {
    #     'n_train': 64,
    #     'batch_size': 5,
    #     'modes': 10,
    #     'width': 128,
    #     'learning_rate': 0.001,
    #     'epochs': 500,
    #     'step_size': 100,
    #     'gamma': 0.1,
    #     'patience': 50,
    #     'freq_print': 1
    # }

    config = {
        'n_train': 64,
        'batch_size': 5,
        'modes': 25,
        'width': 64,
        'learning_rate': 0.001,
        'epochs': 500,
        'step_size': 100,
        'gamma': 0.1,
        'patience': 50,
        'freq_print': 1
    }


    # 15.93%
    # config = {
    #     'n_train': 64,
    #     'batch_size': 5,
    #     'modes': 10,
    #     'width': 128,
    #     'learning_rate': 0.005,
    #     'epochs': 500,
    #     'step_size': 100,
    #     'gamma': 0.1,
    #     'patience': 50,
    #     'freq_print': 1
    # }

    # 21.78%
    # config = {
    #     'n_train': 64,
    #     'batch_size': 5,
    #     'modes': 10,
    #     'width': 128,
    #     'learning_rate': 0.01,
    #     'epochs': 500,
    #     'step_size': 100,
    #     'gamma': 0.1,
    #     'patience': 50,
    #     'freq_print': 1
    # }


    # Prepare data
    training_set, testing_set, test_data = prepare_data("data/train_sol.npy", 
                                                      config['n_train'], 
                                                      config['batch_size'],
                                                      use_library=True)
    
    # Initialize model
    model = FNO(n_modes=(32, 1),
                hidden_channels=64,
                in_channels=2,     
                out_channels=1,    
                spatial_dim=2)
    
    # Train model
    trained_model = train_model(model, training_set, testing_set, config, 
                              "checkpoints/library_fno")
    
    print("Training completed. Model saved in checkpoints/library_fno/best_model.pth")

if __name__ == "__main__":
    main()