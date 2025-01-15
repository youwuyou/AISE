import torch
import numpy as np

from training import AllenCahnDataset
from utils import (
print_bold
)

import json
from pathlib import Path


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
    # TODO: read model config and load automatically


    #==================================================
    # Fine-tune (optional)
    #==================================================


    #==================================================
    # Load testing data
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

    # Standard test set with default samplers
    print_bold(f"In-distribution test set with ɛ = {epsilon_values} and default samplers")
    test_data_dict = np.load(f"{data_folder}/test_sol.npy", allow_pickle=True).item()
    for ic_type in test_data_dict.keys():
        print(f"Evaluating with IC data of function class {ic_type}")
        test_dataset = AllenCahnDataset(test_data_dict[ic_type], epsilon_values, time_points)

    # OOD test set with special samplers but standard eps
    print_bold(f"OOD test set with same ɛ = {epsilon_values}, but special parameters for samplers: {config['ood_params']}")
    test_ood_data_dict = np.load(f"{data_folder}/test_sol_OOD.npy", allow_pickle=True).item()
    for ic_type in test_ood_data_dict.keys():
        print(f"Evaluating with IC data of function class {ic_type}")
        test_ood_dataset = AllenCahnDataset(test_ood_data_dict[ic_type], epsilon_values, time_points)

    # Epsilon test set with extra-, interpolated eps
    print_bold(f"Epsilon test set with special ɛ = {added_epsilon_values} and default samplers")
    test_eps_data_dict = np.load(f"{data_folder}/test_sol_eps.npy", allow_pickle=True).item()
    for ic_type in test_eps_data_dict.keys():
        print(f"Evaluating with IC data of function class {ic_type}")
        test_eps_dataset = AllenCahnDataset(test_eps_data_dict[ic_type], added_epsilon_values, time_points)
            
    #==================================================
    # Evaluation
    #==================================================
    print_bold("Evaluation of the Foundation model")


if __name__ == '__main__':
    main()