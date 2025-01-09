"""
Main module that uses PDE-Find for 2D coupled PDE system
- Finishes prediction of the governing equations of the PDE system
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import shutil

def generate_animation(data, times, X, Y, variable_name, results_dir):
    """Helper function to generate animation for a single variable"""
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    snapshots = data.shape[2]

    # Create frames directory for this variable
    frames_dir = os.path.join(results_dir, f"frames_{variable_name}")
    os.makedirs(frames_dir, exist_ok=True)

    # Generate frames
    plt.figure(figsize=(10, 8))
    for i in range(snapshots):
        plt.clf()  # Clear the current figure
        plt.imshow(data[:, :, i], extent=[x_min, x_max, y_min, y_max],
                  origin='lower', cmap='coolwarm')
        plt.colorbar(label=f'{variable_name}(x,y,t)')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Handle time value display based on T array shape
        if times.ndim == 1:
            time_val = times[i]
        else:
            time_val = times[0, i] if times.shape[0] == 1 else times[i, 0]
            
        plt.title(f'{variable_name} at time step {i}')
        
        # Save frame
        plt.savefig(f'{frames_dir}/frame_{i:04d}.png')
        
        if i % 20 == 0:  # Progress indicator
            print(f"Generated frame {i}/{snapshots} for {variable_name}")

    plt.close()

    # Create animated GIF
    gif_path = os.path.join(results_dir, f'original_data_{variable_name}.gif')
    print(f"Creating GIF for {variable_name}...")
    
    with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
        for i in range(snapshots):
            frame = imageio.imread(f'{frames_dir}/frame_{i:04d}.png')
            writer.append_data(frame)

    # Clean up temporary frames
    shutil.rmtree(frames_dir)
    print(f"GIF saved as '{gif_path}'")

def main():
    system = 3
    path = f'data/{system}.npz'
    name = "Reaction-Diffusion Equation"

    # Create results directory for the current system
    results_dir = f"results/system_{system}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Testing on dataset loaded from {path}")
    vectorial_data = np.load(path)

    # Load the arrays from the file
    U = vectorial_data['u']
    V = vectorial_data['v']
    X = vectorial_data['x']
    Y = vectorial_data['y']
    T = vectorial_data['t']

    # Shape information
    print(f"U shape: {U.shape}")
    print(f"V shape: {V.shape}")
    nx, ny, snapshots = U.shape
    print(f"({nx}, {ny}) 2D spatial points, {snapshots} time snapshots in total")
    print(f"Time array shape: {T.shape}")

    # Generate animations for both U and V
    print("\nProcessing U variable...")
    generate_animation(U, T, X, Y, "u", results_dir)
    
    print("\nProcessing V variable...")
    generate_animation(V, T, X, Y, "v", results_dir)

    print(f"\nAll results saved in {results_dir}")

if __name__ == "__main__":
    main()