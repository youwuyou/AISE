import numpy as np
import matplotlib.pyplot as plt

import torch

def plot_derivatives(model, x_tensor, t_tensor, X, t, functions, snapshot=None, system=1, results_dir='results'):
    """
    Plot the neural network solution and its derivatives
    
    Args:
        model: Neural network model
        x_tensor, t_tensor: Input tensors
        X, t: Spatial and temporal grids
        functions: Dictionary containing u, u_t, and all derivatives as numpy arrays
        snapshot: Time snapshot to plot (if None, uses 1/5th of total time)
        system: System identifier (1 for Burgers, 2 for KdV)
        results_dir: Directory to save plots
    """
    # Choose snapshot if not provided
    if snapshot is None:
        snapshot = functions['u'].shape[1] // 5

    # Create subplots
    n_plots = len(functions)  # All functions including u and u_t
    n_rows = (n_plots + 1) // 2  # Ensure enough rows for all plots
    plt.figure(figsize=(18, 6 * n_rows))

    # Plot True vs Predicted u
    model.eval()
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy().reshape(functions['u'].shape)

    # Plot all functions
    colors = plt.cm.tab10(np.linspace(0, 1, n_plots))
    for i, (key, value) in enumerate(functions.items()):
        plt.subplot(n_rows, 2, i + 1)
        if key == 'u':
            plt.plot(X[:, snapshot], value[:, snapshot], label='True u')
            plt.plot(X[:, snapshot], u_pred[:, snapshot], '--', label='Predicted u')
        else:
            plt.plot(X[:, snapshot], value[:, snapshot], label=key, color=colors[i])
        plt.xlabel('x')
        plt.ylabel(key)
        plt.title(f'Term: {key}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{results_dir}/derivatives.png')
    plt.close()


def format_term(term, coeff):
    """
    Format a term with its coefficient in proper LaTeX notation with subscripts.
    
    Parameters:
    -----------
    term : str
        Term name (e.g., 'u_t', 'u_xx', 'u*u_x')
    coeff : float
        Coefficient value
    
    Returns:
    --------
    str
        LaTeX formatted term
    """
    # Handle coefficient
    coeff_str = '' if coeff == 1 else str(coeff)
    
    # Replace underscores with proper LaTeX subscripts
    # First, handle special case of terms with multiplication
    if '*' in term:
        parts = term.split('*')
        formatted_parts = []
        for part in parts:
            if '_' in part:
                base, sub = part.split('_')
                formatted_parts.append(f"{base}_{{{sub}}}")
            else:
                formatted_parts.append(part)
        term = ''.join(formatted_parts)
    else:
        if '_' in term:
            base, sub = term.split('_')
            term = f"{base}_{{{sub}}}"
    
    return f"{coeff_str}{term}"

def plot_pde_comparison(X, 
                        functions, 
                        lhs_terms, 
                        rhs_terms, 
                        snapshot, 
                        results_dir, 
                        title=None,
                        system="1"):
    """
    Plot PDE terms comparison and residual in separate subplots.
    Also calculates and reports the relative L2 error between LHS and RHS.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Spatial grid points
    functions : dict
        Dictionary containing the computed derivatives (u_t, u_x, etc.)
    lhs_terms : list of tuples
        List of (term_name, coefficient) for left-hand side
        e.g., [('u_t', 1), ('u*u_x', 6)]
    rhs_terms : list of tuples
        List of (term_name, coefficient) for right-hand side
        e.g., [('u_xxx', -1)]
    results_dir : str
        Directory to save the results
    title : str, optional
        Custom title for the plot
    
    Returns:
    --------
    float
        Relative L2 error between LHS and RHS as a percentage
    """    
    
    # Calculate individual terms
    all_terms = lhs_terms + rhs_terms
    term_values = {}
    
    for term, coeff in all_terms:
        if term not in functions:
            raise KeyError(f"Term {term} not found in functions dictionary")
        term_values[term] = coeff * functions[term]
    
    # Calculate LHS, RHS and residual
    lhs = np.zeros_like(X)
    for term, coeff in lhs_terms:
        lhs += coeff * functions[term]
        
    rhs = np.zeros_like(X)
    for term, coeff in rhs_terms:
        rhs += coeff * functions[term]
    
    residual = lhs - rhs
    
    # Calculate relative L2 error
    l2_norm_residual = np.linalg.norm(lhs - rhs, 2)
    l2_norm_lhs =  np.linalg.norm(lhs, 2)
    relative_l2_error = (l2_norm_residual / l2_norm_lhs) * 100  # as percentage
    print(f"Relative L2 error {relative_l2_error}% ")

    # Create figure with subplots
    n_terms = len(all_terms)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot individual terms
    ax1 = fig.add_subplot(gs[0, :])
    for term, coeff in all_terms:
        label = f'${format_term(term, coeff)}$'
        ax1.plot(X[:, snapshot], term_values[term][:, snapshot], label=label)
    ax1.set_title('Individual Terms')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax1.legend()
    
    # Plot LHS vs RHS
    ax2 = fig.add_subplot(gs[1, 0])
    lhs_label = ' + '.join([format_term(term, coeff) 
                           for term, coeff in lhs_terms])
    if rhs_terms:
        rhs_label = ' + '.join([format_term(term, coeff) 
                               for term, coeff in rhs_terms])
        ax2.plot(X[:, snapshot], rhs[:, snapshot], '--', 
                label=f'RHS: ${rhs_label}$', color='crimson')
    
    ax2.plot(X[:, snapshot], lhs[:, snapshot], 
             label=f'LHS: ${lhs_label}$', color='darkblue')
    ax2.set_title(f'LHS vs RHS (Relative L2 Error: {relative_l2_error:.3f}%)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.legend()
    
    # Plot residual
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(X[:, snapshot], residual[:, snapshot], color='green')
    ax3.set_title('Residual')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    
    # Set overall title
    if title is None:
        if rhs_terms:
            title = f'System {system}: ${lhs_label} = {rhs_label}$'
        else:
            title = f'\n${lhs_label} = 0$'
    fig.suptitle(title, fontsize=16, y=0.95, weight='bold')
    
    # Save and close
    plt.savefig(f'{results_dir}/check_sol.png', bbox_inches='tight')
    plt.close()
    
    return relative_l2_error

#============================================
#  2D Visualization
#============================================
import imageio
import shutil

def plot_2d_heatmap_anim(data, times, X, Y, variable_name, results_dir):
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