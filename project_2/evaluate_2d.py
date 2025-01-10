"""
Main module that uses PDE-Find for 2D coupled PDE system
- Finishes prediction of the governing equations of the PDE system
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from visualization import plot_2d_heatmap_anim

def main(create_gif=False):
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


    #=================================================
    # Computing derivatives
    #==================================================
    # # Select different candidate symbols for different systems
    # if system == 1:
    #     symbols = generate_candidate_symbols(
    #         max_x_order=3,     # Up to u_xxx
    #         max_t_order=0,
    #         binary_ops=['mul'],
    #         power_orders=[1],
    #         allowed_mul_orders=[(0,1), (0,2)],
    #         exclude_u_t=True
    #     )
    # else:
    #     symbols = generate_candidate_symbols(
    #         max_x_order=3,     # Up to u_xxx
    #         max_t_order=0,
    #         binary_ops=['mul'],
    #         power_orders=[1],
    #         allowed_mul_orders=[(0,1), (0,2)],
    #         exclude_u_t=True
    #     )

    # derivatives = compute_derivatives_autodiff(model, x_tensor, t_tensor, symbols, 
    #                            include_constant=True,
    #                            include_u=True)

    # # Manually filter out some entries
    # # derivatives.pop('u')
    # # derivatives.pop('u_t')
    # derivatives.pop('u_x')

    # print(f"{len(list(derivatives.keys()))} derivatives keys: {list(derivatives.keys())}")
    # candidates = list(derivatives.keys())


    #==================================================
    # Assemble LSE
    #==================================================
    Theta = build_theta(u_tensor, derivatives)
    u_t   = build_u_t(model, x_tensor, t_tensor)

    c_theta = generalized_condition_number(Theta.cpu().detach().numpy())
    print(f"condition number of Theta is {c_theta}")

    #==================================================
    # Sparse regression for LSE
    #==================================================
    # TODO: we probably need to solve two LSE here?
    # # Assembling Theta and u_t
    # Theta_np = Theta.cpu().detach().numpy()
    # u_t_np = u_t.cpu().detach().numpy()

    # ξ_best = TrainSTRidge(
    #     Theta_np, u_t_np,
    #     lam=1e-6,
    #     d_tol=5e-3,
    #     maxit=10,
    #     STR_iters=10,
    #     l0_penalty=None,
    #     split=0.7,
    #     print_best_tol=True
    # )
    # print(f"Found coefficients {ξ_best}")

    # # After running ridge regression:
    # print_discovered_equation(candidates, ξ_best)


    # Generate animations for both U and V
    if create_gif:
        print("\nProcessing U variable...")
        plot_2d_heatmap_anim(U, T, X, Y, "u", results_dir)
        
        print("\nProcessing V variable...")
        plot_2d_heatmap_anim(V, T, X, Y, "v", results_dir)

    print(f"\nAll results saved in {results_dir}")

if __name__ == "__main__":
    main()