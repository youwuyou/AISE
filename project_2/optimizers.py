"""
This file contains the solver for solving the sparse LSE for 
    - Uₜ = ϴ ξ is solved, where Uₜ and ϴ matrix are provided
    - STRidge applies ridge regression with hard thresholding recursively
"""

import sklearn
import numpy as np
import torch
import torch.linalg as linalg

#==================================================
# Analysis Tools
#==================================================
def generalized_condition_number(A):
    """Computation of the generalized condition number, eligible for complex-valued, overdetermined system"""
    # Compute A^H A
    A_H_A = np.conj(A.T) @ A
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A_H_A)
    # Calculate condition number
    cond = np.sqrt(np.max(eigenvalues) / np.min(eigenvalues))
    return cond

#==================================================
#  Solvers for Sparse LSE (Overdetermined)
#==================================================
def STRidge(Theta, u_t, λ, tol, iters):
    """Implements Algorithm 1 as in supplimentary material of the PDE-Find paper"""
    # Convert to numpy arrays
    Theta_np = Theta.cpu().detach().numpy()
    u_t_np   = u_t.cpu().detach().numpy()

    # Ridge regression
    classifier = sklearn.linear_model.Ridge(fit_intercept=False, max_iter=500)    
    classifier.fit(Theta_np, u_t_np)

    # Select large coefficients
    ξ = classifier.coef_
    bigcoeffs = np.abs(ξ) > tol
    ξ[~bigcoeffs] = 0

    # Recursive call with fewer coefficients
    if iters > 0 and bigcoeffs.any():
        # print(f"current bigcoeffs {bigcoeffs}")
        print(f"current iters {iters}, {ξ}")
        ξ[bigcoeffs] = STRidge(Theta[:, bigcoeffs], u_t, λ, tol, iters-1)

    return ξ