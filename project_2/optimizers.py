"""
This file contains the solver for solving the sparse LSE for 
    - Uₜ = ϴ ξ is solved, where Uₜ and ϴ matrix are provided
    - STRidge applies ridge regression with hard thresholding recursively

Code reference and inspiration:

- TrainSTRidge: https://github.com/snagcliffs/PDE-FIND
"""

import sklearn
import numpy as np
import torch
import torch.linalg as linalg
from sklearn.preprocessing import StandardScaler

# surpress warning for better reporting
import warnings
from scipy.linalg import LinAlgWarning

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
    """Sparse ridge regression with hard thresholding"""
    Theta_np = Theta.cpu().numpy()
    u_t_np   = u_t.cpu().numpy()

    # Suppress the specific LinAlgWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=LinAlgWarning)

        # No intercept, smaller tol, higher max_iter
        classifier = sklearn.linear_model.Ridge(alpha=λ, fit_intercept=False,
                                        tol=1e-5, max_iter=500)
        classifier.fit(Theta_np, u_t_np)

    # Select large coefficients
    ξ = classifier.coef_
    bigcoeffs = np.abs(ξ) > tol
    ξ[~bigcoeffs] = 0

    # Recursive call with fewer coefficients
    if iters > 0 and bigcoeffs.any():
        # print(f"current iters {iters}, {ξ}")
        ξ[bigcoeffs] = STRidge(Theta[:, bigcoeffs], u_t, λ, tol, iters-1)

    return ξ

def TrainSTRidge(ϴ, Ut, λ, d_tol, maxiter=25, STR_iters=10, η=1e-3,
                 split=0.8, print_best_tol=False):
    """
    Disclaimer: This function takes inspiration from the original PDE-Find implementation!
    It implements the Algorithm 2 in the original PDE-Find paper.
    
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.
    """

    # First split the data into training and testing sets
    np.random.seed(0)
    n, _ = ϴ.shape
    train = np.random.choice(n, int(n * split), replace=False)
    test = [i for i in range(n) if i not in train]

    ϴ_train, ϴ_test   = ϴ[train, :], ϴ[test, :]
    Ut_train, Ut_test = Ut[train, :], Ut[test, :]

    # Set an appropriate l⁰-penalty
    η = η * np.linalg.cond(ϴ_train)

    # Get a baseline predictor
    ξ_best   = np.linalg.lstsq(ϴ_train, Ut_train, rcond=None)[0]
    err_best = np.linalg.norm(Ut_test - ϴ_test @ ξ_best, 2) + η * np.count_nonzero(ξ_best)

    # Now search through values of tolerance to find the best predictor
    tol_best = 0
    tol = float(d_tol)

    for curr_iter in range(maxiter):
        # Train and evaluate performance
        ξ = STRidge(torch.tensor(ϴ_train, dtype=torch.float32),
                    torch.tensor(Ut_train, dtype=torch.float32),
                    λ, tol, STR_iters).reshape(-1, 1)
        err = np.linalg.norm(ϴ_test @ ξ - Ut_test, 2) + η * np.count_nonzero(ξ)

        # Is the error still dropping?
        if err <= err_best:
            err_best = err
            ξ_best = ξ
            tol_best = tol
            tol += d_tol
        # Or is tolerance too high?
        else:
            tol = max(0, tol - 2 * d_tol)
            d_tol = 2 * d_tol / (maxiter - curr_iter)
            tol += d_tol

    if print_best_tol:
        print("Optimal tolerance:", tol_best)

    return ξ_best