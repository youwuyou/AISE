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
        print(f"current iters {iters}, {ξ}")
        ξ[bigcoeffs] = STRidge(Theta[:, bigcoeffs], u_t, λ, tol, iters-1)

    return ξ

def TrainSTRidge(R, Ut, lam, d_tol, maxit=25, STR_iters=10, l0_penalty=None,
                 split=0.8, print_best_tol=False):
    """
    Disclaimer: This function takes inspiration from the original PDE-Find implementation!
    
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.
    """

    np.random.seed(0)
    n, _ = R.shape
    train = np.random.choice(n, int(n * split), replace=False)
    test = [i for i in range(n) if i not in train]

    TrainR, TestR = R[train, :], R[test, :]
    TrainY, TestY = Ut[train, :], Ut[test, :]

    if l0_penalty is None:
        l0_penalty = 0.001 * np.linalg.cond(TrainR)

    w_best = np.linalg.lstsq(TrainR, TrainY, rcond=None)[0]
    err_best = np.linalg.norm(TestY - TestR @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0
    tol = float(d_tol)

    for _ in range(maxit):
        w = STRidge(torch.tensor(TrainR, dtype=torch.float32),
                    torch.tensor(TrainY, dtype=torch.float32),
                    lam, tol, STR_iters).reshape(-1, 1)
        err = np.linalg.norm(TestY - TestR @ w, 2) + l0_penalty * np.count_nonzero(w)

        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol += d_tol
        else:
            tol = max(0, tol - 2 * d_tol)
            d_tol = 2 * d_tol / (maxit - _)
            tol += d_tol

    if print_best_tol:
        print("Optimal tolerance:", tol_best)

    return w_best