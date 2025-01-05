import torch
import torch.linalg as linalg

# Ridge regression with hard thresholding
def ridge_regression_with_thresholding(Theta, U_t, lambda_reg=1e-5, threshold=1e-3):
    """
    Solve the Ridge Regression problem with hard thresholding.
    
    Parameters:
    - Theta: The matrix containing the derivative information (n * m, D)
    - U_t: Temporal derivative (n * m, 1)
    - lambda_reg: Regularization parameter for ridge regression
    - threshold: Threshold for hard thresholding
    
    Returns:
    - X: The solution vector (D,)
    """
    
    # Ensure both tensors are on the same device (either CPU or GPU)
    device = Theta.device
    U_t = U_t.to(device)
    
    # Ridge Regression (Θ.T * Θ + λ * I) * X = Θ.T * U_t
    I = torch.eye(Theta.shape[1], device=device)  # Identity matrix of size D
    ridge_term = lambda_reg * I
    X = linalg.solve(Theta.T @ Theta + ridge_term, Theta.T @ U_t)
    
    # Hard thresholding: set small coefficients to zero
    X[torch.abs(X) < threshold] = 0
    
    return X