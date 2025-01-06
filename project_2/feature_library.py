"""
This file contains routines for the assembly of the sparse LSE Uₜ = ϴ ξ for PDE-Find method
- Part 1 contains the helper routines that builds the Theta matrix and the U_t LHS
- Part 2 contains routines that help with specifying candidates (partial derivatives) using sympy
"""

import torch

import sympy as sp
from itertools import product

#========================================================
# Part 1: Building Sparse LSE Uₜ = ϴ ξ
# - ϴ   matrix assembled using derivatives of u
# - Uₜ  vector containing provided spatiotemporal solution
# - ξ   unknown for storing coefficients
#========================================================

def build_theta(u, derivatives):
    n = u.shape[0]  # Number of spatial points
    m = u.shape[1]  # Number of time steps
    print(f"Domain shape is ({u.shape}), consists of n = {n} spatial points and m = {m} time snapshots")
    print(f"n ྾ m = {n * m}")

    D = len(derivatives)  # Number of candidates
    print(f"{D} candidates will be used for assembling matrix Theta")
    
    Theta = torch.zeros((n * m, D))  # D columns
    print(f"Shape of Theta: {Theta.shape}")
    
    # Fill in the columns with candidates
    for idx, derivative_values in enumerate(derivatives.values()):
        Theta[:, idx] = derivative_values.squeeze()

    return Theta

def build_u_t(model, x, t):
    """
    Compute only the temporal derivative u_t
    
    Args:
        model: Neural network model
        x, t: Input tensors
    
    Returns:
        u_t: First order temporal derivative
    """
    t.requires_grad = True
    u = model(x, t)
    
    # Compute temporal derivative
    grad = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grad_outputs=grad, create_graph=True)[0]
    
    return u_t

#==================================================
# Part 2: Symbolic Selection of Candidates
#==================================================

def print_discovered_equation(candidates, ξ, threshold=1e-4):
    """
    Print the discovered equation in a readable format
    
    Args:
        candidates: List of symbolic expressions used
        ξ: Solution vector from ridge regression
        threshold: Minimum coefficient magnitude to include in equation
    """
    print("\nDiscovered equation:")
    print("u_t = ", end="")
    
    # Get non-zero terms
    significant_terms = []
    for i, (coeff, expr) in enumerate(zip(ξ, candidates)):
        if abs(coeff) > threshold:
            term = f"{coeff:.6f}*{expr}" if expr != 'constant' else f"{coeff:.6f}"
            significant_terms.append(term)
    
    # Print equation
    if significant_terms:
        print(" + ".join(significant_terms))
    else:
        print("0")
    
    print("\n")


def simplify_expression(expr):
    # Basic simplification
    simplified = sp.simplify(expr)
    # Further simplifications for associativity and powers
    simplified = simplified.expand()
    simplified = simplified.powsimp()
    simplified = simplified.collect(simplified.free_symbols)
    return simplified

def get_symbol_order(sym):
    """
    Get the derivative order of a symbol by counting x and t derivatives.
    Returns 0 for base symbol 'u'.
    """
    str_sym = str(sym)
    return str_sym.count('x') + str_sym.count('t')

def generate_candidate_symbols(max_x_order=2, 
                        max_t_order=2, 
                        binary_ops=None, 
                        power_orders=None, 
                        allowed_mul_orders=None, 
                        exclude_u_t=True):
    """
    Generate expressions with specified binary operations and powers, with control over multiplication orders
    
    Args:
        max_x_order: Maximum order of spatial derivatives
        max_t_order: Maximum order of temporal derivatives
        binary_ops: List of binary operations ['add', 'mul'] or None for no binary ops
        power_orders: List of power orders [2, 3, ...] or None for no powers
        allowed_mul_orders: List of tuples (order1, order2) specifying which orders can be multiplied
                          e.g. [(0,2)] allows u * u_xx but not u_xx * u_xx
        exclude_u_t: Whether to exclude u_t terms from candidates
    """
    # Define symbols separately for spatial and temporal derivatives
    x_syms = {sp.Symbol(f'u_{"x"*n}') for n in range(1, max_x_order+1)}
    
    # Generate temporal derivatives, excluding u_t if specified
    t_syms = set()

    for n in range(1, max_t_order+1):
        sym = sp.Symbol(f'u_{"t"*n}')
        if not (exclude_u_t and n == 1):  # Skip u_t if exclude_u_t is True
            t_syms.add(sym)
    
    # Combine all symbols
    syms = x_syms.union(t_syms)
    syms.add(sp.Symbol('u'))
    
    # Start with simplified base symbols
    expressions = {simplify_expression(sym) for sym in syms}
    new_exprs = set()
    
    # Generate powers if specified
    if power_orders:
        for e in expressions:
            for power in power_orders:
                if power != 1:  # Skip power 1 as it's the same as the base symbol
                    new_exprs.add(simplify_expression(e**power))
        expressions.update(new_exprs)
    
    # Generate binary operations if specified
    if binary_ops and 'mul' in binary_ops and allowed_mul_orders:
        # Only generate products with allowed order combinations
        new_exprs = set()
        for e1, e2 in product(expressions, expressions):
            if e1 != e2:  # Avoid self-multiplication
                # Skip if either term contains u_t
                if exclude_u_t and ('u_t' in str(e1) or 'u_t' in str(e2)):
                    continue
                    
                order1 = get_symbol_order(e1)
                order2 = get_symbol_order(e2)
                
                # Check if this order combination is allowed
                if (order1, order2) in allowed_mul_orders or (order2, order1) in allowed_mul_orders:
                    prod = simplify_expression(e1 * e2)
                    new_exprs.add(prod)
        expressions.update(new_exprs)
    
    # Final simplification of all expressions
    expressions = {simplify_expression(expr) for expr in expressions}
    
    # Filter out any remaining expressions containing u_t if exclude_u_t is True
    if exclude_u_t:
        expressions = {expr for expr in expressions if 'u_t' not in str(expr)}
    
    return sorted(expressions, key=lambda x: len(str(x)))

if __name__ == "__main__":

    # Generated candidates are just strings, not the numeric values of derivative functions
    candidates = generate_candidate_symbols(
        max_x_order=3,     # Up to u_xxx
        max_t_order=2,     # Up to u_tt
        binary_ops=['mul'],
        power_orders=[1],
        allowed_mul_orders=[(0,1), (0,2)] # e.g. allow only  u * u_x and u * u_xx
    )    

    print(f"Generated {len(candidates)} unique expressions after simplification")
    print(f"{candidates}")