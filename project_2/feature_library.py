"""
This file contains routines for the assembly of the sparse LSE Uₜ = ϴ ξ for PDE-Find method
- Part 1 contains the helper routines that builds the Theta matrix and the U_t LHS
- Part 2 contains routines that help with specifying candidates (partial derivatives) using sympy
- Part 3 contains concrete implementation for derivative computation
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

def print_discovered_equation(candidates, ξ, threshold=1e-4, f_symbol="u"):
    """
    Print the discovered equation in a readable format
    
    Args:
        candidates: List of symbolic expressions used
        ξ: Solution vector from ridge regression
        threshold: Minimum coefficient magnitude to include in equation
        f_symbol: Symbol representing the dependent variable in the equation
    """
    print("\nDiscovered equation:")
    print(f"{f_symbol} = ", end="")
    
    terms = []
    for coeff, expr in zip(ξ, candidates):
        c = float(coeff)  # convert to float
        if abs(c) > threshold:
            if expr == 'constant':
                terms.append(f"{c:.6f}")
            else:
                terms.append(f"{c:.6f}*{expr}")
                
    print(" + ".join(terms) if terms else "0")
    print()

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
        exclude_u_t: Whether to exclude u_t terms from candidates
    """
    # Define symbols separately for spatial and temporal derivatives
    x_syms = {sp.Symbol(f'u_{"x"*n}') for n in range(1, max_x_order+1)}
    
    # Generate temporal derivatives
    t_syms = set()
    for n in range(1, max_t_order+1):
        sym = sp.Symbol(f'u_{"t"*n}')
        if not (exclude_u_t and n == 1):
            t_syms.add(sym)
    
    # Combine all symbols
    syms = x_syms.union(t_syms)
    syms.add(sp.Symbol('u'))
    
    # Start with base symbols
    expressions = set(syms)
    
    # Generate powers first
    powered_expressions = set()
    if power_orders:
        for base_expr in expressions:
            for power in power_orders:
                if power != 1:  # Skip power 1
                    powered_expr = base_expr**power
                    powered_expressions.add(powered_expr)
    
    # Combine base and powered expressions
    expressions.update(powered_expressions)
    
    # Generate binary operations if specified
    if binary_ops and 'mul' in binary_ops and allowed_mul_orders:
        products = set()
        for e1, e2 in product(expressions, expressions):
            if e1 != e2:  # Avoid self-multiplication
                if exclude_u_t and ('u_t' in str(e1) or 'u_t' in str(e2)):
                    continue
                    
                order1 = get_symbol_order(e1)
                order2 = get_symbol_order(e2)
                
                if (order1, order2) in allowed_mul_orders or (order2, order1) in allowed_mul_orders:
                    prod = e1 * e2
                    products.add(prod)
        
        expressions.update(products)
    
    # Filter out u_t terms if needed
    if exclude_u_t:
        expressions = {expr for expr in expressions if 'u_t' not in str(expr)}
    
    return sorted(expressions, key=lambda x: str(x))

#============================================================
# Part 3: Concrete Implementation for derivative computation
#============================================================

import torch
import sympy as sp
from sympy import Pow, Symbol, Mul, Integer

def compute_derivatives_autodiff(model, x, t, symbols, include_constant=True, include_u=True):
    """
    Compute derivatives based on NN model that approximates 1D spatiotemporal u(x,t)
    - symbols: symbolic expressions are derivatives to be computed need to be specified
    """
    results = {}
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    # First compute all basic derivatives we'll need
    basic_derivatives = {}
    max_x_order = max(str(expr).count('x') for expr in symbols)
    max_t_order = max(str(expr).count('t') for expr in symbols)
    
    # Add constant term if requested
    if include_constant:
        results['1'] = torch.ones((u.shape[0] * u.shape[1], 1))
    
    # Add base function u if requested
    if include_u:
        results['u'] = u
    
    # Compute x derivatives
    current = u
    basic_derivatives['u'] = u
    for order in range(1, max_x_order + 1):
        grad = torch.ones_like(current)
        current = torch.autograd.grad(current, x, grad_outputs=grad, create_graph=True)[0]
        basic_derivatives[f'u_{"x" * order}'] = current
    
    # Compute t derivatives
    current = u
    for order in range(1, max_t_order + 1):
        grad = torch.ones_like(current)
        current = torch.autograd.grad(current, t, grad_outputs=grad, create_graph=True)[0]
        basic_derivatives[f'u_{"t" * order}'] = current
    
    # Helper function to convert expression to computation
    def process_expression(expr):
        if isinstance(expr, Symbol):
            # Basic symbol (u, u_x, etc.)
            return basic_derivatives[str(expr)]
        elif isinstance(expr, Pow):
            # Handle power expressions (u^2, u_x^2, etc.)
            base = process_expression(expr.args[0])
            power = expr.args[1]
            # Convert SymPy Integer to Python float for torch.pow
            if isinstance(power, Integer):
                power = float(power)
            return torch.pow(base, power)
        elif isinstance(expr, Mul):
            # Handle multiplication
            result = process_expression(expr.args[0])
            for arg in expr.args[1:]:
                result = result * process_expression(arg)
            return result
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    # Now construct each candidate expression
    for expr in symbols:
        try:
            results[str(expr)] = process_expression(expr)
        except Exception as e:
            print(f"Error processing expression {expr}: {e}")
            raise
                
    return results

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