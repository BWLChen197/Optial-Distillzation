#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:23:55 2023

@author: bruce
"""

import cvxpy as cp
import numpy as np

def partial_transpose(var, dimA, dimB):
    """
    Define the partial transpose of a CVXPY Variable.
    
    Parameters:
    - var: CVXPY Variable representing the matrix to be partially transposed
    - dimA: Dimension of subsystem A
    - dimB: Dimension of subsystem B
    
    Returns:
    - pt_var: CVXPY Variable representing the partial transpose
    """
    d = dimA * dimB
    pt_var = cp.Variable((d, d), hermitian=True)
    
    # Define the constraints for the partial transpose
    constraints = []
    for i in range(dimA):
        for j in range(dimA):
            for k in range(dimB):
                for l in range(dimB):
                    # Original indices
                    row = i * dimB + k
                    col = j * dimB + l
                    # Transposed indices
                    new_row = i * dimB + l
                    new_col = j * dimB + k
                    constraints.append(pt_var[new_row, new_col] == var[row, col])
                    
    return pt_var, constraints

# Now you can use pt_rho and constraints in your CVXPY problem

