#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:24:56 2023

@author: bruce
"""

import numpy as np
from state import isotropic,rStatePhase
from checkInput import isQuantumState

def copies(rho, n):
    if isinstance(n, float):
        n = int(n)
    result = rho
    for _ in range(n - 1):
        result = np.kron(result, rho)
    return result



def permutesystems(X, perm):
    """
    X: Input matrix, assumed to be square and its dimensions a power of 2
    perm: List of integers specifying the new ordering of subsystems
    
    Returns the matrix X with permuted subsystems as specified by 'perm'
    """
    # Calculate the dimension of each subsystem assuming they are all equal
    num_subsystems = len(perm)
    dim_subsystem = int(X.shape[0] ** (1 / num_subsystems))
    
    # Check for invalid input
    if X.shape[0] != X.shape[1]:
        raise ValueError("Input matrix must be square")
    
    if dim_subsystem ** num_subsystems != X.shape[0]:
        raise ValueError("Invalid permutation or input matrix dimensions")
    
    # Reshape into tensor and then transpose
    new_shape = [dim_subsystem]*num_subsystems*2
    new_order = [num_subsystems + perm[i] - 1 for i in range(num_subsystems)] + [perm[i] - 1 for i in range(num_subsystems)]
    X_tensor = np.reshape(X, new_shape)
    X_tensor = np.transpose(X_tensor, new_order)
    
    # Reshape back into matrix
    X_permuted = np.reshape(X_tensor, [X.shape[0], X.shape[1]])
    
    return X_permuted


def sort(rhoBig, n):
    #n: Represents the number of copies of the original density matrix rho 
    #d: Represents the dimension of subsystem A (or B, as they are assumed to be the same). 
    perm = np.concatenate([np.arange(1, 2*n, 2), np.arange(2, 2*n + 1, 2)])
    rhoOut = permutesystems(rhoBig, perm)
    return rhoOut




# X = copies(rStatePhase(0.9), 3)
# Y = permutesystems(X, [1,3,5,2,4,6])
# Y_1 = sort(X,3)
# print(Y-Y_1)
# print(X-Y)
# isQuantumState(Y)