#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:58:19 2023

@author: bruce
"""

import cvxpy as cp
import numpy as np
from state import isotropic,rStatePhase
from checkInput import isQuantumState
from PT import partial_transpose
from Multicopies import copies, sort

#Inputs:
# - *rho* quantum state to be distilled on A and B
# - *nA* dimension of the A system
# - *nB* dimension of the B system
# - *k* desired output dimension of the maximally entangled state
# - *delta* desired success probability

def pptRelax(rho, n, k, delta, verbose=False, eps=1e-4, max_iters=20000):
    nA = 2 ** n
    nB = nA
    d = nA * nB

    # Check if rho is a quantum state
    assert isQuantumState(rho), "The input state must be a valid quantum state."

    # Check dimensions
    assert rho.shape == (d, d), f"Input state doesn't match given dimensions: {d} != {nA * nB}"

    # Check delta
    assert 0 <= delta <= 1, "Success probability must be between 0 and 1."

    # Define identity matrix
    id_matrix = np.eye(d)

    # Define variables
    D = cp.Variable((d, d), hermitian=True)
    E = cp.Variable((d, d), hermitian=True)
    tv = cp.Variable()

    # Define objective and constraints
    objective = cp.Maximize(tv)
    constraints = [
        tv == cp.trace(nA * nB * D @ rho.T),
        D >> 0,
        E >> 0,
        id_matrix / d - (D + E) >> 0,
        cp.trace(nA * nB * rho.T.conj() @ (D + E)) == delta
    ]

    # Add PPT constraints
    DPT, constraint1 = partial_transpose(D, nA, nB)
    EPT, constraint2 = partial_transpose(E, nA, nB)
    constraints += constraint1
    constraints += constraint2
    constraints += [
        DPT + EPT / (k + 1) >> 0,
        -DPT + EPT / (k - 1) >> 0,
        id_matrix / d - (DPT + EPT) >> 0
    ]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=verbose, eps=eps, max_iters=max_iters)

    p_succ = nA * nB * np.trace(rho.T @ (D.value + E.value))
    F = problem.value / p_succ

    return problem, F, p_succ

#Example usage
# rho = isotropic(0.4)
# n = 3
# k = 2
# delta = 1


# X = copies(rho, n)
# rho_new = sort(X,n)
# problem, F, p_succ = pptRelax(rho_new, n, k, delta)
# print(f"Optimal fidelity: {F}, Success probability: {p_succ}")











