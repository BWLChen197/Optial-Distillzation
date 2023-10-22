#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:46:06 2023

@author: bruce
"""

import numpy as np

def isQuantumState(rho, prec=0.00001):
    # Check if Hermitian
    if not isHermitian(rho, prec):
        print("isQuantumState: Matrix is not Hermitian.")
        return False
    
    # Check for positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(rho)
    if np.min(eigenvalues) < -prec:
        print("isQuantumState: Matrix is not positive semidefinite.")
        return False

    # Check the trace
    tr = np.trace(rho)
    if tr < 1 - prec or tr > 1 + prec:
        print("isQuantumState: Matrix does not have trace one.")
        return False

    return True

def isHermitian(rho, prec=0.00001):
    if not np.allclose(rho, np.conj(rho).T, atol=prec):
        return False
    return True

def isUnitary(U):
    # Check dimensions
    d, da = U.shape
    if d != da:
        print("isUnitary: Input is not a square matrix.")
        return False
    
    # Check unitarity
    if not np.allclose(np.dot(U, np.conj(U).T), np.eye(d), atol=1e-6):
        return False
    if not np.allclose(np.dot(np.conj(U).T, U), np.eye(d), atol=1e-6):
        return False
    
    return True

def isPPT(rho, nA, nB, epsilon=1e-15):
    assert isQuantumState(rho), "Input is not a quantum state."
    da, db = rho.shape
    assert da == nA * nB, "Input does not match given dimensions."
    
    # Compute the partial transpose
    rho = rho.reshape(nA, nB, nA, nB)
    rhoPT = rho.transpose(1, 0, 3, 2).reshape(da, db)
    
    # Check for positivity
    eigenvalues = np.linalg.eigvalsh(rhoPT)
    if np.min(eigenvalues) < -epsilon:
        return False
    
    return True

