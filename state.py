#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:39:47 2023

@author: bruce
"""

import numpy as np

# def isotropic(p):
#     rho = np.array([[0.5*p,0,0,0.5*p],
#                    [0,0,0,0],
#                    [0,0,0,0],
#                    [0.5*p,0,0,0.5*p]])+np.array([[0.25*(1-p),0,0,0],
#                                                [0,0.25*(1-p),0,0],
#                                                [0,0,0.25*(1-p),0],
#                                                [0,0,0,0.25*(1-p)]])
#     return rho


def eVec(dim, pos):
    vec = np.zeros(dim)
    vec[pos-1] = 1
    return vec


def bell_states(n):
    """Generate one of the Bell states based on the input n."""
    if n == 1:
        rho = (np.kron(eVec(2, 1), eVec(2, 1)) + np.kron(eVec(2, 2), eVec(2, 2))) / np.sqrt(2)
    elif n == 2:
        rho = (np.kron(eVec(2, 1), eVec(2, 1)) - np.kron(eVec(2, 2), eVec(2, 2))) / np.sqrt(2)
    elif n == 3:
        rho = (np.kron(eVec(2, 1), eVec(2, 2)) + np.kron(eVec(2, 2), eVec(2, 1))) / np.sqrt(2)
    elif n == 4:
        rho = (np.kron(eVec(2, 1), eVec(2, 2)) - np.kron(eVec(2, 2), eVec(2, 1))) / np.sqrt(2)
    else:
        raise ValueError("Invalid value for n. Choose n from [1, 2, 3, 4].")
    return rho

def isotropic(p):
    if p < 0 or p > 1:
        raise ValueError("p should be in the range [0, 1].")
    
    rho = p * np.outer(bell_states(1), bell_states(1)) + (1-p)/4 * np.eye(4)
    return rho




def rStatePhase(p, phi=0.0):
    assert 0 <= p <= 1, "Probabilities must be between 0 and 1."
    
    # Produce the state |01> + e^(i phi) |10>
    e0 = eVec(2, 1)
    e1 = eVec(2, 2)
    vec = (np.kron(e0, e1) + np.exp(1j * phi) * np.kron(e1, e0)) / np.sqrt(2)
    
    if phi == 0 or phi == np.pi:
        vec = np.real(vec)
        
    # Produce a state orthogonal to the one above
    v11 = np.kron(e1, e1)
    
    # Construct the desired mixture
    out = p * np.outer(vec, np.conj(vec)) + (1 - p) * np.outer(v11, np.conj(v11))
    
    return out


