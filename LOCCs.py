#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:43:29 2023

@author: bruce
"""
import numpy as np
from state import isotropic
from fidelity import fidelity

def BBPSSW(rho):
    """
    Calculate the outcome fidelity and the probability of success of the BBPSSW protocol 
    for an isotropic state with initial fidelity F.

    Parameters:
    - F (float): Fidelity of the isotropic state with the maximally entangled state.

    Returns:
    - F_BBPSSW (float): Outcome fidelity after one round of BBPSSW.
    - P_success (float): Probability of success of one round of BBPSSW.
    """
    
    F = fidelity(rho, 1)
    
    # Calculate F_BBPSSW
    numerator = F**2 + ((1-F)/3)**2
    denominator = F**2 + 2*F*(1-F)/3 + 5*((1-F)/3)**2
    F_BBPSSW = numerator / denominator

    # Calculate P_success
    P_success = F**2 + 2*F*(1-F)/3 + 5*((1-F)/3)**2

    return F_BBPSSW, P_success




p = 0.8
F_BBPSSW, P_success = BBPSSW(isotropic(p))
print("Outcome fidelity:", F_BBPSSW)
print("Probability of success:", P_success)
