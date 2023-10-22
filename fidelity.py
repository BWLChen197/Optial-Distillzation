#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:23:42 2023

@author: bruce
"""

from state import bell_states, isotropic
import numpy as np


def fidelity(rho, n):
    """Compute the fidelity of the given density matrix with the nth Bell state."""
    psi = bell_states(1)
    F = np.real(np.dot(psi.conj().T, np.dot(rho, psi)))
    return F


