#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:11:46 2023

@author: bruce
"""

import numpy as np
from state import isotropic,rStatePhase
from Multicopies import copies, sort
from pptrelaxation import pptRelax
import matplotlib.pyplot as plt


n = 3
k = 2
delta = 1
delta1 = 0.75
delta2 = 0.5


step = 0.01
x_axis = np.arange(0.2,1,step )
y_axis = np.zeros(np.size(x_axis))
y_axis1 = np.zeros(np.size(x_axis))
y_axis2 = np.zeros(np.size(x_axis))
j = 0
j1 = 0
j2 = 0

#for p = 1
for a in x_axis:
    rho = isotropic(a)
    X = copies(rho, n)
    rho_new = sort(X,n)
    problem, F, p_succ = pptRelax(rho_new, n, k, delta)
    y_axis[j] = F
    print(f"Optimal fidelity: {F}, Success probability: {p_succ}")
    j += 1

#for p = 0.75
for b in x_axis:
    rho = isotropic(b)
    X = copies(rho, n)
    rho_new = sort(X,n)
    problem, F, p_succ = pptRelax(rho_new, n, k, delta1)
    y_axis1[j1] = F
    print(f"Optimal fidelity: {F}, Success probability: {p_succ}")
    j1 += 1

#for p = 0.5    
for c in x_axis:
    rho = isotropic(c)
    X = copies(rho, n)
    rho_new = sort(X,n)
    problem, F, p_succ = pptRelax(rho_new, n, k, delta2)
    y_axis2[j2] = F
    print(f"Optimal fidelity: {F}, Success probability: {p_succ}")
    j2 += 1


x_axis = 1-x_axis
y_axis = 1-y_axis
y_axis1 = 1-y_axis1
y_axis2 = 1-y_axis2
    
plt.plot(x_axis, y_axis,label = 'p=1')
plt.plot(x_axis, y_axis1,label = 'p=0.75')
plt.plot(x_axis, y_axis2,label = 'p=0.5')

plt.xlabel('Noise'+r'$\gamma$')
plt.ylabel('Distillation Error')
plt.title('Plot of Noise vs Distillation Error')

plt.grid(True)

plt.legend()

plt.show()

np.savetxt('dataforn2.txt', np.column_stack([x_axis,y_axis,y_axis1,y_axis2]))
