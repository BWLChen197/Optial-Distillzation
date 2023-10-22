#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:45:57 2023

@author: bruce
"""

import numpy as np
import matplotlib.pyplot as plt


# Initialize empty lists to store arrays
x_axis = []
y_axis = []
y_axis1 = []
y_axis2 = []

loaded_arrays = np.loadtxt('dataforn3.txt', delimiter=' ')

# Convert lists to NumPy arrays if necessary
x_axis = loaded_arrays[:, 0]
y_axis = loaded_arrays[:, 1]
y_axis1 = loaded_arrays[:, 2]
y_axis2 = loaded_arrays[:, 3]

plt.plot(x_axis, y_axis,label = 'p=1')
plt.plot(x_axis, y_axis1,label = 'p=0.75')
plt.plot(x_axis, y_axis2,label = 'p=0.5')

plt.xlabel('Noise'+r'$\gamma$')
plt.ylabel('Distillation Error')
plt.title('Plot of Noise vs Distillation Error')

plt.grid(True)

plt.legend()

plt.show()
