#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:39:12 2022

Basics of plotting transfer functions in pymor. See also the resources in 
https://docs.pymor.org/2022-1-0/tutorial_bt.html

@author: alessandro
"""

import pymor
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel
from pymor.reductors.h2 import TFIRKAReductor
from pymor.reductors.interpolation import TFBHIReductor, GenericBHIReductor

plt.rcParams['axes.grid'] = True

k = 50
n = 2 * k + 1

E = sps.eye(n, format='lil')
E[0, 0] = E[-1, -1] = 0.5
E = E.tocsc()

d0 = n * [-2 * (n - 1)**2]
d1 = (n - 1) * [(n - 1)**2]
A = sps.diags([d1, d0, d1], [-1, 0, 1], format='lil')
A[0, 0] = A[-1, -1] = -n * (n - 1)
A = A.tocsc()

B = np.zeros((n, 2))
B[:, 0] = 1
B[0, 0] = B[-1, 0] = 0.5
B[0, 1] = n - 1

C = np.zeros((3, n))
C[0, 0] = C[1, k] = C[2, -1] = 1

fom = LTIModel.from_matrices(A, B, C, E=E, sampling_time=1)

# Magnitude Plot

fig, ax = plt.subplots()
w = np.logspace(-2, 8, 300)
_ = fom.transfer_function.mag_plot(w, ax=ax, label='FOM')


hsv = fom.hsv()
fig2, ax2 = plt.subplots()
ax2.semilogy(range(1, len(hsv) + 1), hsv, '.-')
_ = ax2.set_title('Hankel singular values')


tf = fom.transfer_function

tftilde = TFIRKAReductor(fom)
htilde = tftilde.reduce(11) #why it does not go more than 11? Loewner matrix gets ill

# let us use class pymor.reductors.interpolation.TFBHIReductor
# and class pymor.models.transfer_function.TransferFunction(dim_input, dim_output, tf, dtf=None, parameters={}, sampling_time=0, name=None
# for creating a fom object from the contour intergral method
#rom = TFBHIReductor(tf)
#order = 12
#l = np.random.rand(order, 2)
#r = np.random.rand(order, 3)
#htilde = rom.reduce(np.linspace(0,2000,order), l, r)

print(htilde)


w = np.logspace(-2, 8, 300)
_ = htilde.transfer_function.mag_plot(w, ax=ax, linestyle='--', label='ROM TF-IRKA')
_ = ax.legend()


error = fom - htilde
print(error.h2_norm())




