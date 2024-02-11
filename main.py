import numpy as np

from contrlwgeo import geodesic as geo
from contrlwgeo import fidelity as fide
from contrlwgeo.controlSetup3 import control1setup3
from contrlwgeo.control3_step import control3_step
from contrlwgeo.controlSetup1 import control1setup1

import sympy

import matplotlib.pyplot as plt

## testing setup1
qsri = 1 / np.sqrt(3) * np.array([0.0, 0.0, 0.9])
qssf = 1 / np.sqrt(3) * np.array([0.9, 0.0, 0.0])
w0 = 5.0
gamma_0 = 0.00
gamma_c = w0
Nmax = 40
imax = 7
deltat = 0.0030

# Save initial and final states
auxri = qsri
auxsf = qssf

estadoslist, tiempolists, solution, vec_lambda = control1setup1(
    qsri, qssf, Nmax=Nmax, w0=w0, gamma_0=gamma_0, w_c=w0, deltat=deltat
)


print(estadoslist)


vect1 = np.array([0.0, 0.0, 0.9])
vect2 = np.array([0.9, 0.0, 0.0])

print(geo(0.1, vect1, vect2))

print(fide(vect1, vect2))

### data to test control3_step
w0 = 5
gamma_0 = 0.01
gamma_c = 10
deltat = 0.003
lambda_x = sympy.Symbol("lambda_x", real=True)
D_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]  ## Dmatrix of setup 3
vector_lambda = list([])
x, y, z, soln = control3_step(
    vect1, vect2, lambda_x, w0, gamma_0, gamma_c, deltat, D_matrix, vector_lambda
)


# geodesic
stat1 = np.array([0.0, 0.0, 0.9])
stat2 = np.array([0.9, 0.0, 0.0])
tau = 0.2
state = geo(tau, stat1, stat2)
print(np.trace(state).real)
# print(state)
# print(state[0, 0].real)
# print(state[0, 0].imag)
# print(state[0, 1].real)
# print("last geodesic component", state[0, 1].imag)
#
#
# stat1 = np.array([0.0, 0.0, 0.9])
# stat2 = np.array([0.9, 0.0, 0.0])
# muktry = muk.muk(stat1, stat2)
# print(muktry[0])
# print(muktry[1])
# print(muktry[2])
#
# fidtry = fidelity.fidelity(stat1, stat2)
## fidtry = fidelity.fidelity(stat1,stat1)
## fidtry = fidelity.fidelity(stat2,stat2)
#
# print("fidelity", fidtry)
# Create the data.
