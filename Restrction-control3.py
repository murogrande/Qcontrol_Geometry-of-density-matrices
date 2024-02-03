# libraries
import numpy as np
import matplotlib as mpl
from matplotlib import cm

# from sympy.solvers import solve
# from sympy import Symbol
#
# from scipy.optimize import minimize
# from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import qutip

# import mayavi

from contrlWithGeodesics import geodesic

from contrlWithGeodesics import fidelity
from contrlWithGeodesics.pauli_mat_vec import *
from contrlWithGeodesics.utils import delete_less_than_k
from contrlWithGeodesics.controlSetup3 import control1setup3
from contrlWithGeodesics.getTimeFidelity import get_time_fidelity


# Test values
# del(estadoslist, tiempolists, solution)
# qsri = 1/np.sqrt(3)*np.array([1.0, 1.0, 0.9])
# qssf = np.array([0.0, 0.9, 0.0])
qsri = 1 / np.sqrt(3) * np.array([0.7, 0.8, 0.8])
qssf = 1 / np.sqrt(3) * np.array([0.2, 0.9, 0.0])
w0 = 5

gamma_0 = 0.01
gamma_c = 10
Nmax = 30  ### with 20 is not working
imax = 7
deltat = 0.003

# Save initial and final states
auxri = qsri
auxsf = qssf

print(auxri)
