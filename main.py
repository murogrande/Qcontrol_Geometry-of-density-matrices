import numpy as np

from contrlWithGeodesics import geodesic as geo
from contrlWithGeodesics import fidelity as fide
from contrlWithGeodesics.controlSetup3 import control1setup3


import sys

print(sys.path)

vect1 = np.array([0.0, 0.0, 0.9])
vect2 = np.array([0.9, 0.0, 0.0])

print(geo(0.1, vect1, vect2))
print(fide(vect1, vect2))
# Test values
# qsri = 1/np.sqrt(3)*np.array([1.0, 1.0, 0.9])
# qssf = np.array([0.0, 0.9, 0.0])
qsri = 1 / np.sqrt(3) * np.array([0.7, 0.8, 0.8])
qssf = 1 / np.sqrt(3) * np.array([0.2, 0.9, 0.0])
w0 = 5.0
gamma_0 = 0.01
gamma_c = 10
Nmax = 40
imax = 7
deltat = 0.0030

# Save initial and final states
auxri = qsri
auxsf = qssf

estadoslist, tiempolists, solution, vec_lambda = control1setup3(
    qsri, qssf, Nmax=Nmax, deltat=deltat
)

print(estadoslist)

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
