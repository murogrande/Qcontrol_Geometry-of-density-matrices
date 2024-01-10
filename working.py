import geodesic as geo
import muk as muk
import fidelity as fidelity
import numpy as np

stat1 = np.array([0.0, 0.0, 0.9])
stat2 = np.array([0.9, 0.0, 0.0])
tau = 0.2
state = geo.geodesic(tau, stat1, stat2)
print(np.trace(state).real)
print(state)
print(state[0, 0].real)
print(state[0, 0].imag)
print(state[0, 1].real)
print("last geodesic component", state[0, 1].imag)


stat1 = np.array([0.0, 0.0, 0.9])
stat2 = np.array([0.9, 0.0, 0.0])
muktry = muk.muk(stat1, stat2)
print(muktry[0])
print(muktry[1])
print(muktry[2])

fidtry = fidelity.fidelity(stat1, stat2)
# fidtry = fidelity.fidelity(stat1,stat1)
# fidtry = fidelity.fidelity(stat2,stat2)

print("fidelity", fidtry)