import geodesic as geo
import numpy as np 

stat1 = np.array([0.0,0.0,0.9])
stat2 = np.array([1.0,0.0,0.0])
tau = 0.2
state= geo.geodesic(tau,stat1,stat2)
print(np.trace(state).real)
print(state)
print(state[0,0].real)
print(state[0,0].imag)
print(state[0,1].real)
print(state[0,1].imag)