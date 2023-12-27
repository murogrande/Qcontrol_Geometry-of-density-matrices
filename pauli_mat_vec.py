import numpy as np
# Declare variables as arrays
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_y = np.array([[0.0, -1j], [1.0j, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
iden = np.array([[1.0, 0.0], [0.0, 1.0]])
zero = np.array([[1.0], [0.0]])
uno = np.array([[0.0], [1.0]])
iden = np.array([[1.0,0.0],[0.0,1.0]])


def bloch_vector(rho):
    ''' 
    This functions takes a density matrix rho and we get the coordinates on the Bloch sphere
    '''
    vec = np.array([np.trace(rho @ sigma_x),
                    np.trace(rho @ sigma_y),
                    np.trace(rho @ sigma_z)])
    #return vec / np.linalg.norm(vec) #not normzalized becuase it does not work
    return vec 