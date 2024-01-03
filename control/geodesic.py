from pauli_mat_vec import iden, sigma_x, sigma_y, sigma_z
import numpy as np

# __all__ = [geo]


## the geodesic function
def geo(te=0.2, ri=np.array([0.0, 0.0, 0.0]), sf=np.array([0.0, 0.0, 0.0])):
    """This is the function for the geodesic"""
    ###
    # Geodesic given 2 vectors that belongs to the Bloch sphere
    ### INPUT
    # te(int): the parameter of the geodesic. te \in [0,1]
    # ri(vector): numpy array with the coordinates of the ri quantum state
    # sf(vector): numpy array with the coordinates of the sf quantum state
    ### OUTPUT
    # a quantum state at te given by geodesic between ri and sf
    #####################################
    # define the density matrices for the initial and final states
    rho1 = 0.5 * (iden + ri[0] * sigma_x + ri[1] * sigma_y + ri[2] * sigma_z)
    sigma2 = 0.5 * (iden + sf[0] * sigma_x + sf[1] * sigma_y + sf[2] * sigma_z)

    # calculate the square root and its inverse of the final matrix
    # rootrho1 = np.sqrt(1 / (np.trace(rho1) + 2 * np.sqrt(np.linalg.det(rho1)))) * (rho1 + np.sqrt(np.linalg.det(rho1)) * iden)
    # rootinverrho1 = np.linalg.inv(rootrho1)
    rootsigma2 = np.sqrt(
        1 / (np.trace(sigma2) + 2 * np.sqrt(np.linalg.det(sigma2)))
    ) * (sigma2 + np.sqrt(np.linalg.det(sigma2)) * iden)
    rootinversigma2 = np.linalg.inv(rootsigma2)

    # calculate the root product of the geodesic, the fidelity, and theta
    rootproducto = (
        rootsigma2 @ rho1 @ rootsigma2
        + np.sqrt(np.linalg.det(sigma2)) * np.sqrt(np.linalg.det(rho1)) * iden
    ) / (
        np.trace(sigma2 @ rho1)
        + 2 * np.sqrt(np.linalg.det(sigma2)) * np.sqrt(np.linalg.det(rho1))
    ) ** 0.5
    fide = np.trace(rho1 @ sigma2) + np.sqrt(1 - np.trace(rho1 @ rho1)) * np.sqrt(
        1 - np.trace(sigma2 @ sigma2)
    )
    theta0 = np.arccos(np.sqrt(fide))

    # define a function for the geodesic

    return (
        (np.cos(te * theta0) - np.sin(te * theta0) / np.tan(theta0)) ** 2 * rho1
        + (np.sin(te * theta0)) ** 2 / (np.sin(theta0)) ** 2 * sigma2
        + (np.sin(te * theta0))
        / (np.sin(theta0))
        * (np.cos(te * theta0) - (np.sin(te * theta0) / np.tan(theta0)))
        * (
            rootinversigma2 @ rootproducto @ rootsigma2
            + rootsigma2 @ rootproducto @ rootinversigma2
        )
    )
