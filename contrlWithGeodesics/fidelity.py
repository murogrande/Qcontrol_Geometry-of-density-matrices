import numpy as np


def fidelity(rho: np.array, st: np.array):
    """This is the fidelity of two the states rho and st. rho and st are numpy arrays"""
    rho = np.squeeze(
        np.asarray(rho)
    )  ## this line is import when are getting the time in the imax function
    return (
        0.5
        * (
            1
            + np.dot(rho, st)
            + (1 - np.dot(rho, rho)) ** 0.5 * (1 - np.dot(st, st)) ** 0.5
        )
    ).real
