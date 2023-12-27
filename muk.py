import numpy as np


def muk(rk: np.array, s: np.array):
    """muk function of equation (43). It is related to M_k. This funciton helps calculating the
    optimal control parameters
    """
    return (
        s
        - np.dot(rk, rk) * s
        - np.sqrt(1 - np.dot(rk, rk)) * np.sqrt(1 - np.dot(s, s)) * rk
    ) / (
        2
        * (1 - np.dot(rk, rk))
        * np.sqrt(
            0.5
            * (1 + np.dot(rk, s) + np.sqrt((1 - np.dot(rk, rk)) * (1 - np.dot(s, s))))
        )
    )
