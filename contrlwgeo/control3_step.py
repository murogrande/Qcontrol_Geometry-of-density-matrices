from .muk import muk
from sympy.solvers import solve
import numpy as np
import sympy
from scipy.integrate import solve_ivp


### add test for this function
""" Hamiltonian evolution by Δt in time. Using setup3
"""


def control3_step(
    ri,
    sf,
    lambda_x: "sympy.Symbol",
    w0: float,
    gamma_0: float,
    gamma_c: float,
    deltat: float,
    D_matrix,
    vector_lambda,
):

    # calculate vk, ukx, uky, and ukz: conherent and incoherent controls
    vk = lambda_x * 0.25 * (2.0 * muk(ri, sf)[2] - np.dot(ri, D_matrix @ muk(ri, sf)))
    ukx = lambda_x * np.cross(ri, muk(ri, sf))[0]
    uky = lambda_x * np.cross(ri, muk(ri, sf))[1]
    ukz = lambda_x * np.cross(ri, muk(ri, sf))[2]

    # solve for lambda
    solver1 = solve(vk**2 + ukx**2 + uky**2 + ukz**2 - gamma_c**2, lambda_x)

    lamdasol = solver1[1]

    vector_lambda.append(lamdasol)

    auxilvk = (
        lamdasol * 0.25 * (2.0 * muk(ri, sf)[2] - np.dot(ri, D_matrix @ muk(ri, sf)))
    )
    vk = auxilvk if auxilvk > 0 else 0
    ukx = lamdasol * np.cross(ri, muk(ri, sf))[0]
    uky = lamdasol * np.cross(ri, muk(ri, sf))[1]
    ukz = lamdasol * np.cross(ri, muk(ri, sf))[2]

    # Matrix for setup 3
    Bmatrix = np.array(
        [
            [-2.0 * gamma_0 - vk / 2.0, -2.0 * (w0 + ukz), 2.0 * uky],
            [2.0 * (w0 + ukz), -2.0 * gamma_0 - vk / 2.0, -2.0 * ukx],
            [-2.0 * uky, 2.0 * ukx, -vk],
        ]
    )
    qk = [0.0, 0.0, vk]  # qk setup3

    def odes(t, X):
        ## this function defines the differential equation
        rkx, rky, rkz = X
        drkx = (Bmatrix @ [rkx, rky, rkz] + qk)[0]
        drky = (Bmatrix @ [rkx, rky, rkz] + qk)[1]
        drkz = (Bmatrix @ [rkx, rky, rkz] + qk)[2]
        return drkx, drky, drkz

    ## initial conditions of the ODE
    rkx0 = ri[0]
    rky0 = ri[1]
    rkz0 = ri[2]
    # here we solve the ODE
    soln = solve_ivp(odes, (0.0, deltat), (rkx0, rky0, rkz0), dense_output=True)
    # replace to get x,y,z to plot on the Bloch sphere
    x, y, z = soln.sol(deltat)
    return x, y, z, soln
