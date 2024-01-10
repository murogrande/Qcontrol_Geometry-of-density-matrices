from sympy import Symbol
from fidelity import fidelity
from muk import muk
import numpy as np
from scipy.integrate import solve_ivp
from sympy.solvers import solve


def control1setup3(
    ri, sf, Nmax=60, w0=5, gamma_0=0.01, gamma_c=10, deltat=0.003, initime=0.0
):
    """Control setup3 function without imax"""
    ###INPUT
    # ri(np.array): np.array[rix,riy,riz] initial quantum
    # sf(np.array): np.array[rix,riy,riz] final quantum
    # Nmax(int): Nmax iteration that will run the algorithm
    # ...
    # ...t
    # ....
    # OUTPUT
    # c(np.array): list containing the quantum states that will lead to the final state
    # tiempototal(np.array): list containing the time steps
    # soln (solution of odes): solution of odes with parameter t
    #######################################
    lambda_x = Symbol("lambda_x", real=True)  ## simbolic lambda for solving with sympy
    D_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]  ## Dmatrix of setup 3
    c = [ri]
    tiempototal = [initime]
    vector_lambda = list([])
    # iterate to find the lambda value
    helperk = 0
    # auxtime =
    oldri = ri  ## save the initial state ri
    # for k in range(Nmax):
    while (fidelity(oldri, sf) <= fidelity(ri, sf)) and (helperk < Nmax):
        # while (fidelity(oldri, sf) - fidelity(ri, sf)<0.0000001) and (helperk < Nmax):
        # calculate vk, ukx, uky, and ukz
        vk = (
            lambda_x / 4.0 * (2.0 * muk(ri, sf)[2] - np.dot(ri, D_matrix @ muk(ri, sf)))
        )
        ukx = lambda_x * np.cross(ri, muk(ri, sf))[0]
        uky = lambda_x * np.cross(ri, muk(ri, sf))[1]
        ukz = lambda_x * np.cross(ri, muk(ri, sf))[2]
        # print("estado initial",ri)
        # print("estado final",sf)
        # print("muk",muk(ri, sf))
        # print(vk, ukx, uky, ukz)

        # solve for lambda
        solver1 = solve(
            vk**2 + ukx**2 + uky**2 + ukz**2 - gamma_c**2, lambda_x
        )

        lamdasol = solver1[1]

        vector_lambda.append(lamdasol)
        auxilvk = (
            lamdasol / 4 * (2.0 * muk(ri, sf)[2] - np.dot(ri, D_matrix @ muk(ri, sf)))
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
        oldri = ri  ## save the old ri
        ri = np.array([x, y, z])  ## update the ri
        # print("Just checking-the new vector is:",ri)
        # print("Just checking-the vector to get:",sf)
        # print("Just checking-the old fidelity is:",fidelity(oldri,sf))
        # print("Just checking-the fidelity is:",fidelity(ri,sf))
        # print("Just checking-helperk=",helperk)
        c.append(ri)
        initime = initime + deltat
        # print(initime)
        tiempototal.append(initime)
        helperk += 1
    ### eliminate the final states
    c = c[0:-1]
    tiempototal = tiempototal[0:-1]
    return c, tiempototal, soln, vector_lambda
