from sympy import Symbol
from .fidelity import fidelity
from .muk import muk
import numpy as np
from scipy.integrate import solve_ivp
from sympy.solvers import solve

from .control3_step import control3_step


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
    oldri = ri  ## save initial state
    xri, yri, zri, _ = control3_step(
        ri, sf, lambda_x, w0, gamma_0, gamma_c, deltat, D_matrix, vector_lambda
    )  # initialize the new evolve state
    ri = np.array([xri, yri, zri])

    while (fidelity(oldri, sf) <= fidelity(ri, sf)) and (helperk < Nmax):

        x, y, z, soln = control3_step(
            ri, sf, lambda_x, w0, gamma_0, gamma_c, deltat, D_matrix, vector_lambda
        )

        oldri = ri  ## save the old ri
        ri = np.array([x, y, z])
        c.append(ri)
        initime = initime + deltat
        tiempototal.append(initime)
        helperk += 1
    ### eliminate the final states
    c = c[0:-1]
    tiempototal = tiempototal[0:-1]
    return c, tiempototal, soln, vector_lambda


def control1setup3_int_states(
    ri, sf, Nmax=60, w0=5, gamma_0=0.01, gamma_c=10, deltat=0.003, initime=0.0
):
    """Control setup3 with intermediate states. Function without imax"""
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
    oldri = ri  ## save initial state
    xri, yri, zri, solnri = control3_step(
        ri, sf, lambda_x, w0, gamma_0, gamma_c, deltat, D_matrix, vector_lambda
    )  # initialize the new evolve state
    ri = np.array([xri, yri, zri])

    # while (fidelity(oldri, sf) <= fidelity(ri, sf)) and (helperk < Nmax):

    while (fidelity(ri, sf) - fidelity(oldri, sf) >= 0.00001) and (helperk < Nmax):
        """hacer una function solo para continuos y luego para intermediate states"""
        print("k", helperk)
        print(fidelity(ri, sf) - fidelity(oldri, sf))

        x, y, z, soln = control3_step(
            ri, sf, lambda_x, w0, gamma_0, gamma_c, deltat, D_matrix, vector_lambda
        )

        oldri = ri  ## save the old ri
        ri = np.array([x, y, z])
        c.append(ri)
        initime = initime + deltat
        tiempototal.append(initime)
        helperk += 1
    ### eliminate the final states
    c = c[0:-1]
    tiempototal = tiempototal[0:-1]
    return c, tiempototal, soln, vector_lambda
