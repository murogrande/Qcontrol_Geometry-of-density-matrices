import numpy as np
from sympy import Symbol
from .fidelity import fidelity
from scipy.optimize import minimize
from .muk import muk
from sympy.solvers import solve
from scipy.integrate import solve_ivp


def get_time_fidelity(c, tiempototal, soln, imax, sf, w0=5, gamma_0=0.01, gamma_c=10):
    """This is the for loop that contains the imax. This function calculates the
    time of the next evolution using the fidelity"""
    #### INPUT
    # c(list): numpy array that contains the quantum states until Nmax
    # tiempotoal(list): numpy array that contains the time steps
    # soln(solution of odes): this is the solution of odes that come from the previous step
    # sf(list): final quantum states
    # imax(int): number of iterations in order to get the time with fidelity
    # ..
    # ..
    ###OUTPUT
    # c(list): list of quantum states that contains the evolution
    # tiempotoal(list): numpy array that contains the time steps
    #
    #########################################################
    # Time with Fidelity
    auxvar = len(c)
    aux2time = tiempototal[-1]
    lambda_x = Symbol("lambda_x")  ## simbolic lambda for solving with sympy
    D_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
    ]  ## Dmatrix of setup 3
    list_lambda = list([])
    for i in range(imax):
        print("Find time with fidelity")

        def func_to_optimize(t, soln):
            """This is the fidelity that we want to optimize in order to get
            the time. Here, we are minimizing the -1*Fidelity"""
            ## INPUT
            # t(parameter): t is a parameter
            # soln(solution of odes): this solution of odes come from the last step
            # OUTPUT
            # Fidelity that depends on parameter t
            #
            return -1.0 * fidelity(soln.sol(t), sf)

        res = minimize(func_to_optimize, [0.00000001], args=(soln), tol=1e-6)
        timeopt = res.x[0] / 2.0  # time from fidelity

        aux2time += timeopt
        tiempototal.append(aux2time)

        print("new time:", timeopt)
        x1, y1, z1 = soln.sol(timeopt)
        rnew = np.array([x1, y1, z1])
        print("Fidelity", fidelity(rnew, sf))
        ri = rnew
        c.append(ri)

        ############################## we have to use the control1setup3 function here
        print("New quantum state", ri)

        # calculate vk, ukx, uky, and ukz
        vk = (
            lambda_x / 4.0 * (2.0 * muk(ri, sf)[2] - np.dot(ri, D_matrix @ muk(ri, sf)))
        )
        ukx = lambda_x * np.cross(ri, muk(ri, sf))[0]
        uky = lambda_x * np.cross(ri, muk(ri, sf))[1]
        ukz = lambda_x * np.cross(ri, muk(ri, sf))[2]

        # solve for lambda
        solver1 = solve(
            vk**2 + ukx**2 + uky**2 + ukz**2 - gamma_c**2, lambda_x
        )
        lamdasol = solver1[1]
        list_lambda.append(lamdasol)
        auxilvk = (
            lamdasol / 4.0 * (2.0 * muk(ri, sf)[2] - np.dot(ri, D_matrix @ muk(ri, sf)))
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
        qk = [0.0, 0.0, vk]

        def odes(t, X):
            ## this function (internal) defines the differential equation

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
        soln = solve_ivp(odes, (0, timeopt), (rkx0, rky0, rkz0), dense_output=True)

        # replace to get x,y,z to plot on the Bloch sphere
        x, y, z = soln.sol(timeopt)
        ri = np.array([x, y, z])  ## update the ri

    return c, tiempototal, list_lambda
