# Import libraries
import sys
import os
import numpy as np
import scipy as sp

# OdeBase(): An Scipy based solver.
class OdeBase:
    def __init__(
        self,
        n_dims,
        ode_params,
        bc_vec,
        bc_time,
        time_inc,
        n_steps,
        bk_flag,
        intg_type="LSODA",
        tol=1.0e-4,
        n_max_iters=20,
        act_sol=None,
        adj_sol=None,
    ):

        n_times = n_steps + 1

        assert bc_time >= 0, "BC time should be >= 0!"

        if act_sol is not None:
            assert act_sol.shape[1] == n_times, "Incorrect act_sol size!"

        if adj_sol is not None:
            assert adj_sol.shape[1] == n_times, "Incorrect adj_sol size!"

        self.ode_params = ode_params
        self.bc_vec = bc_vec
        self.bc_time = bc_time
        self.n_dims = n_dims
        self.time_inc = time_inc
        self.n_steps = n_steps
        self.bk_flag = bk_flag
        self.intg_type = intg_type
        self.act_sol = act_sol
        self.adj_sol = adj_sol
        self.tol = tol
        self.n_max_iters = n_max_iters
        self.n_times = n_times

        self.sol = np.zeros(shape=(n_dims, n_times), dtype="d")

    def fun(self, t, y):
        pass

    def jac(self, t, y):
        pass

    def solve(self):

        n_dims = self.n_dims
        n_steps = self.n_steps
        n_times = self.n_times
        bc_time = self.bc_time
        time_inc = self.time_inc

        bk_flag = bool(bc_time > 0)

        if bk_flag:
            end_time = bc_time - n_steps * time_inc
        else:
            end_time = bc_time + n_steps * time_inc

        time_span = (bc_time, end_time)
        time_eval = np.linspace(bc_time, end_time, n_times)

        if self.intg_type in ["Radau", "BDF", "LSODA"]:
            jac = self.jac
        else:
            jac = None

        res = sp.integrate.solve_ivp(
            fun=self.fun,
            jac=jac,
            y0=self.bc_vec,
            t_span=time_span,
            t_eval=time_eval,
            method=self.intg_type,
            rtol=self.tol,
        )

        s_flag = res.success

        assert s_flag, "Failed to solve the ODE!"

        assert res.y.shape[1] == n_times, "Internal error!"

        if bk_flag:
            self.sol = np.flip(res.y, 1)
        else:
            self.sol = res.y.copy()

        return s_flag

    def get_sol(self):
        return self.sol
