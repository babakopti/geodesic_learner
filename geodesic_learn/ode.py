# Import libraries
import sys
import os
import time
import numpy as np
import scipy as sp
from scipy.integrate import trapz
from ode_base import OdeBase

# OdeGeoConstOrd1: Geodesic ODE solver; 1st order; const. curv.
class OdeGeoConstOrd1(OdeBase):
    def fun(self, t, y):
        n_dims = self.n_dims
        gamma = self.ode_params
        time_inc = self.time_inc
        n_times = self.n_times

        vals = -np.tensordot(
            gamma, np.tensordot(y, y, axes=0), ((1, 2), (0, 1))
        )

        return vals

    def jac(self, t, y):
        n_dims = self.n_dims
        gamma = self.ode_params

        vals = -2.0 * np.tensordot(gamma, y, axes=((2), (0)))

        return vals


# OdeAdjConstOrd1: Adjoint Geodesic solver; constant curvature
class OdeAdjConstOrd1(OdeBase):
    def fun(self, t, v):

        n_dims = self.n_dims
        gamma = self.ode_params
        time_inc = self.time_inc
        n_times = self.n_times
        act_sol = self.act_sol
        adj_sol = self.adj_sol

        vals = np.zeros(shape=(n_dims), dtype="d")
        ts_id = int(t / time_inc)

        assert ts_id < n_times, "ts_id should be smaller than n_times!"

        adj_vec = np.zeros(shape=(n_dims), dtype="d")
        act_vec = np.zeros(shape=(n_dims), dtype="d")

        for a in range(n_dims):
            adj_vec[a] = adj_sol[a][ts_id]
            act_vec[a] = act_sol[a][ts_id]

        vals = 2.0 * np.tensordot(
            gamma, np.tensordot(v, adj_vec, axes=0), ((0, 2), (0, 1))
        ) + (adj_vec - act_vec)

        return vals

    def jac(self, t, v):

        n_dims = self.n_dims
        time_inc = self.time_inc
        n_times = self.n_times
        gamma = self.ode_params
        adj_sol = self.adj_sol

        vals = np.zeros(shape=(n_dims, n_dims), dtype="d")
        ts_id = int(t / time_inc)

        assert ts_id < n_times, "ts_id should be smaller than n_times!"

        adj_vec = np.zeros(shape=(n_dims), dtype="d")

        for a in range(n_dims):
            adj_vec[a] = adj_sol[a][ts_id]

        vals = 2.0 * np.tensordot(gamma, adj_vec, ((2), (0)))
        vals = np.transpose(vals)

        return vals


# OdeGeoConstOrd2: Geodesic ODE solver; 2nd order; const. curv.
class OdeGeoConstOrd2(OdeBase):
    def fun(self, t, y):
        n_dims = self.n_dims
        gamma = self.ode_params

        vals = y[n_dims:].copy()
        vels = -np.tensordot(
            gamma, np.tensordot(y[n_dims:], y[n_dims:], axes=0), ((1, 2), (0, 1))
        )
        vals = np.concatenate((vals, vels))
        
        return vals

    def jac(self, t, y):
        n_dims = self.n_dims
        gamma = self.ode_params

        vals00 = np.zeros(shape=(n_dims, n_dims), dtype="d")
        vals01 = np.eye(n_dims)
        vals10 = vals00
        vals11 = -2.0 * np.tensordot(gamma, y[n_dims:], axes=((2), (0)))

        vals = np.concatenate(
            (
                np.concatenate((vals00, vals01), axis=1),
                np.concatenate((vals10, vals11), axis=1)
            ),
            axis=0
        )
        
        return vals

    
# OdeAdjConstOrd2: Adjoint Geodesic solver; constant curvature; 2nd order
class OdeAdjConstOrd2(OdeBase):
    def fun(self, t, v):

        n_dims = self.n_dims
        gamma = self.ode_params
        time_inc = self.time_inc
        n_times = self.n_times
        act_sol = self.act_sol
        adj_sol = self.adj_sol

        vals = np.zeros(shape=(n_dims), dtype="d")
        ts_id = int(t / time_inc)

        assert ts_id < n_times, "ts_id should be smaller than n_times!"

        adj_vec = np.zeros(shape=(2 * n_dims), dtype="d")
        act_vec = np.zeros(shape=(n_dims), dtype="d")

        for a in range(2 * n_dims):
            adj_vec[a] = adj_sol[a][ts_id]

        for a in range(n_dims):
            act_vec[a] = act_sol[a][ts_id]

        vals = v[:n_dims].copy()

        tmp1 = np.tensordot(
            gamma, np.tensordot(v[n_dims:], adj_vec[n_dims:], axes=0), ((0, 2), (0, 1))
        )
        tmp2 = adj_vec[:n_dims] - act_vec[:n_dims]
        tmp3 = np.tensordot(
            gamma, np.tensordot(adj_vec[n_dims:], adj_vec[n_dims:], axes=0), ((1, 2), (0, 1))
        )
        tmp4 = np.tensordot(gamma, v[n_dims:], ((0), (0)))
        
        vels = 2.0 * tmp1 - tmp2 - 2.0 * np.tensordot(tmp3, tmp4, ((0), (0)))

        vals = np.concatenate((vals, vels))

        return vals

    def jac(self, t, v):
        n_dims = self.n_dims
        gamma = self.ode_params

        vals00 = np.zeros(shape=(n_dims, n_dims), dtype="d")
        vals01 = np.eye(n_dims)

        tmp = np.tensordot(
            gamma, np.tensordot(adj_vec[n_dims:], adj_vec[n_dims:], axes=0), ((1, 2), (0, 1))
        )
        vals10 = -2.0 * np.tensordot(gamma, tmp, ((1), (0))).transpose()
        vals11 = 2.0 * np.tensordot(gamma, y[n_dims:], axes=((2), (0))).transpose()

        vals = np.concatenate(
            (
                np.concatenate((vals00, vals01), axis=1),
                np.concatenate((vals10, vals11), axis=1)
            ),
            axis=0
        )
        
        return vals

    
