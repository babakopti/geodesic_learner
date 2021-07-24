# Import libraries
import sys
import os
import time
import numpy as np
import scipy as sp
from scipy.integrate import trapz
from ode_base import OdeBase

# OdeGeoConst: Geodesic ODE solver; 1st order; const. curv.
class OdeGeoConst(OdeBase):
    def fun(self, t, y):
        n_dims = self.n_dims
        gamma = self.gamma
        time_inc = self.time_inc
        n_times = self.n_times

        vals = -np.tensordot(
            gamma, np.tensordot(y, y, axes=0), ((1, 2), (0, 1))
        )

        return vals

    def jac(self, t, y):
        n_dims = self.n_dims
        gamma = self.gamma

        vals = -2.0 * np.tensordot(gamma, y, axes=((2), (0)))

        return vals


# OdeAdjConst: Adjoint Geodesic solver; constant curvature
class OdeAdjConst(OdeBase):
    def fun(self, t, v):

        n_dims = self.n_dims
        gamma = self.gamma
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
        gamma = self.gamma
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


# OdeGeo: Geodesic ODE solver
class OdeGeo(OdeBase):
    def fun(self, t, y):

        n_dims = self.n_dims
        time_inc = self.time_inc
        n_times = self.n_times
        ts_id = int(t / time_inc)
        gamma = self.gamma

        vals = -np.tensordot(
            gamma[ts_id], np.tensordot(y, y, axes=0), ((1, 2), (0, 1))
        )

        return vals

    def jac(self, t, y):

        n_dims = self.n_dims
        time_inc = self.time_inc
        ts_id = int(t / time_inc)
        gamma = self.gamma

        vals = -2.0 * np.tensordot(gamma[ts_id], y, axes=((2), (0)))

        return vals


# OdeAdj: Adjoint Geodesic solver
class OdeAdj(OdeBase):
    def fun(self, t, v):

        n_dims = self.n_dims
        time_inc = self.time_inc
        n_times = self.n_times
        act_sol = self.act_sol
        adj_sol = self.adj_sol

        ts_id = int(t / time_inc)
        gamma = self.gamma

        vals = np.zeros(shape=(n_dims), dtype="d")
        ts_id = int(t / time_inc)

        assert ts_id < n_times, "ts_id should be smaller than n_times!"

        adj_vec = np.zeros(shape=(n_dims), dtype="d")
        act_vec = np.zeros(shape=(n_dims), dtype="d")

        for a in range(n_dims):
            adj_vec[a] = adj_sol[a][ts_id]
            act_vec[a] = act_sol[a][ts_id]

        vals = (
            2.0
            * np.tensordot(
                gamma[ts_id],
                np.tensordot(v, adj_vec, axes=0),
                ((0, 2), (0, 1)),
            )
            + (adj_vec - act_vec)
        )

        return vals

    def jac(self, t, v):

        n_dims = self.n_dims
        time_inc = self.time_inc
        n_times = self.n_times
        adj_sol = self.adj_sol

        vals = np.zeros(shape=(n_dims, n_dims), dtype="d")
        ts_id = int(t / time_inc)

        assert ts_id < n_times, "ts_id should be smaller than n_times!"

        gamma = self.gamma

        adj_vec = np.zeros(shape=(n_dims), dtype="d")

        for a in range(n_dims):
            adj_vec[a] = adj_sol[a][ts_id]

        vals = 2.0 * np.tensordot(gamma[ts_id], adj_vec, ((2), (0)))
        vals = np.transpose(vals)

        return vals
