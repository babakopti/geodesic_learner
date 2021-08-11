# Import libraries
import sys
import os
import math
import time
import logging
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import gc
from scipy.integrate import trapz

import const as c
from ode import (
    OdeGeoConstOrd1,
    OdeGeoConstIEOrd1,
    OdeAdjConstIEOrd1,        
    OdeAdjConstOrd1,
    OdeGeoConstOrd2,
    OdeAdjConstOrd2,
    OdeGeoQuadratic,
    OdeAdjQuadratic,
)
from utils import get_logger


class GeodesicLearner:
    def __init__(
        self,
        manifold_type: str = c.CONST_CURVATURE_ORD1,
        opt_method: str = "SLSQP",
        max_opt_iters: int = 100,
        opt_tol: float = 1.0e-8,
        ode_geo_solver: str = "LSODA",
        ode_adj_solver: str = "RK45",
        ode_geo_tol: float = 1.0e-2,
        ode_adj_tol: float = 1.0e-2,
        ode_geo_max_iters: int = 20,
        ode_adj_max_iters: int = 20,
        ode_bc_mode: str = c.END_BC,
        learning_rate: float = 1.0,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        diagonal_metric: bool = True,
        self_relation: bool = False,
        verbose: bool = True,
    ):
        """
        Creates an object for geodesic learning.
        |
        |      Parameters
        |      ----------
        |      manifold_type: str
        |          Type of the manifold. Currently, "const_curvature" and
        |          "free_style" are supported. Default value is "const_curvature".
        |
        |      opt_method: str
        |          Optimization method used. Currently only "SLSQP" is recommended.
        |          Default is "SLSQP".
        |
        |      max_opt_iters: int
        |          Maximum number of optimization iterations. Default is 100.
        |
        |      opt_tol: float
        |          Optimization tolerance. Default is 1.0e-8.
        |
        |      ode_geo_solver: str.
        |          ODE solver for geodesic equations.
        |          "LSODA" is recommended. Default is "LSODA".
        |
        |      ode_adj_solver: str
        |          ODE solver for adjoint equations.
        |          "RK45" is recommended. Default is "RK45".
        |
        |      ode_geo_tol: float
        |          ODE solver tolerance for geodesic equations.
        |          Default is 1.0e-2.
        |
        |      ode_adj_tol: float
        |          ODE solver tolerance for adjoint equations.
        |          Default is 1.0e-2.
        |
        |      ode_geo_max_iters: int
        |          ODE solver max iterations for geodesic equations.
        |          Default is 20.
        |
        |      ode_adj_max_iters: int
        |          ODE solver max iterations for adjoint equations.
        |          Default is 20.
        |
        |      ode_bc_mode: str
        |          Mode of boundary condition during training.
        |          Allowed values are "end_bc" and "start_bc". Default is "end_bc".
        |
        |      learning_rate: float
        |          The learning rate for solving the optimization problem. Default is 1.0.
        |
        |      alpha: float
        |          Regularization strength. Default is 0.
        |
        |      l1_ratio: float
        |          Regularization mixing parameter, with 0 <= l1_ratio <= 1.
        |          For l1_ratio = 0 the penalty is an L2 penalty.
        |          For l1_ratio = 1 it is an L1 penalty.
        |          For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        |          Default is 0.
        |
        |     diagonal_metric: bool
        |          Assume a diagonal metric. Only applies to "constant_curevature" manifolds.
        |          Default is True.
        |
        |     self_relation: bool
        |         Self relation flag. Only applies to "constant_curevature" manifolds.
        |         Default is False.
        |
        |     verbose: bool
        |         Verbosity flag. Default is True.
        |
        |      Returns
        |      -------
        |      self
        """
        self.manifold_type = manifold_type
        self.opt_method = opt_method
        self.max_opt_iters = max_opt_iters
        self.opt_tol = opt_tol
        self.ode_geo_solver = ode_geo_solver
        self.ode_adj_solver = ode_adj_solver
        self.ode_geo_tol = ode_geo_tol
        self.ode_adj_tol = ode_adj_tol
        self.ode_geo_max_iters = ode_geo_max_iters
        self.ode_adj_max_iters = ode_adj_max_iters  
        self.ode_bc_mode = ode_bc_mode
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.diagonal_metric = diagonal_metric
        self.self_relation = self_relation
        self.verbose = verbose
        self.n_dims = None
        self.n_times = None
        self.n_params = None
        self.params = None
        self.act_sol = None

        if verbose:
            level = logging.INFO
        else:
            level = logging.CRITICAL

        self.logger = get_logger("geodesic", level)

    def fit(self, X: np.array, y=None):
        """
        Fit geodesic model.
        |
        |      Parameters
        |      ----------
        |      X : array, shape = (n_times, n_dims)
        |          Training data.
        |
        |      y : This argument is ignored.
        |
        |      Returns
        |      -------
        |      None
        """
        self._init_params(X)
        self._set_params()

    def predict(
        self,
        bc_vec: np.array,
        n_steps: int,
        bk_flag: bool = False,
    ):
        """
        Predict using geodesic model.
        |
        |      Parameters
        |      ----------
        |      bc_vec : array, shape = [n_dims]
        |          The boundary condition array.
        |
        |      n_steps : int
        |          Number of integration steps.
        |
        |      bk_flag: bool
        |          If set to true, them time marching goes backward.
        |          Default: False.
        |
        |      Returns
        |      -------
        |      array, shape = (n_times, n_dims).
        """
        assert self.params is not None

        if bk_flag:
            bc_time = n_steps
        else:
            bc_time = 0.0

        gamma = self._get_ode_params(self.params)

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            assert len(bc_vec) == self.n_dims
            if self.ode_geo_solver == "implicit_euler":
                ode_obj = OdeGeoConstIEOrd1(
                    n_dims=self.n_dims,
                    ode_params=gamma,
                    bc_vec=bc_vec,
                    time_inc=1.0,
                    n_steps=n_steps,
                    bk_flag=bk_flag,
                    tol=self.ode_geo_tol,
                    n_max_iters=self.ode_geo_max_iters,                
                )
            else:
                ode_obj = OdeGeoConstOrd1(
                    n_dims=self.n_dims,
                    ode_params=gamma,
                    bc_vec=bc_vec,
                    bc_time=bc_time,
                    time_inc=1.0,
                    n_steps=n_steps,
                    bk_flag=bk_flag,
                    intg_type=self.ode_geo_solver,
                    tol=self.ode_geo_tol,
                    n_max_iters=self.ode_geo_max_iters,                
                )                
        elif self.manifold_type == c.CONST_CURVATURE_ORD2:
            assert len(bc_vec) == 2 * self.n_dims
            ode_obj = OdeGeoConstOrd2(
                n_dims=self.n_dims,
                ode_params=gamma,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_geo_solver,
                tol=self.ode_geo_tol,
                n_max_iters=self.ode_geo_max_iters,                
            )
        elif self.manifold_type == c.QUADRATIC:
            assert len(bc_vec) == 2 * self.n_dims
            ode_obj = OdeGeoQuadratic(
                n_dims=self.n_dims,
                ode_params=gamma,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_geo_solver,
                tol=self.ode_geo_tol,
                n_max_iters=self.ode_geo_max_iters,                
            )
        else:
            assert False, "Unknow manidold type"

        s_flag = ode_obj.solve()

        if not s_flag:
            self.logger.warning("Geodesic equation did not converge!")
            return None

        return ode_obj.get_sol().transpose()

    def predict_train(self):
        """
        Predict training data.
        |
        |      Returns
        |      -------
        |      array, shape = (n_times, n_dims).
        """
        assert self.params is not None

        return self._get_geo_sol(self.params).transpose()

    def _init_params(self, X):

        n_times = X.shape[0]
        n_dims = X.shape[1]

        self.n_times = n_times
        self.n_dims = n_dims

        # Calculate number of model parameters
        if self.manifold_type in [
            c.CONST_CURVATURE_ORD1,
            c.CONST_CURVATURE_ORD2,
        ]:

            if self.diagonal_metric:
                self.n_params = n_dims * (2 * n_dims - 1)
            else:
                self.n_params = int(n_dims * n_dims * (n_dims + 1) / 2)
            if not self.self_relation:
                self.n_params -= n_dims
        elif self.manifold_type == c.QUADRATIC:
            self.n_params = n_dims
        else:
            assert False, "Unknow manidold type"

        n_gamma_vec = self.n_params

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            self.n_params += n_dims
        else:
            self.n_params += 2 * n_dims

        # Initialize parameters and set BC vec
        self.params = np.zeros(shape=(self.n_params), dtype="d")

        for m in range(self.n_dims):
            if self.ode_bc_mode == c.END_BC:
                self.params[m + n_gamma_vec] = X[self.n_times - 1][m]
                if self.manifold_type != c.CONST_CURVATURE_ORD1:
                    self.params[m + n_gamma_vec + n_dims] = (
                        X[self.n_times - 1][m] - X[self.n_times - 2][m]
                    )
            elif self.ode_bc_mode == c.START_BC:
                self.params[m + n_gamma_vec] = X[0][m]
                if self.manifold_type != c.CONST_CURVATURE_ORD1:
                    self.params[m + n_gamma_vec + n_dims] = X[1][m] - X[0][m]
            else:
                assert False

        self.act_sol = X.transpose()

    def _set_params(self):

        t0 = time.time()

        self.logger.info(
            "Running continuous adjoint optimization to set Christoffel symbols..."
        )

        options = {
            "ftol": self.opt_tol,
            "maxiter": self.max_opt_iters,
            "disp": True,
            "eps": 1.0e-5,
        }

        try:
            opt_obj = scipy.optimize.minimize(
                fun=self._get_obj_func,
                x0=self.params,
                method=self.opt_method,
                jac=self._get_grad,
                options=options,
            )
            s_flag = opt_obj.success

            self.params = opt_obj.x

            self.logger.info("Success: %s", str(s_flag))

        except Exception as exc:
            self.logger.error(exc)
            s_flag = False

        self.logger.info(
            "Setting parameters took %0.2f seconds.",
            time.time() - t0,
        )

        return s_flag

    def _get_obj_func(self, params):

        # Get solution
        sol = self._get_geo_sol(params)

        if sol is None:
            return np.inf

        # Calculate the objective function
        tmp_vec = np.zeros(shape=(self.n_times))
        for m in range(self.n_dims):
            tmp_vec += (sol[m][:] - self.act_sol[m][:]) ** 2

        val = 0.5 * trapz(tmp_vec, dx=1.0)

        # Add regularization term based on gamma
        if self.manifold_type != c.CONST_CURVATURE_ORD1:
            n_gamma_vec = self.n_params - self.n_dims
        else:
            n_gamma_vec = self.n_params - 2 * self.n_dims

        tmp1 = np.linalg.norm(params[:n_gamma_vec], 1)
        tmp2 = np.linalg.norm(params[:n_gamma_vec])
        val += self.alpha * (self.l1_ratio * tmp1 + (1.0 - self.l1_ratio) * tmp2 ** 2)

        del sol
        del tmp_vec

        gc.collect()

        return val

    def _get_grad(self, params):

        if self.manifold_type in [c.CONST_CURVATURE_ORD1, c.CONST_CURVATURE_ORD2]:
            return self._get_grad_const(params)
        elif self.manifold_type == c.QUADRATIC:
            return self._get_grad_quadratic(params)
        else:
            assert False

    def _get_grad_const(self, params):

        n_dims = self.n_dims
        n_times = self.n_times
        alpha = self.alpha
        l1_ratio = self.l1_ratio
        time_inc = 1.0
        xi = lambda a, b: 1.0 if a == b else 2.0
        grad = np.zeros(shape=(self.n_params), dtype="d")

        sol = self._get_geo_sol(params)
        adj_sol = self._get_adj_sol(params, sol)
        gamma = self._get_ode_params(params)

        gamma_id = 0
        for r in range(n_dims):
            for p in range(n_dims):
                for q in range(p, n_dims):

                    if self.diagonal_metric and r != p and r != q and p != q:
                        continue
                    if (not self.self_relation) and r == p and p == q:
                        continue

                    if self.manifold_type == c.CONST_CURVATURE_ORD1:
                        tmp_vec = xi(p, q) * np.multiply(sol[p][:], sol[q][:])
                    elif self.manifold_type == c.CONST_CURVATURE_ORD2:
                        tmp_vec = xi(p, q) * np.multiply(
                            sol[p + n_dims][:], sol[q + n_dims][:]
                        )

                    tmp_vec = np.multiply(tmp_vec, adj_sol[r][:])

                    grad[gamma_id] = trapz(tmp_vec, dx=time_inc) + alpha * (
                        l1_ratio * np.sign(params[gamma_id])
                        + (1.0 - l1_ratio) * 2.0 * params[gamma_id]
                    )

                    gamma_id += 1

        del tmp_vec

        gc.collect()

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            n_gamma_vec = self.n_params - n_dims
        elif self.manifold_type == c.CONST_CURVATURE_ORD2:
            n_gamma_vec = self.n_params - 2 * n_dims

        gamma_vec = params[:n_gamma_vec]

        if self.ode_bc_mode == c.END_BC:
            bc_ind = -1
            bc_fct = 1.0
        else:
            bc_ind = 0
            bc_fct = -1.0

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            for i in range(n_dims):
                grad[i + n_gamma_vec] = bc_fct * adj_sol[i][bc_ind]
        elif self.manifold_type == c.CONST_CURVATURE_ORD2:
            tmp_vec1 = np.empty(shape=(n_dims), dtype="d")
            tmp_vec2 = np.empty(shape=(n_dims), dtype="d")

            for i in range(n_dims):
                tmp_vec1[i] = adj_sol[i][bc_ind]
                tmp_vec2[i] = sol[i + n_dims][bc_ind]

            tmp_vec3 = np.tensordot(
                gamma, np.tensordot(tmp_vec1, tmp_vec2, axes=0), ((0, 1), (0, 1))
            )
            for i in range(n_dims):
                grad[i + n_gamma_vec] = bc_fct * (
                    -adj_sol[i + n_dims][bc_ind] + 2.0 * tmp_vec3[i]
                )
                grad[i + n_gamma_vec + n_dims] = bc_fct * adj_sol[i][bc_ind]

        learning_rate = self.learning_rate
        if learning_rate is None:
            tmp1 = np.linalg.norm(gamma_vec)
            tmp2 = np.linalg.norm(grad)

            learning_rate = 1.0
            if tmp2 > 0:
                learning_rate = min(1.0, math.sqrt(abs(tmp1 ** 2 - n_gamma_vec) / tmp2 ** 2))

        grad = learning_rate * grad

        del sol
        del adj_sol

        gc.collect()

        return grad

    def _get_grad_quadratic(self, params):

        n_dims = self.n_dims
        n_times = self.n_times
        alpha = self.alpha
        l1_ratio = self.l1_ratio
        time_inc = 1.0
        grad = np.zeros(shape=(self.n_params), dtype="d")

        sol = self._get_geo_sol(params)
        adj_sol = self._get_adj_sol(params, sol)
        ode_params = self._get_ode_params(params)

        tmp_vec0 = np.ones(shape=(n_times), dtype="d")
        for m in range(n_dims):
            tmp_vec0 += (params[m] * sol[m][:]) ** 2
            
        tmp_vec0 = 1.0 / tmp_vec0

        tmp_vec1 = np.zeros(shape=(n_times), dtype="d")
        for m in range(n_dims):
            tmp_vec1 += params[m] * sol[m][:] * adj_sol[m][:]

        tmp_vec2 = np.zeros(shape=(n_times), dtype="d")
        for a in range(n_dims):
            tmp_vec2 += params[a] * sol[a + n_dims][:] ** 2

        for l in range(n_dims):

            tmp_vec = (
                tmp_vec0 * tmp_vec1 * sol[l + n_dims][:] ** 2
                + tmp_vec0 * tmp_vec2 * sol[l][:] * adj_sol[l][:]
                - 2.0 * tmp_vec0 ** 2 * tmp_vec1 * tmp_vec2 * params[l] * sol[l][:] ** 2
            )

            grad[l] = trapz(tmp_vec, dx=time_inc) + alpha * (
                l1_ratio * np.sign(params[l]) + (1.0 - l1_ratio) * 2.0 * params[l]
            )

        del tmp_vec, tmp_vec2

        gc.collect()

        if self.ode_bc_mode == c.END_BC:
            bc_ind = -1
            bc_fct = 1.0
        else:
            bc_ind = 0
            bc_fct = -1.0

        for r in range(n_dims):
            grad[r + n_dims] = bc_fct * (
                -adj_sol[r + n_dims][bc_ind]
                + 2.0 * tmp_vec0[bc_ind] * tmp_vec1[bc_ind] * params[r] * sol[r + n_dims][bc_ind]
            )
            grad[r + 2 * n_dims] = bc_fct * adj_sol[r][bc_ind]

        del sol
        del adj_sol
        del tmp_vec0, tmp_vec1

        gc.collect()

        return grad

    def _get_geo_sol(self, params):

        n_steps = self.n_times - 1

        if self.ode_bc_mode == c.END_BC:
            bc_time = n_steps
            bk_flag = True
        else:
            bc_time = 0.0
            bk_flag = False

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            n_gamma_vec = self.n_params - self.n_dims
        else:
            n_gamma_vec = self.n_params - 2 * self.n_dims

        gamma_vec = params[:n_gamma_vec]
        bc_vec = params[n_gamma_vec:]

        ode_params = self._get_ode_params(params)

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            if self.ode_geo_solver == "implicit_euler":            
                ode_obj = OdeGeoConstIEOrd1(
                    n_dims=self.n_dims,
                    ode_params=ode_params,
                    bc_vec=bc_vec,
                    time_inc=1.0,
                    n_steps=n_steps,
                    bk_flag=bk_flag,
                    tol=self.ode_geo_tol,
                    n_max_iters=self.ode_geo_max_iters,                
                )
            else:
                ode_obj = OdeGeoConstOrd1(
                    n_dims=self.n_dims,
                    ode_params=ode_params,
                    bc_vec=bc_vec,
                    bc_time=bc_time,
                    time_inc=1.0,
                    n_steps=n_steps,
                    bk_flag=bk_flag,
                    intg_type=self.ode_geo_solver,
                    tol=self.ode_geo_tol,
                    n_max_iters=self.ode_geo_max_iters,                
                )
        elif self.manifold_type == c.CONST_CURVATURE_ORD2:
            ode_obj = OdeGeoConstOrd2(
                n_dims=self.n_dims,
                ode_params=ode_params,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_geo_solver,
                tol=self.ode_geo_tol,
                n_max_iters=self.ode_geo_max_iters,                
            )
        elif self.manifold_type == c.QUADRATIC:
            ode_obj = OdeGeoQuadratic(
                n_dims=self.n_dims,
                ode_params=ode_params,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_geo_solver,
                tol=self.ode_geo_tol, 
                n_max_iters=self.ode_geo_max_iters,               
            )
        else:
            assert False, "Unknow manidold type"

        s_flag = ode_obj.solve()

        if not s_flag:
            self.logger.warning("Geodesic equation did not converge!")
            return None

        return ode_obj.get_sol()

    def _get_adj_sol(self, params, geo_sol):

        n_steps = self.n_times - 1

        ode_params = self._get_ode_params(params)

        if self.ode_bc_mode == c.END_BC:
            bc_time = 0.0
            bk_flag = False
        else:
            bc_time = n_steps
            bk_flag = True

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            bc_vec = np.zeros(shape=(self.n_dims), dtype="d")
            if self.ode_geo_solver == "implicit_euler":
                adj_ode_obj = OdeAdjConstIEOrd1(
                    n_dims=self.n_dims,
                    ode_params=ode_params,
                    bc_vec=bc_vec,
                    time_inc=1.0,
                    n_steps=n_steps,
                    bk_flag=bk_flag,
                    tol=self.ode_adj_tol,
                    act_sol=self.act_sol,
                    adj_sol=geo_sol,
                    n_max_iters=self.ode_adj_max_iters,                
                )
            else:
                adj_ode_obj = OdeAdjConstOrd1(
                    n_dims=self.n_dims,
                    ode_params=ode_params,
                    bc_vec=bc_vec,
                    bc_time=bc_time,
                    time_inc=1.0,
                    n_steps=n_steps,
                    bk_flag=bk_flag,
                    intg_type=self.ode_adj_solver,
                    tol=self.ode_adj_tol,
                    act_sol=self.act_sol,
                    adj_sol=geo_sol,
                    n_max_iters=self.ode_adj_max_iters,                
                )                
        elif self.manifold_type == c.CONST_CURVATURE_ORD2:
            bc_vec = np.zeros(shape=(2 * self.n_dims), dtype="d")
            adj_ode_obj = OdeAdjConstOrd2(
                n_dims=self.n_dims,
                ode_params=ode_params,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_adj_solver,
                tol=self.ode_adj_tol,
                act_sol=self.act_sol,
                adj_sol=geo_sol,
                n_max_iters=self.ode_adj_max_iters,                
            )
        elif self.manifold_type == c.QUADRATIC:
            bc_vec = np.zeros(shape=(2 * self.n_dims), dtype="d")
            adj_ode_obj = OdeAdjQuadratic(
                n_dims=self.n_dims,
                ode_params=ode_params,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_adj_solver,
                tol=self.ode_adj_tol,
                act_sol=self.act_sol,
                adj_sol=geo_sol,
                n_max_iters=self.ode_adj_max_iters,                
            )
        else:
            assert False, "Unknow manifold type"

        s_flag = adj_ode_obj.solve()

        if not s_flag:
            self.logger.warning("Adjoint equation did not converge!")
            return None

        return adj_ode_obj.get_sol()

    def _get_ode_params(self, params):

        if self.manifold_type in [c.CONST_CURVATURE_ORD1, c.CONST_CURVATURE_ORD2]:
            return self._get_gamma_const(params)
        elif self.manifold_type == c.QUADRATIC:
            return params[: self.n_dims]
        else:
            assert False

    def _get_gamma_const(self, params):

        n_dims = self.n_dims
        n_times = self.n_times

        if self.manifold_type == c.CONST_CURVATURE_ORD1:
            n_gamma_vec = self.n_params - n_dims
        elif self.manifold_type == c.CONST_CURVATURE_ORD2:
            n_gamma_vec = self.n_params - 2 * n_dims

        gamma_vec = params[:n_gamma_vec]
        gamma = np.zeros(shape=(n_dims, n_dims, n_dims), dtype="d")

        gamma_id = 0
        for m in range(n_dims):
            for a in range(n_dims):
                for b in range(a, n_dims):

                    if self.diagonal_metric and m != a and m != b and a != b:
                        continue
                    elif (not self.self_relation) and m == a and a == b:
                        continue
                    else:
                        gamma[m][a][b] = gamma_vec[gamma_id]
                        gamma[m][b][a] = gamma_vec[gamma_id]
                        gamma_id += 1

        return gamma

    def _get_curvature_const(self, params):

        gamma = self._get_gamma_const(params)

        curve_const = np.tensordot(gamma, gamma, axes=((1), (0))) 
        curve_const = curve_const - np.swapaxes(curve_const, 1, 3)

        return curve_const
