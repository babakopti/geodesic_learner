# Import libraries
import sys
import os
import math
import time
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import gc

from scipy.integrate import trapz

from ode import OdeGeoConst, OdeAdjConst, OdeGeo, OdeAdj


class GeodesicLearner:
    def __init__(
        self,
        manifold_type: str = "const_curvature",
        opt_method: str = "SLSQP",
        maxOptItrs: int = 100,
        opt_tol: float = 1.0e-8,
        ode_geo_solver: str = "LSODA",
        ode_adj_solver: str = "RK45",
        ode_geo_tol: float = 1.0e-2,
        ode_adj_tol: float = 1.0e-2,
        ode_bc_mode: str = "end_bc",
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
        |          Type of the manifold. Currently, "constant_curvature" and
        |          "free" are supported. Default value is "constant_curvature".
        |
        |      opt_method: str
        |          Optimization method used. Currently only "SLSQP" is recommended.
        |          Default is "SLSQP".
        |
        |      maxOptItrs: int
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
        |      ode_bc_mode: str
        |          Mode of boundary condition during training.
        |          Allowed values are "end_bc" and "start_bc". Default is "end_bc".
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
        self.opt_method = opt_method
        self.max_opt_itrs = max_opt_itrs
        self.opt_tol = opt_tol
        self.ode_geo_solver = ode_geo_solver
        self.ode_adj_solver = ode_adj_solver
        self.ode_geo_tol = ode_geo_tol
        self.ode_adj_tol = ode_adj_tol
        self.ode_bc_mode = ode_bc_mode
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
        start_time: float,
        end_time: float,
        time_inc: float,
        bc_vec: np.array,
    ):
        """
        Predict using geodesic model.
        |
        |      Parameters
        |      ----------
        |      start_time : float
        |          Start time for ODE solution.
        |
        |      end_time : float
        |          End time for ODE solution.
        |
        |      n_times : int
        |          Number of times in (start_time, end_time) interval.
        |
        |      bc_vec : array, shape = [n_dims]
        |          The boundary condition array imposed at "start_time".
        |
        |      Returns
        |      -------
        |      array, shape = (n_times, n_dims).
        """
        assert self.params is not None

    def _init_params(self, X):

        n_times = X.shape[0]
        n_dims = X.shape[1]

        self.n_times = n_times
        self.n_dims = n_dims

        # Calculate number of model parameters
        if self.manifold_type == "const_curvature":

            if self.diagonal_metric:
                self.n_params = n_dims * (2 * n_dims - 1)
            else:
                self.n_params = int(n_dims * n_dims * (n_dims + 1) / 2)

            if not self.self_relation:
                self.n_params -= n_dims

        if self.manifold_type == "free":
            self.n_params = n_times * int(
                n_dims * n_dims * (n_dims + 1) / 2
            )

        self.n_params += n_dims

        # Initialize parameters and set BC vec
        self.params = np.zeros(shape=(self.n_params), dtype="d")

        n_gamma_vec = self.n_params - self.n_dims

        for m in range(self.n_dims):
            if self.ode_bc_mode == "end_bc":
                self.params[m + n_gamma_vec] = X[self.n_times - 1][m]
            elif self.ode_bc_mode == "start_bc":
                self.params[m + n_gamma_vec] = X[0][m]
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
        }

        try:
            bounds = [(-1.0, 1.0) for i in range(self.n_params)]
            opt_obj = scipy.optimize.minimize(
                fun=self._get_obj_func,
                x0=self.params,
                method=self.opt_type,
                jac=self._get_grad,
                bounds=bounds,
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
        n_gamma_vec = self.n_params - self.n_dims
        tmp1 = np.linalg.norm(params[:n_gamma_vec], 1)
        tmp2 = np.linalg.norm(params[:n_gamma_vec])
        val += self.alpha * (
            self.l1_ratio * tmp1 + (1.0 - self.l1_ratio) * tmp2 ** 2
        )

        del sol
        del tmp_vec

        gc.collect()

        return val

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

        gamma_id = 0
        for r in range(n_dims):
            for p in range(n_dims):
                for q in range(p, n_dims):

                    if (
                        self.diagonal_metric
                        and r != p
                        and r != q
                        and p != q
                    ):
                        continue
                    if (not self.self_relation) and r == p and p == q:
                        continue

                    tmp_vec = xi(p, q) * np.multiply(sol[p][:], sol[q][:])
                    tmp_vec = np.multiply(tmp_vec, adj_sol[r][:])

                    grad[gamma_id] = trapz(
                        tmp_vec, dx=time_inc
                    ) + alpha * (
                        l1_ratio * np.sign(params[gamma_id])
                        + (1.0 - l1_ratio) * 2.0 * params[gamma_id]
                    )

                    gamma_id += 1

        del sol
        del tmp_vec

        gc.collect()

        n_gamma_vec = self.n_params - n_dims
        gamma_vec = params[:n_gamma_vec]

        tmp1 = np.linalg.norm(gamma_vec)
        tmp2 = np.linalg.norm(grad)

        fct = 1.0
        if tmp2 > 0:
            fct = min(
                1.0, math.sqrt(abs(tmp1 ** 2 - n_gamma_vec) / tmp2 ** 2)
            )

        for i in range(n_dims):
            if self.ode_bc_mode == "end_bc":
                grad[i + n_gamma_vec] = adj_sol[i][-1]
            else:
                grad[i + n_gamma_vec] = -adj_sol[i][0]

        grad = fct * grad

        del adj_sol

        gc.collect()
        
        return grad

    def _get_geo_sol(self, params):

        n_steps = self.n_times - 1

        if self.ode_bc_mode == "end_bc":
            bc_time = n_steps
            bk_flag = True
        else:
            bc_time = 0.0
            bk_flag = False

        n_gamma_vec = self.n_params - self.n_dims
        gamma_vec = params[:n_gamma_vec]
        bc_vec = params[n_gamma_vec:]

        gamma = self._get_gamma(gamma_vec)

        if self.manifold_type == "const_curvature":
            ode_obj = OdeGeoConst(
                gamma=gamma,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_geo_solver,
                tol=self.ode_geo_tol,
            )
        else:
            ode_obj = OdeGeo(
                gamma=gamma,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_geo_solver,
                tol=self.ode_geo_tol,
            )

        s_flag = ode_obj.solve()

        if not s_flag:
            self.logger.warning("Geodesic equation did not converge!")
            return None

        return ode_obj.get_sol()

    def _get_adj_sol(self, params, geo_sol):

        n_steps = self.n_times - 1

        n_gamma_vec = self.n_params - self.n_dims
        gamma_vec = params[:n_gamma_vec]
        gamma = self._get_gamma(gamma_vec)
        bc_vec = np.zeros(shape=(self.n_dims), dtype="d")

        if self.ode_bc_mode == "end_bc":
            bc_time = 0.0
            bk_flag = False
        else:
            bc_time = n_steps
            bk_flag = True

        if self.manifold_type == "const_curvature":
            adj_ode_obj = OdeAdjConst(
                gamma=gamma,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_adj_solver,
                tol=self.ode_adj_tol,
                act_sol=self.act_sol,
                adj_sol=geo_sol,
            )
        else:
            adj_ode_obj = OdeAdj(
                gamma=gamma,
                bc_vec=bc_vec,
                bc_time=bc_time,
                time_inc=1.0,
                n_steps=n_steps,
                bk_flag=bk_flag,
                intg_type=self.ode_adj_solver,
                tol=self.ode_adj_tol,
                act_sol=self.act_sol,
                adj_sol=geo_sol,
            )

        s_flag = adj_ode_obj.solve()

        if not s_flag:
            self.logger.warning("Adjoint equation did not converge!")
            return None

        return adj_ode_obj.get_sol()

    def _get_gamma(self, gamma_vec):

        if self.manifold_type == "const_curvature":
            return self._get_gamma_const(gamma_vec)
        else:
            return self._get_gamma_var(gamma_vec)

    def _get_gamma_const(self, gamma_vec):

        n_dims = self.n_dims
        n_times = self.n_times
        gamma = np.zeros(shape=(n_dims, n_dims, n_dims), dtype="d")

        gamma_id = 0
        for m in range(n_dims):
            for a in range(n_dims):
                for b in range(a, n_dims):

                    if (
                        self.diagonal_metric
                        and m != a
                        and m != b
                        and a != b
                    ):
                        continue
                    elif (not self.srel_relation) and m == a and a == b:
                        continue
                    else:
                        gamma[m][a][b] = gamma_vec[gamma_id]
                        gamma[m][b][a] = gamma_vec[gamma_id]
                        gamma_id += 1

        return gamma

    def _get_gamma_var(self, gamma_vec):

        n_dims = self.n_dims
        n_times = self.n_times
        gamma = np.zeros(
            shape=(n_times, n_dims, n_dims, n_dims), dtype="d"
        )

        assert n_times > 0, "Number of times should be positive!"

        n_tmp = int(len(gamma_vec) / n_times)

        gamma_id = 0
        for m in range(n_dims):
            for a in range(n_dims):
                for b in range(a, n_dims):

                    if (
                        self.diagonal_metric
                        and m != a
                        and m != b
                        and a != b
                    ):
                        continue
                    elif (not self.self_relation) and m == a and a == b:
                        continue
                    else:
                        for ts_id in range(self.n_times):
                            gamma[ts_id][m][a][b] = gamma_vec[
                                ts_id * n_tmp + gamma_id
                            ]
                            gamma[ts_id][m][b][a] = gamma_vec[
                                ts_id * n_tmp + gamma_id
                            ]

                    gamma_id += 1

        return gamma
