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
import pickle as pk
import dill
import gc

from scipy.integrate import trapz
from scipy.optimize import line_search
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

sys.path.append( os.path.abspath( '../' ) )

from ode import OdeGeoConst, OdeAdjConst, OdeGeo, OdeAdj 

class GeodesicLearner():

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
        ode_bc_mode: str = "end",            
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
        |          Allowed values are "end" and "start". Default is "end".
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
        self.bc_vec = None
        self.X = None
        
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
        pass

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
        pass

    def _prep_params(self, X):
        
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
            self.n_params = n_times * int(n_dims * n_dims * (n_dims + 1) / 2)

        self.n_params += n_dims

        # Initialize parameters and set BC vec
        self.params = np.zeros(shape=(self.n_params), dtype="d")
        self.bc_vec = np.zeros(shape=(self.n_dims), dtype="d")

        n_gamma_vec = self.n_params - self.n_dims
        
        for m in range(self.n_dims):
            if self.ode_bc_mode == "end":
                self.bc_vec[m] = X[self.n_times-1][m]
            elif self.ode_bc_mode == "start":
                self.bc_vec[m] = X[0][m]
            else:
                assert False
                
            self.params[m + n_gamma_vec] = self.bc_vec[m]
        
    def get_params(self):
        pass
    
