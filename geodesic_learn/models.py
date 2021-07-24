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
        self._prep(X)
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

    def _prep(self, X):

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

        self.act_sol = X.transpose()
        
    def _set_params(self):

        t0 = time.time()
        
        self.logger.info(
            "Running continuous adjoint optimization to set Christoffel symbols..."
        )

        options  = {
            "ftol": self.opt_tol,
            'maxiter': self.max_opt_iters,
            "disp": True,
        }

        try:
            bounds = [(-1.0, 1.0) for i in range(self.n_params)]
            optObj = scipy.optimize.minimize(
                fun=self._get_obj_func, 
                x0=self.params,
                method=self.opt_type, 
                jac=self._get_grad,
                bounds=bounds,
                options=options,
            )
            sFlag = optObj.success
    
            self.params = optObj.x

            self.logger.info("Success: %s", str(sFlag))

        except Exception as exc:
            self.logger.error(exc)
            sFlag = False

        self.logger.info(
            "Setting parameters took %0.2f seconds.", 
            time.time() - t0,
        )
    
    def _get_obj_func(self, params):

        # Get solution
        sol = self._get_geo_sol(params)
        
        if sol is None:
            return np.inf

        # Calculate the objective function
        tmp_vec = np.zeros(shape=(self.n_times))
        for m in range(self.n_dims):
            tmp_vec += (sol[m][:] - act_sol[m][:])**2 

        val = 0.5 * trapz(tmp_vec, dx=1.0)

        # Add regularization term based on gamma
        n_gamma_vec = self.n_params - self.n_dims
        tmp1 = np.linalg.norm(params[:n_gamma_vec], 1)
        tmp2 = np.linalg.norm(params[:n_gamma_vec])
        val += self.alpha * (
            self.l1_ratio * tmp1 + (1.0 - self.l1_ratio) * tmp2**2
        )

        del sol
        del tmp_vec
        
        gc.collect()

        return val

    def _get_geo_sol(self, params):

        nDims    = self.nDims
        nSteps   = self.nSteps

        if self.endBcFlag:
            bcTime = nSteps   
        else:
            bcTime = 0.0

        nGammaVec = self.nParams - nDims
        GammaVec = params[:nGammaVec]
        bcVec = params[nGammaVec:]
        
        Gamma = self.getGammaArray(GammaVec)

        self.logger.debug( 'Solving geodesic...' )

        odeObj   = OdeGeoConst( Gamma    = Gamma,
                                bcVec    = bcVec,
                                bcTime   = bcTime,
                                timeInc  = 1.0,
                                nSteps   = self.nSteps,
                                intgType = 'LSODA',
                                tol      = GEO_TOL,
                                srcCoefs = self.srcCoefs,
                                srcTerm  = self.srcTerm,
                                verbose  = self.verbose       )

        sFlag = odeObj.solve()

        if not sFlag:
            self.logger.warning( 'Geodesic equation did not converge!' )
            return None
        
        return odeObj
    
    def _get_adj_sol( self, params, ode_obj ):

        nDims    = self.nDims

        nGammaVec = self.nParams - nDims
        GammaVec = params[:nGammaVec]
        
        Gamma    = self.getGammaArray( GammaVec )
        sol      = odeObj.getSol()
        bcVec    = np.zeros( shape = ( nDims ), dtype = 'd' )
        bkFlag   = not self.endBcFlag

        self.logger.debug( 'Solving adjoint geodesic equation...' )

        adjOdeObj = OdeAdjConst( Gamma    = Gamma,
                                 bcVec    = bcVec,
                                 bcTime    = 0.0,
                                 timeInc   = 1.0,
                                 nSteps    = self.nSteps,
                                 intgType  = 'RK45',
                                 actSol    = self.actSol,
                                 adjSol    = sol,
                                 tol       = ADJ_TOL,
                                 varCoefs  = self.varCoefs,
                                 atnCoefs  = self.atnCoefs,
                                 verbose   = self.verbose       )

        sFlag  = adjOdeObj.solve()

        if not sFlag:
            self.logger.warning( 'Adjoint equation did not converge!' )
            return None

        self.statHash[ 'adjOdeTime' ] += time.time() - t0

        self.logger.debug( 'Adjoint equation: %0.2f seconds.', 
                           time.time() - t0 ) 

        return adjOdeObj
