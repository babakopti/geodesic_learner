import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.append("..")

from models import GeodesicLearner
from ode import OdeGeoConstOrd1

# Instantiate
geodesic_learner = GeodesicLearner(
    manifold_type="const_curvature_first_order",
    opt_method="SLSQP",
    max_opt_iters=300,
    opt_tol=1.0e-8,
    ode_geo_solver="LSODA",
    ode_adj_solver="RK45",
    ode_geo_tol=1.0e-6,
    ode_adj_tol=1.0e-6,
    ode_bc_mode="end_bc",
    alpha=1.0e-6,
    l1_ratio=0.0,
    diagonal_metric=True,
    self_relation=False,
    verbose=True,
)

# Generate X
n_dims = 2
n_steps = 100
n_gamma_vec = n_dims * (2 * n_dims - 1) - n_dims
gamma_vec = np.random.uniform(low=-0.01, high=0.01, size=(n_gamma_vec,))
params = np.concatenate([gamma_vec, np.zeros(shape=(n_dims), dtype="d")])
geodesic_learner.n_dims = n_dims
geodesic_learner.n_params = n_gamma_vec + n_dims
gamma1 = geodesic_learner._get_gamma(params)
bc_vec = np.ones(shape=(n_dims))

ode_obj = OdeGeoConstOrd1(
    n_dims=n_dims,
    ode_params=gamma1,
    bc_vec=bc_vec,
    bc_time=n_steps,
    time_inc=1.0,
    n_steps=n_steps,
    bk_flag=True,
    intg_type="LSODA",
    tol=1.0e-6,
)

s_flag = ode_obj.solve()

assert s_flag

X = ode_obj.get_sol().transpose()

geodesic_learner.n_dims = None
geodesic_learner.n_params = None

# Fit
geodesic_learner.fit(X)

gamma2 = geodesic_learner._get_gamma(geodesic_learner.params)

print("Gamma Diff:", np.linalg.norm(gamma1-gamma2)/np.linalg.norm(gamma1))

X_prd = geodesic_learner.predict_train()

act_sol = X.transpose()
prd_sol = X_prd.transpose()

for m in range(n_dims):
    plt.plot(act_sol[m], "b.")
    plt.plot(prd_sol[m], "r")
    plt.legend(["Actual", "Predicted"])
    plt.show()
