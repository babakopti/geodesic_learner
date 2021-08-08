import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapz

sys.path.append("..")

from models import GeodesicLearner
from ode import OdeGeoQuadratic

# z = 0.5 * alpha * x^2 + 0.5 * beta * y^2
n_dims = 2
n_times = 50
n_steps = n_times - 1

u0 = 8.0
v0 = 0.0
xi0 = 0.2
eta0 = 0.0

alpha = -1.0
beta = 1.0

params1 = [alpha, beta]

# Generate training data
bc_vec = np.array([u0, v0, xi0, eta0])
ode_obj = OdeGeoQuadratic(
    n_dims=n_dims,
    ode_params=params1,
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

sol = ode_obj.get_sol()
us = sol[0][:]
vs = sol[1][:]
xis = sol[2][:]
etas = sol[3][:]

# Plot surface and geodesic curve
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = alpha * X**2 + beta * Y**2
zs = alpha * us**2 + beta * vs**2
ax.plot(us, vs, zs, "r", linewidth=2)
ax.plot_surface(
    X, Y, Z, linewidth=1, antialiased=True, alpha=0.7,
)

plt.show()

X = np.empty(shape=(n_times, n_dims))
for m in range(n_dims):
    for ts_id in range(n_times):
        X[ts_id][m] = sol[m][ts_id]

# Train
geodesic_learner = GeodesicLearner(
    manifold_type="quadratic_surface",
    opt_method="SLSQP",
    max_opt_iters=300,
    opt_tol=1.0e-8,
    ode_geo_solver="LSODA",
    ode_adj_solver="RK45",
    ode_geo_tol=1.0e-6,
    ode_adj_tol=1.0e-6,
    ode_bc_mode="end_bc",
    alpha=1.0e-8,
    l1_ratio=0.0,
    diagonal_metric=True,
    self_relation=False,
    verbose=True,
)
geodesic_learner.fit(X)

params2 = geodesic_learner.params

print("params1:", params1)
print("params2:", params2)
