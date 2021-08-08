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
xi0 = -0.2
eta0 = 0.0

alpha = -1.0
beta = 1.0

params = [alpha, beta]

# Use quadratic ODE solver
bc_vec = np.array([u0, v0, xi0, eta0])
ode_obj = OdeGeoQuadratic(
    n_dims=n_dims,
    ode_params=params,
    bc_vec=bc_vec,
    bc_time=0.0,
    time_inc=1.0,
    n_steps=n_steps,
    bk_flag=False,
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

# Train
