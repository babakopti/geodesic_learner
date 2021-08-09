import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.append("..")

from models import GeodesicLearner

df = pd.read_pickle("data/dfFile_2021-02-01 09:30:04.pkl")

df["Date"] = df["Date"].apply(lambda x: pd.to_datetime(x).date())
df = df.groupby("Date", as_index=False).mean()
df = df.interpolate(method="linear")
df = df.dropna()
min_trn_date = pd.to_datetime("2020-02-05 09:00:00")
max_trn_date = pd.to_datetime("2020-12-31 09:00:00")

df = df[(df.Date >= min_trn_date) & (df.Date <= max_trn_date)]

col_names = [
    "SPY",
    "MVV",
    "AGQ",
    "BOIL",
    "UST",
]

X = np.array(df[col_names])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#X = np.load("data/X_test_1.npy")

geodesic_learner = GeodesicLearner(
    manifold_type="const_curvature_first_order",
    opt_method="SLSQP",
    max_opt_iters=300,
    opt_tol=1.0e-8,
    ode_geo_solver="LSODA",
    ode_adj_solver="RK45",
    ode_geo_tol=1.0e-2,
    ode_adj_tol=1.0e-2,
    ode_bc_mode="end_bc",
    learning_rate=1.0e-4,
    alpha=1.0e-8,
    l1_ratio=0.0,
    diagonal_metric=True,
    self_relation=False,
    verbose=True,
)

geodesic_learner.fit(X)

print(geodesic_learner.params)

X_prd = geodesic_learner.predict_train()

act_sol = X.transpose()
prd_sol = X_prd.transpose()

for m in range(len(col_names)):
    plt.plot(act_sol[m], "b")
    plt.plot(prd_sol[m], "r")
    plt.xlabel("time")
    plt.ylabel(col_names[m])
    plt.show()
