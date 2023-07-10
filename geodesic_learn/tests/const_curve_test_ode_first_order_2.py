import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split, KFold,
)

sys.path.append("..")

from models import GeodesicLearner

# df = pd.read_pickle("data/dfFile_5assets_all_dates.pkl")

# df["Date"] = df["Date"].apply(lambda x: pd.to_datetime(x).date())
# df = df.groupby("Date", as_index=False).mean()
# df = df.interpolate(method="linear")
# df = df.dropna()

col_names = [
    "SPY",
    "MVV",
    "AGQ",
    "BOIL",
    "UST",
]

#X = np.array(df[col_names])

#X_train, X_test = train_test_split(X, test_size=0.05, shuffle=False)

#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

X_train = np.load(open("data/X_train.npy", "rb"))
X_test = np.load(open("data/X_test.npy", "rb"))

geodesic_learner = GeodesicLearner(
    manifold_type="const_curvature_first_order",
    opt_method="SLSQP",
    max_opt_iters=300,
    opt_tol=1.0e-8,
    ode_geo_solver="implicit_euler",
    ode_adj_solver="implicit_euler",
    ode_geo_tol=1.0e-6,
    ode_adj_tol=1.0e-2,
    ode_geo_max_iters=200,
    ode_adj_max_iters=200,
    ode_bc_mode="end_bc",
    learning_rate=None,
    alpha=10.0,
    l1_ratio=0.0,
    diagonal_christoffel=False,   
    diagonal_metric=True,
    self_relation=False,
    verbose=True,
)

geodesic_learner.fit(X_train)

print(geodesic_learner.params)

X_train_prd = geodesic_learner.predict_train()
X_test_prd = geodesic_learner.predict(
    bc_vec=X_test[0],
    n_steps=X_test.shape[0] - 1,
)
np.save(open("X_train_prd.npy", "wb"), X_train_prd)
np.save(open("X_test_prd.npy", "wb"), X_test_prd)

act_vals_train = X_train.transpose() #scaler.inverse_transform(X_train).transpose()
act_vals_test = X_test.transpose() #scaler.inverse_transform(X_test).transpose()
prd_vals_train = X_train_prd.transpose() #scaler.inverse_transform(X_train_prd).transpose()
prd_vals_test = X_test_prd.transpose() #scaler.inverse_transform(X_test_prd).transpose()    

test_scores = []
for m in range(len(col_names)):
    y_true_train = act_vals_train[m]
    y_true_test = act_vals_test[m]    
    y_pred_train = prd_vals_train[m]
    y_pred_test = prd_vals_test[m]    
    
    print(
        r2_score(y_true_train, y_pred_train),
        r2_score(y_true_test, y_pred_test)
    )
    test_scores.append(r2_score(y_true_test, y_pred_test))

print("Average test score:", np.mean(test_scores))    
sys.exit()
for m in range(len(col_names)):    
    plt.plot(act_vals_train[m], "b")
    plt.plot(prd_vals_train[m], "r")    
    plt.xlabel("time")
    plt.ylabel(col_names[m])
    plt.show()    

    plt.plot(prd_vals_test[m], "g")
    plt.plot(act_vals_test[m], "b")
    plt.xlabel("time")
    plt.ylabel(col_names[m])    
    plt.show()

    
