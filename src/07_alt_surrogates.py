"""
Eksperimen 3a — Ruang desain alternatif dari surrogate lain (uji ketahanan).
  space_rf.csv  : Random Forest (500 pohon)
  space_rbf.csv : RBF thin-plate spline (scipy), input distandarkan
Target dalam log, metrik dihitung pada skala asli (split 90/10 sama dengan tahap 1).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import common as c

D = c.ROOT / "data"
d = pd.read_csv(D / "dataset.csv")
d.columns = ["mat", "r_idx", "n", "h_idx", "disp", "cost"]
d["mat"] -= 1
grid = pd.read_csv(D / "space.csv")[["mat", "r_idx", "n", "h_idx", "r", "h"]]


def feats(df):
    X = np.column_stack([df["r_idx"] / 8, (df["n"] - 4) / 8, df["h_idx"] / 25]
                        + [(df["mat"] == m).astype(float) for m in range(6)])
    return X


X, Xg = feats(d), feats(grid)
makers = {
    "rf": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=0, n_jobs=-1),
    "rbf": None,
}
rows = []
for name in ["rf", "rbf"]:
    out = grid.copy()
    for t in ["disp", "cost"]:
        y = np.log(d[t].values)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.10, random_state=42)
        if name == "rf":
            fit = lambda A, b: makers["rf"]().fit(A, b).predict
        else:
            fit = lambda A, b: RBFInterpolator(A, b, kernel="thin_plate_spline", smoothing=1e-3)
        f = fit(Xtr, ytr)
        p, tt = np.exp(f(Xte)), np.exp(yte)
        rows.append(dict(surrogate=name, objective=t, RMSE=np.sqrt(mean_squared_error(tt, p)), R2_test=r2_score(tt, p)))
        out[t] = np.exp(fit(X, y)(Xg))
    out.to_csv(D / f"space_{name}.csv", index=False)
m = pd.DataFrame(rows)
m.to_csv(c.ROOT / "results" / "alt_surrogate_metrics.csv", index=False)
print(m.round(5).to_string(index=False))
