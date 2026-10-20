"""
Tahap 1 — Data, surrogate, dan ruang desain lengkap.

Input : dataset.csv dari Pereira & Leitao (2026), Zenodo doi:10.5281/zenodo.18344135 (CC BY 4.0)
Output: data/space.csv        -> 12.636 desain + prediksi kedua objektif (surrogate)
        data/pareto_true.csv  -> Pareto front "sebenarnya" (berbasis surrogate)
        results/surrogate_metrics.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "dataset.csv"
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

# ---------- 1. Data (kode integer, sesuai README penulis) ----------
d = pd.read_csv(RAW)
d.columns = ["mat", "r_idx", "n", "h_idx", "disp", "cost"]
d["mat"] = d["mat"] - 1                        # 0..5
d["r"] = 0.035 + 0.005 * d["r_idx"]            # m
d["h"] = 0.5 + 0.05 * d["h_idx"]               # m
print(f"Data: {len(d)} baris, {d[['mat','r_idx','n','h_idx']].drop_duplicates().shape[0]} desain unik")


def features(df):
    X = pd.DataFrame({"r": df["r"], "n": df["n"], "h": df["h"]})
    for m in range(6):
        X[f"mat_{m}"] = (df["mat"] == m).astype(float)
    return X.values


X = features(d)
# Target di-log: displacement & biaya bersifat multiplikatif (biaya = volume x harga material)
targets = {"disp": np.log(d["disp"].values), "cost": np.log(d["cost"].values)}

# ---------- 2. Surrogate (GBT) + evaluasi 90/10 dan 5-fold CV ----------
rows, models = [], {}
for name, y in targets.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.10, random_state=42)
    gbt = GradientBoostingRegressor(n_estimators=800, max_depth=4, learning_rate=0.05,
                                    subsample=0.9, random_state=42)
    gbt.fit(Xtr, ytr)
    p, t = np.exp(gbt.predict(Xte)), np.exp(yte)    # metrik di skala asli
    cv = cross_val_score(gbt, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    rows.append(dict(objective=name, MSE=mean_squared_error(t, p),
                     RMSE=np.sqrt(mean_squared_error(t, p)), MAE=mean_absolute_error(t, p),
                     R2_test=r2_score(t, p), R2_cv5_mean=cv.mean(), R2_cv5_std=cv.std()))
    gbt.fit(X, y)                                     # model final: semua data
    models[name] = gbt
met = pd.DataFrame(rows)
met.to_csv(ROOT / "results" / "surrogate_metrics.csv", index=False)
print(met.round(5).to_string(index=False))

# ---------- 3. Evaluasi seluruh ruang desain ----------
grid = pd.DataFrame([(m, ri, n, hi) for m in range(6) for ri in range(9)
                     for n in range(4, 13) for hi in range(26)],
                    columns=["mat", "r_idx", "n", "h_idx"])
grid["r"] = 0.035 + 0.005 * grid["r_idx"]
grid["h"] = 0.5 + 0.05 * grid["h_idx"]
Xg = features(grid)
grid["disp"] = np.exp(models["disp"].predict(Xg))
grid["cost"] = np.exp(models["cost"].predict(Xg))
grid.to_csv(ROOT / "data" / "space.csv", index=False)
print(f"Ruang desain: {len(grid)} desain dievaluasi")

# ---------- 4. Pareto front sebenarnya ----------
F = grid[["disp", "cost"]].values
front = NonDominatedSorting().do(F, only_non_dominated_front=True)
pf = grid.iloc[front].sort_values("disp")
pf.to_csv(ROOT / "data" / "pareto_true.csv", index=False)
print(f"Pareto front sebenarnya: {len(pf)} desain")
print("Material pada front:", pf["mat"].value_counts().sort_index().to_dict())
