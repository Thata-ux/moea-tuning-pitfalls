"""
Q2-D — Verifikasi penurunan: mengapa titik referensi yang jauh memangkas n_runs.

Untuk front 2 objektif A (ternormalisasi) dan titik referensi r = (R, R) di luar front:
    HV(A; R) = (R - m1)(R - m2) - U(A)
dengan m_k = min_A f_k (titik ekstrem) dan U(A) = luas tak-terdominasi di dalam kotak
[m1, u1] x [m2, u2] (tidak bergantung pada R). Maka
    HV = R^2 - R (m1 + m2) + (m1 m2 - U)
    E[HV] ~ R^2,  SD[HV] -> R * SD(m1 + m2)   untuk R besar
    CV ~ SD(m1 + m2) / R   =>   n_runs (aturan A) ~ (z CV / e)^2 ~ R^-2
Untuk R besar, HV dan peringkatnya ditentukan oleh m1 + m2, yaitu kedua titik ekstrem front.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm as N, kendalltau
from pymoo.indicators.hv import HV
import common as c

OUT = c.ROOT / "results"
fronts = np.load(OUT / "baseline_final_fronts.npy", allow_pickle=True)
runs = pd.read_csv(OUT / "baseline_runs.csv")
Z = N.ppf(0.975)
rows, slopes = [], []
for a in ["NSGA2", "SPEA2", "SMS-EMOA"]:
    Fs = [c.norm(c.nd(F)) for F, al in zip(fronts, runs.alg) if al == a]
    m = np.array([F.min(0) for F in Fs])                  # m1, m2 per run
    ns = []
    for R in [1.1, 1.5, 2, 3, 5, 10, 20, 50]:
        hv = np.array([HV(ref_point=np.array([R, R]))(F) for F in Fs])
        # identitas: U = (R-m1)(R-m2) - HV harus konstan terhadap R
        U = (R - m[:, 0]) * (R - m[:, 1]) - hv
        cv_obs = hv.std(ddof=1) / hv.mean()
        cv_pred = (m.sum(1)).std(ddof=1) / R              # aproksimasi orde pertama
        n_obs = (Z * cv_obs / 0.002) ** 2
        ns.append((R, n_obs))
        rows.append(dict(alg=a, R=R, cv_obs=cv_obs, cv_pred_large_R=cv_pred, n_runs_obs=n_obs,
                         U_mean=U.mean(), tau_hv_vs_extremes=kendalltau(hv, -m.sum(1)).statistic))
    lg = np.log([x for x in ns if x[0] >= 5])
    slopes.append(dict(alg=a, slope_log_n_vs_log_R=np.polyfit(lg[:, 0], lg[:, 1], 1)[0]))

df = pd.DataFrame(rows)
df.to_csv(OUT / "theory_check.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(5).to_string(index=False))
sl = pd.DataFrame(slopes)
sl.to_csv(OUT / "theory_slopes.csv", index=False)
print("\nKemiringan log n_runs vs log R (R >= 5); teori = -2:")
print(sl.round(3).to_string(index=False))
# U harus identik antar-R (cek identitas)
chk = df.groupby("alg").U_mean.agg(["min", "max"])
print("\nCek identitas U (min = maks per algoritma):\n", chk.round(8).to_string())
