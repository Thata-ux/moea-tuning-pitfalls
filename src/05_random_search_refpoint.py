"""
Eksperimen 1 — Apakah pilihan titik referensi HV mengubah HASIL HPO?

Random search 500 konfigurasi per algoritma (ruang hyperparameter Bagian 4.1 artikel),
10 run per konfigurasi dengan seed bersama (common random numbers).
Setiap front akhir dinilai dengan dua konvensi:
  - NORM: objektif dinormalisasi (ideal/nadir front sebenarnya), ref = (1,1; 1,1)
  - FAR : objektif mentah, ref = titik terjauh ruang desain (konvensi penulis)
Pembanding derau: reliabilitas split-half (run 0–4 vs 5–9) dalam konvensi yang sama.
"""
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import common as c

N_CFG, N_RUNS = 500, 10
OUT = c.ROOT / "results"
POP_ITERS = [(p, 300 // p) for p in (5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100)]
GRID_GA = dict(eta_cr=np.arange(5, 31, 1), p_cr=np.round(np.arange(0.7, 0.901, 0.05), 2),
               eta_m=np.arange(5, 51, 1), p_m=np.round(np.arange(0.05, 0.501, 0.05), 2))
GRID_DE = dict(F=np.round(np.arange(0.3, 9.001, 0.1), 1), CR=np.round(np.arange(0.1, 1.001, 0.1), 1),
               eta=np.arange(10, 51, 1), p_m=np.round(np.arange(0.05, 0.501, 0.05), 2),
               delta=np.round(np.arange(0.5, 0.901, 0.1), 1))
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]


def sample_configs(alg, rng):
    grid = GRID_DE if alg == "MOEA/D-DE" else GRID_GA
    rows = []
    for k in range(N_CFG):
        pop, it = POP_ITERS[rng.integers(len(POP_ITERS))]
        hp = {name: rng.choice(vals).item() for name, vals in grid.items()}
        rows.append(dict(alg=alg, cfg=k, pop_size=pop, n_iters=it, **hp))
    return rows


def job(row):
    hp = {k: row[k] for k in (GRID_DE if row["alg"] == "MOEA/D-DE" else GRID_GA)}
    out = []
    for s in range(N_RUNS):
        _, h, i, F = c.run(row["alg"], int(row["pop_size"]), int(row["n_iters"]), s, **hp)
        out.append(dict(alg=row["alg"], cfg=row["cfg"], seed=s, hv_norm=h, hv_far=c.hv_far(F), igd=i))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    cfgs = pd.DataFrame([r for a in ALGS for r in sample_configs(a, rng)])
    cfgs.to_csv(OUT / "rs_configs.csv", index=False)
    with Pool() as p:
        res = p.map(job, cfgs.to_dict("records"), chunksize=4)
    runs = pd.DataFrame([r for chunk in res for r in chunk])
    runs.to_csv(OUT / "rs_runs.csv", index=False)
    print("selesai:", len(runs), "run")
