"""
Validasi rule B — tahap 1: data kebenaran.
100 konfigurasi per algoritma (sampel acak dari 500 konfigurasi random search, rs_configs.csv).
Tiap konfigurasi: 100 run "kebenaran" (seed 1000–1099) + 50 run "pemilihan" (seed 2000–2049),
dua kumpulan seed yang terpisah agar estimasi pemilihan tidak bias terhadap kebenaran.
Output: results/ruleB_runs.csv (alg, cfg, seed, pool, hv_norm, hv_far)
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
import common as c

OUT = c.ROOT / "results"
N_CFG = 100
TRUTH = list(range(1000, 1100))
SELECT = list(range(2000, 2050))
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]


def hp_of(row):
    keys = ["F", "CR", "eta", "p_m", "delta"] if row["alg"] == "MOEA/D-DE" else ["eta_cr", "p_cr", "eta_m", "p_m"]
    return {k: row[k] for k in keys}


def job(row):
    out = []
    for pool, seeds in (("truth", TRUTH), ("select", SELECT)):
        for s in seeds:
            _, h, _, F = c.run(row["alg"], int(row["pop_size"]), int(row["n_iters"]), s, **hp_of(row))
            out.append(dict(alg=row["alg"], cfg=row["cfg"], seed=s, pool=pool, hv_norm=h, hv_far=c.hv_far(F)))
    return out


if __name__ == "__main__":
    cfgs = pd.read_csv(OUT / "rs_configs.csv")
    rng = np.random.default_rng(7)
    sub = pd.concat([g.loc[rng.choice(g.index, N_CFG, replace=False)] for _, g in cfgs.groupby("alg", sort=False)])
    sub.to_csv(OUT / "ruleB_configs.csv", index=False)
    with Pool() as p:
        res = p.map(job, sub.to_dict("records"), chunksize=2)
    pd.DataFrame([r for chunk in res for r in chunk]).to_csv(OUT / "ruleB_runs.csv", index=False)
    print("selesai:", sum(len(x) for x in res), "run")
