"""
Usulan: eps relatif terhadap SEBARAN HV antarkonfigurasi (bukan rata-rata HV).

n_runs = (z * sigma_dalam / (k * sigma_antar))^2
  sigma_dalam : SD HV antar-run pada konfigurasi default (100 run, baseline)
  sigma_antar : SD rata-rata HV antar 500 konfigurasi random search
  k           : resolusi yang diinginkan (mis. 0,1 = mampu membedakan 1/10 sebaran antarkonfigurasi)
Dibandingkan dengan aturan artikel (eps = % rata-rata HV) pada dua titik referensi.
Juga: reliabilitas peringkat yang sebenarnya dicapai dengan n run (Spearman-Brown dari split-half).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm as N, kendalltau
from pymoo.indicators.hv import HV
import common as c

OUT = c.ROOT / "results"
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
Z = N.ppf(0.975)
base = pd.read_csv(OUT / "baseline_runs.csv")
fronts = np.load(OUT / "baseline_final_fronts.npy", allow_pickle=True)
hv_far_base = np.array([c.hv_far(F) for F in fronts])
rs = pd.read_csv(OUT / "rs_runs.csv")

rows = []
for a in ALGS:
    msk = base.alg.values == a
    m = rs[rs.alg == a].groupby("cfg")[["hv_norm", "hv_far"]].mean()
    for lab, within, between in [("ref 1,1 (norm)", base.hv[msk], m.hv_norm),
                                 ("titik terjauh (mentah)", hv_far_base[msk], m.hv_far)]:
        sw, sb, mu = np.std(within, ddof=1), between.std(ddof=1), np.mean(within)
        rows.append(dict(alg=a, konvensi=lab,
                         n_artikel_eps0_2pct=int(np.ceil((Z * sw / (0.002 * mu)) ** 2)),
                         n_usulan_k0_1=int(np.ceil((Z * sw / (0.10 * sb)) ** 2)),
                         n_usulan_k0_2=int(np.ceil((Z * sw / (0.20 * sb)) ** 2)),
                         rasio_dalam_antar=sw / sb))
df = pd.DataFrame(rows)
df.to_csv(OUT / "invariant_nruns.csv", index=False)
print(df.round(3).to_string(index=False))

# Reliabilitas peringkat (Kendall tau) vs jumlah run, dari 10 run random search
print("\nKesepakatan peringkat 500 konfigurasi antara dua kelompok run independen berukuran n (ref 1,1):")
rel = []
for a in ALGS:
    r = rs[rs.alg == a]
    row = {"alg": a}
    for n in (1, 2, 3, 5):
        A = r[r.seed < n].groupby("cfg").hv_norm.mean()
        B = r[(r.seed >= 5) & (r.seed < 5 + n)].groupby("cfg").hv_norm.mean()
        row[f"n={n}"] = kendalltau(A, B).statistic
    rel.append(row)
rel = pd.DataFrame(rel)
rel.to_csv(OUT / "ranking_reliability.csv", index=False)
print(rel.round(3).to_string(index=False))
