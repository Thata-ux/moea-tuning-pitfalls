"""
Tahap 3 — Sensitivitas jumlah run (CLT) terhadap titik referensi hypervolume.

Hipotesis: n_runs = (z*sigma/eps)^2 dengan eps relatif terhadap rata-rata HV setara dengan
n_runs = (z*CV/e)^2. Titik referensi yang menjauh menambah "luas konstan" pada HV sehingga CV
mengecil dan n_runs ikut mengecil, padahal kualitas solusi tidak berubah.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm as N
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import common as c

ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
OUT = c.ROOT / "results"
DIAG = OUT / "diagnostics"  # quick-look plots (labels partly in Indonesian)
DIAG.mkdir(parents=True, exist_ok=True)
fronts = np.load(OUT / "baseline_final_fronts.npy", allow_pickle=True)
runs = pd.read_csv(OUT / "baseline_runs.csv")
Z95 = N.ppf(0.975)
E_REL = 0.002            # eps = 0,2% dari rata-rata HV (tengah rentang 0,1–0,26% di artikel)

# Pereira & Leitao, Table 1 (CV = std/mean)
PAPER_CV = {"NSGA2": 18.27 / 20128.28, "SPEA2": 40.19 / 20037.75,
            "SMS-EMOA": 19.29 / 20138.41, "MOEA/D-DE": 2321.58 / 16903.34}

rows = []
for r in [1.05, 1.1, 1.5, 2, 3, 5, 10, 20]:
    ind = HV(ref_point=np.array([r, r]))
    hvs = np.array([ind(c.norm(c.nd(F))) for F in fronts])
    for a in ALGS:
        h = hvs[runs.alg.values == a]
        cv = h.std(ddof=1) / h.mean()
        rows.append(dict(ref=r, alg=a, hv_mean=h.mean(), hv_std=h.std(ddof=1), cv_pct=100 * cv,
                         n_runs_95=int(np.ceil((Z95 * cv / E_REL) ** 2))))
df = pd.DataFrame(rows)
df.to_csv(OUT / "refpoint_sensitivity.csv", index=False)
piv = df.pivot(index="ref", columns="alg", values="n_runs_95")[ALGS]
print("Jumlah run minimum (CI 95%, eps = 0,2% dari rata-rata HV) vs titik referensi (ternormalisasi):")
print(piv.to_string())
print("\nCV (%) per titik referensi:")
print(df.pivot(index="ref", columns="alg", values="cv_pct")[ALGS].round(3).to_string())

paper_n = {a: int(np.ceil((Z95 * cv / E_REL) ** 2)) for a, cv in PAPER_CV.items()}
print("\nUntuk pembanding, CV artikel (%):", {a: round(100 * v, 3) for a, v in PAPER_CV.items()})
print("n_runs artikel pada aturan yang sama:", paper_n)

# Urutan algoritma berdasarkan rata-rata HV, apakah berubah dengan titik referensi?
rank = df.pivot(index="ref", columns="alg", values="hv_mean")[ALGS].rank(axis=1, ascending=False).astype(int)
print("\nPeringkat rata-rata HV per titik referensi:")
print(rank.to_string())

fig, ax = plt.subplots(figsize=(7, 4.2))
for a, col in zip(ALGS[:3], ["#1b9e77", "#7570b3", "#d95f02"]):
    s = df[df.alg == a]
    ax.plot(s.ref, s.n_runs_95, "o-", label=a, color=col, lw=2)
    ax.axhline(paper_n[a], color=col, ls=":", lw=1)
ax.set(xscale="log", yscale="log", xlabel="Titik referensi HV (ruang ternormalisasi, r = r1 = r2)",
       ylabel="Jumlah run minimum (CI 95%)",
       title="Jumlah run dari CLT bergantung pada titik referensi\n(garis putus: nilai setara dari Table 1 artikel)")
ax.grid(alpha=.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig(DIAG / "fig_refpoint_nruns.png", dpi=160)
