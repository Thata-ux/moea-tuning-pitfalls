"""Gambar ulang Fig. 6 (dosis-respons) dari dose_response_summary.csv: legenda di bawah, label tidak bertumpuk."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11})
OUT = c.ROOT / "results"
S = pd.read_csv(OUT / "dose_response_summary.csv")
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE raw", "MOEA/D-DE online"]
col = {"NSGA2": "#1b9e77", "SPEA2": "#7570b3", "SMS-EMOA": "#d95f02",
       "MOEA/D-DE raw": "#66a61e", "MOEA/D-DE online": "#1f78b4"}
lab = {"MOEA/D-DE raw": "MOEA/D-DE, raw Tchebycheff", "MOEA/D-DE online": "MOEA/D-DE, normalized Tchebycheff",
       "NSGA2": "NSGA-II", "SPEA2": "SPEA2", "SMS-EMOA": "SMS-EMOA"}
orig = np.log10((c.NADIR - c.IDEAL)[1] / (c.NADIR - c.IDEAL)[0])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, part, title in zip(axes, ["DTLZ2", "Truss"],
                           ["(a) DTLZ2, 1,500 evaluations", "(b) Skylight truss, 300 evaluations"]):
    for a in ALGS:
        s = S[(S.part == part) & (S.alg == a)].sort_values("log10_ratio")
        ax.errorbar(s.log10_ratio, s["mean"], yerr=s["std"], marker="o", ms=4, capsize=2, lw=1.8,
                    color=col[a], label=lab[a], ls="--" if a.endswith("raw") else "-")
    ax.axvline(0, color="grey", lw=.8, ls=":")
    ax.text(0.1, 24, "balanced", fontsize=8, color="grey")
    if part == "Truss":
        ax.axvline(orig, color="k", lw=.8, ls="--")
        ax.text(orig - 0.15, 24, "original units", fontsize=8, ha="right")
    ax.set(title=title, xlabel="log₁₀(range of f₂ / range of f₁) seen by the algorithm", ylim=(20, 102))
    ax.grid(alpha=.3)
axes[0].set_ylabel("% of true-front HV (mean ± SD, 30 runs)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "figures" / "Fig7_scale_dose_response.png", dpi=300)
print("ok")
