"""
Analisis Q2-F (E5): kualitas sejati konfigurasi terpilih per metode HPO.
Hanya ulangan yang lengkap untuk semua (algoritma, metode) yang dipakai.
Output: results/hpo_summary.csv, results/hpo_tests.csv, figures_en/Fig5_hpo_methods.png
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11})
OUT = c.ROOT / "results"
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
NAME = {"NSGA2": "NSGA-II", "SPEA2": "SPEA2", "SMS-EMOA": "SMS-EMOA", "MOEA/D-DE": "MOEA/D-DE"}
METHODS = ["RS", "TPE", "GP-BO", "SMAC", "RS+SH"]
COL = {"RS": "#888888", "TPE": "#1b9e77", "GP-BO": "#7570b3", "SMAC": "#d95f02", "RS+SH": "#e7298a"}

d = pd.read_csv(OUT / "hpo_sessions.csv").drop_duplicates(["alg", "method", "rep"])
full = d.groupby("rep").size()
reps = sorted(full[full == len(ALGS) * len(METHODS)].index)
d = d[d.rep.isin(reps)]
print(f"ulangan lengkap: {len(reps)} ({reps})")

# referensi: konfigurasi terbaik yang diketahui per algoritma (kebenaran E4 atau hasil HPO, mana yang lebih tinggi)
truth = pd.read_csv(OUT / "ruleB_runs.csv")
best_e4 = truth[truth.pool == "truth"].groupby(["alg", "cfg"]).hv_norm.mean().groupby("alg").max()
best = {a: max(best_e4[a], d[d.alg == a].true_hv.max()) for a in ALGS}
d["regret_pp"] = d.apply(lambda r: 100 * (best[r.alg] - r.true_hv) / c.HV_TRUE, axis=1)

S = d.groupby(["alg", "method"]).agg(true_pct_mean=("true_hv_pct", "mean"), true_pct_sd=("true_hv_pct", "std"),
                                     regret_mean=("regret_pp", "mean"), regret_median=("regret_pp", "median"),
                                     runs_used=("runs_used", "mean")).reset_index()
S.to_csv(OUT / "hpo_summary.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(3).to_string(index=False))

# peringkat rata-rata per algoritma (1 = terbaik) berdasarkan regret rata-rata
S["rank"] = S.groupby("alg").regret_mean.rank()
print("\nPeringkat rata-rata metode (lintas 4 algoritma):")
print(S.groupby("method")["rank"].mean().sort_values().round(2).to_string())

T = []
for a in ALGS:
    rs = d[(d.alg == a) & (d.method == "RS")].regret_pp
    for m in METHODS[1:]:
        x = d[(d.alg == a) & (d.method == m)].regret_pp
        T.append(dict(alg=a, method=m, regret_mean=x.mean(), rs_regret_mean=rs.mean(),
                      p_vs_RS=mannwhitneyu(x, rs, alternative="two-sided").pvalue))
T = pd.DataFrame(T)
T.to_csv(OUT / "hpo_tests.csv", index=False)
print("\nUji vs random search (Mann–Whitney dua arah):")
print(T.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
for ax, a in zip(axes, ALGS):
    for i, m in enumerate(METHODS):
        y = d[(d.alg == a) & (d.method == m)].regret_pp.values
        x = np.full(len(y), i) + np.linspace(-0.12, 0.12, len(y))
        ax.scatter(x, y, s=22, color=COL[m], alpha=.75, zorder=3)
        ax.hlines(y.mean(), i - 0.28, i + 0.28, color=COL[m], lw=2.2, zorder=4)
    ax.set_xticks(range(len(METHODS)), METHODS, rotation=0, fontsize=9)
    ax.set(title=NAME[a], ylim=(0, None))
    ax.grid(alpha=.3, axis="y")
axes[0].set_ylabel("Regret of selected configuration\n(pp of true-front HV, 100 new seeds)")
fig.suptitle(f"HPO strategies at an equal budget of 500 runs ({len(reps)} sessions each; bar = mean)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "figures" / "Fig5_hpo_methods.png", dpi=300)
print("ok")
