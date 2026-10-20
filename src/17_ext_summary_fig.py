"""Ringkasan + Fig. 7 untuk Eksperimen Q2-A (3 objektif dan suite RE)."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import benchlib as b

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11})
OUT = b.ROOT / "results"
R = pd.read_csv(OUT / "ext_runs.csv")
S = pd.read_csv(OUT / "ext_summary.csv")
NR = pd.read_csv(OUT / "ext_nruns.csv")

# Temuan 2 — uji dua arah online vs raw, budget 3000
rows = []
for pb, g in R[R.evals == 3000].groupby("problem", sort=False):
    raw, onl = g[g.alg == "MOEA/D-DE raw"].hv_pct, g[g.alg == "MOEA/D-DE online"].hv_pct
    rows.append(dict(problem=pb, gain=onl.mean() - raw.mean(),
                     p_two_sided=mannwhitneyu(onl, raw, alternative="two-sided").pvalue,
                     raw=raw.mean(), online=onl.mean(), nsga2=g[g.alg == "NSGA2"].hv_pct.mean()))
G = pd.DataFrame(rows).sort_values("gain")
G["verdict"] = np.where((G.p_two_sided < 0.05) & (G.gain > 0), "better",
                        np.where((G.p_two_sided < 0.05) & (G.gain < 0), "worse", "no difference"))
G.to_csv(OUT / "ext_moead_gain_3000.csv", index=False)
print(G.round(3).to_string(index=False))
print("\nverdict counts:", G.verdict.value_counts().to_dict())

# Temuan 1 — rasio n_runs NORM/FAR
NR["log10_ratio"] = np.log10(NR.n_norm.clip(lower=1) / NR.n_far.clip(lower=1))
big = (NR.log10_ratio.abs() > np.log10(1.5)).sum()
print(f"\nn_runs NORM/FAR: {big} dari {len(NR)} pasangan berubah > 1,5x; rentang rasio "
      f"{10 ** NR.log10_ratio.min():.2f}–{10 ** NR.log10_ratio.max():.0f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
cols = {"better": "#1f78b4", "worse": "#e31a1c", "no difference": "#bbbbbb"}
y = np.arange(len(G))
a1.barh(y, G.gain, color=[cols[v] for v in G.verdict])
a1.set_yticks(y, G.problem)
a1.axvline(0, color="k", lw=.8)
for yi, g in zip(y, G.gain):
    a1.text(g + (1 if g >= 0 else -1), yi, f"{g:+.1f}", va="center", ha="left" if g >= 0 else "right", fontsize=8)
a1.set(xlabel="HV gain of normalized over raw MOEA/D-DE (percentage points, 3,000 evaluations)",
       title="(a) Effect of normalization, 30 runs")
a1.set_xlim(G.gain.min() - 6, G.gain.max() + 8)
from matplotlib.patches import Patch
a1.legend(handles=[Patch(color=cols[k], label=k) for k in ["better", "no difference", "worse"]],
          fontsize=8, loc="lower right", title="Mann–Whitney, p < 0.05", title_fontsize=8)
a1.grid(alpha=.3, axis="x")

order = G.problem.tolist()
mk = {"NSGA2": ("o", "#1b9e77", "NSGA-II"), "SPEA2": ("s", "#7570b3", "SPEA2"), "SMS-EMOA": ("^", "#d95f02", "SMS-EMOA")}
for a, (m, cl, lb) in mk.items():
    s = NR[NR.alg == a].set_index("problem").loc[order]
    a2.scatter(10 ** s.log10_ratio, np.arange(len(order)) + {"NSGA2": -.2, "SPEA2": 0, "SMS-EMOA": .2}[a],
               marker=m, color=cl, label=lb, s=28)
a2.set_yticks(np.arange(len(order)), order)
a2.axvline(1, color="k", lw=.8)
a2.axvspan(1 / 1.5, 1.5, color="grey", alpha=.15, lw=0)
a2.set(xscale="log", xlabel="Rule-A run count, NORM ÷ FAR (log scale)",
       title="(b) Run count under two reference points")
a2.grid(alpha=.3, axis="x", which="both")
a2.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "figures" / "Fig8_extended_benchmarks.png", dpi=300)
print("ok")
