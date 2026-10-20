"""English figures for the manuscript (JUTIF). Rebuilt from saved results; MOEA/D-DE traces are re-run (fast)."""
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

OUT = c.ROOT / "results" / "figures"
OUT.mkdir(exist_ok=True)
R = c.ROOT / "results"
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
NAME = {"NSGA2": "NSGA-II", "SPEA2": "SPEA2", "SMS-EMOA": "SMS-EMOA", "MOEA/D-DE": "MOEA/D-DE"}
COL = {"NSGA2": "#1b9e77", "SPEA2": "#7570b3", "SMS-EMOA": "#d95f02", "MOEA/D-DE": "#66a61e"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11})


def moead_job(a):
    mode, s = a
    tr, h, i, F = c.run_moead_de(20, 15, s, normalize=mode, **c.DEFAULTS["MOEA/D-DE"])
    return mode, s, tr, F


if __name__ == "__main__":
    # Fig. 1 — baseline convergence
    tr = np.load(R / "baseline_traces.npz")
    it = np.arange(1, 16)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for a in ALGS:
        m = tr[a.replace("/", "_")].mean(0)
        (a2 if a == "MOEA/D-DE" else a1).plot(it, m, lw=2, color=COL[a], label=NAME[a])
    a2.plot(it, tr["NSGA2"].mean(0), lw=1, color=COL["NSGA2"], alpha=.45, label="NSGA-II (reference)")
    for ax, t in ((a1, "(a) NSGA-II, SPEA2, SMS-EMOA"), (a2, "(b) MOEA/D-DE")):
        ax.axhline(c.HV_TRUE, ls=":", color="grey", lw=1, label="True Pareto front")
        ax.set(xlabel="Iteration", ylabel="Mean normalized HV (100 runs)", title=t)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "Fig1_baseline_convergence.png", dpi=300)

    # Fig. 2 — run count vs reference point
    rp = pd.read_csv(R / "refpoint_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    for a in ALGS[:3]:
        s = rp[rp.alg == a]
        ax.plot(s.ref, s.n_runs_95, "o-", lw=2, color=COL[a], label=NAME[a])
    ax.axhline(7, color="k", ls="--", lw=1)
    ax.text(1.06, 8.2, "n_runs = 7 used in the original study", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="Reference point r (normalized objectives, r₁ = r₂)",
           ylabel="Minimum runs (rule A, 95% CI)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "Fig2_runs_vs_reference_point.png", dpi=300)

    # Fig. 3 — random search under two conventions
    rs = pd.read_csv(R / "rs_runs.csv")
    S = pd.read_csv(R / "rs_refpoint_summary.csv").set_index("alg")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, a in zip(axes, ALGS):
        m = rs[rs.alg == a].groupby("cfg")[["hv_norm", "hv_far"]].mean()
        q = m.quantile(0.5)
        top = m[(m.hv_norm >= q.hv_norm) & (m.hv_far >= q.hv_far)]
        ax.scatter(top.hv_far, top.hv_norm, s=8, alpha=.5, color="#888", label="configuration")
        bn, bf = m.hv_norm.idxmax(), m.hv_far.idxmax()
        if bn == bf:
            ax.scatter(m.hv_far[bn], m.hv_norm[bn], s=110, marker="*", color="#7b3294", zorder=3,
                       label="best (same in both)")
        else:
            ax.scatter(m.hv_far[bn], m.hv_norm[bn], s=60, color="#1f78b4", zorder=3, label="best under NORM")
            ax.scatter(m.hv_far[bf], m.hv_norm[bf], s=60, marker="D", color="#e7298a", zorder=3, label="best under FAR")
        ax.set(title=f"({'abcd'[ALGS.index(a)]}) {NAME[a]}, τ = {S.loc[a, 'tau_norm_vs_far']:.2f}",
               xlabel="Mean HV, FAR (raw)", ylabel="Mean HV, NORM")
        ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "Fig3_random_search_two_conventions.png", dpi=300)

    # Fig. 4 — MOEA/D-DE normalization
    with Pool() as p:
        res = p.map(moead_job, [(m, s) for m in ("none", "online", "oracle") for s in range(100)])
    lab = {"none": "Raw Tchebycheff (Metaheuristics.jl)", "online": "Online normalization",
           "oracle": "Oracle normalization"}
    cl = {"none": "#66a61e", "online": "#e7298a", "oracle": "#1f78b4"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for m in ("none", "online", "oracle"):
        t = np.array([x for mm, _, x, _ in res if mm == m])
        a1.plot(it, t.mean(0), lw=2, color=cl[m], label=lab[m])
    a1.plot(it, tr["NSGA2"].mean(0), lw=1.5, ls="--", color="grey", label="NSGA-II baseline")
    a1.set(xlabel="Iteration", ylabel="Mean normalized HV (100 runs)", title="(a) Convergence")
    a1.grid(alpha=.3); a1.legend(fontsize=8)
    a2.plot(c.PF_TRUE[:, 1], c.PF_TRUE[:, 0], ".", color="lightgrey", ms=4, label="True Pareto front")
    for m in ("none", "oracle"):
        F = c.nd([x for mm, s, _, x in res if mm == m and s == 0][0])
        a2.plot(F[:, 1], F[:, 0], "o", color=cl[m], ms=6, alpha=.85, label=lab[m])
    a2.set(xlabel="Material cost", ylabel="Maximum displacement (m)", yscale="log", title="(b) Final front, run 0")
    a2.grid(alpha=.3); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "Fig6_moead_normalization.png", dpi=300)

    # Fig. 5 — benchmarks
    t2 = pd.read_csv(R / "bench_moead_normalization.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for k, (ax, (pb, e)) in enumerate(zip(axes, [("DTLZ2", 300), ("DTLZ2", 1500), ("ZDT1", 1500)])):
        sub = t2[(t2.problem == pb) & (t2.evals == e)].set_index("version").loc[["seimbang", "timpang"]]
        x = np.arange(2)
        b1 = ax.bar(x - .2, sub.mentah, .4, color="#66a61e", label="Raw Tchebycheff")
        b2 = ax.bar(x + .2, sub.norm, .4, color="#1f78b4", label="Normalized Tchebycheff")
        for b in (b1, b2):
            ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
        ax.set_xticks(x, ["balanced scales", "mismatched scales\n(f₂ × 10⁴)"])
        ax.set(title=f"({'abc'[k]}) {pb}, {e} evaluations", ylim=(0, 110)); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("% of true-front HV (mean, 100 runs)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / "FigS1_benchmarks_moead.png", dpi=300)
    print("ok")
