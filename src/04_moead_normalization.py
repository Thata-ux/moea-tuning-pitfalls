"""
Tahap 4 — Mengapa MOEA/D-DE gagal? Uji hipotesis alternatif: skala objektif.

Pereira & Leitao (Bag. 9) menduga Pareto front tidak seragam (decomposition mismatch).
Hipotesis alternatif: Tchebycheff di Metaheuristics.jl memakai objektif mentah, padahal
displacement ~1e-2..1e-1 dan biaya ~1e2..1e4, sehingga hampir semua subproblem hanya
"melihat" biaya. Diuji: none (asli) vs online vs oracle normalization, 100 run, setting default.
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

N_RUNS, POP, ITERS = 100, 20, 15
OUT = c.ROOT / "results"
DIAG = OUT / "diagnostics"  # quick-look plots (labels partly in Indonesian)
DIAG.mkdir(parents=True, exist_ok=True)
MODES = ["none", "online", "oracle"]
LABEL = {"none": "Tanpa normalisasi\n(seperti Metaheuristics.jl)", "online": "Normalisasi online",
         "oracle": "Normalisasi oracle\n(batas atas)"}


def job(a):
    mode, seed = a
    tr, h, i, Ff = c.run_moead_de(POP, ITERS, seed, normalize=mode, **c.DEFAULTS["MOEA/D-DE"])
    return mode, seed, tr, h, i, Ff


if __name__ == "__main__":
    with Pool() as p:
        res = p.map(job, [(m, s) for m in MODES for s in range(N_RUNS)])
    df = pd.DataFrame([dict(mode=m, seed=s, hv=h, igd=i, n_front=len(c.nd(F))) for m, s, _, h, i, F in res])
    df.to_csv(OUT / "moead_normalization_runs.csv", index=False)
    base = pd.read_csv(OUT / "baseline_runs.csv")
    nsga = base.loc[base.alg == "NSGA2", "hv"]

    summ = df.groupby("mode").agg(hv_mean=("hv", "mean"), hv_std=("hv", "std"),
                                  igd_mean=("igd", "mean"), front_size=("n_front", "mean")).loc[MODES]
    summ["hv_ratio_%"] = 100 * summ.hv_mean / c.HV_TRUE
    print(summ.round(4).to_string())
    for m in ["online", "oracle"]:
        p = mannwhitneyu(df[df["mode"] == m].hv, df[df["mode"] == "none"].hv, alternative="greater").pvalue
        q = mannwhitneyu(df[df["mode"] == m].hv, nsga, alternative="two-sided").pvalue
        print(f"{m:7s} vs none: p = {p:.2e} | vs NSGA2 baseline: p = {q:.3g}")
    print(f"NSGA2 baseline HV mean = {nsga.mean():.4f}")
    summ.to_csv(OUT / "moead_normalization_summary.csv")

    # Gambar: HV per iterasi + front akhir contoh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    cols = {"none": "#66a61e", "online": "#e7298a", "oracle": "#1f78b4"}
    it = np.arange(1, ITERS + 1)
    for m in MODES:
        tr = np.array([t for mm, _, t, *_ in res if mm == m])
        ax1.plot(it, tr.mean(0), lw=2, color=cols[m], label=LABEL[m].replace("\n", " "))
    tr_n = np.load(OUT / "baseline_traces.npz")["NSGA2"]
    ax1.plot(it, tr_n.mean(0), lw=1.5, ls="--", color="grey", label="NSGA2 (baseline)")
    ax1.set(xlabel="Iterasi", ylabel="Rata-rata HV (100 run)", title="MOEA/D-DE: efek normalisasi objektif")
    ax1.grid(alpha=.3); ax1.legend(fontsize=8)

    pf = c.PF_TRUE
    ax2.plot(pf[:, 1], pf[:, 0], ".", color="lightgrey", ms=4, label="Pareto front sebenarnya")
    seed_show = 0
    for m in ["none", "oracle"]:
        Ff = [F for mm, s, *_, F in res if mm == m and s == seed_show][0]
        Ff = c.nd(Ff)
        ax2.plot(Ff[:, 1], Ff[:, 0], "o", color=cols[m], ms=6, alpha=.8, label=LABEL[m].replace("\n", " "))
    ax2.set(xlabel="Biaya", ylabel="Displacement maks.", yscale="log",
            title=f"Front akhir MOEA/D-DE (run {seed_show})")
    ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIAG / "fig_moead_normalization.png", dpi=160)
