"""
Tahap 2 — Baseline (setting default) dan estimasi jumlah run dengan CLT.
Replikasi Bagian 3.2–3.3, Table 1, Fig. 2–3 dari Pereira & Leitao (2026).

pop_size=20, n_iters=15 (300 evaluasi), 100 run independen per algoritma,
populasi awal LHS yang sama untuk semua algoritma pada run ke-i.
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import norm as N
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
N_RUNS, POP, ITERS = 100, 20, 15
OUT = c.ROOT / "results"
DIAG = OUT / "diagnostics"  # quick-look plots (labels partly in Indonesian)
DIAG.mkdir(parents=True, exist_ok=True)
COLORS = {"NSGA2": "#1b9e77", "SMS-EMOA": "#d95f02", "SPEA2": "#7570b3", "MOEA/D-DE": "#66a61e"}


def job(args):
    alg, seed = args
    trace, hv_fin, igd_fin, Ffin = c.run(alg, POP, ITERS, seed, **c.DEFAULTS[alg])
    return alg, seed, trace, hv_fin, igd_fin, Ffin


if __name__ == "__main__":
    with Pool() as p:
        res = p.map(job, [(a, s) for a in ALGS for s in range(N_RUNS)])

    final = pd.DataFrame([dict(alg=a, seed=s, hv=h, igd=i, hv_ratio=h / c.HV_TRUE)
                          for a, s, _, h, i, _ in res])
    final.to_csv(OUT / "baseline_runs.csv", index=False)
    traces = {a: np.array([t for aa, _, t, _, _, _ in res if aa == a]) for a in ALGS}
    np.save(OUT / "baseline_final_fronts.npy", np.array([F for *_, F in res], dtype=object), allow_pickle=True)
    np.savez(OUT / "baseline_traces.npz", **{a.replace("/", "_"): v for a, v in traces.items()})

    # ---------- Ringkasan ----------
    summ = final.groupby("alg").agg(hv_mean=("hv", "mean"), hv_std=("hv", "std"),
                                    hv_ratio_mean=("hv_ratio", "mean"),
                                    igd_mean=("igd", "mean"), igd_std=("igd", "std")).loc[ALGS]
    summ.to_csv(OUT / "baseline_summary.csv")
    print(f"HV Pareto front sebenarnya = {c.HV_TRUE:.4f}\n")
    print(summ.round(4).to_string(), "\n")

    # ---------- Table 1: jumlah run minimum, n = (z*sigma/eps)^2 ----------
    # eps dinyatakan relatif terhadap HV sebenarnya, setara 0,10–0,25% seperti di artikel
    eps_pct = [0.10, 0.15, 0.20, 0.25]
    rows = []
    for e in eps_pct:
        eps = e / 100 * c.HV_TRUE
        row = {"eps_%": e}
        for a in ALGS:
            s = summ.loc[a, "hv_std"]
            for ci in (90, 95):
                z = N.ppf(1 - (1 - ci / 100) / 2)
                row[f"{a} {ci}%"] = int(np.ceil((z * s / eps) ** 2))
        rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "table1_nruns.csv", index=False)
    print("Jumlah run minimum (CLT):")
    print(t1.to_string(index=False))

    # ---------- Fig. 2: rata-rata HV per iterasi ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    it = np.arange(1, ITERS + 1)
    for a in ALGS:
        m = traces[a].mean(0)
        (ax2 if a == "MOEA/D-DE" else ax1).plot(it, m, label=a, color=COLORS[a], lw=2)
    ax2.plot(it, traces["NSGA2"].mean(0), color=COLORS["NSGA2"], lw=1, alpha=.4, label="NSGA2 (pembanding)")
    for ax, t in ((ax1, "NSGA2, SPEA2, SMS-EMOA"), (ax2, "MOEA/D-DE")):
        ax.axhline(c.HV_TRUE, ls=":", color="grey", lw=1, label="HV front sebenarnya")
        ax.set(xlabel="Iterasi", ylabel="Hypervolume (ternormalisasi)", title=t)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Rata-rata HV per iterasi — baseline default (100 run)")
    fig.tight_layout(); fig.savefig(DIAG / "fig2_baseline_hv_per_iter.png", dpi=160)

    # ---------- Fig. 3: variabilitas HV akhir ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1.4]})
    data = [final.loc[final.alg == a, "hv"] for a in ALGS]
    b = ax1.boxplot(data[:3], tick_labels=ALGS[:3], patch_artist=True)
    for patch, a in zip(b["boxes"], ALGS[:3]):
        patch.set_facecolor(COLORS[a]); patch.set_alpha(.6)
    b = ax2.boxplot(data, tick_labels=["NSGA2", "SPEA2", "SMS", "MOEA/D"], patch_artist=True)
    for patch, a in zip(b["boxes"], ALGS):
        patch.set_facecolor(COLORS[a]); patch.set_alpha(.6)
    ax1.set(ylabel="HV akhir", title="Tiga algoritma terbaik (zoom)")
    ax2.set(title="Semua algoritma")
    for ax in (ax1, ax2):
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("Variabilitas HV akhir — baseline default (100 run)")
    fig.tight_layout(); fig.savefig(DIAG / "fig3_baseline_variability.png", dpi=160)
    print("\nGambar disimpan di results/")
