"""
Analisis Eksperimen 1 — dampak titik referensi pada hasil random search.
Masukan: results/rs_configs.csv, results/rs_runs.csv (dari 05_random_search_refpoint.py)
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

OUT = c.ROOT / "results"
DIAG = OUT / "diagnostics"  # quick-look plots (labels partly in Indonesian)
DIAG.mkdir(parents=True, exist_ok=True)
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
cfgs = pd.read_csv(OUT / "rs_configs.csv")
runs = pd.read_csv(OUT / "rs_runs.csv")
HP_GA = ["pop_size", "eta_cr", "p_cr", "eta_m", "p_m"]
HP_DE = ["pop_size", "F", "CR", "eta", "p_m", "delta"]


def hp_of(row):
    keys = ["F", "CR", "eta", "p_m", "delta"] if row["alg"] == "MOEA/D-DE" else ["eta_cr", "p_cr", "eta_m", "p_m"]
    return {k: row[k] for k in keys}


def reeval(args):
    row, s = args
    _, h, _, F = c.run(row["alg"], int(row["pop_size"]), int(row["n_iters"]), s, **hp_of(row))
    return h, c.hv_far(F)


summary, hp_corr, picks, best_pairs = [], [], [], []
for a in ALGS:
    r = runs[runs.alg == a]
    m = r.groupby("cfg")[["hv_norm", "hv_far"]].mean()
    A = r[r.seed < 5].groupby("cfg")[["hv_norm", "hv_far"]].mean()
    B = r[r.seed >= 5].groupby("cfg")[["hv_norm", "hv_far"]].mean()
    tau = lambda x, y: kendalltau(x, y).statistic
    top = lambda s, k=10: set(s.nlargest(k).index)
    bn, bf = m.hv_norm.idxmax(), m.hv_far.idxmax()
    rank_n = m.hv_norm.rank(ascending=False)
    rank_f = m.hv_far.rank(ascending=False)
    summary.append(dict(
        alg=a,
        tau_norm_vs_far=tau(m.hv_norm, m.hv_far),
        tau_splithalf_norm=tau(A.hv_norm, B.hv_norm),
        tau_splithalf_far=tau(A.hv_far, B.hv_far),
        tau_cross_half=(tau(A.hv_norm, B.hv_far) + tau(A.hv_far, B.hv_norm)) / 2,
        top10_overlap=len(top(m.hv_norm) & top(m.hv_far)),
        same_best=bn == bf,
        rank_under_norm_of_far_best=int(rank_n[bf]),
        rank_under_far_of_norm_best=int(rank_f[bn]),
    ))
    hp = HP_DE if a == "MOEA/D-DE" else HP_GA
    cc = cfgs[cfgs.alg == a].set_index("cfg").loc[m.index]
    for h in hp:
        hp_corr.append(dict(alg=a, hyperparameter=h,
                            rho_norm=spearmanr(cc[h], m.hv_norm).statistic,
                            rho_far=spearmanr(cc[h], m.hv_far).statistic))
    for lab, k in [("terbaik-NORM", bn), ("terbaik-FAR", bf)]:
        row = cfgs[(cfgs.alg == a) & (cfgs.cfg == k)].iloc[0].to_dict()
        picks.append(dict(alg=a, pilihan=lab, cfg=k, **{h: row[h] for h in hp}, n_iters=row["n_iters"]))
    best_pairs.append((a, cfgs[(cfgs.alg == a) & (cfgs.cfg == bn)].iloc[0].to_dict(),
                       cfgs[(cfgs.alg == a) & (cfgs.cfg == bf)].iloc[0].to_dict(), bn == bf))

S = pd.DataFrame(summary)
S.to_csv(OUT / "rs_refpoint_summary.csv", index=False)
print("Kesepakatan peringkat 500 konfigurasi (Kendall tau):")
print(S.round(3).to_string(index=False))
H = pd.DataFrame(hp_corr)
H["arah_berubah"] = np.sign(H.rho_norm) != np.sign(H.rho_far)
H.to_csv(OUT / "rs_hp_correlation.csv", index=False)
print("\nKorelasi Spearman hyperparameter vs HV rata-rata:")
print(H.round(3).to_string(index=False))
P = pd.DataFrame(picks)
P.to_csv(OUT / "rs_best_configs.csv", index=False)
print("\nKonfigurasi terbaik per konvensi:")
print(P.to_string(index=False))

# ---- Re-evaluasi kedua konfigurasi terbaik dengan 100 seed baru (100..199) ----
re = []
with Pool() as pool:
    for a, rn, rf, same in best_pairs:
        if same:
            re.append(dict(alg=a, catatan="konfigurasi terbaik sama"))
            continue
        xn = np.array(pool.map(reeval, [(rn, s) for s in range(100, 200)]))
        xf = np.array(pool.map(reeval, [(rf, s) for s in range(100, 200)]))
        re.append(dict(alg=a,
                       NORM_hv_norm=xn[:, 0].mean(), FAR_hv_norm=xf[:, 0].mean(),
                       p_dinilai_norm=wilcoxon(xn[:, 0], xf[:, 0]).pvalue,
                       NORM_hv_far=xn[:, 1].mean(), FAR_hv_far=xf[:, 1].mean(),
                       p_dinilai_far=wilcoxon(xn[:, 1], xf[:, 1]).pvalue,
                       NORM_pct_true=100 * xn[:, 0].mean() / c.HV_TRUE,
                       FAR_pct_true=100 * xf[:, 0].mean() / c.HV_TRUE))
R = pd.DataFrame(re)
R.to_csv(OUT / "rs_best_reevaluation.csv", index=False)
print("\nRe-evaluasi 100 run baru (NORM = terbaik menurut ref 1,1; FAR = terbaik menurut titik terjauh):")
print(R.round(4).to_string(index=False))

# ---- Gambar ----
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, a in zip(axes, ALGS):
    r = runs[runs.alg == a]
    m = r.groupby("cfg")[["hv_norm", "hv_far"]].mean()
    ax.scatter(m.hv_far, m.hv_norm, s=8, alpha=.5, color="#888")
    bn, bf = m.hv_norm.idxmax(), m.hv_far.idxmax()
    ax.scatter(m.hv_far[bn], m.hv_norm[bn], s=80, color="#1f78b4", label="terbaik ref 1,1", zorder=3)
    ax.scatter(m.hv_far[bf], m.hv_norm[bf], s=80, color="#e7298a", marker="D", label="terbaik titik terjauh", zorder=3)
    t = S.loc[S.alg == a, "tau_norm_vs_far"].item()
    ax.set(title=f"{a} (τ = {t:.2f})", xlabel="HV, titik terjauh (mentah)", ylabel="HV, ref 1,1 (ternormalisasi)")
    q = m.quantile(0.5)
    ax.set_xlim(left=q.hv_far); ax.set_ylim(bottom=q.hv_norm)
    ax.grid(alpha=.3)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("500 konfigurasi random search: penilaian dengan dua titik referensi (separuh teratas ditampilkan)")
fig.tight_layout(); fig.savefig(DIAG / "fig_rs_refpoint.png", dpi=160)
