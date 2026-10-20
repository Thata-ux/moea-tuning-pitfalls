"""
Validasi rule B — tahap 2: analisis pemilihan konfigurasi.

Kebenaran: mu_c = rata-rata HV (NORM) 100 run kebenaran per konfigurasi; c* = argmax mu_c.
Simulasi: untuk n run per konfigurasi (diambil tanpa pengembalian dari 50 run pemilihan, 500 ulangan),
pilih argmax rata-rata sampel lalu ukur
  - regret = (mu_c* - mu_terpilih) / HV_front_sebenarnya x 100  (poin persen)
  - P(terpilih termasuk top-5 sejati)
  - Kendall tau peringkat sampel vs peringkat sejati
Uji konsistensi rule B: rule B dengan resolusi k menargetkan galat ~ k * sigma_b. Dihitung n empiris terkecil
dengan E[regret] <= k * sigma_b_sejati, lalu dibandingkan dengan n yang diprediksi rule B.
Pembanding racing: successive halving (SH) dengan total budget = n x 100 run.
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm as N
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11})
OUT = c.ROOT / "results"
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
NAME = {"NSGA2": "NSGA-II", "SPEA2": "SPEA2", "SMS-EMOA": "SMS-EMOA", "MOEA/D-DE": "MOEA/D-DE"}
NS = [1, 2, 3, 5, 7, 10, 15, 20]  # n <= 20: di atas itu sampel ulang dari 50 run pemilihan terlalu tumpang-tindih
REPS = 500
Z = N.ppf(0.975)
rng = np.random.default_rng(11)

runs = pd.read_csv(OUT / "ruleB_runs.csv")
base = pd.read_csv(OUT / "baseline_runs.csv")
rs = pd.read_csv(OUT / "rs_runs.csv")
fronts = np.load(OUT / "baseline_final_fronts.npy", allow_pickle=True)
hv_far_base = np.array([c.hv_far(F) for F in fronts])


def rule_ns(a):
    """n dari aturan A (NORM, FAR) dan rule B (k = 0,1; 0,2), dihitung dari data yang tersedia SEBELUM validasi
    (baseline default 100 run + random search 500 konfigurasi x 10 run)."""
    msk = base.alg.values == a
    m = rs[rs.alg == a].groupby("cfg")[["hv_norm", "hv_far"]].mean()
    out = {}
    for conv, w, bcol in (("NORM", base.hv.values[msk], "hv_norm"), ("FAR", hv_far_base[msk], "hv_far")):
        sw, mu, sb = np.std(w, ddof=1), np.mean(w), m[bcol].std(ddof=1)
        out[f"A_{conv}"] = int(np.ceil((Z * sw / (0.002 * mu)) ** 2))
        for k in (0.1, 0.2):
            out[f"B{k}_{conv}"] = int(np.ceil((Z * sw / (k * sb)) ** 2))
    return out


def successive_halving(sel, budget, rng):
    """SH sederhana: babak berulang, separuh terbawah dieliminasi; budget dibagi rata per babak."""
    n_cfg, pool = sel.shape
    alive = np.arange(n_cfg)
    rounds = int(np.ceil(np.log2(n_cfg))) + 1
    used = np.zeros(n_cfg, int)
    sums = np.zeros(n_cfg)
    order = np.array([rng.permutation(pool) for _ in range(n_cfg)])
    spent = 0
    for r in range(rounds):
        per = max(1, (budget // rounds) // len(alive))
        for i in alive:
            take = min(per, pool - used[i])
            if take > 0:
                idx = order[i, used[i]:used[i] + take]
                sums[i] += sel[i, idx].sum()
                used[i] += take
                spent += take
        means = sums[alive] / np.maximum(used[alive], 1)
        if len(alive) == 1:
            break
        keep = max(1, len(alive) // 2)
        alive = alive[np.argsort(-means)[:keep]]
    return alive[0], spent


rows, curves, sh_rows, cons = [], [], [], []
for a in ALGS:
    r = runs[runs.alg == a]
    truth = r[r.pool == "truth"].pivot(index="cfg", columns="seed", values="hv_norm")
    sel = r[r.pool == "select"].pivot(index="cfg", columns="seed", values="hv_norm").loc[truth.index].values
    mu = truth.mean(axis=1).to_numpy()
    best = mu.max()
    top5 = set(np.argsort(-mu)[:5])
    sb_true = mu.std(ddof=1)
    for n in NS:
        reg, t5, tau = [], [], []
        for _ in range(REPS):
            idx = np.array([rng.choice(sel.shape[1], n, replace=False) for _ in range(sel.shape[0])])
            est = np.take_along_axis(sel, idx, 1).mean(1)
            k = int(np.argmax(est))
            reg.append(best - mu[k]); t5.append(k in top5)
            if _ < 100:
                tau.append(kendalltau(est, mu).statistic)
        curves.append(dict(alg=a, n=n, regret_pp=100 * np.mean(reg) / c.HV_TRUE,
                           regret_p90_pp=100 * np.quantile(reg, .9) / c.HV_TRUE,
                           p_top5=np.mean(t5), tau=np.mean(tau), mean_regret_raw=np.mean(reg)))
    C = pd.DataFrame([x for x in curves if x["alg"] == a])
    # SH dengan budget setara n x 100
    for n in (3, 7, 10, 20):
        reg, t5 = [], []
        for _ in range(REPS):
            k, spent = successive_halving(sel, n * sel.shape[0], rng)
            reg.append(best - mu[k]); t5.append(k in top5)
        sh_rows.append(dict(alg=a, budget_equiv_n=n, regret_pp=100 * np.mean(reg) / c.HV_TRUE, p_top5=np.mean(t5)))
    # konsistensi rule B
    rn = rule_ns(a)
    for kk in (0.1, 0.2):
        ok = C[C.mean_regret_raw <= kk * sb_true]
        n_emp = int(ok.n.min()) if len(ok) else None
        cons.append(dict(alg=a, k=kk, sigma_b_true=sb_true, n_empirical=n_emp if n_emp else ">20",
                         n_ruleB_NORM=rn[f"B{kk}_NORM"], n_ruleB_FAR=rn[f"B{kk}_FAR"],
                         n_ruleA_NORM=rn["A_NORM"], n_ruleA_FAR=rn["A_FAR"]))
    rows.append(dict(alg=a, **rn, sigma_b_true=sb_true))

CV = pd.DataFrame(curves); CV.to_csv(OUT / "ruleB_selection_curves.csv", index=False)
SH = pd.DataFrame(sh_rows); SH.to_csv(OUT / "ruleB_successive_halving.csv", index=False)
CS = pd.DataFrame(cons); CS.to_csv(OUT / "ruleB_consistency.csv", index=False)
pd.set_option("display.width", 220)
print("Kurva pemilihan (regret dalam poin persen HV front sebenarnya):")
print(CV.pivot(index="n", columns="alg", values="regret_pp")[ALGS].round(3).to_string())
print("\nP(top-5):")
print(CV.pivot(index="n", columns="alg", values="p_top5")[ALGS].round(2).to_string())
print("\nKendall tau:")
print(CV.pivot(index="n", columns="alg", values="tau")[ALGS].round(2).to_string())
print("\nSuccessive halving (budget setara n x 100):")
print(SH.round(3).to_string(index=False))
print("\nKonsistensi rule B (n empiris agar E[regret] <= k * sigma_b sejati):")
print(CS.to_string(index=False))


def at_n(a, n):
    """Interpolasi regret pada n (log-linear), n di luar rentang -> nilai tepi."""
    s = CV[CV.alg == a].sort_values("n")
    return float(np.interp(np.log(min(max(n, 1), 20)), np.log(s.n), s.regret_pp))


fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, a in zip(axes, ALGS):
    s = CV[CV.alg == a]
    ax.plot(s.n, s.regret_pp, "o-", color="#444", lw=1.8, ms=4, label="Fixed n runs per configuration")
    h = SH[SH.alg == a]
    ax.plot(h.budget_equiv_n, h.regret_pp, "D", color="#e7298a", ms=6, label="Successive halving, same total budget")
    rn = [x for x in rows if x["alg"] == a][0]
    ymax = max(s.regret_pp.max(), h.regret_pp.max()) * 1.18
    ax.set_ylim(0, ymax)
    off = []
    for j, (key, lb, colr, ls) in enumerate([("A_FAR", "Rule A, FAR", "#d62728", "--"),
                                             ("A_NORM", "Rule A, NORM", "#777777", ":"),
                                             ("B0.2_NORM", "Rule B, k = 0.2", "#1f78b4", "-.")]):
        n = rn[key]
        if n <= 20:
            ax.axvline(n, color=colr, ls=ls, lw=1.3)
            ax.text(n * 1.04, ymax * (0.97 - 0.16 * j), f"{lb}\nn = {n}", color=colr, fontsize=7.5, va="top")
        else:
            off.append(f"{ {'A_FAR': 'A-FAR', 'A_NORM': 'A-NORM', 'B0.2_NORM': 'B'}[key]} n = {n:,}")
    sub = ("\noff scale: " + ", ".join(off[:2]) + ("\n" + ", ".join(off[2:]) if len(off) > 2 else "")) if off else "\n "
    ax.set_title(NAME[a] + sub, fontsize=9.5)
    ax.set(xscale="log", xlabel="Runs per configuration (n)",
           ylabel="Mean regret (pp of true-front HV)" if a == "NSGA2" else None)
    ax.grid(alpha=.3, which="both")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("Selecting the best of 100 configurations: regret against an independent ground truth (500 resamples)", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "figures" / "Fig4_ruleB_validation.png", dpi=300)
print("ok")
