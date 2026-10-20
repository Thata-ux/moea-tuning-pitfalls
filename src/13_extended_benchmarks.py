"""
Eksperimen Q2-A — Generalisasi ke 3 objektif dan problem dunia nyata.
Problem: DTLZ2 3 objektif (seimbang dan f3 x 1e4) + 12 problem RE (rasio skala alami 1,5 sampai 1e7).
Algoritma: NSGA-II, SPEA2, SMS-EMOA, MOEA/D-DE (mentah | online | oracle). 30 run, budget 300 dan 3000.
Output: results/ext_runs.csv, ext_summary.csv, ext_nruns.csv
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import norm as N, mannwhitneyu
import benchlib as b

OUT = b.ROOT / "results"
N_RUNS = 30
PROBS = [("DTLZ2-3", None), ("DTLZ2-3", (1.0, 1.0, 1e4))] + [(n, None) for n in b.RE_NAMES]
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE raw", "MOEA/D-DE online", "MOEA/D-DE oracle"]
BUDGETS = [300, 3000]


def label(name, scale):
    return name if scale is None else f"{name} (f3x1e4)"


def job(t):
    name, scale, alg, seed, ev = t
    prob, pf, far = b.build(name, scale)
    if alg.startswith("MOEA/D"):
        mode = alg.split()[-1]
        mode = "none" if mode == "raw" else mode
        F = b.run_moead(prob, seed, ev, normalize=mode, oracle=(pf.min(0), pf.max(0)))
    else:
        F = b.run_ga(alg, prob, seed, ev)
    return dict(problem=label(name, scale), n_obj=prob.n_obj, alg=alg, seed=seed, evals=ev,
                hv_norm=b.hv_norm(F, pf), hv_far=b.hv_far(F, far))


if __name__ == "__main__":
    tasks = [(n, s, a, k, e) for n, s in PROBS for a in ALGS for k in range(N_RUNS) for e in BUDGETS]
    with Pool() as p:
        rows = p.map(job, tasks, chunksize=6)
    df = pd.DataFrame(rows)
    ref_hv = {label(n, s): b.hv_norm(b.build(n, s)[1], b.build(n, s)[1]) for n, s in PROBS}
    ratio = {label(n, s): b.scale_ratio(b.build(n, s)[1]) for n, s in PROBS}
    df["hv_pct"] = 100 * df.hv_norm / df.problem.map(ref_hv)
    df.to_csv(OUT / "ext_runs.csv", index=False)

    # Temuan 2: MOEA/D-DE mentah vs normalisasi
    rows = []
    for (pb, ev), g in df.groupby(["problem", "evals"], sort=False):
        m = lambda a: g[g.alg == a].hv_pct
        rows.append(dict(problem=pb, n_obj=g.n_obj.iloc[0], evals=ev, scale_ratio=ratio[pb],
                         NSGA2=m("NSGA2").mean(), SPEA2=m("SPEA2").mean(), SMS_EMOA=m("SMS-EMOA").mean(),
                         MOEAD_raw=m("MOEA/D-DE raw").mean(), MOEAD_online=m("MOEA/D-DE online").mean(),
                         MOEAD_oracle=m("MOEA/D-DE oracle").mean(),
                         p_online_gt_raw=mannwhitneyu(m("MOEA/D-DE online"), m("MOEA/D-DE raw"),
                                                      alternative="greater").pvalue))
    S = pd.DataFrame(rows)
    S["gain_online_minus_raw"] = S.MOEAD_online - S.MOEAD_raw
    S.to_csv(OUT / "ext_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))

    # Temuan 1: n_runs aturan A, ref 1,1 vs titik terjauh (budget 3000)
    z = N.ppf(0.975)
    nr = []
    for (pb, a), g in df[(df.evals == 3000) & df.alg.isin(["NSGA2", "SPEA2", "SMS-EMOA"])].groupby(["problem", "alg"], sort=False):
        f = lambda col: int(np.ceil((z * g[col].std(ddof=1) / g[col].mean() / 0.002) ** 2))
        nr.append(dict(problem=pb, alg=a, n_norm=f("hv_norm"), n_far=f("hv_far")))
    NR = pd.DataFrame(nr)
    NR["ratio"] = NR.n_norm / NR.n_far.clip(lower=1)
    NR.to_csv(OUT / "ext_nruns.csv", index=False)
    print("\nn_runs (aturan A, budget 3000): median rasio NORM/FAR =", round(NR.ratio.median(), 1),
          "| min", round(NR.ratio.min(), 2), "| maks", round(NR.ratio.max(), 1))
    print(NR.pivot(index="problem", columns="alg", values="ratio").round(1).to_string())
