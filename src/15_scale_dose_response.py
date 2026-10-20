"""
Q2-B — Kurva dosis–respons: performa vs rasio skala objektif.
A) DTLZ2 2 objektif, f2 x s, s = 10^-6 ... 10^6 (1500 evaluasi, 30 run)
B) Kasus truss asli, biaya x s (efektif: rasio rentang biaya/displacement), 300 evaluasi, 30 run
HV selalu dihitung pada skala asli/ternormalisasi, sehingga hanya perilaku algoritma yang berubah.
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.sms import SMSEMOA
import benchlib as b
import common as c

OUT = c.ROOT / "results"
N_RUNS = 30
EXP_A = [-6, -4, -2, 0, 2, 4, 6]
EXP_B = [-8, -6, -5, -4, -3, -2, 0, 2]
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE raw", "MOEA/D-DE online"]


class TrussScaled(c.TrussProblem):
    def __init__(self, s):
        super().__init__()
        self.sv = np.array([1.0, s])

    def _evaluate(self, X, out, *a, **k):
        out["F"] = c.evaluate(X) * self.sv


def job_a(t):
    e, alg, seed = t
    prob, pf, _ = b.build("DTLZ2-2", (1.0, 10.0 ** e))
    if alg.startswith("MOEA/D"):
        F = b.run_moead(prob, seed, 1500, normalize="none" if alg.endswith("raw") else "online")
    else:
        F = b.run_ga(alg, prob, seed, 1500)
    return dict(part="DTLZ2", exp=e, alg=alg, seed=seed, hv_pct=100 * b.hv_norm(F, pf) / b.hv_norm(pf, pf))


def job_b(t):
    e, alg, seed = t
    s = 10.0 ** e
    if alg.startswith("MOEA/D"):
        _, h, _, _ = c.run_moead_de(20, 15, seed, normalize="none" if alg.endswith("raw") else "online",
                                    obj_scale=(1.0, s), **c.DEFAULTS["MOEA/D-DE"])
    else:
        cls = {"NSGA2": NSGA2, "SPEA2": SPEA2, "SMS-EMOA": SMSEMOA}[alg]
        hp = c.DEFAULTS[alg]
        a = cls(pop_size=20, sampling=c.lhs_init(20, seed),
                crossover=SBX(prob=hp["p_cr"], eta=hp["eta_cr"], repair=c.RoundRepair()),
                mutation=PM(prob=1.0, prob_var=hp["p_m"], eta=hp["eta_m"], repair=c.RoundRepair()),
                eliminate_duplicates=False)
        F = minimize(TrussScaled(s), a, ("n_eval", 300), seed=seed).algorithm.pop.get("F") / np.array([1.0, s])
        h = c.hv(F)
    return dict(part="Truss", exp=e, alg=alg, seed=seed, hv_pct=100 * h / c.HV_TRUE)


if __name__ == "__main__":
    with Pool() as p:
        ra = p.map(job_a, [(e, a, k) for e in EXP_A for a in ALGS for k in range(N_RUNS)], chunksize=5)
        rb = p.map(job_b, [(e, a, k) for e in EXP_B for a in ALGS for k in range(N_RUNS)], chunksize=5)
    df = pd.DataFrame(ra + rb)
    rng_truss = (c.NADIR - c.IDEAL)
    # rasio efektif = rentang objektif-2 / rentang objektif-1 yang dilihat algoritma (log10)
    df["log10_ratio"] = np.where(df.part == "DTLZ2", df.exp,
                                 df.exp + np.log10(rng_truss[1] / rng_truss[0]))
    df.to_csv(OUT / "dose_response_runs.csv", index=False)
    S = df.groupby(["part", "log10_ratio", "alg"]).hv_pct.agg(["mean", "std"]).reset_index()
    S.to_csv(OUT / "dose_response_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.pivot_table(index=["part", "log10_ratio"], columns="alg", values="mean").round(1).to_string())

    col = {"NSGA2": "#1b9e77", "SPEA2": "#7570b3", "SMS-EMOA": "#d95f02",
           "MOEA/D-DE raw": "#66a61e", "MOEA/D-DE online": "#1f78b4"}
    lab = {"MOEA/D-DE raw": "MOEA/D-DE, raw Tchebycheff", "MOEA/D-DE online": "MOEA/D-DE, normalized",
           "NSGA2": "NSGA-II", "SPEA2": "SPEA2", "SMS-EMOA": "SMS-EMOA"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, part, title in zip(axes, ["DTLZ2", "Truss"], ["(a) DTLZ2, 1,500 evaluations",
                                                          "(b) Skylight truss, 300 evaluations"]):
        for a in ALGS:
            s = S[(S.part == part) & (S.alg == a)].sort_values("log10_ratio")
            ax.errorbar(s.log10_ratio, s["mean"], yerr=s["std"], marker="o", ms=4, capsize=2, lw=1.8,
                        color=col[a], label=lab[a], ls="--" if a.endswith("raw") else "-")
        ax.axvline(0, color="grey", lw=.8, ls=":")
        if part == "Truss":
            ax.axvline(np.log10(rng_truss[1] / rng_truss[0]), color="k", lw=.8, ls="--")
            ax.text(np.log10(rng_truss[1] / rng_truss[0]) + .1, 8, "original\nunits", fontsize=8)
        ax.set(title=title, xlabel="log₁₀ (range of f₂ / range of f₁) seen by the algorithm")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("% of true-front HV (mean ± SD, 30 runs)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "Fig7_scale_dose_response.png", dpi=300)
