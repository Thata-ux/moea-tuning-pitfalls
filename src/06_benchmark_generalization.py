"""
Eksperimen 2 — Generalisasi kedua temuan ke problem benchmark kontinu.

Problem: ZDT1 (10 var) dan DTLZ2 (2 objektif, 6 var), masing-masing dalam dua versi:
  - seimbang : skala asli (kedua objektif ~[0,1])
  - timpang  : f2 dikali 1e4 (meniru displacement ~1e-2 vs biaya ~1e3 pada kasus truss)
Budget 300 evaluasi (pop 20 x 15), 100 run, populasi awal LHS bersama per run.
Temuan 1: CV & n_runs vs titik referensi (ternormalisasi 1,1..20 dan "terjauh" mentah).
Temuan 2: MOEA/D-DE Tchebycheff mentah vs ternormalisasi (budget 300 dan 1500).
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import qmc, norm as N, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.util.ref_dirs import get_reference_directions
import common as c

OUT = c.ROOT / "results"
DIAG = OUT / "diagnostics"  # quick-look plots (labels partly in Indonesian)
DIAG.mkdir(parents=True, exist_ok=True)
N_RUNS, POP = 100, 20
SCALE = 1e4
Z95 = N.ppf(0.975)
E_REL = 0.002


def make(name):
    return get_problem("zdt1", n_var=10) if name == "ZDT1" else get_problem("dtlz2", n_var=6, n_obj=2)


class Scaled(Problem):
    def __init__(self, base, s):
        super().__init__(n_var=base.n_var, n_obj=2, xl=base.xl, xu=base.xu)
        self.base, self.s = base, np.array([1.0, s])

    def _evaluate(self, X, out, *a, **k):
        out["F"] = self.base.evaluate(X) * self.s


def setup(name, scaled):
    base = make(name)
    s = SCALE if scaled else 1.0
    prob = Scaled(base, s)
    t = np.linspace(0, 1, 500)
    if name == "ZDT1":                       # f2 = 1 - sqrt(f1)
        pf0 = np.column_stack([t, 1 - np.sqrt(t)])
    else:                                    # DTLZ2, 2 objektif: seperempat lingkaran
        pf0 = np.column_stack([np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)])
    pf = pf0 * np.array([1.0, s])
    rng = np.random.default_rng(0)
    big = prob.evaluate(prob.xl + rng.random((20000, prob.n_var)) * (prob.xu - prob.xl))
    return prob, pf, big.max(0)


def nd(F):
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def lhs(prob, n, seed):
    u = qmc.LatinHypercube(d=prob.n_var, seed=seed).random(n)
    return prob.xl + u * (prob.xu - prob.xl)


def run_ga(alg, prob, seed, evals):
    cls = {"NSGA2": NSGA2, "SPEA2": SPEA2, "SMS-EMOA": SMSEMOA}[alg]
    a = cls(pop_size=POP, sampling=lhs(prob, POP, seed), crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=1.0, prob_var=0.25, eta=20), eliminate_duplicates=False)
    return minimize(prob, a, ("n_eval", evals), seed=seed).algorithm.pop.get("F")


def run_moead(prob, seed, evals, scale_vec=None, F_=0.5, CR=1.0, eta=20, p_m=0.25, delta=0.9):
    """Sama dengan c.run_moead_de (padanan Metaheuristics.jl), versi kontinu umum."""
    rng = np.random.default_rng(seed)
    W = get_reference_directions("das-dennis", 2, n_partitions=POP - 1)
    Nn, D = len(W), prob.n_var
    T, n_r = max(3, round(0.2 * Nn)), max(2, round(0.05 * Nn))
    B = np.argsort(((W[:, None] - W[None]) ** 2).sum(-1), axis=1, kind="stable")[:, 1:T + 1]
    xl, xu = prob.xl, prob.xu
    X = lhs(prob, Nn, seed)
    Fx = prob.evaluate(X)
    z = Fx.min(0)
    s = np.ones(2) if scale_vec is None else scale_vec
    used = Nn
    while used < evals:
        for i in range(Nn):
            if used >= evals:
                break
            P = list(B[i]) if rng.random() < delta else list(range(Nn))
            r2 = rng.choice([p for p in P if p != i])
            r3 = rng.choice([p for p in P if p not in (i, r2)])
            y = X[i].copy()
            for j in range(D):
                if rng.random() < CR:
                    y[j] = X[i][j] + F_ * (X[r2][j] - X[r3][j])
                if rng.random() < p_m:
                    u = rng.random()
                    dq = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 - 2 * u) ** (1 / (eta + 1))
                    y[j] += dq * (xu[j] - xl[j])
            bad = (y < xl) | (y > xu)
            y[bad] = xl[bad] + rng.random(bad.sum()) * (xu - xl)[bad]
            fy = prob.evaluate(y[None])[0]
            used += 1
            z = np.minimum(z, fy)
            cnt = 0
            for k in rng.permutation(P):
                if cnt >= n_r:
                    break
                if np.max(W[k] * np.abs(fy - z) / s) <= np.max(W[k] * np.abs(Fx[k] - z) / s):
                    X[k], Fx[k] = y, fy
                    cnt += 1
    return Fx


def job(args):
    pname, scaled, alg, seed, evals = args
    prob, pf, far = setup(pname, scaled)
    if alg == "MOEA/D-DE":
        F = run_moead(prob, seed, evals)
    elif alg == "MOEA/D-DE (norm)":
        F = run_moead(prob, seed, evals, scale_vec=pf.max(0) - pf.min(0))
    else:
        F = run_ga(alg, prob, seed, evals)
    return pname, scaled, alg, seed, evals, nd(F)


if __name__ == "__main__":
    PROBS = [("ZDT1", False), ("ZDT1", True), ("DTLZ2", False), ("DTLZ2", True)]
    GA = ["NSGA2", "SPEA2", "SMS-EMOA"]
    MO = ["MOEA/D-DE", "MOEA/D-DE (norm)"]
    tasks = [(p, s, a, k, 300) for p, s in PROBS for a in GA + MO for k in range(N_RUNS)]
    tasks += [(p, s, a, k, 1500) for p, s in PROBS for a in MO for k in range(N_RUNS)]
    with Pool() as pool:
        res = pool.map(job, tasks, chunksize=8)

    info = {(p, s): setup(p, s) for p, s in PROBS}
    rows = []
    for pname, scaled, alg, seed, evals, F in res:
        prob, pf, far = info[(pname, scaled)]
        ideal, nadir = pf.min(0), pf.max(0)
        Fn = (F - ideal) / (nadir - ideal)
        row = dict(problem=pname, version="timpang" if scaled else "seimbang", alg=alg, seed=seed,
                   evals=evals, hv_far=HV(ref_point=far)(F))
        for r in (1.1, 2, 5, 10, 20):
            row[f"hv_{r}"] = HV(ref_point=np.array([r, r]))(Fn)
        row["hv_ratio_%"] = 100 * row["hv_1.1"] / HV(ref_point=np.array([1.1, 1.1]))((pf - ideal) / (nadir - ideal))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bench_runs.csv", index=False)

    # ---- Temuan 1: n_runs vs titik referensi (budget 300, tiga GA) ----
    t1 = []
    for (p, v, a), g in df[(df.evals == 300) & df.alg.isin(GA)].groupby(["problem", "version", "alg"]):
        row = dict(problem=p, version=v, alg=a)
        for col, lab in [("hv_1.1", "ref 1.1"), ("hv_2", "ref 2"), ("hv_5", "ref 5"),
                         ("hv_10", "ref 10"), ("hv_20", "ref 20"), ("hv_far", "terjauh (mentah)")]:
            cv = g[col].std(ddof=1) / g[col].mean()
            row[lab] = int(np.ceil((Z95 * cv / E_REL) ** 2))
        t1.append(row)
    t1 = pd.DataFrame(t1)
    t1.to_csv(OUT / "bench_nruns_refpoint.csv", index=False)
    print("Temuan 1 — n_runs (CI 95%, eps 0,2% rata-rata HV):")
    print(t1.to_string(index=False))

    # ---- Temuan 2: MOEA/D-DE mentah vs ternormalisasi ----
    t2 = []
    for (p, v, e), g in df[df.alg.isin(MO)].groupby(["problem", "version", "evals"]):
        a = g[g.alg == "MOEA/D-DE"]["hv_ratio_%"]
        b = g[g.alg == "MOEA/D-DE (norm)"]["hv_ratio_%"]
        nsga = df[(df.problem == p) & (df.version == v) & (df.alg == "NSGA2")]["hv_ratio_%"]
        t2.append(dict(problem=p, version=v, evals=e, mentah=a.mean(), norm=b.mean(),
                       p_value=mannwhitneyu(b, a, alternative="greater").pvalue,
                       nsga2_300=nsga.mean()))
    t2 = pd.DataFrame(t2)
    t2.to_csv(OUT / "bench_moead_normalization.csv", index=False)
    print("\nTemuan 2 — MOEA/D-DE, % HV front sebenarnya:")
    print(t2.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, p in zip(axes, ["ZDT1", "DTLZ2"]):
        sub = t2[(t2.problem == p) & (t2.evals == 300)]
        x = np.arange(2)
        ax.bar(x - 0.2, sub.mentah, 0.4, label="Tchebycheff mentah", color="#66a61e")
        ax.bar(x + 0.2, sub.norm, 0.4, label="Tchebycheff ternormalisasi", color="#1f78b4")
        ax.scatter(x, sub.nsga2_300, marker="_", s=600, color="k", label="NSGA2", zorder=3)
        ax.set_xticks(x, ["seimbang", f"timpang (f2 x {SCALE:.0e})"])
        ax.set(title=f"{p} — MOEA/D-DE (300 evaluasi)", ylabel="% HV front sebenarnya", ylim=(0, 105))
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(DIAG / "fig_bench_moead.png", dpi=160)
