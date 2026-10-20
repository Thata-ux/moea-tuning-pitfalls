"""
Pustaka benchmark umum (2 dan 3 objektif) untuk eksperimen lanjutan (Q2).
- Problem: DTLZ2 (pymoo) dan suite RE (Tanabe & Ishibuchi, 2020; MIT, third_party/reproblems)
- Algoritma: NSGA-II, SPEA2, SMS-EMOA (pymoo) dan MOEA/D-DE (padanan Metaheuristics.jl)
- Metrik: HV ternormalisasi (ideal/nadir front referensi, r = 1,1) dan HV mentah dengan titik terjauh
"""
import sys
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.stats import qmc
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

ROOT = Path(__file__).resolve().parents[1]
RE_DIR = ROOT / "third_party" / "reproblems"
sys.path.insert(0, str(RE_DIR))
import reproblem as RE  # noqa: E402

RE_NAMES = ["RE21", "RE22", "RE23", "RE24", "RE25", "RE31", "RE32", "RE33", "RE34", "RE35", "RE36", "RE37"]


class Wrapped(Problem):
    """Problem pymoo dengan skala objektif opsional (f_m * s_m)."""

    def __init__(self, n_var, n_obj, xl, xu, fn, scale=None):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=np.asarray(xl, float), xu=np.asarray(xu, float))
        self.fn = fn
        self.s = np.ones(n_obj) if scale is None else np.asarray(scale, float)

    def _evaluate(self, X, out, *a, **k):
        out["F"] = self.fn(np.atleast_2d(X)) * self.s


@lru_cache(maxsize=None)
def build(name, scale=None):
    """Kembalikan (problem, front referensi, titik terjauh). scale = tuple pengali per objektif."""
    if name.startswith("RE"):
        p = getattr(RE, name)()
        fn = lambda X, p=p: np.array([p.evaluate(x) for x in X])
        prob = Wrapped(p.n_variables, p.n_objectives, p.lbound, p.ubound, fn, scale)
        pf = np.loadtxt(RE_DIR / "approximated_Pareto_fronts" / f"reference_points_{name}.dat")
    else:  # "DTLZ2-2" atau "DTLZ2-3"
        m = int(name.split("-")[1])
        base = get_problem("dtlz2", n_var=m + 5, n_obj=m)
        fn = lambda X, b=base: b.evaluate(X)
        prob = Wrapped(base.n_var, m, base.xl, base.xu, fn, scale)
        rd = get_reference_directions("das-dennis", m, n_partitions=99 if m == 2 else 30)
        pf = rd / np.linalg.norm(rd, axis=1, keepdims=True)
    pf = pf * prob.s
    rng = np.random.default_rng(0)
    big = prob.evaluate(prob.xl + rng.random((5000, prob.n_var)) * (prob.xu - prob.xl))
    far = np.maximum(big.max(0), pf.max(0))
    return prob, pf, far


def pop_size(n_obj):
    return 20 if n_obj == 2 else 21


def weights(n_obj):
    return get_reference_directions("das-dennis", n_obj, n_partitions=19 if n_obj == 2 else 5)


def nd(F):
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def lhs(prob, n, seed):
    u = qmc.LatinHypercube(d=prob.n_var, seed=seed).random(n)
    return prob.xl + u * (prob.xu - prob.xl)


def run_ga(alg, prob, seed, evals):
    cls = {"NSGA2": NSGA2, "SPEA2": SPEA2, "SMS-EMOA": SMSEMOA}[alg]
    n = pop_size(prob.n_obj)
    a = cls(pop_size=n, sampling=lhs(prob, n, seed), crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=1.0, prob_var=0.25, eta=20), eliminate_duplicates=False)
    return minimize(prob, a, ("n_eval", evals), seed=seed).algorithm.pop.get("F")


def run_moead(prob, seed, evals, normalize="none", oracle=None,
              F_=0.5, CR=1.0, eta=20, p_m=0.25, delta=0.9):
    """MOEA/D-DE seperti Metaheuristics.jl; normalize: none | online | oracle (oracle = (ideal, nadir))."""
    rng = np.random.default_rng(seed)
    W = weights(prob.n_obj)
    N, D = len(W), prob.n_var
    T, n_r = max(3, round(0.2 * N)), max(2, round(0.05 * N))
    B = np.argsort(((W[:, None] - W[None]) ** 2).sum(-1), axis=1, kind="stable")[:, 1:T + 1]
    xl, xu = prob.xl, prob.xu
    X = lhs(prob, N, seed)
    Fx = prob.evaluate(X)
    z = Fx.min(0)
    used = N
    while used < evals:
        for i in range(N):
            if used >= evals:
                break
            P = list(B[i]) if rng.random() < delta else list(range(N))
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
            if normalize == "none":
                s = np.ones(prob.n_obj)
            elif normalize == "oracle":
                s = oracle[1] - oracle[0]
            else:
                s = np.maximum(np.vstack([Fx, fy]).max(0) - z, 1e-12)
            cnt = 0
            for k in rng.permutation(P):
                if cnt >= n_r:
                    break
                if np.max(W[k] * np.abs(fy - z) / s) <= np.max(W[k] * np.abs(Fx[k] - z) / s):
                    X[k], Fx[k] = y, fy
                    cnt += 1
    return Fx


def hv_norm(F, pf, r=1.1):
    ideal, nadir = pf.min(0), pf.max(0)
    return float(HV(ref_point=np.full(pf.shape[1], r))((nd(F) - ideal) / (nadir - ideal)))


def hv_far(F, far):
    return float(HV(ref_point=far)(nd(F)))


def scale_ratio(pf):
    rng_ = pf.max(0) - pf.min(0)
    return float(rng_.max() / rng_.min())
