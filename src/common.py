"""
Komponen bersama: problem truss (lookup surrogate), LHS bersama, hypervolume,
algoritma pymoo (NSGA2, SPEA2, SMS-EMOA), dan MOEA/D-DE (Li & Zhang, 2009).

Representasi: 4 variabel integer [mat, r_idx, n, h_idx]
  mat 0..5, r_idx 0..8, n 4..12, h_idx 0..25  -> 6*9*9*26 = 12.636 desain
Objektif (diminimalkan): displacement maksimum, biaya material.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc

from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.util.ref_dirs import get_reference_directions

ROOT = Path(__file__).resolve().parents[1]
XL = np.array([0, 0, 4, 0])
XU = np.array([5, 8, 12, 25])

import os
# Ruang desain yang dipakai; bisa diganti lewat env var SPACE (mis. space_rf.csv) untuk uji ketahanan surrogate
SPACE_FILE = os.environ.get("SPACE", "space.csv")
_space = pd.read_csv(ROOT / "data" / SPACE_FILE)
F_SPACE = _space[["disp", "cost"]].values
PF_TRUE = F_SPACE[NonDominatedSorting().do(F_SPACE, only_non_dominated_front=True)]
PF_TRUE = PF_TRUE[np.argsort(PF_TRUE[:, 0])]
IDEAL, NADIR = PF_TRUE.min(0), PF_TRUE.max(0)
REF = np.array([1.1, 1.1])                     # titik referensi pada ruang ternormalisasi
REF_FAR = F_SPACE.max(0)                       # "titik terjauh" pada objektif mentah (seperti penulis)


def norm(F):
    return (F - IDEAL) / (NADIR - IDEAL)


_hv = HV(ref_point=REF)
_igd = IGD(norm(PF_TRUE))
HV_TRUE = _hv(norm(PF_TRUE))


def to_index(X):
    X = np.clip(np.round(X).astype(int), XL, XU)
    m, r, n, h = X.T
    return ((m * 9 + r) * 9 + (n - 4)) * 26 + h


def evaluate(X):
    return F_SPACE[to_index(X)]


def nd(F):
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def hv(F):
    return float(_hv(norm(nd(F))))


_hv_far = HV(ref_point=REF_FAR)


def hv_far(F):
    """HV pada objektif mentah dengan titik referensi terjauh (konvensi Pereira & Leitao)."""
    return float(_hv_far(nd(F)))


def igd(F):
    return float(_igd(norm(nd(F))))


class TrussProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=2, xl=XL.astype(float), xu=XU.astype(float))

    def _evaluate(self, X, out, *a, **k):
        out["F"] = evaluate(X)


class RoundRepair(Repair):
    def _do(self, problem, X, **kwargs):
        return np.clip(np.round(X), XL, XU).astype(float)


def lhs_init(pop_size, seed):
    """Populasi awal LHS yang sama untuk semua algoritma pada run ke-i (seperti di artikel)."""
    u = qmc.LatinHypercube(d=4, seed=seed).random(pop_size)
    return np.floor(XL + u * (XU - XL + 1)).clip(XL, XU).astype(float)


class HVTrace(Callback):
    """HV himpunan non-dominated dari populasi saat ini, per iterasi."""
    def __init__(self):
        super().__init__()
        self.trace = []

    def notify(self, algorithm):
        self.trace.append(hv(algorithm.pop.get("F")))


# ---------------- Default hyperparameter (Metaheuristics.jl, Bagian 3.1 artikel) ----------------
DEFAULTS = {
    "NSGA2":    dict(eta_cr=20, p_cr=0.9, eta_m=20, p_m=0.25),
    "SPEA2":    dict(eta_cr=20, p_cr=0.9, eta_m=20, p_m=0.25),
    "SMS-EMOA": dict(eta_cr=20, p_cr=0.9, eta_m=20, p_m=0.25),
    "MOEA/D-DE": dict(F=0.5, CR=1.0, eta=20, p_m=0.25, delta=0.9),
}


def run_pymoo(name, pop_size, n_iters, seed, eta_cr, p_cr, eta_m, p_m):
    cls = {"NSGA2": NSGA2, "SPEA2": SPEA2, "SMS-EMOA": SMSEMOA}[name]
    alg = cls(pop_size=pop_size,
              sampling=lhs_init(pop_size, seed),
              crossover=SBX(prob=p_cr, eta=eta_cr, repair=RoundRepair()),
              mutation=PM(prob=1.0, prob_var=p_m, eta=eta_m, repair=RoundRepair()),
              eliminate_duplicates=False)
    cb = HVTrace()
    res = minimize(TrussProblem(), alg, ("n_eval", pop_size * n_iters),
                   seed=seed, callback=cb, verbose=False)
    Ffin = res.algorithm.pop.get("F")
    return cb.trace, hv(Ffin), igd(Ffin), Ffin


def run_moead_de(pop_size, n_iters, seed, F=0.5, CR=1.0, eta=20, p_m=0.25, delta=0.9,
                 T=None, n_r=None, weights=None, normalize="none", obj_scale=None):
    """MOEA/D-DE (Li & Zhang, 2009), dipadankan dengan Metaheuristics.jl (MOEAD_DE.jl):
      - tetangga B(i) = T bobot terdekat TANPA i sendiri; subproblem diproses berurutan 1..N
      - DE: v = x_i + F (x_r2 - x_r3), crossover binomial per variabel (tanpa j_rand)
      - mutasi polinomial per variabel dengan peluang p_m; perbaikan batas = nilai acak dalam batas
      - pengganti hingga n_r anggota P (acak), aturan g(child) <= g(parent)
      - Tchebycheff: g = max_k w_k |f_k - z_k|  (normalize="none", seperti aslinya)
    T dan n_r mengikuti penyesuaian penulis: T=max(3, round(.2N)), n_r=max(2, round(.05N)).
    Genotipe kontinu dalam [XL, XU]; dibulatkan saat dievaluasi.

    normalize: "none"   -> objektif mentah (perilaku Metaheuristics.jl)
               "online" -> (f - z) / (z_nad - z), z_nad = maksimum populasi saat ini
               "oracle" -> normalisasi dengan ideal/nadir Pareto front sebenarnya (batas atas)
    """
    rng = np.random.default_rng(seed)
    W = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1) if weights is None else weights
    N = len(W)
    T = T or max(3, round(0.2 * N))
    n_r = n_r or max(2, round(0.05 * N))
    D2 = ((W[:, None, :] - W[None, :, :]) ** 2).sum(-1)
    B = np.argsort(D2, axis=1, kind="stable")[:, 1:T + 1]
    span = (XU - XL).astype(float)

    # obj_scale: pengali objektif yang dilihat algoritma (uji rasio skala); HV dihitung pada skala asli
    sv = np.ones(2) if obj_scale is None else np.asarray(obj_scale, float)
    ev = lambda Xq: evaluate(Xq) * sv

    X = lhs_init(N, seed)
    Fx = ev(X)
    z = Fx.min(0)

    def scale(Fpop):
        if normalize == "none":
            return np.ones(2)
        if normalize == "oracle":
            return (NADIR - IDEAL) * sv
        return np.maximum(Fpop.max(0) - z, 1e-12)

    trace = [hv(Fx / sv)]
    for _ in range(n_iters - 1):
        for i in range(N):
            P = list(B[i]) if rng.random() < delta else list(range(N))
            r2 = rng.choice([p for p in P if p != i])
            r3 = rng.choice([p for p in P if p not in (i, r2)])
            y = X[i].copy()
            for j in range(4):
                if rng.random() < CR:
                    y[j] = X[i][j] + F * (X[r2][j] - X[r3][j])
                if rng.random() < p_m:
                    u = rng.random()
                    dq = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 - 2 * u) ** (1 / (eta + 1))
                    y[j] += dq * span[j]
            out = (y < XL) | (y > XU)
            y[out] = XL[out] + rng.random(out.sum()) * span[out]
            fy = ev(y[None])[0]
            z = np.minimum(z, fy)
            s = scale(np.vstack([Fx, fy]))
            c = 0
            for k in rng.permutation(P):
                if c >= n_r:
                    break
                if np.max(W[k] * np.abs(fy - z) / s) <= np.max(W[k] * np.abs(Fx[k] - z) / s):
                    X[k], Fx[k] = y, fy
                    c += 1
        trace.append(hv(Fx / sv))
    return trace, hv(Fx / sv), igd(Fx / sv), (Fx / sv).copy()


def run(name, pop_size, n_iters, seed, **hp):
    if name == "MOEA/D-DE":
        return run_moead_de(pop_size, n_iters, seed, **hp)
    return run_pymoo(name, pop_size, n_iters, seed, **hp)
