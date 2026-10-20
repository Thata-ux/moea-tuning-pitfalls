"""
Q2-F — Perbandingan metode HPO pada budget run yang sama.

Metode (budget = 500 run MOEA per sesi HPO, ruang hyperparameter = grid Bagian 4.1 artikel asli):
  RS     : random search, 100 konfigurasi x 5 run
  TPE    : Optuna TPESampler, 100 trial x 5 run
  GP-BO  : scikit-optimize Optimizer (Gaussian process, Expected Improvement), 100 trial x 5 run
  SMAC   : SMAC3 HPO facade (random forest + intensifier), 500 trial (1 trial = 1 run dengan 1 seed)
  RS+SH  : 100 konfigurasi acak + successive halving (alokasi run adaptif), total 500 run
Objektif = HV ternormalisasi (ref 1,1) pada problem truss (surrogate GBT). Seed run selalu baru.
Konfigurasi terpilih dievaluasi ulang dengan 100 seed baru (seed 50000+) -> kualitas sejati.
10 ulangan per metode x algoritma. Hasil ditulis per sesi ke results/hpo_sessions.csv (bisa dilanjutkan).
"""
import os
import sys
import json
import zlib
import shutil
import tempfile
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
import common as c

warnings.filterwarnings("ignore")
OUT = c.ROOT / "results"
FILE = OUT / "hpo_sessions.csv"
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE"]
METHODS = ["RS", "TPE", "GP-BO", "SMAC", "RS+SH"]
REPS = 10
BUDGET_RUNS = 500
RUNS_PER_EVAL = 5
POPS = [5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100]
# grid diskret: nama -> daftar nilai
GRID_GA = {"eta_cr": list(range(5, 31)), "p_cr": [0.7, 0.75, 0.8, 0.85, 0.9],
           "eta_m": list(range(5, 51)), "p_m": [round(0.05 * i, 2) for i in range(1, 11)]}
GRID_DE = {"F": [round(0.3 + 0.1 * i, 1) for i in range(88)], "CR": [round(0.1 * i, 1) for i in range(1, 11)],
           "eta": list(range(10, 51)), "p_m": [round(0.05 * i, 2) for i in range(1, 11)],
           "delta": [0.5, 0.6, 0.7, 0.8, 0.9]}


def grid(alg):
    g = {"pop_idx": list(range(len(POPS)))}
    g.update(GRID_DE if alg == "MOEA/D-DE" else GRID_GA)
    return g


def decode(alg, idx):
    """idx: dict nama -> indeks dalam grid -> (pop, iters, hp)."""
    g = grid(alg)
    pop = POPS[int(idx["pop_idx"])]
    hp = {k: g[k][int(idx[k])] for k in g if k != "pop_idx"}
    return pop, 300 // pop, hp


class Runner:
    """Menjalankan MOEA dengan seed yang selalu baru; menghitung run yang terpakai."""

    def __init__(self, alg, seed0):
        self.alg, self.next_seed, self.used = alg, seed0, 0

    def run(self, idx, seed=None):
        pop, it, hp = decode(self.alg, idx)
        if seed is None:
            seed, self.next_seed = self.next_seed, self.next_seed + 1
        self.used += 1
        return c.run(self.alg, pop, it, int(seed), **hp)[1]

    def mean(self, idx, n):
        return float(np.mean([self.run(idx) for _ in range(n)]))


def m_rs(alg, R, rng):
    g = grid(alg)
    best, bidx = -1, None
    for _ in range(BUDGET_RUNS // RUNS_PER_EVAL):
        idx = {k: int(rng.integers(len(v))) for k, v in g.items()}
        v = R.mean(idx, RUNS_PER_EVAL)
        if v > best:
            best, bidx = v, idx
    return bidx


def m_optuna(alg, R, rng, kind):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    g = grid(alg)
    seed = int(rng.integers(1 << 30))
    sampler = (optuna.samplers.TPESampler(seed=seed, n_startup_trials=10) if kind == "TPE"
               else optuna.samplers.GPSampler(seed=seed, n_startup_trials=10))
    st = optuna.create_study(direction="maximize", sampler=sampler)

    def obj(t):
        idx = {k: t.suggest_int(k, 0, len(v) - 1) for k, v in g.items()}
        return R.mean(idx, RUNS_PER_EVAL)

    st.optimize(obj, n_trials=BUDGET_RUNS // RUNS_PER_EVAL)
    return {k: int(v) for k, v in st.best_params.items()}


def m_gp(alg, R, rng):
    from skopt import Optimizer
    from skopt.space import Integer
    g = grid(alg)
    keys = list(g)
    opt = Optimizer([Integer(0, len(g[k]) - 1) for k in keys], base_estimator="GP", acq_func="EI",
                    n_initial_points=10, random_state=int(rng.integers(1 << 30)))
    for _ in range(BUDGET_RUNS // RUNS_PER_EVAL):
        x = opt.ask()
        idx = {k: int(v) for k, v in zip(keys, x)}
        opt.tell(x, -R.mean(idx, RUNS_PER_EVAL))
    best = opt.Xi[int(np.argmin(opt.yi))]
    return {k: int(v) for k, v in zip(keys, best)}


def m_smac(alg, R, rng):
    from ConfigSpace import ConfigurationSpace, Integer
    from smac import HyperparameterOptimizationFacade as HPO, Scenario
    g = grid(alg)
    cs = ConfigurationSpace(seed=int(rng.integers(1 << 30)))
    cs.add([Integer(k, (0, len(v) - 1)) for k, v in g.items()])
    tmp = tempfile.mkdtemp(prefix="smac_")
    sc = Scenario(cs, deterministic=False, n_trials=BUDGET_RUNS, seed=int(rng.integers(1 << 30)),
                  output_directory=tmp)

    def f(config, seed=0):
        return -R.run(dict(config))          # SMAC meminimalkan; seed baru dari Runner

    smac = HPO(sc, f, overwrite=True, logging_level=40)
    inc = smac.optimize()
    shutil.rmtree(tmp, ignore_errors=True)
    return {k: int(v) for k, v in dict(inc).items()}


def m_rs_sh(alg, R, rng):
    g = grid(alg)
    cands = [{k: int(rng.integers(len(v))) for k, v in g.items()} for _ in range(100)]
    alive = list(range(len(cands)))
    sums, cnt = np.zeros(len(cands)), np.zeros(len(cands), int)
    rounds = int(np.ceil(np.log2(len(cands)))) + 1
    for _ in range(rounds):
        per = max(1, (BUDGET_RUNS // rounds) // len(alive))
        for i in alive:
            for _k in range(per):
                if R.used >= BUDGET_RUNS:
                    break
                sums[i] += R.run(cands[i]); cnt[i] += 1
        if len(alive) == 1 or R.used >= BUDGET_RUNS:
            break
        means = sums[alive] / np.maximum(cnt[alive], 1)
        alive = [alive[j] for j in np.argsort(-means)[:max(1, len(alive) // 2)]]
    means = sums[alive] / np.maximum(cnt[alive], 1)
    return cands[alive[int(np.argmax(means))]]


def session(task):
    alg, method, rep = task
    rng = np.random.default_rng(zlib.crc32(f"{alg}|{method}|{rep}".encode()))  # deterministik antarproses
    R = Runner(alg, seed0=100000 + 10000 * rep)
    fn = {"RS": m_rs, "TPE": lambda a, r, g: m_optuna(a, r, g, "TPE"),
          "GP-BO": m_gp, "SMAC": m_smac, "RS+SH": m_rs_sh}[method]
    idx = fn(alg, R, rng)
    pop, it, hp = decode(alg, idx)
    true = [c.run(alg, pop, it, 50000 + s, **hp)[1] for s in range(100)]
    row = dict(alg=alg, method=method, rep=rep, runs_used=R.used, pop_size=pop, n_iters=it,
               hp=json.dumps(hp), true_hv=float(np.mean(true)), true_hv_pct=100 * float(np.mean(true)) / c.HV_TRUE)
    return row


if __name__ == "__main__":
    done = set()
    if FILE.exists():
        d = pd.read_csv(FILE)
        done = set(zip(d.alg, d.method, d.rep))
    tasks = [(a, m, r) for r in range(REPS) for a in ALGS for m in METHODS if (a, m, r) not in done]
    print("sesi tersisa:", len(tasks), flush=True)
    with Pool() as p:
        for row in p.imap_unordered(session, tasks):
            pd.DataFrame([row]).to_csv(FILE, mode="a", header=not FILE.exists(), index=False)
            print(row["alg"], row["method"], row["rep"], row["runs_used"], round(row["true_hv_pct"], 3), flush=True)
