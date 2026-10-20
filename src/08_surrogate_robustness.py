"""
Eksperimen 3b — Apakah temuan bergantung pada surrogate?
Untuk tiap ruang desain (GBT, RF, RBF): baseline 4 algoritma (100 run), n_runs (CLT) pada
ref ternormalisasi 1,1 vs titik terjauh mentah, dan MOEA/D-DE mentah vs normalisasi oracle.
Pemakaian: python 08_surrogate_robustness.py   (driver; tiap ruang dijalankan di subprocess)
"""
import json
import os
import subprocess
import sys

SPACES = {"GBT (utama)": "space.csv", "Random Forest": "space_rf.csv", "RBF": "space_rbf.csv"}


def job(a):
    import common as c
    alg, s, mode = a
    if alg == "MOEA/D-DE":
        _, h, _, F = c.run_moead_de(20, 15, s, normalize=mode, **c.DEFAULTS[alg])
    else:
        _, h, _, F = c.run(alg, 20, 15, s, **c.DEFAULTS[alg])
    return alg, mode, h, c.hv_far(F)


def worker():
    from multiprocessing import Pool
    import numpy as np
    from scipy.stats import norm as N
    import common as c

    algs = ["NSGA2", "SPEA2", "SMS-EMOA"]
    tasks = [(a, s, "none") for a in algs + ["MOEA/D-DE"] for s in range(100)]
    tasks += [("MOEA/D-DE", s, "oracle") for s in range(100)]
    with Pool() as p:
        res = p.map(job, tasks)
    z = N.ppf(0.975)
    out = {"pareto_size": len(c.PF_TRUE)}
    for a in algs + ["MOEA/D-DE"]:
        hn = np.array([h for aa, m, h, _ in res if aa == a and m == "none"])
        hf = np.array([f for aa, m, _, f in res if aa == a and m == "none"])
        out[a] = dict(hv_pct=100 * hn.mean() / c.HV_TRUE,
                      n_runs_ref1_1=int(np.ceil((z * hn.std(ddof=1) / hn.mean() / 0.002) ** 2)),
                      n_runs_far=int(np.ceil((z * hf.std(ddof=1) / hf.mean() / 0.002) ** 2)))
    ho = np.array([h for aa, m, h, _ in res if m == "oracle"])
    out["MOEA/D-DE (norm)"] = dict(hv_pct=100 * ho.mean() / c.HV_TRUE)
    print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
        sys.exit()
    import pandas as pd
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for label, f in SPACES.items():
        r = subprocess.run([sys.executable, __file__, "--worker"], cwd=here, capture_output=True,
                           text=True, env={**os.environ, "SPACE": f}, check=True)
        o = json.loads(r.stdout.strip().splitlines()[-1])
        for a in ["NSGA2", "SPEA2", "SMS-EMOA", "MOEA/D-DE", "MOEA/D-DE (norm)"]:
            rows.append(dict(surrogate=label, pareto_size=o["pareto_size"], alg=a, **o[a]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(here, "..", "results", "surrogate_robustness.csv"), index=False)
    print(df.round(2).to_string(index=False))
