"""
Q2-G — Verifikasi penurunan untuk M objektif (M = 3).

Proposisi: untuk front A (ternormalisasi), m = titik ideal A, u = titik maksimum A, r = (R,...,R) dengan R >= max u,
    HV(A; R) = R^M - R^(M-1) * sum_k m_k + O(R^(M-2)).
Konsekuensi: CV -> SD(sum_k m_k) / R  dan  n_A ∝ R^-2 untuk semua M; pada R besar peringkat HV = peringkat -sum_k m_k.
Uji: DTLZ2-3 dan RE33, RE34, RE37 (3 objektif), NSGA-II / SPEA2 / SMS-EMOA, 30 run, 3000 evaluasi.
Diukur: kemiringan log n_A vs log R (R >= 5), CV teramati vs SD(sum m)/R, Kendall tau(HV, -sum m),
dan sisa orde-dua: (prod(R - m_k) - HV) / R^(M-2) harus mendekati konstan.
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import norm as N, kendalltau
from pymoo.indicators.hv import HV
import benchlib as b

OUT = b.ROOT / "results"
PROBS = ["DTLZ2-3", "RE33", "RE34", "RE37"]
ALGS = ["NSGA2", "SPEA2", "SMS-EMOA"]
RS = [1.1, 2, 3, 5, 10, 20, 50]
Z = N.ppf(0.975)


def job(t):
    pb, a, seed = t
    prob, pf, _ = b.build(pb)
    F = b.nd(b.run_ga(a, prob, seed, 3000))
    lo, hi = pf.min(0), pf.max(0)
    Fn = (F - lo) / (hi - lo)
    return pb, a, seed, Fn


if __name__ == "__main__":
    with Pool() as p:
        res = p.map(job, [(pb, a, s) for pb in PROBS for a in ALGS for s in range(30)], chunksize=4)
    rows, slopes = [], []
    for pb in PROBS:
        for a in ALGS:
            Fs = [F for q, aa, _, F in res if q == pb and aa == a]
            m = np.array([F.min(0) for F in Fs])
            M = m.shape[1]
            Rmin = max(F.max() for F in Fs)                    # R harus di luar semua front
            ns = []
            for R in RS:
                if R < Rmin and R != 1.1:
                    continue
                hv = np.array([HV(ref_point=np.full(M, R))(F) for F in Fs])
                cv = hv.std(ddof=1) / hv.mean()
                resid = (np.prod(R - m, axis=1) - hv) / R ** (M - 2)
                n = (Z * cv / 0.002) ** 2
                ns.append((R, n))
                rows.append(dict(problem=pb, alg=a, R=R, cv_obs=cv, cv_pred=m.sum(1).std(ddof=1) / R,
                                 n_runs=n, tau_hv_vs_sum_minima=kendalltau(hv, -m.sum(1)).statistic,
                                 second_order_resid_mean=resid.mean(), second_order_resid_sd=resid.std(ddof=1)))
            big = np.array([x for x in ns if x[0] >= 5])
            if len(big) >= 3:
                slopes.append(dict(problem=pb, alg=a, slope=np.polyfit(np.log(big[:, 0]), np.log(big[:, 1]), 1)[0]))
    D = pd.DataFrame(rows); D.to_csv(OUT / "theory_M3_check.csv", index=False)
    S = pd.DataFrame(slopes); S.to_csv(OUT / "theory_M3_slopes.csv", index=False)
    pd.set_option("display.width", 200)
    print(D[D.R.isin([1.1, 10, 50])].round(4).to_string(index=False))
    print("\nKemiringan log n vs log R (R >= 5), teori = -2:")
    print(S.round(3).to_string(index=False))
    print("\nRingkasan: kemiringan median %.3f, rentang %.3f..%.3f; tau(R=50) median %.3f, min %.3f" % (
        S.slope.median(), S.slope.min(), S.slope.max(),
        D[D.R == 50].tau_hv_vs_sum_minima.median(), D[D.R == 50].tau_hv_vs_sum_minima.min()))
