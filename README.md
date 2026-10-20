# Reference points and objective scaling in multi-objective algorithm tuning — code, data and results

This archive contains everything needed to reproduce the experiments, tables and figures of the article

> [Author names] (2026). *Reference points and objective scaling: two hidden choices that change the
> conclusions of multi-objective algorithm tuning.* Manuscript submitted to *Soft Computing*.

**DOI of this archive:** https://doi.org/10.5281/zenodo.[XXXXXXX]

## What the study is about

When multi-objective evolutionary algorithms (MOEAs) are tuned for an expensive engineering problem, three
choices are rarely reported: the **reference point** used to compute the hypervolume (HV), whether objectives
are **normalized** inside decomposition-based algorithms such as MOEA/D, and **how many independent runs**
each configuration receives. Starting from a published study (Pereira & Leitão, 2026) and its open data, this
work

1. replicates that study in Python, using surrogate models and an exhaustive evaluation of all 12,636 designs
   of a skylight truss problem to obtain the true Pareto front;
2. derives, for any number of objectives, how the HV and the number of runs certified by a common
   Central-Limit-Theorem rule depend on the reference point, and checks the result on 2- and 3-objective
   problems;
3. shows that the reported failure of MOEA/D-DE is explained mainly by Tchebycheff scalarization on
   unnormalized objectives, a default in seven widely used libraries;
4. validates run-allocation rules against an independent ground truth and compares five tuning strategies
   (random search, TPE, Gaussian-process Bayesian optimization, SMAC, racing) at equal cost.

No simulator is needed: all truss evaluations come from surrogate models trained on the published data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd src
python 01_build_space.py        # surrogates + full design space + true Pareto front   (~1 min)
python 02_baseline.py           # replication: 4 algorithms x 100 runs                 (<1 min)
python 12_figures.py            # Figs. 1, 2, 3, 6 of the article                      (~1 min)
```

All scripts are run from `src/` and read/write relative to the archive root, so the folder can be placed
anywhere. Every result file used in the article is already included in `results/`; re-running a script
overwrites its own outputs.

## Folder structure

```
data/                 input dataset and the surrogate-based design spaces
  dataset.csv           2,400 Frame3DD evaluations from Pereira (2026)  [third party, CC BY 4.0]
  space.csv             all 12,636 designs predicted by the main surrogate (gradient boosted trees)
  space_rf.csv          same, random-forest surrogate
  space_rbf.csv         same, radial-basis-function surrogate
  pareto_true.csv       true Pareto front of space.csv (251 designs)
src/                  Python scripts, numbered in the order they were run
  common.py             truss problem, algorithms (pymoo + MOEA/D-DE re-implementation), HV conventions
  benchlib.py           benchmark and RE problems, algorithms for continuous problems
results/              CSV results of every experiment
  figures/              publication figures, named as in the article (Fig1 ... Fig8, FigS1)
  diagnostics/          quick-look plots written by some scripts (created on demand)
third_party/reproblems/ RE real-world problem suite (Tanabe & Ishibuchi, 2020)   [third party, MIT]
docs/                 development notes kept during the study (in Indonesian)
```

## Experiments, scripts and outputs

Experiment IDs follow Table 2 of the article. Run times were measured on a 2-core machine.

| Exp. | Question | Scripts | Main outputs | In the article | Time |
|---|---|---|---|---|---|
| — | Surrogates, design space, true front | `01_build_space.py`, `07_alt_surrogates.py` | `surrogate_metrics.csv`, `alt_surrogate_metrics.csv`, `data/space*.csv` | Sect. 3.1, Table 9 | 1 min |
| E1 | Replication of the original baseline | `02_baseline.py` | `baseline_*.csv`, `table1_nruns.csv` | Table 3, Fig. 1 | <1 min |
| E2 | Run count vs reference point | `03_refpoint_sensitivity.py`, `14_theory_check.py`, `22_theory_M_objectives.py` | `refpoint_sensitivity.csv`, `theory_*.csv` | Sect. 4.2, Fig. 2, Appendix A | 2 min |
| E3 | Does the reference point change tuning results? | `05_random_search_refpoint.py`, `09_rs_analysis.py`, `10_invariant_nruns.py` | `rs_*.csv`, `invariant_nruns.csv`, `ranking_reliability.csv` | Table 4, Fig. 3 | 20 min |
| E4 | How many runs does selection need? | `18_ruleB_validation_runs.py`, `19_ruleB_validation_analysis.py` | `ruleB_*.csv` | Table 5, Fig. 4 | 25 min |
| E5 | Five tuning strategies at equal cost | `20_hpo_comparison.py`, `21_hpo_analysis.py` | `hpo_*.csv` | Table 6, Fig. 5 | 45 min per 5 sessions |
| E6 | MOEA/D-DE and objective scaling | `04_moead_normalization.py`, `15_scale_dose_response.py`, `16_fig7_replot.py` | `moead_normalization_*.csv`, `dose_response_*.csv` | Table 7, Figs. 6–7 | 5 min |
| E7 | Normalization on 14 benchmark and real-world problems | `13_extended_benchmarks.py`, `17_ext_summary_fig.py` | `ext_*.csv` | Sect. 4.2, 4.5, Fig. 8 | 10 min |
| E8 | Robustness to the surrogate | `08_surrogate_robustness.py` | `surrogate_robustness.csv` | Table 9 | <1 min |
| E9 | Library defaults | manual inspection of source code (versions/commits in the CSV) | `library_survey.csv` | Table 8 | — |
| Suppl. | Early benchmark check (ZDT1, DTLZ2), not reported in the article | `06_benchmark_generalization.py` | `bench_*.csv`, `figures/FigS1_benchmarks_moead.png` | — | 2 min |

Figures: `12_figures.py` draws Figs. 1, 2, 3 and 6; Fig. 4 comes from `19_…`, Fig. 5 from `21_…`, Fig. 7
from `16_…` (or `15_…`), and Fig. 8 from `17_…`.

## Main result files

| File | One row per | Key columns |
|---|---|---|
| `baseline_runs.csv` | run of an algorithm with default settings | `alg`, `seed`, `hv` (normalized HV), `igd`, `hv_ratio` (share of true-front HV) |
| `rs_runs.csv` | run of a random-search configuration | `alg`, `cfg`, `seed`, `hv_norm` (NORM convention), `hv_far` (FAR convention), `igd` |
| `ruleB_runs.csv` | run used for the ground-truth study | `alg`, `cfg`, `seed`, `pool` (`truth`/`select`), `hv_norm`, `hv_far` |
| `hpo_sessions.csv` | tuning session | `alg`, `method`, `rep`, `runs_used`, chosen `hp`, `true_hv_pct` (re-evaluated with 100 new seeds) |
| `ext_runs.csv` | run on a benchmark or RE problem | `problem`, `n_obj`, `alg`, `seed`, `evals`, `hv_norm`, `hv_far`, `hv_pct` |
| `library_survey.csv` | library | version or commit, default scalarization, normalization option, evidence (file and line) |

HV conventions: **NORM** = objectives normalized by the ideal and nadir points of the reference front,
reference point (1.1, …, 1.1); **FAR** = raw objectives, reference point at the worst value of the design space
(or of 5,000 random solutions for benchmarks), as in the original study.

## Reproducibility notes

* **Seeds.** All runs use fixed seeds (written in the result files), so the CSVs can be regenerated exactly
  with the library versions in `requirements.txt`. Other versions of pymoo or scikit-learn may change
  individual values but not the conclusions.
* **Surrogates instead of the simulator.** Truss objectives are surrogate predictions (hold-out
  R² ≥ 0.999); absolute HV values therefore differ slightly from those in Pereira & Leitão (2026).
* **E5 sessions.** The analysis uses the 5 repetitions (0–4) completed for every algorithm–strategy pair;
  `hpo_sessions.csv` also holds two sessions of repetition 5, which are ignored. `20_hpo_comparison.py`
  resumes where it stopped, so more repetitions can be added.
* **Binary file.** `results/baseline_final_fronts.npy` stores the final fronts of E1 as a NumPy object
  array and is read with `allow_pickle=True`; only load it from this archive.
* **Language.** The article and this README are in English; some code comments and diagnostic plots are in
  Indonesian.

## How to cite

Please cite both the article and this archive; machine-readable metadata are in `CITATION.cff`. If you use
`data/dataset.csv` or the RE problems, also cite their original sources (see `NOTICE.md`).

## Licence

Code: MIT (`LICENSE`). Results and derived data: CC BY 4.0 (`LICENSE-DATA.md`). Third-party data and code
keep their own licences (`NOTICE.md`).
