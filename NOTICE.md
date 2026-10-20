# Third-party material and attribution

This archive redistributes two third-party resources. Both licences permit redistribution; the
attribution below is required.

## 1. Skylight truss evaluation dataset — `data/dataset.csv`

* **Source:** Pereira, I. (2026). *DataOptimizingOptimizers: Data produced for the paper Optimizing the
  Optimizers* (version 2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18344135
* **Related article:** Pereira, I., & Leitão, A. (2026). Optimizing the optimizers: Hyperparameter
  optimization for multi-objective metaheuristics. *Advanced Engineering Informatics*, 74, 104690.
  https://doi.org/10.1016/j.aei.2026.104690
* **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
* **Changes made:** none to `dataset.csv`, which is included unchanged (2,400 Frame3DD evaluations,
  integer-coded variables). The files `data/space*.csv` and `data/pareto_true.csv` are derived from it
  by surrogate modelling (`src/01_build_space.py`, `src/07_alt_surrogates.py`).
* The original authors are not associated with this work and do not endorse it.

## 2. RE real-world multi-objective problem suite — `third_party/reproblems/`

* **Source:** Tanabe, R., & Ishibuchi, H. (2020). An easy-to-use real-world multi-objective optimization
  problem suite. *Applied Soft Computing*, 89, 106078. https://doi.org/10.1016/j.asoc.2020.106078
  Code and reference fronts: https://github.com/ryojitanabe/reproblems
* **Files included:** `reproblem.py` (Python implementation), `approximated_Pareto_fronts/`,
  `ideal_nadir_points/`.
* **Licence:** MIT License, copyright Ryoji Tanabe; the full text is in
  `third_party/reproblems/LICENSE.txt` and must be kept with the files.
* **Changes made:** none.

## Software used but not redistributed

pymoo, scikit-learn, Optuna, SMAC3, scikit-optimize, NumPy, SciPy, pandas and Matplotlib are installed
from PyPI (`requirements.txt`) and are not included here. The MOEA/D-DE implementation in
`src/common.py` and `src/benchlib.py` was written in Python for this study, following the published
algorithm (Li & Zhang, 2009) and the behaviour of Metaheuristics.jl 3.5.0 (MIT licence,
https://github.com/jmejia8/Metaheuristics.jl).
