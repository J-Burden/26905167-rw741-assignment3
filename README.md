# RW741 Assignment 3: ensemble learning on UNSW-NB15

Random forest compared with bagged trees, a classification tree and a
k-nearest neighbours model on the edited UNSW-NB15 network traffic dataset.
Student number: 26905167.

## Data
Place `networkTraffic.csv` (supplied on STEMlearn) in `data/`, or set the
environment variable `NT_SRC` to its path. Results are written to `results/`
(override with `NT_WORK`).

## Execution order
| Step | Script | Purpose |
|---|---|---|
| 1 | `code/stage2_knn.py` | KNN pre-processing variants and k / weighting sweep (5-fold CV, development partition) |
| 2 | `code/stage2_tree.py` | Tree grid, cost-complexity sweep, encoding check |
| 3 | `code/stage3_rf.py` | Forest stage A (q x class weights, diversity), stage B (leaf size, depth), stage C (number of trees) |
| 4 | `code/stage3b_rf_encoding.py` | Forest encoding check (integer codes vs indicators) |
| 5 | `code/stage4_final.py` | Final evaluation on the held-out generalisation set, 10 seeds, bootstrap and McNemar tests |
| 6 | `code/analysis_conflicts.py` | Label-conflict analysis, majority-label ceiling, post-hoc analysis |
| 7 | `code/make_figs.py tuning` and `code/make_figs.py final` | Figures |

Run every script from inside `code/`. All fitted transformations are
learned inside the cross-validation folds (see `common.py`).

## Environment
Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2. Total run time is about 2.5 hours on two cores.

## Report
`report/26905167RW741assignment3.pdf`. To rebuild: in `report/`, run
`pdflatex main`, `bibtex main`, `makeglossaries main`, then `pdflatex main` twice.
