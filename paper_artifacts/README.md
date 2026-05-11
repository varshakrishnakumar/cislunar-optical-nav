# `paper_artifacts/` — canonical CSV + figure + summary bundle

This directory is the **reviewer-facing** snapshot of every n=1000
production artifact the journal manuscript leans on. It is tracked in
git (the broader `results/mc/` tree is gitignored to keep the
working-developer footprint small). On a fresh clone, every paper
figure can be regenerated from the contents of this directory; no
Monte Carlo re-run required.

If a paper number drifts from what is reported here, the bug is either
in the source CSV, the aggregator script, or the manuscript prose.
This directory is the single source of truth.

## Contents

- `csv/` — canonical n=1000 production CSVs, one per filename-namespaced
  source directory under `results/mc/`. The script-side filename convention is
  `<source_dir>__<original_basename>.csv` so the origin is recoverable from the
  filename alone.
- `figures/` — canonical PNGs accepted into the manuscript.
- `summaries/` — canonical text summaries (`.txt`) emitted by the
  aggregator scripts.

## Canonical CSVs and what they back

| CSV | Backs |
| --- | --- |
| `phase_d_production__06c_baseline_results.csv` | Tab.~3 (baseline MC), Fig.~12 curve C1 (CR3BP baseline), Fig.~12 curve C4 (uncorrected reference) |
| `phase_d_production_spice__06c_baseline_results.csv` | Fig.~12 curve C2 (SPICE point-mass truth), Fig.~19 (SPICE comparison) |
| `phase_f_landmarks_pointing__06r_landmarks_under_pointing_degradation.csv` | Tab.~9, Fig.~16 (Landmarks Under Pointing Degradation 3×5 grid), Fig.~12 curve C3 (Moon+L2 landmarks) |
| `phase_g_estimator_ablation__06s_estimator_ablation.csv` | Tab.~13, Fig.~20 (EKF/IEKF/UKF ablation) |
| `phase_h_central_ecdf__06t_success_ecdf_central.csv` | Tab.~5, Fig.~12 (success-vs-tolerance curves interpolated) |

## Rebuilding the central figure from this directory

```bash
python scripts/06t_success_ecdf_central.py
```

The driver auto-detects `paper_artifacts/csv/` and uses it as the
preferred source; if missing it falls back to the developer tree under
`results/mc/`.

## Canonical numbers

These should match the manuscript exactly (or to the rounding shown):

| metric | value |
|--------|------:|
| Baseline median terminal miss | **61.64 km** |
| Baseline p95 terminal miss    | **228.27 km** |
| Baseline pass rate @ 390 km   | 99.10% |
| Baseline pass rate @ 39 km    | 32.80% |
| Baseline pass rate @ 25 km    | 18.10% |
| SPICE-truth median miss       | 182.88 km |
| Moon + L2 landmarks median    | 25.07 km |
| Fixed-pointing collapse miss  | 493.73 km |
| EKF / IEKF / UKF median miss  | 61.61 / 61.64 / 91.38 km |
| UKF NEES band fraction        | 99.90% |

Production runs are at `q_a = 1.0e-14` with `1 LU = 389,703.2648 km`
(JPL periodic-orbits catalog characteristic length). See
[`scripts/_paper_constants.py`](../scripts/_paper_constants.py) for the
canonical constants and [`docs/experiment_manifest.md`](../docs/experiment_manifest.md)
for the full figure/table → script mapping.
