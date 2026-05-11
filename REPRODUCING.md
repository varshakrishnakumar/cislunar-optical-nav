# Reproducing the cislunar bearing-only OpNav paper

This file is the reviewer-facing entry point. It is intentionally short.
For the full mapping of figures and tables to scripts, configs, seed
counts, and approximate runtime, see
[`docs/experiment_manifest.md`](docs/experiment_manifest.md).

The manuscript and its compiled PDF live at
[`reports/final-report/final_report.tex`](reports/final-report/final_report.tex)
and [`reports/final-report/final_report.pdf`](reports/final-report/final_report.pdf).

## 1. Environment

Tested with **Python 3.13.2** on Apple Silicon. Other 3.11+ Pythons
should work but only 3.13.2 is verified.

```bash
python3.13 -m venv .cisopt
source .cisopt/bin/activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` is the exact dependency set the production
n=1000 CSVs were generated against; `requirements.txt` is the
human-curated subset. Use the lock file for reviewer-grade
reproduction.

For the SPICE truth and scenario visualization, also place two NAIF
kernels in `data/kernels/` (not in version control, license + size):

```bash
mkdir -p data/kernels
curl -sL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls \
  -o data/kernels/naif0012.tls
curl -sL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp \
  -o data/kernels/de442s.bsp
```

## 2. Quick smoke (≈ 2 min)

```bash
# Single-trial sanity check that all three estimator paths run end-to-end:
python -c "
import sys; sys.path[:0] = ['src','scripts']
import numpy as np
from importlib import import_module
m = import_module('06_midcourse_ekf_correction')
common = dict(mu=0.0121505856, t0=0., tf=6., tc=2., dt_meas=0.02,
              sigma_px=1., dropout_prob=0., seed=42,
              dx0=np.array([1e-4]*6), est_err=np.array([1e-4]*6),
              camera_mode='estimate_tracking',
              return_debug=False, accumulate_gramian=False)
for fk in ('iekf','ekf','ukf'):
    out = m.run_case(filter_kind=fk, **common)
    print(f'{fk:>5}: miss={out[\"miss_ekf\"]:.3e}  iters={out[\"iters_used_mean\"]:.1f}')
"

# Smoke-scale Monte Carlo (n=4 per cell, ~50 s wall):
python scripts/06s_estimator_ablation.py --n-seeds 4 --n-workers -1 --out results/mc/06s_smoke
python scripts/06r_landmarks_under_pointing_degradation.py --n-seeds 4 --n-workers -1 --out results/mc/06r_smoke
```

If any of these raise, fix that before attempting production.

## 3. Rebuild all paper figures from existing CSVs (≈ 1 s)

The central ECDF (Figure 12, Table 5) and the experiment manifest
require no new Monte Carlo — they consume the canonical CSVs in
`paper_artifacts/csv/` that ship with the repository:

```bash
python scripts/06t_success_ecdf_central.py
```

`06t` reads from `paper_artifacts/csv/` by default. If you have re-run
a production driver under `results/mc/` and want the central figure to
pick up the new numbers, either copy the new CSVs into
`paper_artifacts/csv/` or delete the existing ones — the driver will
fall back to the local `results/mc/` tree automatically.

See [`paper_artifacts/README.md`](paper_artifacts/README.md) for the
canonical CSV bundle and the per-CSV table/figure backing.

## 4. Rerun production Monte Carlo (≈ 3 h wall on 8 cores, plus 8 h SPICE)

The full set of n=1000 production runs that back the manuscript:

```bash
# Headline baselines (Section 11):
python scripts/06_monte_carlo.py     --n-seeds 1000 --n-workers -1 --out results/mc/phase_d_production
python scripts/06f_compare_truth_modes.py --n-seeds 1000 --n-workers 1  --out results/mc/phase_d_production_spice  # SPICE single-process

# Realism extensions (Section 13):
python scripts/06q_attitude_noise_sweep.py    --n-seeds 1000 --n-workers -1 --out results/mc/phase_e_attitude_noise
python scripts/06m_parallax_vs_range.py        --n-seeds 1000 --n-workers -1 --out results/mc/phase_e_parallax
python scripts/06n_landmarks.py                --n-seeds 1000 --n-workers -1 --out results/mc/phase_e_landmarks         --landmark-case synthetic_6
python scripts/06n_landmarks.py                --n-seeds 1000 --n-workers -1 --out results/mc/phase_e_landmarks_catalog --landmark-case catalog_craters_6
python scripts/06k_p0_sensitivity.py           --n-seeds 1000 --n-workers -1 --out results/mc/phase_e_P0
python scripts/06l_multi_revolution.py         --n-seeds 100  --n-workers -1 --out results/mc/phase_e_multi_rev
python scripts/06p_measurement_delay.py        --n-seeds 1000 --n-workers -1 --out results/mc/phase_e_delay
python scripts/06j_covariance_ellipses.py                                    --out results/mc/phase_e_covariance

# Phase-4 / Phase-5 / Phase-6 additions:
python scripts/06r_landmarks_under_pointing_degradation.py --n-seeds 1000 --n-workers -1 --out results/mc/phase_f_landmarks_pointing
python scripts/06s_estimator_ablation.py                   --n-seeds 1000 --n-workers -1 --out results/mc/phase_g_estimator_ablation
python scripts/06t_success_ecdf_central.py                                                 --out results/mc/phase_h_central_ecdf
```

Validate the n=1000 Phase-4 result with the four-gate Tier-2 validator
before treating it as canonical:

```bash
python scripts/_validate_06r_tier2.py --csv results/mc/phase_f_landmarks_pointing/06r_landmarks_under_pointing_degradation.csv
```

(Validator passes 4/4 against the production CSV shipped with the repo.)

## 5. Rebuild the PDF

```bash
cd reports/final-report
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex   # second pass for cross-references
```

Two passes are required so the cross-references and bibliography
resolve. The build artifacts (`*.aux`, `*.log`, `*.fls`,
`*.fdb_latexmk`, `*.out`) are deliberately gitignored — only
`final_report.tex` and `final_report.pdf` are tracked.

## 6. Headline numbers to verify

If reproduction succeeds, these n=1000 production numbers should match
the manuscript exactly (or to the rounding shown):

| metric | expected value |
|--------|---------------:|
| Baseline median terminal miss | 61.64 km |
| Baseline p95 terminal miss    | 228.27 km |
| Baseline pass rate @ 390 km   | 99.10% |
| Baseline pass rate @ 39 km    | 32.80% |
| Baseline pass rate @ 25 km    | 18.10% |
| SPICE-truth median miss       | 182.88 km |
| Moon + L2 landmarks median    | 25.07 km |
| Fixed-pointing collapse miss  | 493.73 km |
| EKF / IEKF / UKF median miss  | 61.61 / 61.64 / 91.38 km |
| UKF NEES band fraction        | 99.90% |

These match the canonical CSVs in `paper_artifacts/csv/` and the
manuscript exactly. The conversion `1 LU = 389,703.2648 km` and the
canonical `q_a = 1.0e-14` operating point live in
[`scripts/_paper_constants.py`](scripts/_paper_constants.py); change
either constant only with deliberate intent because every km-valued
table in the manuscript depends on it.

If any of these drift by more than the rounding shown, regenerate the
manifest entry, examine seed differences, and check for environment
drift (NumPy / SciPy version skew is the most common cause).
