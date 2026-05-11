# Experiment Manifest — cislunar-optical-nav

This file is the reviewer-grade index of every figure, table, and headline
number in the journal manuscript at [`reports/final-report/final_report.tex`](../reports/final-report/final_report.tex).
For each artifact it lists: the script that produced the input data, the
output directory under `results/mc/`, the matched-seed Monte Carlo
population size, the relevant config knobs, and the approximate
single-machine wall time on the reference 8-core machine the production
runs were generated on (Apple Silicon, no GPU, Python 3.13.2, see
[`requirements-lock.txt`](../requirements-lock.txt)).

This manifest is the single source of truth for "what generated what."
If a manuscript artifact is not listed here, it was either generated
inline in `final_report.tex` (configuration tables) or is a static
diagram (`pipeline_diagram.png`, `bearing_concept_visual.png`,
`phase01_slide_visual.png`).

## Conventions

- All scripts live under [`scripts/`](../scripts/) and are invoked from the
  repository root.  Default `--n-seeds 1000` and `--n-workers -1` (process
  pool sized to `cpu_count()`).
- All output paths are relative to the repository root.
- Random seeds are deterministic per trial via
  `mc.sampler.make_trial_rng(base_seed, trial_id)` — the same `--base-seed`
  reproduces the exact production CSVs bit-for-bit on the same Python /
  NumPy / SciPy build.
- Wall times are end-to-end including process-pool startup and result
  serialization; CR3BP single-trial cost is roughly 0.4 s wall on this
  machine when amortized over a full 1000-seed batch.
- "post-hoc" wall times are for re-rendering only; they read existing
  CSVs and produce the figure / table without rerunning Monte Carlo.

## Sections 1–8 — narrative figures (no Monte Carlo)

| ID | Artifact | Script | Notes |
| -- | -------- | ------ | ----- |
| Fig 1 | `pipeline_diagram.png` | `scripts/10_pipeline_diagram.py` | Closed-loop architecture diagram (static) |
| Fig 2 | `phase01_slide_visual.png` | `scripts/11_phase01_slide_visual.py` (consumes `results/seeds/spice_nrho_seed.csv`) | SPICE-backed scenario visualization |
| Fig 3 | `bearing_concept_visual.png` | `scripts/12_bearing_concept_visual.py` | Bearing-only measurement geometry (static) |
| Tab 1 | Camera/measurement parameters | hand-coded in `final_report.tex` | Configuration only |

## Section 9 — Bearing-only observability (CR3BP)

| ID | Artifact | Script | Output | n | Wall |
| -- | -------- | ------ | ------ | -:| ---- |
| Fig 4 | `observability.png` | `scripts/06_ekf_diagnostics.py` | `results/diagnostics/` | 1 trial | <30 s |

## Section 10 — Active camera pointing

| ID | Artifact | Script | Output | n | Wall |
| -- | -------- | ------ | ------ | -:| ---- |
| Tab 2, Fig 7 | Fixed-vs-active table + figure | `scripts/07_active_tracking.py` | `results/active_tracking/` | 1 trial × 301 epochs | <1 min |

## Section 11 — Monte Carlo Results

| ID | Artifact | Driver | Output dir | n | Wall |
| -- | -------- | ------ | ---------- | -:| ---- |
| Tab 3 + Fig 8 | Baseline n=1000 headline | `scripts/06_monte_carlo.py` | `results/mc/phase_d_production/` | 1 × 1000 | ~30 m |
| Tab 4 + Fig 9 | Process-noise sweep ($q_a$) | `scripts/06_monte_carlo.py` (with `--q-acc-sweep`) | `results/mc/phase_b_*/` | 5–8 × 1000 | ~3 h total |

## Section 12 — Sensitivity Studies (legacy n=80)

| ID | Artifact | Driver | Output dir | n | Wall |
| -- | -------- | ------ | ---------- | -:| ---- |
| Fig 10 | Pixel-noise sensitivity | `scripts/06_sensitivity_mc.py` | `results/sensitivity/` | $\sim$5 × 80 | ~10 m |
| Fig 11 | Correction-time ($t_c$) sensitivity | `scripts/06_sensitivity_mc.py` | `results/sensitivity/` | $\sim$8 × 80 | ~12 m |

## Section 13 — Realism and Sensitivity Extensions (production)

| ID | Artifact | Driver | Output dir | n | Wall |
| -- | -------- | ------ | ---------- | -:| ---- |
| Tab 5, Fig 12 | **Central ECDF + success-vs-tolerance** | `scripts/06t_success_ecdf_central.py` | `results/mc/phase_h_central_ecdf/` | post-hoc (4 curves × 1000) | <1 s |
| Tab 6, Fig 13 | Attitude-noise sweep | `scripts/06q_attitude_noise_sweep.py` | `results/mc/phase_e_attitude_noise/` | 7 × 1000 | ~30 m |
| Fig 14 | Cumulative-parallax vs range error | `scripts/06m_parallax_vs_range.py` | `results/mc/phase_e_parallax/` | $\sim$8000 points | ~25 m |
| Tab 7, Tab 8, Fig 15 | L1 (synthetic) and L2 (catalog) landmarks | `scripts/06n_landmarks.py` (run twice with `--landmark-case synthetic_6` / `--landmark-case catalog_craters_6`) | `results/mc/phase_e_landmarks/`, `results/mc/phase_e_landmarks_catalog/` | 3 × 1000 each | ~25 m each |
| Tab 9, Fig 16 | **Landmarks under pointing degradation (3×5 grid)** | `scripts/06r_landmarks_under_pointing_degradation.py` | `results/mc/phase_f_landmarks_pointing/` | 15 × 1000 | ~3 m |
| Fig 17 | Filter-covariance ellipses at burn | `scripts/06j_covariance_ellipses.py` | `results/mc/phase_e_covariance/` | 1 trial (debug) | <10 s |
| Tab 10 | Initial-covariance scaling sweep | `scripts/06k_p0_sensitivity.py` | `results/mc/phase_e_P0/` | 5 × 1000 | ~1 h |
| Fig 18 | Multi-revolution stability | `scripts/06l_multi_revolution.py` | `results/mc/phase_e_multi_rev/` | 3 × 100 | ~30 m |
| Tab 11 | Measurement-delay sweep | `scripts/06p_measurement_delay.py` | `results/mc/phase_e_delay/` | 7 × 1000 | ~50 m |

## Section 14 — Verification and Stress Testing

| ID | Artifact | Driver | Output dir | n | Wall |
| -- | -------- | ------ | ---------- | -:| ---- |
| Tab 12 | V&V / consistency table | aggregated from existing CSVs | (post-hoc) | — | — |
| Fig 19 | SPICE point-mass truth comparison | `scripts/06f_compare_truth_modes.py` | `results/mc/phase_d_production_spice/` | 1 × 1000 | ~8 h (SPICE single-process) |
| **Tab 13, Fig 20** | **Estimator ablation EKF / IEKF / UKF** | `scripts/06s_estimator_ablation.py` | `results/mc/phase_g_estimator_ablation/` | 3 × 1000 | ~5 m |
| Tab 14 | Compute budget for production drivers | (this manifest) | — | — | — |

## SPICE truth assets

The SPICE truth comparison and the scenario visualization both depend on
two SPICE kernels that are NOT in version control (license + size):
- `data/kernels/de442s.bsp` (NASA NAIF DE442 short-arc planetary ephemeris)
- `data/kernels/naif0012.tls` (NAIF leapseconds kernel)

Download both from
[NAIF generic kernels](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/)
and place them under `data/kernels/` before invoking `06f_compare_truth_modes.py`
or `11_phase01_slide_visual.py`.  See [`docs/spice_vs_cr3bp.md`](spice_vs_cr3bp.md)
for the exact "what uses SPICE vs CR3BP" matrix.

## Validation gates

| Gate | Script | Used by |
| ---- | ------ | ------- |
| 06r 4-check Tier-2 validator | `scripts/_validate_06r_tier2.py` | Pre-production gate for Section 13 / Tab 9 / Fig 16 |

The 06s ablation does not have a separate validator script; its three
cells are inspected directly by reading the summary `.txt` produced by
`scripts/06s_estimator_ablation.py`.

## Headline reproducibility numbers (n=1000 production)

These are the canonical numbers the manuscript leans on; if a
re-run drifts, the cause should be tracked down before publishing.

| metric | value | source |
| ------ | -----:| ------ |
| Baseline median terminal miss [km] | 60.80 | `phase_d_production/06c_baseline_results.csv`, `miss_ekf` median ×384400 |
| Baseline p95 terminal miss [km] | 225.16 | same |
| Baseline pass rate at 390 km | 99.10% | same |
| Baseline pass rate at 39 km | 32.90% | same |
| SPICE-truth median miss [km] | 182.88 | `phase_d_production_spice/06c_baseline_results.csv`, `miss_ekf` median |
| Moon + L2 landmarks median miss [km] | 24.73 | `phase_f_landmarks_pointing/06r_landmarks_under_pointing_degradation.csv` filtered to (`moon_plus_landmarks_L2`, `active_ideal`) |
| Fixed-pointing collapse miss [km] | 487.01 | same CSV, any cell with `pt_mode=fixed` |
| EKF / IEKF median miss [km] | 60.78 / 60.80 | `phase_g_estimator_ablation/06s_estimator_ablation.csv` |
| UKF NEES band fraction | 99.90% | same |
