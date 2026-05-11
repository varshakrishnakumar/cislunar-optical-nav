# Cislunar Optical Navigation

Bearing-only optical navigation and midcourse-correction analysis for cislunar
transfers. This repository contains the source code, configs, and canonical
n=1000 production artifacts behind a journal manuscript on **guidance-coupled
evaluation of a minimum-information Moon-center bearing architecture** in a
cislunar transfer, with reality checks on attitude noise, pointing degradation,
multi-revolution stability, parallax-vs-range observability, $P_0$
initialization, measurement delay, synthetic and catalog-backed landmark
geometry, and SPICE point-mass truth.

- **Manuscript:** [`reports/final-report/final_report.tex`](reports/final-report/final_report.tex)
  → compiled PDF at [`reports/final-report/final_report.pdf`](reports/final-report/final_report.pdf)
- **Reviewer-grade reproduction:** [`REPRODUCING.md`](REPRODUCING.md)
- **Canonical CSV/figure/summary bundle:** [`paper_artifacts/`](paper_artifacts/)
- **Figure-to-script manifest:** [`docs/experiment_manifest.md`](docs/experiment_manifest.md)
- **Locked paper constants** (`KM_PER_LU = 389703.2648`, `Q_ACC_CANONICAL = 1e-14`):
  [`scripts/_paper_constants.py`](scripts/_paper_constants.py)

There are two layers in this repository:

- **Layer 1:** the original optical-navigation implementation under
  `src/` and `scripts/`, including the numbered 06\* drivers that produced
  the journal manuscript's n=1000 production runs.
- **Layer 2:** `src/cisopt/`, a config-driven experiment framework built on
  top of the Layer-1 dynamics / sensor / estimator / guidance primitives.
  The journal manuscript currently uses Layer 1; Layer 2 is the staging
  ground for the next paper's experiments.

## What This Project Does

The nominal pipeline simulates a spacecraft in an Earth-Moon cislunar scenario,
generates camera-based Moon bearing measurements, estimates the spacecraft
state with EKF / IEKF / UKF variants, and computes a single-impulse midcourse
correction from the estimated state. The outputs are not just state-error
plots: the framework tracks guidance-relevant metrics such as terminal miss,
burn magnitude, Delta-V inflation, NIS, NEES, measurement validity rate, and
observability directions.

Current supported experiment pieces:

- CR3BP halo-L1 and NRHO-style scenarios.
- SPICE-backed halo-L1 scenario when kernels are available.
- Camera bearing sensor with pixel noise, dropout, active/fixed pointing, and
  optional realism wrappers.
- EKF, IEKF, and UKF estimator implementations.
- Single-impulse correction targeting.
- Monte Carlo, ablation, estimator-zoo, observability, and
  navigation-error-to-burn-error coupling studies.
- Figure generation for the onboard/session report.

## Quickstart

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The checked-in `run.sh` is a convenience wrapper for the local `.cisopt` virtual
environment used during development:

```bash
./run.sh run_mc.py configs/examples/halo_l1_baseline.yaml --n-trials 8 --workers 1
```

If you are using your own virtual environment, call the scripts with `python3`
instead.

## Quickstart For The Journal Manuscript

The reviewer-facing reproduction flow lives in
[`REPRODUCING.md`](REPRODUCING.md). The short version:

```bash
python3.13 -m venv .cisopt
source .cisopt/bin/activate
pip install -r requirements-lock.txt

# Rebuild every paper figure / table from canonical CSVs (~1 s):
python scripts/06t_success_ecdf_central.py

# Rebuild the PDF:
cd reports/final-report
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
```

Full production Monte Carlo regeneration (≈ 3 h CR3BP, ≈ 8 h SPICE) is
documented in [`REPRODUCING.md`](REPRODUCING.md) section 4 and the per-
driver wall times are reported in
[`docs/experiment_manifest.md`](docs/experiment_manifest.md).

The SPICE point-mass truth comparison and the scenario visualization both
expect two NAIF kernels at `data/kernels/naif0012.tls` and
`data/kernels/de442s.bsp`. These are external (license + size) and are
not in version control; see REPRODUCING for the curl commands. CR3BP-only
reproduction needs no kernels.

## Config-Driven Experiments With `cisopt`

The main experiment entry points are top-level scripts:

```bash
# Run one configured trial.
python3 run_experiment.py configs/examples/halo_l1_baseline.yaml

# Run a Monte Carlo sweep.
python3 run_mc.py configs/examples/halo_l1_baseline.yaml --n-trials 100 --workers 4

# Run a named-axis ablation sweep.
python3 run_ablation.py configs/examples/halo_l1_ablation.yaml --workers 4

# Map navigation error into correction-burn error.
python3 run_coupling.py configs/examples/halo_l1_baseline.yaml \
  --sigma-r 1e-5,1e-4,1e-3,1e-2 \
  --sigma-v 0.0,1e-5 \
  --n-samples 30 \
  --with-observability
```

Default outputs go under `results/cisopt/`. Single trials write
`summary.json` and, when `output.save_debug` is true, `timeseries.npz`.
Monte Carlo and ablation runs write Parquet tables plus run metadata:

```text
results/cisopt/<config-name>/
results/cisopt/mc/<config-name>/trials.parquet
results/cisopt/ablation/<config-name>/trials.parquet
results/cisopt/coupling/<config-name>/coupling_random.parquet
```

Example configs live in `configs/examples/`:

- `halo_l1_baseline.yaml`: baseline CR3BP halo-L1 optical-nav trial.
- `halo_l1_spice_baseline.yaml`: SPICE-backed halo-L1 trial.
- `nrho_baseline.yaml`: CR3BP NRHO-style trial.
- `halo_l1_ablation.yaml`: pointing, process-noise, and pixel-noise ablation.

## `cisopt` Architecture Map

```text
src/cisopt/
  protocols.py       Shared Scenario / Sensor / Estimator / Guidance interfaces
  config.py          JSON / YAML / TOML config loading and dotted-path patching
  scenarios/         halo_l1_cr3bp, halo_l1_spice, nrho_cr3bp
  sensors/           camera_bearing and measurement-realism wrappers
  estimators/        EKF, IEKF, UKF
  guidance/          single-impulse correction targeting
  runner/            one-trial experiment loop
  results/           single-trial artifact storage
  sweeps/            Monte Carlo tables and query helpers
  ablation/          Cartesian-product ablation engine
  observability/     Gramian and weak-direction analysis
  coupling/          navigation-error -> burn-error maps
  viz/               report plotting API
```

The framework is intentionally registry-based: each config names a scenario,
sensor, estimator, and guidance module, then `cisopt.runner.run_trial` wires the
pieces together through the shared protocols.

## Current Paper Claims

The journal manuscript is built around a single locked thesis:

> A sparse Moon-center bearing stream is guidance-useful when target
> visibility is actively maintained, but the architecture's utility must
> be judged by terminal correction performance, NEES, and visibility
> jointly — not by innovation consistency alone — and is bounded by
> explicit dynamical, measurement, and timing assumptions.

The three guidance-coupled contributions:

1. **Mission-output evaluation framework** — terminal miss, $\Delta V$
   inflation, NEES, NIS, and visibility fraction reported jointly per
   scenario, replacing the more common estimator-only scorecard.
2. **Active pointing as part of the navigation architecture** — a
   1000-trial demonstration that estimate-driven pointing converts a
   fragile passive camera into a closed-loop measurement source, plus a
   pointing-degradation ladder that bounds where this benefit holds.
3. **Bounded-realism characterization of the minimum-information bearing
   architecture** — reality checks on terminal tolerance, attitude noise,
   multi-revolution stability, parallax-vs-range, $P_0$ sensitivity,
   measurement delay, synthetic and catalog-backed landmark geometry, and
   SPICE point-mass truth.

The canonical numbers for the manuscript are in
[`paper_artifacts/README.md`](paper_artifacts/README.md). The data and
figures behind these claims are generated by the numbered `06*` scripts
under `scripts/`; the older `reproduce_paper.py` + `make_report.py` +
`reports/onboard/` flow is preserved for the earlier class-project report
but is not the journal-manuscript pipeline.

## Where Things Live

```text
configs/examples/       Reproducible experiment configs
src/cisopt/             Config-driven publication framework
src/dynamics/           CR3BP, point-mass, variational, SPICE ephemeris models
src/nav/                Original EKF and bearing-measurement primitives
src/cv/                 Camera, pointing, and synthetic measurement helpers
src/guidance/           Original targeting utilities
src/orbits/             JPL periodic-orbit catalog helpers and conversion tools
src/diagnostics/        Filter health, Jacobian, STM, and plotting diagnostics
src/vision/             Blob detection and vision plotting helpers
scripts/                Numbered legacy/demo pipeline and slide visuals
reports/onboard/        Current cisopt session report, data, and generated figures
reports/final-report/   Original ASTE 581 final report
results/                Local/generated outputs, ignored by git
data/kernels/           Local SPICE kernels, ignored by git
data/cache/             Cached JPL periodic-orbit queries, ignored by git
```

## Original Numbered-Script Pipeline

The older script flow is still useful for demos, diagnostics, and class-project
history. It is kept as a legacy/class-project pipeline, while `cisopt` is the
main reproducible experiment layer.

Orbit seed and SPICE propagation:

```bash
python3 scripts/00_fetch_periodic_orbits.py --libr 2 --branch S --limit 10 --print-inertial

python3 scripts/check_spice_ephemeris.py \
  --kernel data/kernels/naif0012.tls \
  --kernel data/kernels/de442s.bsp \
  --epoch "2026 APR 10 00:00:00 TDB"

python3 scripts/01_propagate_jpl_seed_spice.py \
  --kernel data/kernels/naif0012.tls \
  --kernel data/kernels/de442s.bsp \
  --epoch "2026 APR 10 00:00:00 TDB" \
  --libr 2 --branch S \
  --out-csv results/seeds/spice_nrho_seed.csv \
  --out-relative-csv results/seeds/spice_nrho_seed_relative.csv \
  --out-plot results/seeds/spice_nrho_seed.png
```

Navigation, Monte Carlo, and vision demos:

```bash
python3 scripts/06_monte_carlo.py \
  --study baseline \
  --n-trials 50 \
  --plots-dir results/mc/baseline_live

python3 scripts/08_feature_tracking_demo.py \
  --skip-video \
  --plots-dir results/demos \
  --metrics-csv results/vision/08_feature_metrics.csv

python3 scripts/08_blob_centroid_demo.py \
  --input-dir results/videos/08_feature_frames \
  --output-dir results/vision/08_blob_centroid_demo

python3 scripts/08_moon_blob_demo.py \
  --input-dir results/videos/08_feature_frames \
  --roi 0 0 660 552 \
  --skip-video \
  --output-dir results/vision/08_moon_blob_demo
```

## Running The Sim

For the current framework, edit or copy a config under `configs/examples/`, then
run one of the top-level drivers. A minimal single-run loop is:

```bash
python3 run_experiment.py configs/examples/halo_l1_baseline.yaml --out results/cisopt/sandbox
```

The config controls:

- `scenario`: trajectory model and timing.
- `sensor`: camera intrinsics, pixel noise, dropout, and pointing mode.
- `estimator`: EKF / IEKF / UKF choice and tuning parameters.
- `guidance`: correction-burn solver.
- `trial`: random seed and injected/estimated initial uncertainty.
- `output`: artifact directory and debug-timeseries saving.

For repeated trials, use `run_mc.py`; for named parameter sweeps, use
`run_ablation.py`; for paper-grade regeneration, use `reproduce_paper.py`.

## Future Sensor Extensions

Pulsar navigation should fit as another `cisopt` sensor module rather than a
separate project. Plausible extensions:

- Optical bearing / feature bearings.
- X-ray pulsar timing.
- Multi-body optical triangulation.
- Fused optical + pulsar navigation.
