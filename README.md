# Cislunar Optical Navigation

## Orbit Seeds and High-Fidelity Dynamics

Fetch Earth-Moon halo-family seeds from the JPL periodic-orbits catalog:

```bash
python3 scripts/00_fetch_periodic_orbits.py --libr 2 --branch S --limit 10 --print-inertial
```

JPL seed queries are cached under `data/cache/jpl_periodic_orbits/` by default. Add `--refresh-cache` to update a cached query or `--no-cache` to force a one-off live request.

High-fidelity propagation is moving toward a SPICE-backed point-mass model. Install the optional SPICE dependency, then keep large kernels under `data/kernels/`:

```bash
python3 -m pip install ".[high-fidelity]"
python3 scripts/check_spice_ephemeris.py \
  --kernel data/kernels/naif0012.tls \
  --kernel data/kernels/de442s.bsp \
  --epoch "2026 APR 10 00:00:00 TDB"
```

Use a JPL seed directly in the SPICE-backed propagator:

```bash
python3 scripts/01_propagate_jpl_seed_spice.py \
  --kernel data/kernels/naif0012.tls \
  --kernel data/kernels/de442s.bsp \
  --epoch "2026 APR 10 00:00:00 TDB" \
  --libr 2 --branch S \
  --out-csv results/spice_nrho_seed.csv \
  --out-relative-csv results/spice_nrho_seed_relative.csv \
  --out-plot results/plots/spice_nrho_seed.png
```

The relative CSV and plot report include Earth-relative, Moon-relative, and Earth-Moon synodic diagnostics, which are more useful for cislunar navigation than raw Solar System barycentric J2000 coordinates alone.

The model interface now supports both CR3BP and high-fidelity point-mass dynamics, so guidance and EKF propagation can migrate without changing their outer APIs.

The old CR3BP-only propagation starter lives in `scripts/demo_cr3bp_baseline.py`; the numbered scripts are reserved for the seed-to-ephemeris pipeline.

## Analysis and Presentation Outputs

Run the IEKF/midcourse Monte Carlo with plots that explain both navigation performance and measurement quality:

```bash
python3 scripts/06_monte_carlo.py \
  --study baseline \
  --n-trials 50 \
  --plots-dir results/plots/06_baseline
```

The relative Delta-V inflation plot uses `(|Delta-V_EKF| / |Delta-V_perfect| - 1) * 100`. For example, `+5.6%` means the IEKF-based burn magnitude was 5.6% larger than the perfect-information single-impulse burn for the same targeting problem, not that it achieved 5.6% of optimal performance.

The 06-series plots now include terminal miss, burn error, valid bearing-update rate, and mean NIS. Mean NIS is useful as an IEKF consistency check: for a well-tuned 2-D bearing residual, values near 2 are expected, while large values suggest residual/model inconsistency or outliers.

Run the synthetic vision measurement demo with quantitative outputs:

```bash
python3 scripts/08_feature_tracking_demo.py \
  --skip-video \
  --plots-dir results/plots \
  --metrics-csv results/vision/08_feature_metrics.csv
```

This demo is now framed as a centroid, angular-radius, and landmark-track measurement preview rather than a full SLAM claim. It writes a metrics CSV with pixel centroid error, angular radius in mrad, range proxy in km, visible landmark count, and landmark optical flow.

For image folders or ROI-based crops, use:

```bash
python3 scripts/08_blob_centroid_demo.py \
  --input-dir results/videos/08_feature_frames \
  --output-dir results/vision/08_blob_centroid_demo

python3 scripts/08_moon_blob_demo.py \
  --input-dir results/videos/08_feature_frames \
  --roi 0 0 660 552 \
  --skip-video \
  --output-dir results/vision/08_moon_blob_demo
```

The blob demos write per-frame CSV metrics and summary figures using pixel units, including detection rate, centroid location, blob area, enclosing radius, and shape compactness.
