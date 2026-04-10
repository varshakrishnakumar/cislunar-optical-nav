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
  --libr 2 --branch S --out-csv results/spice_nrho_seed.csv
```

The model interface now supports both CR3BP and high-fidelity point-mass dynamics, so guidance and EKF propagation can migrate without changing their outer APIs.

The old CR3BP-only propagation starter lives in `scripts/demo_cr3bp_baseline.py`; the numbered scripts are reserved for the seed-to-ephemeris pipeline.
