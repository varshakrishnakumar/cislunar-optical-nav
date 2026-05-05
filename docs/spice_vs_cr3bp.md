# SPICE vs CR3BP — what each model does in this pipeline

This pipeline can run two truth models. The estimator (the EKF / IEKF
dynamics) **always uses CR3BP**. SPICE only enters as truth and as the
ephemeris source for visualization. Treating them as a controlled
**model-mismatch experiment** is the load-bearing claim of the realism
story.

## TL;DR (matrix view)

| Subsystem                     | Model used                     |
| ----------------------------- | ------------------------------ |
| **Estimator dynamics (EKF)**  | CR3BP (always)                 |
| **Process Jacobian / STM**    | CR3BP (always)                 |
| **Targeting / guidance solve** | CR3BP (always — ND or km via lunit/tunit scaling) |
| **Truth propagation**         | CR3BP **or** SPICE point-mass (Sun + Earth + Moon) |
| **Moon ephemeris** (used by camera + bearing geom) | constant in CR3BP rotating frame; SPICE J2000 vectors via `spkpos` when truth=spice |
| **Visualization (animations / plots)** | SPICE for the dimensional Earth/Moon trajectory in `animate_phases_2_3.py`; CR3BP rotating-frame plots elsewhere |
| **Initial-condition seed**    | CR3BP halo seed (transformed to J2000 km via `orbits.spice_bridge` when truth=spice) |
| **Measurement model**         | Pinhole bearing camera; same code under both truth modes |
| **Filter belief about truth dynamics** | Always CR3BP — under SPICE, the gap is the **un-modeled error** the filter must absorb |

## Why CR3BP for the filter, even under SPICE truth

Switching the EKF to SPICE would erase the contribution: an EKF tuned
on the same truth it propagates is the trivial case. The interesting
question is whether a **CR3BP-only filter** (the kind a real onboard
guidance computer can run) is still consistent and operationally useful
when truth has the full Earth/Moon/Sun gravity, ephemeris drift, and
non-axisymmetric synodic frame.

Phase D headline: under n=1000 SPICE truth, the CR3BP-only filter is
mostly consistent (NIS in-band, NEES band-fraction drops 89% → 70%
in the tails) but pays a **3.0× median miss inflation** (61 → 183 km).
That's the publishable story — the model gap is an *operational* cost,
not a *consistency* failure.

## How to switch modes (CLI)

Every analysis driver in `scripts/06_*.py` accepts `--truth=cr3bp` (the
default — preserves all published numbers) or `--truth=spice`. SPICE
runs land in `_spice`-suffixed sibling output directories so the
CR3BP artifacts stay untouched. Each CSV row gets a `truth_mode` column.

```bash
# Legacy CR3BP/CR3BP run (default)
python scripts/06_monte_carlo.py --study baseline --n-trials 1000

# SPICE truth, CR3BP filter — the realism comparison
python scripts/06_monte_carlo.py --study baseline --n-trials 1000 \
    --truth spice --n-workers 1
```

## Performance / threading caveats

SPICE's C library is not thread-safe for `spkpos`; always pass
`--n-workers 1` (or run a process pool externally). At ~30 s per trial
the n=1000 SPICE Monte Carlo takes ~8 hours single-threaded.

## Where the truth boundary actually lives

The dispatcher is `_analysis_common.load_midcourse_run_case(truth=...)`,
which returns `run_case` for CR3BP and a wrapped `run_case_spice` for
SPICE. Both functions accept the same dimensionless CR3BP parameters
(`mu`, `t0`, `tf`, `tc`, `dt_meas`, `q_acc`, …) and the SPICE wrapper
internally converts them to seconds and km using SPICE's per-epoch
length unit (`lunit_km = ‖r_moon − r_earth‖` at epoch).

## What is *not* SPICE-aware (intentional)

- The targeting / guidance solver works in whichever units the truth
  passes it. Under SPICE truth, the same single-impulse solver runs in
  km / km·s on the J2000 inertial frame. There is no separate guidance
  model.
- The Bearing measurement model is identical under both truths — only
  the inputs (`r_sc`, `r_body`) change units.
- Sensitivity / fine-tune scripts (`06_sensitivity*`, `06e_fine_tune`,
  `06_q_acc_sweep`) accept `--truth` but their plot labels still say
  "ND" — Phase C unit-axis migration is paper-headline-only.

## When to run which

- **CR3BP-only**: tuning sweeps, q_acc / σ_px sensitivity, anything
  where you want clean, reproducible, fast iteration. Default for
  paper-headline figures except where the realism story is the point.
- **SPICE truth**: Phase D headline, scaling-vs-tf, anything where the
  manuscript needs to make a "real-flight" claim.

See `memory/project_spice_migration.md` for the per-phase landed-state
rollup and the empirical numbers behind the model-mismatch headline.
