# Analysis drivers (06i – 06q)

Each script in `scripts/` is a stand-alone analysis driver that calls
the underlying `run_case` / `run_case_spice` (in
[scripts/06_midcourse_ekf_correction.py](../scripts/06_midcourse_ekf_correction.py))
through `_analysis_common.load_midcourse_run_case(truth=...)`. All
support `--truth=cr3bp` (default) or `--truth=spice`. Outputs go to
`results/mc/<script-default-or-cli>` with a `_spice` suffix added when
`--truth=spice`.

| script | phase | what it answers | key outputs |
| ------ | ----- | --------------- | ----------- |
| [06i_tolerance_sweep.py](../scripts/06i_tolerance_sweep.py) | 1.2 | At what miss tolerance does the system stop being a pass? | `06i_tolerance_sweep.png` (success-rate vs tolerance curve), `.txt` table |
| [06j_covariance_ellipses.py](../scripts/06j_covariance_ellipses.py) | 2.5 | What does the filter believe about its uncertainty at the burn? | `06j_covariance_ellipses_seed*.png` (six 2-D 1σ/3σ panels for position and velocity) |
| [06k_p0_sensitivity.py](../scripts/06k_p0_sensitivity.py) | 2.6 | Is the headline sensitive to initial covariance? | `06k_p0_sensitivity.png` (4-panel boxplots: miss, pos_err, NEES, NIS), `.csv`, `.txt` |
| [06l_multi_revolution.py](../scripts/06l_multi_revolution.py) | 2.7 | Does the EKF stay stable over 2–4 halo periods? | `06l_multi_revolution.png` (pos_err + NEES vs time, IQR envelope), `.txt` |
| [06m_parallax_vs_range.py](../scripts/06m_parallax_vs_range.py) | 3.8 | How does range observability scale with LOS angular sweep? | `06m_parallax_vs_range.png` (3-panel scatter: range_err / pos_err / miss vs cumulative parallax with power-law fit), `.txt` includes both `net` and `cumulative` fits |
| [06n_landmarks.py](../scripts/06n_landmarks.py) | 3.9 | Do synthetic landmarks reduce dependence on Moon-center pointing? | `06n_landmarks.png` (3-panel boxplot, three configs: `moon_only`, `landmarks_only`, `moon_plus_landmarks`), `.txt` |
| [06o_pointing_bias_lag.py](../scripts/06o_pointing_bias_lag.py) | 4.10 | What level of pointing bias / control-loop lag breaks the filter? | `06o_pointing_bias_lag.png` (4-panel: bias→miss, lag→miss, bias→NIS, lag→NIS), `.txt` |
| [06p_measurement_delay.py](../scripts/06p_measurement_delay.py) | 4.11 | Can innovations look fine while terminal miss explodes? (delay study) | `06p_measurement_delay.png` (miss vs delay alongside NIS/NEES vs delay) |
| [06q_attitude_noise_sweep.py](../scripts/06q_attitude_noise_sweep.py) | 1.1 | At what σ_attitude does graceful degradation break down? | `06q_attitude_noise_sweep.png` (miss + NIS + NEES across σ_att grid) |

## Common conventions

- `--n-seeds <N>` — Monte Carlo replication count. Smoke tests use
  4–8; production uses 30–100 (CR3BP) or 30 (SPICE — single-threaded
  ~30 s/trial).
- `--truth {cr3bp,spice}` — selects the truth model. EKF dynamics
  always stay CR3BP; see [docs/spice_vs_cr3bp.md](spice_vs_cr3bp.md).
- `--out <dir>` — output directory. SPICE runs land in `<dir>_spice/`.
- `--base-seed <int>` — same seed across configs/sweeps in a script
  reuses identical injection / estimation perturbations, so per-trial
  deltas between configs are meaningful.

## Realism / sensitivity knobs

These flow either through the script-specific CLI flag or through the
underlying `run_case` kwargs (and via `MonteCarloConfig` for
`scripts/06_monte_carlo.py`):

| knob | type | flows through `MonteCarloConfig` | flag in `06_monte_carlo.py` |
| ---- | ---- | -------------------------------- | --------------------------- |
| `sigma_att_rad` | per-step Gaussian attitude noise | yes | `--sigma-att-deg` / `--sigma-att-rad` |
| `bias_att_rad` | constant axis-angle pointing bias | yes (3-tuple) | `--bias-att-deg-y` (y-axis bias) |
| `pointing_lag_steps` | camera tracks `x̂[k − N]` | yes | `--pointing-lag-steps` |
| `meas_delay_steps` | truth used to build measurement is from N steps ago (Moon ephemeris is also delayed under SPICE for geometric consistency) | yes | `--meas-delay-steps` |
| `P0_scale` | multiplier on the initial filter covariance diagonal | yes | `--P0-scale` |
| `landmark_case` | named synthetic-landmark case (`none`, `synthetic_6`, `synthetic_12`) | yes | `--landmark-case` |
| `disable_moon_center` | skip Moon-center bearing entirely | yes | `--disable-moon-center` |

For the synthetic-landmark cases, the runner internally feeds the
right keyword to `run_case` (CR3BP — `landmark_positions` in ND units)
or `run_case_spice` (SPICE — `landmark_offsets_km`) based on the
case_fn's signature.

## Output schemas

`run_case` / `run_case_spice` return:

- `valid_rate` — fraction of `k=1..k_tc` epochs where any update
  (Moon or landmark) was accepted. Excludes index 0 (initial state).
- `valid_rate_moon` — same fraction restricted to Moon-center accepts.
- `valid_rate_landmarks` — average accepted-landmark fraction per
  scheduled epoch. NaN when no landmarks are configured.
- `nis_mean` — Moon-center NIS only (NaN under landmarks-only). The
  legacy MC CSV uses this column.
- `nis_mean_all` — pooled NIS across Moon + every accepted landmark
  update.
- `nis_mean_landmarks` — pooled NIS over landmark accepts only.
- `parallax_net_rad` — endpoint-to-endpoint LOS angle. Use only on
  short monotonic arcs.
- `parallax_cumulative_rad` — sum of step-to-step LOS angle deltas.
  Correct under any geometry; preferred for paper figures.
- `parallax_total_rad` — alias for `parallax_net_rad` (kept for
  backward compat with already-written 06m artifacts).
- `range_err_tc` — `|‖r̂(tc) − r_moon‖ − ‖r(tc) − r_moon‖|`.

Debug dict also exposes:

- `W_obs` — total information Gramian.
- `W_obs_moon` — Moon-only contribution.
- `W_obs_landmarks` — landmark contribution. `W_obs ≡ W_obs_moon +
  W_obs_landmarks` (verified by smoke-test).
- `accepted_moon_arr`, `accepted_landmarks_arr`, `epoch_accepted_arr`
  — per-epoch accept counters used by the three valid-rate aggregations.
- `los_inertial_hist`, `range_truth_hist`, `range_estimate_hist` — the
  raw arrays the parallax / range diagnostics are built from.
- `P_tc`, `P_full_hist` (only when `P_cov_history=True`) — for
  covariance-ellipse rendering.

## Production run recipes

```bash
# Tight tolerance check (no extra runs needed; reads existing CSVs)
python scripts/06i_tolerance_sweep.py \
    --csv results/mc/phase_d_production/06c_baseline_results.csv \
    --csv results/mc/phase_d_production_spice/06c_baseline_results.csv \
    --label "CR3BP truth" --label "SPICE truth" \
    --out results/mc/phase_d_tolerance_sweep

# Attitude-noise headline (CR3BP, n=30)
python scripts/06q_attitude_noise_sweep.py --n-seeds 30 \
    --out results/mc/phase_e_attitude_noise

# Parallax vs range (CR3BP, n=30, sweep tc)
python scripts/06m_parallax_vs_range.py --n-seeds 30 \
    --out results/mc/phase_e_parallax

# Landmarks comparison (CR3BP, n=60 — same seeds across three configs)
python scripts/06n_landmarks.py --n-seeds 60 \
    --out results/mc/phase_e_landmarks

# Multi-revolution stability (CR3BP, n=20 per tf)
python scripts/06l_multi_revolution.py --n-seeds 20 \
    --tf-list 6 9 12 --out results/mc/phase_e_multi_rev

# P0 sensitivity (CR3BP, n=80)
python scripts/06k_p0_sensitivity.py --n-seeds 80 \
    --out results/mc/phase_e_P0

# Pointing bias / lag (CR3BP, n=30 each)
python scripts/06o_pointing_bias_lag.py --n-seeds 30 \
    --out results/mc/phase_e_bias_lag

# Measurement delay (CR3BP, n=30)
python scripts/06p_measurement_delay.py --n-seeds 30 \
    --out results/mc/phase_e_delay

# Covariance ellipses (single trial, baseline + perturbed seeds)
for s in 7 11 23; do
  python scripts/06j_covariance_ellipses.py --seed $s \
      --out results/mc/phase_e_covariance
done
```

For SPICE counterparts, append `--truth spice --n-workers 1` and
expect ~30 s/trial wall-clock. Prioritize: tolerance sweep, attitude
noise sweep, landmarks comparison, measurement delay.
