from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    add_periodic_orbit_query_args,
    ensure_src_on_path,
    kernel_paths,
    repo_path,
    selected_branches,
    selected_libration_points,
    write_state_history_csv,
)

ensure_src_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a JPL Earth-Moon halo seed and propagate it with SPICE-backed point-mass dynamics."
    )
    parser.add_argument("--kernel", action="append", required=True, help="SPICE kernel path; repeat for LSK/SPK/etc.")
    parser.add_argument("--epoch", default="2026 APR 10 00:00:00 TDB")
    add_periodic_orbit_query_args(parser)
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--duration-days", type=float, default=None)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--max-step-days", type=float, default=0.05)
    parser.add_argument("--target", action="append", default=None, help="Point-mass body; default SUN, EARTH, MOON.")
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np

    from dynamics.integrators import propagate
    from dynamics.spice_ephemeris import make_spice_point_mass_dynamics
    from orbits import collect_periodic_orbit_candidates
    from orbits.spice_bridge import periodic_orbit_record_to_spice_inertial_state

    libration_points = selected_libration_points(args)
    branches = selected_branches(args, default=("S",))
    records = collect_periodic_orbit_candidates(
        system=args.system,
        family=args.family,
        libration_points=libration_points,
        branches=branches,
        period_min_days=args.period_min_days,
        period_max_days=args.period_max_days,
        target_period_days=args.target_period_days,
        stability_max=args.stability_max,
        timeout_s=args.timeout_s,
        cache_dir=repo_path(args.cache_dir),
        refresh_cache=bool(args.refresh_cache),
        no_cache=bool(args.no_cache),
    )
    if not records:
        print("No matching JPL periodic-orbit seeds found.", file=sys.stderr)
        return 1
    if args.candidate_index < 1 or args.candidate_index > len(records):
        print(
            f"--candidate-index must be in [1, {len(records)}], got {args.candidate_index}",
            file=sys.stderr,
        )
        return 2

    record = records[args.candidate_index - 1]
    targets = tuple(args.target) if args.target else ("SUN", "EARTH", "MOON")
    ephemeris, dynamics = make_spice_point_mass_dynamics(
        kernels=kernel_paths(args.kernel),
        epoch=args.epoch,
        targets=targets,
    )

    try:
        x0 = periodic_orbit_record_to_spice_inertial_state(record, ephemeris)
        duration_days = float(args.duration_days) if args.duration_days is not None else record.period_days
        duration_s = duration_days * 86400.0
        t_eval = np.linspace(0.0, duration_s, int(args.samples))

        res = propagate(
            dynamics.eom,
            (0.0, duration_s),
            x0,
            t_eval=t_eval,
            method="DOP853",
            rtol=float(args.rtol),
            atol=float(args.atol),
            max_step=float(args.max_step_days) * 86400.0,
        )
        if not res.success:
            raise RuntimeError(f"SPICE-backed propagation failed: {res.message}")

        final_state = res.x[-1]
        earth_state_0 = ephemeris.state_km_s("EARTH", 0.0)
        moon_state_0 = ephemeris.state_km_s("MOON", 0.0)
        earth_state_f = ephemeris.state_km_s("EARTH", duration_s)
        moon_state_f = ephemeris.state_km_s("MOON", duration_s)

        moon_rel_0 = x0[:3] - moon_state_0[:3]
        moon_rel_f = final_state[:3] - moon_state_f[:3]
        earth_rel_0 = x0[:3] - earth_state_0[:3]
        earth_rel_f = final_state[:3] - earth_state_f[:3]
        moon_rel_change = moon_rel_f - moon_rel_0
        earth_rel_change = earth_rel_f - earth_rel_0

        print(f"seed_libration_point {record.libration_point}")
        print(f"seed_branch {record.branch}")
        print(f"seed_period_days {record.period_days:.12f}")
        print(f"seed_stability {record.stability:.12g}")
        print(f"seed_jacobi {record.jacobi:.16e}")
        print(f"epoch_et {ephemeris.epoch_et:.9f}")
        print("x0_j2000_bary_km_km_s", " ".join(f"{v:.16e}" for v in x0))
        print("xf_j2000_bary_km_km_s", " ".join(f"{v:.16e}" for v in final_state))
        print(f"moon_rel_r0_norm_km {np.linalg.norm(moon_rel_0):.9f}")
        print(f"moon_rel_rf_norm_km {np.linalg.norm(moon_rel_f):.9f}")
        print(f"moon_rel_position_change_norm_km {np.linalg.norm(moon_rel_change):.9f}")
        print(f"earth_rel_r0_norm_km {np.linalg.norm(earth_rel_0):.9f}")
        print(f"earth_rel_rf_norm_km {np.linalg.norm(earth_rel_f):.9f}")
        print(f"earth_rel_position_change_norm_km {np.linalg.norm(earth_rel_change):.9f}")
        print(f"nfev {res.nfev}")

        if args.out_csv is not None:
            outpath = write_state_history_csv(args.out_csv, res.t, res.x)
            print(f"wrote_csv {outpath}")
    finally:
        ephemeris.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
