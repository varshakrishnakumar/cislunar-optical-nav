from __future__ import annotations

import argparse

from _common import (
    add_periodic_orbit_query_args,
    ensure_src_on_path,
    repo_path,
    selected_branches,
    selected_libration_points,
)

ensure_src_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Earth-Moon halo orbit seeds from JPL's Periodic Orbits API."
    )
    add_periodic_orbit_query_args(parser)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--theta-rad", type=float, default=0.0)
    parser.add_argument("--print-inertial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from orbits import (
        collect_periodic_orbit_candidates,
        normalized_synodic_to_inertial_state,
        normalized_to_dimensional_state,
    )

    libration_points = selected_libration_points(args)
    branches = selected_branches(args, default=("S", "N"))

    candidates = collect_periodic_orbit_candidates(
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
    candidates = candidates[: max(0, int(args.limit))]

    if not candidates:
        print("No matching periodic orbits found.")
        return 1

    print(
        "idx libr branch period_days stability jacobi "
        "x y z vx vy vz "
        "x_syn_km y_syn_km z_syn_km vx_syn_km_s vy_syn_km_s vz_syn_km_s"
        + (
            " x_inertial_km y_inertial_km z_inertial_km "
            "vx_inertial_km_s vy_inertial_km_s vz_inertial_km_s"
            if args.print_inertial
            else ""
        )
    )
    for idx, record in enumerate(candidates, start=1):
        state_dim = normalized_to_dimensional_state(record.state_norm, record.system)
        state_norm = " ".join(f"{v:.16e}" for v in record.state_norm)
        state_dim_text = " ".join(f"{v:.16e}" for v in state_dim)
        state_inertial_text = ""
        if args.print_inertial:
            state_inertial = normalized_synodic_to_inertial_state(
                record.state_norm,
                record.system,
                theta_rad=float(args.theta_rad),
            )
            state_inertial_text = " " + " ".join(f"{v:.16e}" for v in state_inertial)
        libr = "-" if record.libration_point is None else str(record.libration_point)
        branch = "-" if record.branch is None else record.branch
        print(
            f"{idx} {libr} {branch} {record.period_days:.9f} "
            f"{record.stability:.9g} {record.jacobi:.16e} "
            f"{state_norm} {state_dim_text}{state_inertial_text}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
