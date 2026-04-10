from __future__ import annotations

import argparse

from _common import ensure_src_on_path, kernel_paths

ensure_src_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check SPICE kernels for high-fidelity cislunar propagation."
    )
    parser.add_argument("--kernel", action="append", required=True, help="SPICE kernel path; repeat for LSK/SPK/etc.")
    parser.add_argument("--epoch", default="2026 APR 10 00:00:00 TDB")
    parser.add_argument("--target", action="append", default=None, help="Point-mass body; default SUN, EARTH, MOON.")
    parser.add_argument(
        "--state",
        type=float,
        nargs=6,
        default=[400000.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        metavar=("X", "Y", "Z", "VX", "VY", "VZ"),
        help="Barycentric inertial state in km and km/s for acceleration check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from dynamics.spice_ephemeris import make_spice_point_mass_dynamics

    targets = tuple(args.target) if args.target else ("SUN", "EARTH", "MOON")
    ephemeris, dynamics = make_spice_point_mass_dynamics(
        kernels=kernel_paths(args.kernel),
        epoch=args.epoch,
        targets=targets,
    )

    try:
        print(f"epoch_et {ephemeris.epoch_et:.9f}")
        for target in targets:
            r = ephemeris.position_km(target.upper(), 0.0)
            print(f"{target.upper()}_pos_km {r[0]:.9f} {r[1]:.9f} {r[2]:.9f}")

        dxdt = dynamics.eom(0.0, args.state)
        print("state_derivative", " ".join(f"{v:.16e}" for v in dxdt))
    finally:
        ephemeris.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
