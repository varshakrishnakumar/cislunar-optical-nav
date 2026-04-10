from __future__ import annotations

from pathlib import Path

from .jpl_periodic_orbits import (
    PeriodicOrbitQuery,
    fetch_periodic_orbits,
    fetch_periodic_orbits_cached,
)
from .types import PeriodicOrbitRecord


def collect_periodic_orbit_candidates(
    *,
    system: str = "earth-moon",
    family: str = "halo",
    libration_points: list[int] | tuple[int, ...] = (2,),
    branches: list[str] | tuple[str, ...] = ("S",),
    period_min_days: float | None = 5.0,
    period_max_days: float | None = 8.0,
    target_period_days: float = 6.56,
    stability_max: float | None = None,
    timeout_s: float = 30.0,
    cache_dir: str | Path = "data/cache/jpl_periodic_orbits",
    refresh_cache: bool = False,
    no_cache: bool = False,
) -> list[PeriodicOrbitRecord]:
    records: list[PeriodicOrbitRecord] = []
    for libr in libration_points:
        for branch in branches:
            query = PeriodicOrbitQuery(
                system=system,
                family=family,
                libration_point=libr,
                branch=branch,
                period_min=period_min_days,
                period_max=period_max_days,
                period_units="d",
                stability_max=stability_max,
            )
            if no_cache:
                catalog = fetch_periodic_orbits(query, timeout_s=timeout_s)
            else:
                catalog = fetch_periodic_orbits_cached(
                    query,
                    cache_dir=cache_dir,
                    timeout_s=timeout_s,
                    refresh=refresh_cache,
                )
            records.extend(catalog.records)

    return rank_periodic_orbit_candidates(
        records,
        target_period_days=target_period_days,
    )


def rank_periodic_orbit_candidates(
    records: list[PeriodicOrbitRecord],
    *,
    target_period_days: float = 6.56,
) -> list[PeriodicOrbitRecord]:
    return sorted(
        records,
        key=lambda record: (
            abs(record.period_days - target_period_days),
            record.stability,
        ),
    )
