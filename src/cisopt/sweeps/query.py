"""Lightweight query helpers over Parquet trial tables.

Phase B keeps these pure-python (lists of dicts) so pandas stays optional.
Phase D will add a pandas-based ``to_dataframe()`` for richer analysis.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .storage import read_trials


def load(path: str | Path) -> list[dict[str, Any]]:
    return read_trials(path)


def filter_rows(
    rows: Iterable[dict[str, Any]],
    **predicates: Any,
) -> list[dict[str, Any]]:
    """Return rows matching all predicates.

    Each predicate value can be a literal (equality) or a 1-arg callable.
    Example: filter_rows(rows, combo_id="combo_07", nis_mean=lambda x: x < 3)
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        ok = True
        for key, want in predicates.items():
            got = r.get(key)
            if callable(want):
                if not want(got):
                    ok = False
                    break
            elif got != want:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def group_by(
    rows: Iterable[dict[str, Any]],
    key: str | Callable[[dict[str, Any]], Any],
) -> dict[Any, list[dict[str, Any]]]:
    keyfn = (lambda r: r.get(key)) if isinstance(key, str) else key
    out: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(keyfn(r), []).append(r)
    return out


def summarize(
    rows: Iterable[dict[str, Any]],
    metric: str,
) -> dict[str, float]:
    vals = [
        float(r[metric]) for r in rows
        if metric in r and r[metric] is not None and not math.isnan(float(r[metric]))
    ]
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "median": float("nan")}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "median": statistics.median(vals),
    }


def summarize_by_combo(
    rows: Iterable[dict[str, Any]],
    metric: str,
) -> dict[str, dict[str, float]]:
    return {
        combo_id: summarize(group, metric)
        for combo_id, group in group_by(rows, "combo_id").items()
    }
