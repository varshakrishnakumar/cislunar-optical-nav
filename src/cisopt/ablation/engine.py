"""Cartesian product expansion of an AblationCfg into per-combo overrides."""

from __future__ import annotations

import itertools
from typing import Any


def combos_from_axes(axes: dict[str, list[Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Expand axes into ``(combo_id, overrides)`` pairs.

    ``combo_id`` is a stable, human-readable string of the axis values in the
    same order as the axes dict (Python dicts are insertion-ordered).
    """
    if not axes:
        return []

    keys = list(axes.keys())
    value_lists = [axes[k] for k in keys]

    out: list[tuple[str, dict[str, Any]]] = []
    for idx, combo in enumerate(itertools.product(*value_lists)):
        overrides = {k: v for k, v in zip(keys, combo)}
        combo_id = f"c{idx:04d}_" + "_".join(_token(v) for v in combo)
        out.append((combo_id, overrides))
    return out


def _token(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (int, float)):
        return f"{value:g}".replace("-", "n").replace(".", "p")
    s = str(value)
    return "".join(c if c.isalnum() else "_" for c in s)[:24]
