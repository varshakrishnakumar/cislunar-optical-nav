from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import ExperimentCfg


@dataclass(frozen=True)
class AblationCfg:
    """Cartesian product over named axes, each axis a list of values applied
    via dotted-path overrides on top of base_cfg."""

    base_cfg: ExperimentCfg
    axes: dict[str, list[Any]] = field(default_factory=dict)
    n_trials_per_combo: int = 20
    base_seed: int = 7
    n_workers: int = 1

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("AblationCfg requires at least one axis")
        for k, v in self.axes.items():
            if not isinstance(v, list) or not v:
                raise ValueError(f"Axis {k!r} must be a non-empty list, got {v!r}")
        if self.n_trials_per_combo <= 0:
            raise ValueError(
                f"n_trials_per_combo must be > 0, got {self.n_trials_per_combo}"
            )
