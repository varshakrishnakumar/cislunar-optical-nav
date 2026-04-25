from __future__ import annotations

from dataclasses import dataclass

from ..config import ExperimentCfg


@dataclass(frozen=True)
class MCSweepCfg:
    base_cfg: ExperimentCfg
    n_trials: int = 100
    base_seed: int = 7
    n_workers: int = 1
    save_per_trial_artifacts: bool = False

    def __post_init__(self) -> None:
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}")
        if self.n_workers == 0:
            raise ValueError("n_workers must be != 0 (use -1 for all cores)")


def make_trial_seed(base_seed: int, trial_id: int) -> int:
    return int((int(base_seed) * 1000003 + int(trial_id) * 9176) % (2**32 - 1))
