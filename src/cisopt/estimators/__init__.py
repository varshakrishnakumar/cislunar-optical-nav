from typing import Callable

from ..config import EstimatorCfg
from ..protocols import Estimator, Scenario
from .ekf import EKFEstimator, build_ekf
from .iekf import IEKFEstimator, build_iekf
from .ukf import UKFEstimator, build_ukf


_REGISTRY: dict[str, Callable[[dict, Scenario], Estimator]] = {
    "ekf": build_ekf,
    "iekf": build_iekf,
    "ukf": build_ukf,
}


def register_estimator(name: str, builder: Callable[[dict, Scenario], Estimator]) -> None:
    _REGISTRY[name] = builder


def build_estimator(cfg: EstimatorCfg, scenario: Scenario) -> Estimator:
    if cfg.name not in _REGISTRY:
        raise KeyError(
            f"Unknown estimator {cfg.name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[cfg.name](cfg.params, scenario)


__all__ = [
    "EKFEstimator",
    "IEKFEstimator",
    "UKFEstimator",
    "build_ekf",
    "build_iekf",
    "build_ukf",
    "build_estimator",
    "register_estimator",
]
