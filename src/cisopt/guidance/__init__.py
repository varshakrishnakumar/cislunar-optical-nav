from typing import Callable

from ..config import GuidanceCfg
from ..protocols import Guidance, Scenario
from .single_impulse import SingleImpulseGuidance, build_single_impulse


_REGISTRY: dict[str, Callable[[dict, Scenario], Guidance]] = {
    "single_impulse": build_single_impulse,
}


def register_guidance(name: str, builder: Callable[[dict, Scenario], Guidance]) -> None:
    _REGISTRY[name] = builder


def build_guidance(cfg: GuidanceCfg, scenario: Scenario) -> Guidance:
    if cfg.name not in _REGISTRY:
        raise KeyError(
            f"Unknown guidance {cfg.name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[cfg.name](cfg.params, scenario)


__all__ = [
    "SingleImpulseGuidance",
    "build_single_impulse",
    "build_guidance",
    "register_guidance",
]
