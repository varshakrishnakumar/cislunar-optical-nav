from typing import Callable

from ..config import ScenarioCfg
from ..protocols import Scenario
from .halo_l1_cr3bp import HaloL1CR3BPScenario, build_halo_l1_cr3bp
from .nrho_cr3bp import NRHOCR3BPScenario, build_nrho_cr3bp


def _build_halo_l1_spice(params: dict):
    from .halo_l1_spice import build_halo_l1_spice
    return build_halo_l1_spice(params)


_REGISTRY: dict[str, Callable[[dict], Scenario]] = {
    "halo_l1_cr3bp": build_halo_l1_cr3bp,
    "halo_l1_spice": _build_halo_l1_spice,
    "nrho_cr3bp": build_nrho_cr3bp,
}


def register_scenario(name: str, builder: Callable[[dict], Scenario]) -> None:
    _REGISTRY[name] = builder


def build_scenario(cfg: ScenarioCfg) -> Scenario:
    if cfg.name not in _REGISTRY:
        raise KeyError(
            f"Unknown scenario {cfg.name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[cfg.name](cfg.params)


__all__ = [
    "HaloL1CR3BPScenario",
    "NRHOCR3BPScenario",
    "build_halo_l1_cr3bp",
    "build_nrho_cr3bp",
    "build_scenario",
    "register_scenario",
]
