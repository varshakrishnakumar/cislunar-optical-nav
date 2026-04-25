from typing import Callable

from ..config import SensorCfg
from ..protocols import Scenario, Sensor
from .camera_bearing import CameraBearingSensor, build_camera_bearing


_REGISTRY: dict[str, Callable[[dict, Scenario], Sensor]] = {
    "camera_bearing": build_camera_bearing,
}


def register_sensor(name: str, builder: Callable[[dict, Scenario], Sensor]) -> None:
    _REGISTRY[name] = builder


def build_sensor(cfg: SensorCfg, scenario: Scenario) -> Sensor:
    if cfg.name not in _REGISTRY:
        raise KeyError(
            f"Unknown sensor {cfg.name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[cfg.name](cfg.params, scenario)


__all__ = [
    "CameraBearingSensor",
    "build_camera_bearing",
    "build_sensor",
    "register_sensor",
]
