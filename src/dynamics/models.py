from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cr3bp import CR3BP
from .state import unpack_state_and_stm


Array = np.ndarray


class DynamicsModel(Protocol):
    name: str

    def eom(self, t: float, x: Array) -> Array:
        ...

    def eom_with_stm(self, t: float, z: Array) -> Array:
        ...


@dataclass(frozen=True)
class CR3BPDynamics:
    mu: float
    tiny: float = 1e-12
    name: str = "CR3BP"

    def __post_init__(self) -> None:
        object.__setattr__(self, "system", CR3BP(mu=float(self.mu), tiny=float(self.tiny)))

    def eom(self, t: float, x: Array) -> Array:
        return self.system.eom(t, x)

    def eom_with_stm(self, t: float, z: Array) -> Array:
        x, phi = unpack_state_and_stm(z)
        dxdt = self.system.eom(t, x)
        dphidt = self.system.A_matrix(t, x) @ phi
        return np.concatenate([dxdt, dphidt.reshape(-1, order="F")])
