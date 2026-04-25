from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Measurement:
    t_s: float
    valid: bool
    payload: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateEstimate:
    t_s: float
    x: Array
    P: Array

    def copy(self) -> "StateEstimate":
        return StateEstimate(t_s=float(self.t_s), x=self.x.copy(), P=self.P.copy())


@runtime_checkable
class Dynamics(Protocol):
    name: str

    def eom(self, t: float, x: Array) -> Array: ...
    def eom_with_stm(self, t: float, z: Array) -> Array: ...


class Scenario(Protocol):
    name: str
    dynamics: Dynamics
    t0_s: float
    tf_s: float
    tc_s: float
    dt_meas_s: float

    def initial_truth(self, *, dx0: Array | None = None) -> Array: ...
    def initial_estimate(self, *, est_err: Array | None = None) -> StateEstimate: ...
    def target_position(self) -> Array: ...
    def body_position(self, body: str, t_s: float) -> Array: ...


@runtime_checkable
class Sensor(Protocol):
    name: str

    def measure(
        self,
        t_s: float,
        x_truth: Array,
        x_estimate: Array | None,
        *,
        rng: np.random.Generator,
    ) -> Measurement: ...


@runtime_checkable
class Estimator(Protocol):
    name: str

    def predict(
        self, t1: float, est: StateEstimate,
    ) -> tuple[StateEstimate, dict[str, Any]]:
        """Return propagated estimate and an info dict.

        The info dict should include 'Phi_step' (the discrete-time STM over
        [est.t_s, t1]) when available so observability accumulators can build
        the Gramian. Estimators without an STM may omit it.
        """

    def update(
        self,
        est: StateEstimate,
        meas: Measurement,
    ) -> tuple[StateEstimate, dict[str, Any]]:
        """Return updated estimate and an info dict.

        Info should include 'accepted' (bool), 'nis' (float), and (when the
        update was accepted) 'H' so observability can grow the Gramian.
        """


@runtime_checkable
class Guidance(Protocol):
    name: str

    def solve(
        self,
        est: StateEstimate,
        scenario: Scenario,
    ) -> tuple[Array, dict[str, Any]]: ...
