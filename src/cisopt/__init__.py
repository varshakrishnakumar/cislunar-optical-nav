"""cisopt: config-driven experiment framework for cislunar optical nav.

Wraps the existing primitives in dynamics/, nav/, guidance/, cv/ behind a
small set of protocols (Scenario, Sensor, Estimator, Guidance) so experiments
can be expressed as configs and composed into Monte Carlo / ablation studies.
"""

from .protocols import (
    Dynamics,
    Estimator,
    Guidance,
    Measurement,
    Scenario,
    Sensor,
    StateEstimate,
)

__all__ = [
    "Dynamics",
    "Estimator",
    "Guidance",
    "Measurement",
    "Scenario",
    "Sensor",
    "StateEstimate",
]
