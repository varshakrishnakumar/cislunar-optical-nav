from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScenarioCfg:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorCfg:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EstimatorCfg:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuidanceCfg:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialCfg:
    seed: int = 7
    sigma_r_inj: float = 0.0
    sigma_v_inj: float = 0.0
    sigma_r_est: float = 0.0
    sigma_v_est: float = 0.0
    planar_only: bool = False


@dataclass(frozen=True)
class OutputCfg:
    out_dir: str = "results/cisopt"
    save_debug: bool = True


@dataclass(frozen=True)
class ExperimentCfg:
    name: str
    scenario: ScenarioCfg
    sensor: SensorCfg
    estimator: EstimatorCfg
    guidance: GuidanceCfg
    trial: TrialCfg = field(default_factory=TrialCfg)
    output: OutputCfg = field(default_factory=OutputCfg)


def _coerce_subcfg(raw: dict[str, Any], cls):
    return cls(name=str(raw["name"]), params=dict(raw.get("params", {})))


def _coerce_trial(raw: dict[str, Any] | None) -> TrialCfg:
    raw = dict(raw or {})
    return TrialCfg(**raw)


def _coerce_output(raw: dict[str, Any] | None) -> OutputCfg:
    raw = dict(raw or {})
    return OutputCfg(**raw)


def from_dict(raw: dict[str, Any]) -> ExperimentCfg:
    required = ("name", "scenario", "sensor", "estimator", "guidance")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"ExperimentCfg missing required keys: {missing}")
    return ExperimentCfg(
        name=str(raw["name"]),
        scenario=_coerce_subcfg(raw["scenario"], ScenarioCfg),
        sensor=_coerce_subcfg(raw["sensor"], SensorCfg),
        estimator=_coerce_subcfg(raw["estimator"], EstimatorCfg),
        guidance=_coerce_subcfg(raw["guidance"], GuidanceCfg),
        trial=_coerce_trial(raw.get("trial")),
        output=_coerce_output(raw.get("output")),
    )


def load_config(path: str | Path) -> ExperimentCfg:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        raw = json.loads(p.read_text())
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Loading YAML configs requires PyYAML. "
                "Install with: pip install pyyaml"
            ) from exc
        raw = yaml.safe_load(p.read_text())
    elif suffix == ".toml":
        import tomllib
        raw = tomllib.loads(p.read_text())
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    return from_dict(raw)


def to_dict(cfg: ExperimentCfg) -> dict[str, Any]:
    return asdict(cfg)


def _set_dotted(target: dict[str, Any], path: str, value: Any) -> None:
    """Walk dotted-path parents (must exist as dicts) and set the leaf.

    The parent path is verified to catch typos in known structure (e.g.
    ``sensorr.params.x`` raises). The leaf itself is allowed to be new so
    sensors / estimators can grow opt-in params without the YAML needing to
    declare every defaulted field.
    """
    keys = path.split(".")
    cur = target
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise KeyError(f"patch_cfg: path {path!r} not found at segment {k!r}")
        cur = cur[k]
    cur[keys[-1]] = value


def patch_cfg(cfg: ExperimentCfg, overrides: dict[str, Any]) -> ExperimentCfg:
    """Return a new ExperimentCfg with dotted-path overrides applied.

    Example: patch_cfg(cfg, {"trial.seed": 42, "estimator.params.q_acc": 1e-11})
    """
    raw = to_dict(cfg)
    for path, value in overrides.items():
        _set_dotted(raw, path, value)
    return from_dict(raw)


def config_hash(cfg: ExperimentCfg) -> str:
    """Stable SHA256 hash of an ExperimentCfg's canonical JSON form.

    Used for results-DB metadata so identical configs hash identically across
    runs; floats and dict ordering are normalized via sort_keys + repr.
    """
    payload = json.dumps(to_dict(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
