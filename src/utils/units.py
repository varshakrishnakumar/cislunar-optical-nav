"""Unit conventions for CR3BP-vs-SPICE analysis pipelines.

The CR3BP pipeline runs in dimensionless rotating-frame units (DU/TU); the
SPICE pipeline runs in km / km·s / s. Every numeric threshold, axis
label, and consistency gate that the analysis layer applies must know
which mode is active. This module is the single source of truth.

Use ``RunUnits.for_truth("cr3bp" | "spice")`` to get a dataclass with:
- ``length_km``, ``time_s``, ``velocity_kmps``  — multiplicative factors
   from the run's native units to km / s / km·s. CR3BP factors use the
   canonical Earth–Moon system constants; SPICE factors use the
   per-epoch lunit/tunit produced by ``run_case_spice`` and should be
   overridden via ``RunUnits.for_spice(lunit_km=..., tunit_s=...)`` for
   per-epoch fidelity.
- ``length_label`` / ``time_label`` / ``velocity_label`` — short labels
   like ``"ND"`` or ``"km"`` for axis/legend strings.
- ``frame_label`` — coordinate frame, e.g. ``"CR3BP rotating"`` or
   ``"J2000"``, used in axis labels per the Phase C convention
   ``[unit, frame]``.

The rescale_threshold helper takes a threshold expressed in CR3BP-ND
(the legacy convention) and returns its equivalent in the run's native
units, so callers don't have to bake unit math into their gating code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

# ── Earth–Moon canonical constants ───────────────────────────────────────────
# CR3BP uses these to non-dimensionalize. The same numbers also let us
# convert CR3BP-ND outputs to physical km when stitching together
# CR3BP- and SPICE-mode comparison plots.
_LU_EM_KM = 384_400.0                     # Earth–Moon mean distance
_TU_EM_DAYS = 4.343                       # T_moon / 2π in days
_TU_EM_S = _TU_EM_DAYS * 86_400.0         # ≈ 375 234 s
_VU_EM_KMPS = _LU_EM_KM / _TU_EM_S         # ≈ 1.024 km/s

TruthMode = Literal["cr3bp", "spice"]


@dataclass(frozen=True)
class RunUnits:
    """Unit-conversion facts for a single Monte Carlo run.

    All factors convert *from* the run's native units *to* SI:
        length_km       — multiply by this to convert run lengths → km
        time_s          — multiply by this to convert run times   → s
        velocity_kmps   — multiply by this to convert run vels    → km/s

    For CR3BP runs the factors are the canonical EM normalization. For
    SPICE runs the factors are 1.0 in their respective dimensions
    (native = SI), but ``length_km_per_nd`` still exposes the EM scale
    so legacy ND-tuned thresholds can be rescaled.
    """
    truth: TruthMode
    length_km: float
    time_s: float
    velocity_kmps: float
    length_label: str
    time_label: str
    velocity_label: str
    frame_label: str

    # Always-on conversion from CR3BP-ND legacy thresholds to km, regardless
    # of which mode this run uses. Lets us keep config files in ND.
    length_km_per_nd: float = _LU_EM_KM
    velocity_kmps_per_nd: float = _VU_EM_KMPS

    @classmethod
    def for_truth(cls, truth: TruthMode, **overrides) -> "RunUnits":
        truth = (truth or "cr3bp").lower()  # type: ignore[assignment]
        if truth == "cr3bp":
            base = cls(
                truth="cr3bp",
                length_km=_LU_EM_KM,
                time_s=_TU_EM_S,
                velocity_kmps=_VU_EM_KMPS,
                length_label="ND",
                time_label="TU",
                velocity_label="ND",
                frame_label="CR3BP rotating",
            )
        elif truth == "spice":
            base = cls(
                truth="spice",
                length_km=1.0,
                time_s=1.0,
                velocity_kmps=1.0,
                length_label="km",
                time_label="s",
                velocity_label="km/s",
                frame_label="J2000",
            )
        else:
            raise ValueError(f"Unknown truth mode: {truth!r}")
        return replace(base, **overrides) if overrides else base

    @classmethod
    def for_spice(cls, *, lunit_km: float, tunit_s: float) -> "RunUnits":
        """SPICE units with explicit per-epoch lunit/tunit (overrides the
        canonical EM constants when comparing across epochs)."""
        vunit = lunit_km / tunit_s
        return cls(
            truth="spice",
            length_km=1.0,
            time_s=1.0,
            velocity_kmps=1.0,
            length_label="km",
            time_label="s",
            velocity_label="km/s",
            frame_label="J2000",
            length_km_per_nd=lunit_km,
            velocity_kmps_per_nd=vunit,
        )

    # ── threshold helpers ────────────────────────────────────────────────────
    def length_threshold_native(self, value_nd: float) -> float:
        """Convert a length threshold expressed in CR3BP-ND units to the
        run's native units. Use this for callers that have hard-coded
        ND tolerances (e.g. ``--tol 1e-3`` meaning ND length).

        CR3BP run: returns value_nd unchanged.
        SPICE run: returns value_nd * lunit_km (= value in km).
        """
        if self.truth == "cr3bp":
            return float(value_nd)
        return float(value_nd) * self.length_km_per_nd

    def velocity_threshold_native(self, value_nd: float) -> float:
        if self.truth == "cr3bp":
            return float(value_nd)
        return float(value_nd) * self.velocity_kmps_per_nd

    # ── label helpers (Phase C `[unit, frame]` convention) ───────────────────
    def length_axis(self, quantity: str = "length") -> str:
        return f"{quantity}  [{self.length_label}, {self.frame_label}]"

    def velocity_axis(self, quantity: str = "velocity") -> str:
        return f"{quantity}  [{self.velocity_label}, {self.frame_label}]"

    def time_axis(self, quantity: str = "time") -> str:
        return f"{quantity}  [{self.time_label}]"

    @property
    def miss_axis_km(self) -> str:
        """Axis label for a miss-distance plot rendered in km, regardless
        of native units. CR3BP runs have to be scaled by length_km_per_nd
        before plotting; SPICE runs plot natively."""
        return "miss to target  [km]"
