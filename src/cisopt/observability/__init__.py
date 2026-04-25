"""First-class observability analysis (item 4 from the refactor brief).

The discrete-time observability Gramian for a linearised system is
    W = Σ_k Φ(t_k, t_0)ᵀ H_kᵀ H_k Φ(t_k, t_0)
where Φ is the state-transition matrix and H_k the measurement Jacobian. Its
eigenvalues quantify how observable each direction in state space is — small
eigenvalues mark "weak directions" that bearing-only measurements struggle to
resolve.

This module turns those quantities into reusable functions and a runtime
accumulator that the cisopt runner can plug into.
"""

from .gramian import (
    GramianResult,
    accumulate_gramian,
    compute_gramian,
    condition_number,
    rank,
    weak_directions,
)

__all__ = [
    "GramianResult",
    "accumulate_gramian",
    "compute_gramian",
    "condition_number",
    "rank",
    "weak_directions",
]
