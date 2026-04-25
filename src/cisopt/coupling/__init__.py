"""Guidance--navigation coupling sandbox (item 8 from the refactor brief).

The question this module answers: given a navigation error at the correction
epoch, how does the resulting Δv differ from the perfect-information Δv, and
how does that propagate into terminal miss distance?

It's a guidance-only analysis -- no filter loop, just the targeting solver --
so it sweeps far faster than the full Monte Carlo runner. The shape of the
output is intentionally compatible with the structured-results storage in
``cisopt.sweeps.storage`` so coupling rows can sit next to MC trial rows in
the same Parquet table when desired.
"""

from .maps import (
    CouplingRow,
    coupling_grid_random,
    coupling_grid_structured,
    navigation_to_burn,
)

__all__ = [
    "CouplingRow",
    "coupling_grid_random",
    "coupling_grid_structured",
    "navigation_to_burn",
]
