"""Monte Carlo utilities for cislunar_optical_nav.

Keep this package lightweight: types + sampling + runner + metrics + stats + IO.
"""

from .types import MonteCarloConfig, TrialInput, TrialResult
from .sampler import make_trial_rng, sample_injection_error, sample_estimation_error
from .runner import run_monte_carlo
from .stats import summarize_results, success_rate
from .io import save_results_csv, load_results_csv

__all__ = [
    "MonteCarloConfig",
    "TrialInput",
    "TrialResult",
    "make_trial_rng",
    "sample_injection_error",
    "sample_estimation_error",
    "run_monte_carlo",
    "summarize_results",
    "success_rate",
    "save_results_csv",
    "load_results_csv",
]
