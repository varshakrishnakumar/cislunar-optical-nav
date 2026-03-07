from __future__ import annotations

from .types import MonteCarloConfig


def make_baseline_mc_config(
    *,
    mu: float,
    t0: float,
    tf: float,
    tc: float,
    dt_meas: float,
    sigma_px: float = 1.5,
    n_trials: int = 200,
    base_seed: int = 7,
) -> MonteCarloConfig:
    return MonteCarloConfig(
        mu=mu,
        t0=t0,
        tf=tf,
        tc=tc,
        dt_meas=dt_meas,
        sigma_px=sigma_px,
        dropout_prob=0.0,
        tracking_attitude=True,
        n_trials=n_trials,
        base_seed=base_seed,
        study_name="baseline",
    )


def make_dropout_mc_config(
    *,
    mu: float,
    t0: float,
    tf: float,
    tc: float,
    dt_meas: float,
    sigma_px: float = 1.5,
    dropout_prob: float = 0.05,
    n_trials: int = 200,
    base_seed: int = 7,
) -> MonteCarloConfig:
    return MonteCarloConfig(
        mu=mu,
        t0=t0,
        tf=tf,
        tc=tc,
        dt_meas=dt_meas,
        sigma_px=sigma_px,
        dropout_prob=dropout_prob,
        tracking_attitude=True,
        n_trials=n_trials,
        base_seed=base_seed,
        study_name="dropout",
    )


def make_no_tracking_mc_config(
    *,
    mu: float,
    t0: float,
    tf: float,
    tc: float,
    dt_meas: float,
    sigma_px: float = 1.5,
    dropout_prob: float = 0.05,
    n_trials: int = 200,
    base_seed: int = 7,
) -> MonteCarloConfig:
    return MonteCarloConfig(
        mu=mu,
        t0=t0,
        tf=tf,
        tc=tc,
        dt_meas=dt_meas,
        sigma_px=sigma_px,
        dropout_prob=dropout_prob,
        tracking_attitude=False,
        n_trials=n_trials,
        base_seed=base_seed,
        study_name="no_tracking",
    )


def make_high_noise_mc_config(
    *,
    mu: float,
    t0: float,
    tf: float,
    tc: float,
    dt_meas: float,
    sigma_px: float = 5.0,
    n_trials: int = 200,
    base_seed: int = 7,
) -> MonteCarloConfig:
    return MonteCarloConfig(
        mu=mu,
        t0=t0,
        tf=tf,
        tc=tc,
        dt_meas=dt_meas,
        sigma_px=sigma_px,
        dropout_prob=0.0,
        tracking_attitude=True,
        n_trials=n_trials,
        base_seed=base_seed,
        study_name="high_noise",
    )
