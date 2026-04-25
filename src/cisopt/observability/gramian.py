from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class GramianResult:
    W: Array
    eigvals: Array
    eigvecs: Array
    condition_number: float
    rank: int
    weak_directions: Array

    @property
    def smallest_eig(self) -> float:
        return float(self.eigvals[0])

    @property
    def largest_eig(self) -> float:
        return float(self.eigvals[-1])


def accumulate_gramian(
    Phi_steps: Iterable[Array],
    H_per_step: Iterable[Array | None],
    *,
    n: int = 6,
) -> Array:
    """Build the discrete-time observability Gramian from per-step Φ and H.

    Phi_steps[k] is Φ(t_k, t_{k-1}) — single-step STM. H_per_step[k] is the
    measurement Jacobian at step k, or None if no measurement was accepted.
    """
    W = np.zeros((n, n), dtype=float)
    Phi_cum = np.eye(n, dtype=float)
    for Phi_step, H in zip(Phi_steps, H_per_step):
        Phi_cum = np.asarray(Phi_step, dtype=float) @ Phi_cum
        if H is None:
            continue
        H_arr = np.asarray(H, dtype=float)
        W = W + Phi_cum.T @ H_arr.T @ H_arr @ Phi_cum
    return W


def compute_gramian(W: Array, *, rank_tol: float | None = None) -> GramianResult:
    W_arr = np.asarray(W, dtype=float)
    W_sym = 0.5 * (W_arr + W_arr.T)
    eigvals, eigvecs = np.linalg.eigh(W_sym)
    cond = condition_number(W_sym, eigvals=eigvals)
    rk = rank(W_sym, eigvals=eigvals, tol=rank_tol)
    weak = weak_directions(W_sym, eigvals=eigvals, eigvecs=eigvecs, n=2)
    return GramianResult(
        W=W_sym,
        eigvals=eigvals,
        eigvecs=eigvecs,
        condition_number=cond,
        rank=rk,
        weak_directions=weak,
    )


def condition_number(W: Array, *, eigvals: Array | None = None) -> float:
    if eigvals is None:
        eigvals = np.linalg.eigvalsh(0.5 * (W + W.T))
    smallest = float(eigvals[0])
    largest = float(eigvals[-1])
    if smallest <= 0.0 or not np.isfinite(largest):
        return float("inf")
    return float(largest / smallest)


def rank(W: Array, *, eigvals: Array | None = None, tol: float | None = None) -> int:
    if eigvals is None:
        eigvals = np.linalg.eigvalsh(0.5 * (W + W.T))
    largest = float(eigvals[-1])
    eps = float(np.finfo(float).eps) if largest > 0.0 else 0.0
    if tol is None:
        tol = max(eps * largest * len(eigvals), 0.0)
    return int(np.sum(eigvals > tol))


def weak_directions(
    W: Array,
    *,
    n: int = 2,
    eigvals: Array | None = None,
    eigvecs: Array | None = None,
) -> Array:
    """Return the eigenvectors associated with the n smallest eigenvalues.

    Columns of the returned array are the weak-direction unit vectors.
    """
    if eigvals is None or eigvecs is None:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (W + W.T))
    n = max(1, min(int(n), eigvecs.shape[1]))
    return np.asarray(eigvecs[:, :n], dtype=float)
