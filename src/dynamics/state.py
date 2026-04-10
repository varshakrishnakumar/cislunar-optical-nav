from __future__ import annotations

import numpy as np


Array = np.ndarray


def pack_state_and_stm(x: Array, phi: Array | None = None) -> Array:
    state = np.asarray(x, dtype=float).reshape(6)
    if phi is None:
        phi = np.eye(6, dtype=float)
    phi_arr = np.asarray(phi, dtype=float).reshape(6, 6)
    return np.concatenate([state, phi_arr.reshape(-1, order="F")])


def unpack_state_and_stm(z: Array) -> tuple[Array, Array]:
    state_and_stm = np.asarray(z, dtype=float).reshape(42)
    state = state_and_stm[:6].copy()
    phi = state_and_stm[6:].reshape(6, 6, order="F").copy()
    return state, phi
