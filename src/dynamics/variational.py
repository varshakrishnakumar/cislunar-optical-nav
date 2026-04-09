import numpy as np
from dynamics.cr3bp import CR3BP

def cr3bp_eom_with_stm(t, z, mu):
    sys = CR3BP(mu=mu)
    x = z[:6]
    phi = z[6:].reshape((6, 6), order="F")
    dxdt = sys.eom(t, x)
    A = sys.A_matrix(t, x)
    dphidt = A @ phi
    return np.concatenate([dxdt, dphidt.reshape(-1, order="F")])

def cr3bp_eom(t: float, x: np.ndarray, mu: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(6,)
    z = np.concatenate([x, np.eye(6, dtype=float).reshape(-1, order="F")])
    return cr3bp_eom_with_stm(t, z, mu)[:6]
