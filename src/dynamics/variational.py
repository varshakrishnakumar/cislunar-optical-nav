from functools import lru_cache

from dynamics.models import CR3BPDynamics


def cr3bp_eom_with_stm(t, z, mu):
    return _cr3bp_model(float(mu)).eom_with_stm(t, z)


@lru_cache(maxsize=16)
def _cr3bp_model(mu: float) -> CR3BPDynamics:
    return CR3BPDynamics(mu=mu)
