"""Variance-preserving spatial PFA; plain Python state, no model parameters."""
import math

import torch
from torch.nn import functional as F


def correlated_normal(reference, rho, generator=None, correlated_generator=None):
    """Unit-variance Gaussian field with circular 3x3 spatial correlation.

    With rho=0 and no generator this preserves the original randn_like call.
    The two fields are independent; normalization is the kernel L2 norm,
    never an empirical per-image normalization.
    """
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("PFA rho must be finite and in [0, 1]")
    if generator is None:
        independent = torch.randn_like(reference)
    else:
        independent = torch.randn(reference.shape, dtype=reference.dtype,
                                  device=reference.device, generator=generator)
    if rho == 0:
        return independent
    if reference.ndim != 4 or min(reference.shape[-2:]) < 3:
        raise ValueError("Spatial PFA requires NCHW fields with H,W >= 3")
    field = torch.randn(reference.shape, dtype=reference.dtype,
                        device=reference.device, generator=correlated_generator)
    correlated = 3.0 * F.avg_pool2d(F.pad(field, (1, 1, 1, 1), mode="circular"), 3, stride=1)
    return math.sqrt(1 - rho) * independent + math.sqrt(rho) * correlated


class FeaturePerturbation:
    """Independent per-level/component RNG streams, explicitly checkpointed.

    Default rho=0/seed=None keeps legacy global-RNG behavior exactly.
    Paired B0/B1 experiments supply the same seed, including for rho=0.
    """
    def __init__(self, rho=0.0, seed=None):
        if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("PFA rho must be finite and in [0, 1]")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("PFA seed must be a nonnegative integer or None")
        self.rho = float(rho)
        self.seed = 0 if rho > 0 and seed is None else seed
        self.generators = {}

    def generator(self, reference, level, component):
        if self.seed is None:
            return None
        key = (str(reference.device), level, component)
        if key not in self.generators:
            self.generators[key] = torch.Generator(device=reference.device).manual_seed(
                self.seed + 1009 * level + 100003 * component)
        return self.generators[key]

    def __call__(self, reference, level):
        rho = self.rho if level in (3, 4) else 0.0
        return correlated_normal(reference, rho, self.generator(reference, level, 0),
                                 self.generator(reference, level, 1) if rho else None)

    def state_dict(self):
        if self.seed is None:
            return None
        return {"rho": self.rho, "seed": self.seed,
                "generators": {key: value.get_state().cpu().clone()
                               for key, value in self.generators.items()}}

    def load_state_dict(self, state):
        if state is None:
            if self.seed is not None:
                raise RuntimeError("Missing independent PFA RNG recovery state")
            return
        if state["rho"] != self.rho or state["seed"] != self.seed:
            raise RuntimeError("PFA RNG configuration mismatch")
        self.generators = {}
        for key, value in state["generators"].items():
            generator = torch.Generator(device=key[0])
            generator.set_state(value.cpu())
            self.generators[key] = generator
