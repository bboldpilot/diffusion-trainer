"""The scoring objective, restated so training and selection use the identical arithmetic.

Rectified flow with unit noise: noisy = (1 - sigma) * x + sigma * eps, velocity target = eps - x.
The model returns x0 (denoised); the velocity it implies is (noisy - denoised) / sigma. The score is the
per-sample mean of the squared velocity error, averaged over 16 sigma midpoints and 16 noises, and then
0.5 * captioned + 0.5 * empty-prompt.

Noise seeds are derived from the image hash, stratum and noise index. Training images use the same
derivation on their own hashes, which keeps the internal replica structurally identical to the metric
without touching anything the evaluator will draw.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

BANDS = 16
TEXT_WEIGHT = 0.5
MASTER_SEED = 42


def sigma_of(band: int, bands: int = BANDS) -> float:
    return (band + 0.5) / bands


def case_seed(image_hash: str, band: int, noise_index: int, master: int = MASTER_SEED) -> int:
    key = f"{master}:{image_hash}:{band}:{noise_index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def seeded_noise(shape, seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def noisy_and_target(x: torch.Tensor, noise: torch.Tensor, sigma: float):
    return (1.0 - sigma) * x + sigma * noise, noise - x


def velocity_error(noisy: torch.Tensor, denoised: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-sample squared velocity error, shape (batch,)."""
    if not 0.0 < sigma < 1.0:
        raise ValueError("sigma must lie strictly inside (0, 1)")
    predicted = (noisy.float() - denoised.float()) / sigma
    return (predicted - target.float()).square().flatten(1).mean(1)


def weighted_score(text_mean: float, no_text_mean: float, text_weight: float = TEXT_WEIGHT) -> float:
    return text_weight * text_mean + (1.0 - text_weight) * no_text_mean


@dataclass(frozen=True)
class Case:
    image_index: int
    band: int

    @property
    def sigma(self) -> float:
        return sigma_of(self.band)
