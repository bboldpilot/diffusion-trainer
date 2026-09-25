"""Internal replica of the score on held-out training images.

Same strata, same seed derivation (on the holdout image's own hash), same per-sample error and the same
0.5 / 0.5 caption / empty weighting. The number of noises per stratum is configurable because the replica
runs many times during training; the structure is otherwise the metric's.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from . import objective as obj
from .data import Item

Forward = Callable[[Item, str, torch.Tensor, float], torch.Tensor]


class HoldoutReplica:
    def __init__(self, items: list[Item], bands: int = obj.BANDS, noises: int = 1, noise_batch: int = 2,
                 text_weight: float = obj.TEXT_WEIGHT):
        if not items:
            raise ValueError("holdout replica needs at least one image")
        if noises < 1 or noise_batch < 1:
            raise ValueError("noise counts must be positive")
        self.items, self.bands, self.noises, self.noise_batch, self.text_weight = items, bands, noises, noise_batch, text_weight

    @torch.no_grad()
    def score(self, forward: Forward, device) -> dict:
        started = time.time()
        per_mode = {"text": [], "no_text": []}
        band_sum = np.zeros(self.bands)
        band_count = 0
        for item in self.items:
            x = item.scaled.to(device)
            for mode in per_mode:
                prompt = item.caption if mode == "text" else ""
                losses = []
                for band in range(self.bands):
                    sigma = obj.sigma_of(band, self.bands)
                    band_losses = []
                    for start in range(0, self.noises, self.noise_batch):
                        stop = min(self.noises, start + self.noise_batch)
                        noise = torch.cat([obj.seeded_noise(x.shape, obj.case_seed(item.sha256, band, k)) for k in range(start, stop)]).to(device)
                        x_batch = x.expand(noise.shape[0], *x.shape[1:]) if x.shape[0] == 1 else x
                        noisy, target = obj.noisy_and_target(x_batch, noise, sigma)
                        denoised = forward(item, prompt, noisy, sigma)
                        band_losses += obj.velocity_error(noisy, denoised, target, sigma).tolist()
                    losses += band_losses
                    band_sum[band] += float(np.mean(band_losses))
                band_count += 1
                per_mode[mode].append(float(np.mean(losses)))
        per_image = [obj.weighted_score(t, n, self.text_weight) for t, n in zip(per_mode["text"], per_mode["no_text"])]
        return {
            "score": float(np.mean(per_image)),
            "text": float(np.mean(per_mode["text"])),
            "no_text": float(np.mean(per_mode["no_text"])),
            "per_image": per_image,
            "worst": float(np.max(per_image)),
            "p75": float(np.quantile(per_image, 0.75)),
            "per_band": (band_sum / max(1, band_count)).tolist(),
            "seconds": time.time() - started,
        }
