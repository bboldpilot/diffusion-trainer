#!/usr/bin/env python3
"""Select a trained ideogram4 adapter using only images withheld from its training.

The base is a diagnostic control; only trained candidates may be submitted.
Every candidate must load all its LoRA modules and receive a finite score.
Original model weights are restored exactly between candidates, including errors.
An incomplete comparison produces no winner file.

This no-text proxy differs from the validator's CFG and sigma schedule. Earlier
validator-on-holdout results are not proof of this implementation's fidelity.
The previous proxy replay ranked two bad candidates correctly; agreement on good
candidates across tasks, and selection within the reserve, still require GPU tests.

    holdout_select.py --base-model /cache/models/... --holdout-dir /dataset/holdout \
        --candidate name=/path/a.safetensors --candidate other=/path/b.safetensors \
        --deadline-epoch 1788974588 --out /tmp/winner.txt
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_calibration import (  # noqa: E402  (same container dir)
    AI_TOOLKIT_ROOT,
    adjust_size,
    collect_pairs,
    denoise_schedule,
    eval_seeds,
    load_lora,
)

# validator/evaluation/constants.py EVAL_DEFAULTS["ideogram4"]
EVAL_STEPS = 30
EVAL_DENOISE = 0.75
# Seeds/images: the validator uses 5 seeds over its own held-out set. Selection only needs the
# ORDER, and the reserve is small, so the default is 2 seeds over up to 8 holdout images; both
# are overridable and the deadline stops the loop before it eats the upload margin.
DEFAULT_IMAGES = 8
DEFAULT_SEEDS = 2


def log(msg: str) -> None:
    print(f"[holdout] {msg}", flush=True)


class Ideogram4NoTextScorer:
    """ideogram4 loaded once; scores whatever adapter is currently applied, no-text branch only."""

    def __init__(self, holdout_dir: str, base_model_path: str, model_kwargs: dict,
                 images: int = DEFAULT_IMAGES, seeds_n: int = DEFAULT_SEEDS):
        import numpy as np
        import torch
        from PIL import Image

        sys.path.insert(0, str(AI_TOOLKIT_ROOT))
        from toolkit.config_modules import ModelConfig
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import get_ideogram4_sigmas

        self.np, self.torch, self.Image = np, torch, Image
        self.sigmas = get_ideogram4_sigmas
        self.device = torch.device("cuda")
        model_config = ModelConfig(name_or_path=base_model_path, arch="ideogram4",
                                   dtype="bf16", model_kwargs=dict(model_kwargs or {}))
        self.model = Ideogram4Model(device=self.device, model_config=model_config, dtype="bf16")
        self.model.load_model()
        self.empty_embeds = self.model.get_prompt_embeds("")
        self.pairs = collect_pairs(holdout_dir, images)
        self.seeds = eval_seeds(seeds_n)
        self.per_candidate: float | None = None
        log(f"scorer ready: {len(self.pairs)} holdout image(s), seeds {self.seeds}, "
            f"{EVAL_STEPS} steps, denoise {EVAL_DENOISE}, no-text branch only")

    def _to_tensor(self, image):
        array = self.np.asarray(image.convert("RGB"), self.np.float32) / 127.5 - 1.0
        return self.torch.from_numpy(array).permute(2, 0, 1).to(self.device)

    def _to_image(self, tensor):
        array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5)
        array = array.round().clamp(0, 255).to("cpu").numpy().astype(self.np.uint8)
        return self.Image.fromarray(array.transpose(1, 2, 0))

    def _l2(self, reference, produced):
        a = self.np.asarray(reference.convert("RGB"), self.np.float32) / 255.0
        b = self.np.asarray(produced.convert("RGB"), self.np.float32) / 255.0
        return float(self.np.mean((a - b) ** 2))

    def _sample(self, latents, schedule):
        torch = self.torch
        with torch.no_grad():
            latents = latents.clone()
            for current, following in zip(schedule[:-1], schedule[1:]):
                t = torch.full((latents.shape[0],), current * 1000.0, device=self.device)
                velocity = self.model.get_noise_prediction(
                    latents.to(self.model.torch_dtype), t, self.empty_embeds)
                latents = latents + (following - current) * velocity.float()
            return self._to_image(self.model.decode_latents(latents)[0])

    def fits(self, deadline: float) -> bool:
        return self.per_candidate is None or time.time() + self.per_candidate <= deadline

    def score(self) -> float:
        """Mean no-text L2 of the adapter currently installed on the model."""
        torch = self.torch
        started = time.time()
        losses = []
        for image_path, _caption in self.pairs:
            reference = self.Image.open(image_path)
            width, height = adjust_size(*reference.size)
            reference = reference.resize((width, height), self.Image.Resampling.LANCZOS)
            latent = self.model.encode_images([self._to_tensor(reference)])
            schedule = denoise_schedule(
                lambda n: self.sigmas(n, width, height), EVAL_STEPS, EVAL_DENOISE)
            for seed in self.seeds:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn(latent.shape, generator=generator,
                                    device=self.device, dtype=torch.float32)
                start = schedule[0]
                noised = latent.float() * (1.0 - start) + noise * start
                losses.append(self._l2(reference, self._sample(noised, schedule)))
        self.per_candidate = time.time() - started
        return float(self.np.mean(losses))


def _lora_roots(model) -> list:
    """Modules the LoRA key names may be rooted at, most specific first.

    ai-toolkit names its keys after the DENOISER tree (`diffusion_model.layers.0.attention.qkv`
    for ideogram4), and on an ai-toolkit model wrapper that tree is `wrapper.model` -- the wrapper
    itself has no `layers`. 12 Sep P2-verify: `_resolve(model, ...)` walked from the wrapper, all
    204 modules raised AttributeError, every candidate was "skipped (no LoRA key matched)" and the
    selection exited 1 -- the third way this selector could fail while the run still shipped an
    artefact. Trying the wrapper too keeps whatever already resolves (krea2 calibration) working.
    """
    roots, seen = [], set()
    for candidate in (getattr(model, "model", None), getattr(model, "unet", None),
                      getattr(model, "transformer", None), model):
        if candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            roots.append(candidate)
    return roots


@contextmanager
def candidate_weights(model, deltas):
    """Apply all LoRA modules and restore exact original weights on every exit.

    Subtracting a delta from rounded BF16 weights does not undo its addition.
    CPU snapshots keep candidates independent without another GPU model copy.
    """
    import torch

    from lora_calibration import _resolve

    roots = _lora_roots(model)
    if not deltas:
        raise ValueError('Candidate contains no complete LoRA modules')
    targets = []
    seen = set()
    for module_name, (a, b) in deltas.items():
        target = None
        for root in roots:
            try:
                target = _resolve(root, module_name)
                break
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
        if target is None or not hasattr(target, "weight"):
            raise ValueError(f'LoRA module does not match model: {module_name}')
        if id(target.weight) in seen:
            raise ValueError(f'Duplicate LoRA target: {module_name}')
        seen.add(id(target.weight))
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f'Invalid LoRA dimensions: {module_name}')
        if b.shape[0] * a.shape[1] != target.weight.numel():
            raise ValueError(f'LoRA shape does not match model: {module_name}')
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise ValueError(f'Nonfinite LoRA weights: {module_name}')
        targets.append((target.weight, a, b))
    originals = []
    try:
        with torch.no_grad():
            for weight, a, b in targets:
                originals.append((weight, weight.detach().to(device='cpu', copy=True)))
                delta = (b.to(torch.float32) @ a.to(torch.float32)).to(weight.dtype)
                weight.add_(delta.view_as(weight))
                if not torch.isfinite(weight).all().item():
                    raise ValueError('Nonfinite merged model weights')
        yield len(targets)
    finally:
        with torch.no_grad():
            for weight, original in originals:
                weight.copy_(original)


def load_candidate(path, device, dtype):
    from safetensors import safe_open
    pairs = {}
    with safe_open(path, framework='pt', device='cpu') as weights:
        for key in weights.keys():
            module, separator, tail = key.rpartition('.lora_')
            if separator and tail in ('A.weight', 'B.weight'):
                pairs.setdefault(module, set()).add(tail[0])
    if not pairs or any(sides != {'A', 'B'} for sides in pairs.values()):
        raise ValueError('Missing or incomplete LoRA A/B pairs')
    deltas = load_lora(path, device, dtype)
    if len(deltas) != len(pairs):
        raise ValueError('LoRA loader did not preserve every module')
    return deltas


def valid_score(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('Candidate loss must be finite and nonnegative')
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--holdout-dir", required=True)
    ap.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--model-kwargs", default="{}")
    ap.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--deadline-epoch", type=float, default=0.0)
    ap.add_argument("--out", help="write the winning candidate's path here")
    ap.add_argument("--report", help="write the full ranking as json here")
    a = ap.parse_args()
    for path in (a.out, a.report):
        if path:
            Path(path).unlink(missing_ok=True)

    cands = []
    for item in a.candidate:
        name, _, path = item.partition("=")
        if name and name != 'base' and name not in dict(cands) and path and Path(path).is_file():
            cands.append((name, path))
        else:
            raise ValueError(f'Invalid, duplicate or missing candidate: {name}')
    if not cands:
        log("no candidates — nothing to select"); return 1
    if not collect_pairs(a.holdout_dir, a.images):
        log(f"holdout empty: {a.holdout_dir} — selection skipped"); return 1

    deadline = a.deadline_epoch or (time.time() + 3600.0)
    scorer = Ideogram4NoTextScorer(a.holdout_dir, a.base_model, json.loads(a.model_kwargs),
                                   images=a.images, seeds_n=a.seeds)

    results = {}
    # The base is a candidate too: on 9 Sep the untrained model beat two of three trained runs on
    # the validator's own set, so a recipe that cannot see that is flying blind.
    results["base"] = valid_score(scorer.score())
    log(f"base (no adapter): {results['base']:.6f}")

    for name, path in cands:
        if not scorer.fits(deadline):
            log(f"deadline: {name} not scored ({int(deadline - time.time())}s left)")
            return 2  # Partial comparisons must not publish a winner.
        started = time.time()
        deltas = load_candidate(path, scorer.device, scorer.torch.float32)
        with candidate_weights(scorer.model, deltas) as n:
            results[name] = valid_score(scorer.score())
            log(f"{name}: {results[name]:.6f} ({n} modules)")
        scorer.per_candidate = time.time() - started

    trained = {k: v for k, v in results.items() if k != "base"}
    if not trained:
        log("no trained candidate scored — leaving the artefact as ai-toolkit saved it"); return 1
    best = min(trained, key=trained.get)
    order = " < ".join(f"{k} {v:.6f}" for k, v in sorted(results.items(), key=lambda kv: kv[1]))
    log(f"ranking: {order}")
    log(f"WINNER {best} ({trained[best]:.6f}); base is "
        f"{'BETTER — training hurt on the holdout' if results['base'] < trained[best] else 'behind, training helped'}")
    if a.report:
        Path(a.report).write_text(json.dumps({"scores": results, "winner": best}, indent=1))
    if a.out:
        Path(a.out).write_text(dict(cands)[best])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
