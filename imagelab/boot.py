"""ComfyUI boot and model loading through its public API, configured for training.

Static VRAM (no dynamic offload) so weight wrappers run in the plain cast path; smart memory off; the
text-encoder compute dtype set per family the way the evaluator's process would pick it.
"""
from __future__ import annotations

import gc
import os
import re
import sys

import numpy as np
import torch

from .families import FAMILIES, FP8_DTYPES

_STATE: dict = {}


def boot(comfy_root: str, text_enc_dtype: str = "fp16"):
    if _STATE:
        return _STATE["mm"]
    comfy_root = os.path.abspath(comfy_root)
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    import comfy.cli_args as cli

    args = cli.args
    args.disable_dynamic_vram = True
    args.highvram = True
    args.disable_smart_memory = True
    args.fp16_text_enc = text_enc_dtype == "fp16"
    args.bf16_text_enc = text_enc_dtype == "bf16"
    args.fp32_text_enc = text_enc_dtype == "fp32"
    args.fp8_e4m3fn_text_enc = False
    args.fp8_e5m2_text_enc = False
    import comfy.model_management as mm

    if hasattr(mm, "in_training"):
        mm.in_training = True
    _STATE["mm"] = mm
    return mm


def free_gpu() -> None:
    mm = _STATE.get("mm")
    if mm is not None:
        mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()


def load_diffusion(state: dict, family: str):
    """Frozen base model as a ComfyUI ModelPatcher; verifies the unit-noise rectified-flow contract."""
    import comfy.model_sampling
    import comfy.sd

    fp8 = any(value.dtype in FP8_DTYPES for value in state.values())
    options = {} if fp8 else {"dtype": torch.bfloat16}
    model = comfy.sd.load_diffusion_model_state_dict(state, model_options=options, disable_dynamic=True)
    if model is None:
        raise RuntimeError("ComfyUI could not identify the diffusion model")
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    if family == "flux":
        guard_inplace_residuals(model.model.diffusion_model)
    spec = FAMILIES[family]
    if spec.shift is not None:
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        model = ModelSamplingAuraFlow().patch_aura(model, spec.shift)[0]
    sampling = model.model.model_sampling
    if not isinstance(sampling, comfy.model_sampling.CONST) or getattr(sampling, "noise_scale", 1.0) != 1.0:
        raise ValueError("model is not unit-noise rectified flow")
    return model


def guard_inplace_residuals(diffusion_model) -> int:
    """ComfyUI's flux blocks add their residuals in place (`img += ...`, `txt += ...`, `x += ...`) onto the
    block's own inputs, after a LayerNorm has already saved that tensor for its backward. Inference never
    notices; a backward pass stops with an autograd version error, and under activation checkpointing the
    recomputation would silently see the already-updated inputs. While gradients are recorded, each block
    therefore works on copies of its hidden-state inputs and each of its LayerNorms normalises a copy, so
    every tensor saved for backward stays as it was. Copies are exact: forward values do not change, and
    with gradients off (holdout scoring) nothing is wrapped at all."""
    guarded = 0
    for name, block in diffusion_model.named_modules():
        if not re.fullmatch(r"(double_blocks|single_blocks)\.\d+", name) or getattr(block, "_imagelab_guarded", False):
            continue

        def block_forward(*args, _forward=block.forward, **kwargs):
            if torch.is_grad_enabled():
                args = tuple(a.clone() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
                kwargs = {k: (v.clone() if k in ("img", "txt", "x") and torch.is_tensor(v) else v) for k, v in kwargs.items()}
            return _forward(*args, **kwargs)

        block.forward = block_forward
        for norm in block.modules():
            if isinstance(norm, torch.nn.LayerNorm):
                def norm_forward(x, _forward=norm.forward):
                    return _forward(x.clone()) if torch.is_grad_enabled() else _forward(x)

                norm.forward = norm_forward
        block._imagelab_guarded = True
        guarded += 1
    if not guarded:
        raise RuntimeError("flux: no double/single blocks found to guard against in-place residual updates")
    return guarded


def load_clip(states: list[dict], family: str):
    import comfy.sd
    import comfy.utils

    prepared = []
    for state in states:
        if any(key.endswith(".comfy_quant") or key == "scaled_fp8" for key in state):
            state, _ = comfy.utils.convert_old_quants(state, model_prefix="", metadata={})
        prepared.append(state)
    clip_type = getattr(comfy.sd.CLIPType, FAMILIES[family].clip_type)
    return comfy.sd.load_text_encoder_state_dicts(prepared, embedding_directory=None, clip_type=clip_type,
                                                  model_options={}, disable_dynamic=True)


def encode_prompt(clip, text: str, family: str):
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    guidance = FAMILIES[family].guidance
    if guidance is not None:
        cond = [[c[0], {**c[1], "guidance": guidance}] for c in cond]
    return cond


class LatentEncoder:
    """Deterministic posterior-mean encode, float32, matching the evaluator's two VAE paths."""

    def __init__(self, kind: str, payload):
        self.kind = kind
        if kind == "comfy":
            import comfy.sd

            self.vae = comfy.sd.VAE(sd=payload, dtype=torch.float32)
            self.vae.first_stage_model.eval().requires_grad_(False)
            regularizer = getattr(self.vae.first_stage_model, "regularization", None)
            if regularizer is not None and getattr(regularizer, "sample", False):
                raise ValueError("VAE must encode posterior means")
        else:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(payload, torch_dtype=torch.float32).eval().requires_grad_(False).to("cuda")

    @torch.inference_mode()
    def encode(self, image) -> torch.Tensor:
        if self.kind == "comfy":
            pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
            latent = self.vae.encode(pixels).float().cpu()
        else:
            array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 127.5 - 1.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda")
            latent = self.vae.encode(tensor).latent_dist.mode().float().cpu()
        if not torch.isfinite(latent).all():
            raise ValueError("non-finite latent")
        return latent

    def free(self) -> None:
        self.vae = None
        free_gpu()


def make_guider(model, cond):
    import comfy.samplers

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(cond, [])
    guider.set_cfg(1.0)
    return guider


def processed_conds(guider, cond, latent: torch.Tensor, device, seed: int = 42):
    import comfy.sampler_helpers
    import comfy.samplers

    conds = {"positive": comfy.sampler_helpers.convert_cond(cond)}
    return comfy.samplers.process_conds(guider.inner_model, torch.zeros_like(latent), conds, device, latent, None, seed)
