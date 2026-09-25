"""Model families: how each one is conditioned by the evaluator and where its weights sit in the
trainer container.

The evaluator loads every family through ComfyUI (UNETLoader / CLIPLoader / VAE) with a fixed CLIP type,
an optional guidance value (flux) and an optional AuraFlow shift (qwen-image), and refuses anything that is
not unit-noise rectified flow. The trainer container instead receives the task's HF snapshot under
/cache/models and the family text encoder under /cache/hf_cache, with no network; the two VAE files the
evaluator uses for ideogram4 / krea2 / qwen-image are baked into the image at build time.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class Family:
    name: str
    clip_type: str            # comfy.sd.CLIPType member name
    text_enc_dtype: str       # comfy text encoder compute dtype flag
    guidance: float | None = None
    shift: float | None = None
    baked_vae: str | None = None
    hf_text_encoder: str | None = None


FAMILIES = {
    "ideogram4": Family("ideogram4", "IDEOGRAM4", "fp16", baked_vae="flux2-vae.safetensors",
                        hf_text_encoder="Qwen/Qwen3-VL-8B-Instruct"),
    "krea2": Family("krea2", "KREA2", "fp16", baked_vae="qwen_image_vae.safetensors",
                    hf_text_encoder="Qwen/Qwen3-VL-4B-Instruct"),
    "qwen-image": Family("qwen-image", "QWEN_IMAGE", "bf16", shift=3.0, baked_vae="qwen_image_vae.safetensors"),
    # flux: the evaluator encodes with the Flux.1 AE file through ComfyUI (Comfy-Org/z_image_turbo
    # split_files/vae/ae.safetensors == rayonlabs/FLUX.1-dev ae.safetensors, identical LFS sha256, fp32).
    # z-image: NOT that file. The evaluator's z-image branch (diffusion.load_base, fb8cff26 .. main dd8e898b) encodes with
    # the task repo's own diffusers vae/ (AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=float32),
    # posterior mode) — see Assets.vae(); the baked ae.safetensors is only a last resort when vae/ is absent.
    "z-image": Family("z-image", "LUMINA2", "fp16", baked_vae="ae.safetensors"),
    "flux": Family("flux", "FLUX", "fp16", guidance=3.5, baked_vae="ae.safetensors"),
}

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def normalize_family(model_type: str) -> str:
    name = model_type.lower().replace("_", "-")
    if name not in FAMILIES:
        raise ValueError(f"unsupported image model type {model_type!r}")
    return name


# ----------------------------------------------------------------------------------------- weights

def _shard_files(directory: str) -> list[str]:
    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index = os.path.join(directory, index_name)
        if os.path.exists(index):
            with open(index) as handle:
                weight_map = json.load(handle)["weight_map"]
            return [os.path.join(directory, name) for name in sorted(set(weight_map.values()))]
    return sorted(glob.glob(os.path.join(directory, "*.safetensors")))


def load_shards(files: list[str]) -> dict:
    state = {}
    for path in files:
        state.update(load_file(path, device="cpu"))
    return state


# The evaluator's base rule (validator/evaluation/image_artifacts.prepare_base, #1369 .. fb8cff26): the base is the
# repo-root `.safetensors` larger than 5 GiB, and there must be exactly one. The trainer container sees the same
# repo root, so the same threshold applies here. A repo with several such files (mikeyandfriends/PixelWave_FLUX.1-dev_03:
# bf16 22.2 GiB, fp8 11.1 GiB, nf4 5.7 GiB) is ambiguous to that code, which raises ValueError at fb8cff26 — yet the
# task (87f69ea0) was scored. Pinning the evaluator to the bf16 file reproduces both published losses (+0.041% offset,
# residual +-0.045%; results/image-0924-families/qwen-pw), so production loaded the bf16 file; largest-first picks it.
EVALUATOR_BASE_MIN_BYTES = 5 * 1024**3


def root_checkpoints(model_dir: str, min_bytes: int | None = None) -> list[str]:
    min_bytes = EVALUATOR_BASE_MIN_BYTES if min_bytes is None else min_bytes
    files = [f for f in glob.glob(os.path.join(model_dir, "*.safetensors")) if os.path.getsize(f) > min_bytes]
    return sorted(files, key=os.path.getsize, reverse=True)


def header_keys(path: str) -> set[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        return set(handle.keys())


def prefixed(state: dict, prefix: str) -> dict:
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def cast_bf16(state: dict) -> dict:
    return {key: (value.to(torch.bfloat16) if value.is_floating_point() and value.dtype != torch.bfloat16 else value)
            for key, value in state.items()}


def dequantize_rowwise_fp8(state: dict, roundtrip_fp8: bool = True) -> dict:
    """Diffusers-style fp8 shards carry a per-row scale next to each weight. Multiply it back; then,
    optionally, round through fp8 again so the trainer sees weights on the evaluator's fp8 grid rather
    than the exact bf16 values the evaluator never sees."""
    out = {}
    for key, value in state.items():
        if key.endswith(("weight_scale", "scale_weight")):
            continue
        scale = None
        if key.endswith(".weight"):
            scale = state.get(key + "_scale")
            if scale is None:
                scale = state.get(key.replace(".weight", ".scale_weight"))
        if scale is not None and value.dtype in FP8_DTYPES:
            weight = value.float() * scale.float().reshape(-1, *([1] * (value.dim() - 1)))
            if roundtrip_fp8:
                weight = weight.to(torch.float8_e4m3fn).float()
            out[key] = weight.to(torch.bfloat16)
        elif value.dtype in FP8_DTYPES:
            out[key] = value.float().to(torch.bfloat16)
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------------- evaluator's fp8 text encoder
# The evaluator conditions with Comfy-Org's fp8-scaled text encoders (ideogram4: qwen3vl_8b_fp8_scaled, krea2:
# qwen3vl_4b_fp8_scaled, qwen-image: qwen_2.5_vl_7b_fp8_scaled). The trainer container only holds the bf16 HF
# weights, but the fp8 file is a deterministic function of them: per-tensor scale = amax(|W|) * (1/416) in
# fp32 (the 2026 files) or amax / 416 (qwen-image's older format), round-to-nearest to e4m3fn. Re-deriving it
# here makes the trainer see the evaluator's conditioning bit for bit (verified offline against the published
# files: 1254/1254 and 1217/1217 tensors byte-identical). Any failure falls back to the bf16 weights.
# The legacy qwen-image file also carries a per-layer `scale_input` (358 tensors, every one exactly 1.0, read from
# the file on 25 Sep). ComfyUI's convert_old_quants drops a 1.0 input scale in both loaders (ours: boot.load_clip),
# so the conditioning is unchanged; it is emitted anyway so the rebuilt state is the evaluator's file key for key
# (qwen-image 3b6a0a8e text_encoder/ vs qwen_2.5_vl_7b_fp8_scaled: 1446/1446 byte-identical with it, 1088/1446 + 358
# missing without; results/image-0925-qwen-fp8te/te-check.json).
_TE_LM = re.compile(r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
_TE_VIS = re.compile(r"^visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate|up|down)_proj)|merger\.mlp\.\d+)\.weight$")
_TE_QUANT = json.dumps({"format": "float8_e4m3fn", "full_precision_matrix_mult": False}).encode()


def evaluator_fp8_text_encoder(state: dict, family: str) -> dict:
    legacy = family == "qwen-image"
    out = {key.replace("model.language_model.", "model.", 1): value for key, value in state.items()}
    converted = 0
    for key in list(out):
        if not (_TE_LM.match(key) or (legacy and _TE_VIS.match(key))):
            continue
        weight = out[key].float()
        amax = weight.abs().max()
        scale = amax / 416 if legacy else amax * (1.0 / 416)
        out[key] = (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        base = key[: -len(".weight")]
        if legacy:
            out[base + ".scale_weight"] = scale.reshape(()).float()
            out[base + ".scale_input"] = torch.ones((), dtype=torch.float32)
        else:
            out[base + ".weight_scale"] = scale.reshape(()).float()
            out[base + ".comfy_quant"] = torch.tensor(list(_TE_QUANT), dtype=torch.uint8)
        converted += 1
    if legacy:
        out["scaled_fp8"] = torch.empty(0, dtype=torch.float8_e4m3fn)
    if not converted:
        raise ValueError("no text-encoder linear weights matched; layout unexpected")
    return out


class Assets:
    """Resolve the family's diffusion weights, text-encoder state dicts and VAE from the container."""

    def __init__(self, family: str, model_dir: str, hf_cache: str, baked_dir: str, fp8_roundtrip: bool = True,
                 fp8_text_encoder: bool = False):
        self.family = normalize_family(family)
        self.spec = FAMILIES[self.family]
        self.model_dir, self.hf_cache, self.baked_dir = model_dir, hf_cache, baked_dir
        self.fp8_roundtrip = fp8_roundtrip
        self.fp8_text_encoder = fp8_text_encoder
        self.text_encoder_mode = "bf16"
        self._checkpoint = None
        self._root_keys = None

    # flux arrives in three layouts, told apart by the root file's header alone (nothing is loaded):
    #   1. a standalone single-file checkpoint bundling diffusion + text encoders + vae
    #      (`model.diffusion_model.*`, `text_encoders.clip_l|t5xxl.transformer.*`, `vae.*`);
    #   2. a diffusers snapshot whose root file is the diffusion model alone with bare BFL keys
    #      (rayonlabs/FLUX.1-dev flux1-dev.safetensors, next to the evaluator's own ae.safetensors);
    #   3. a diffusers snapshot whose root files are the diffusion model alone WITH the `model.diffusion_model.`
    #      prefix and nothing else (mikeyandfriends/PixelWave_FLUX.1-dev_03 bf16/fp8/nf4 files, 780 tensors each).
    # Each component therefore comes from the root file only when that file carries it; otherwise from the
    # snapshot's text_encoder/ + text_encoder_2/ (and the baked / diffusers VAE). Layout 3 used to be read as a
    # bundle, which handed the text-encoder loader two empty state dicts.
    def base_checkpoint(self) -> str:
        roots = root_checkpoints(self.model_dir)
        if not roots:
            raise FileNotFoundError(f"no root checkpoint (> 5 GiB .safetensors) under {self.model_dir}")
        return roots[0]

    def _root_has(self, prefix: str) -> bool:
        if self._root_keys is None:
            self._root_keys = header_keys(self.base_checkpoint())
        return any(key.startswith(prefix) for key in self._root_keys)

    def _baked_vae(self) -> str | None:
        name = self.spec.baked_vae
        path = os.path.join(self.baked_dir, name) if (name and self.baked_dir) else None
        return path if path and os.path.exists(path) else None

    def _bundle(self) -> dict:
        if self._checkpoint is None:
            self._checkpoint = load_shards([self.base_checkpoint()])
        return self._checkpoint

    def diffusion_state(self) -> dict:
        family = self.family
        if family == "ideogram4":
            shards = _shard_files(os.path.join(self.model_dir, "transformer"))
            if not shards:
                raise FileNotFoundError(f"no transformer shards under {self.model_dir}")
            return dequantize_rowwise_fp8(load_shards(shards), roundtrip_fp8=self.fp8_roundtrip)
        if family == "flux":
            if self._root_has("model.diffusion_model."):
                return cast_bf16(prefixed(self._bundle(), "model.diffusion_model."))
            return cast_bf16(self._bundle())
        roots = root_checkpoints(self.model_dir)
        if family == "krea2":
            preferred = [path for path in roots if "raw" in os.path.basename(path).lower()] or roots
            if not preferred:
                raise FileNotFoundError(f"no krea2 checkpoint under {self.model_dir}")
            return cast_bf16(load_shards(preferred[:1]))
        if not roots:
            raise FileNotFoundError(f"no root checkpoint under {self.model_dir}")
        state = load_shards(roots[:1])
        if any(value.dtype in FP8_DTYPES for value in state.values()):
            return state
        return cast_bf16(state)

    def _hf_snapshot(self, repo: str) -> str:
        flat = os.path.join(self.hf_cache, repo.replace("/", "--"))
        if os.path.isdir(flat) and _shard_files(flat):
            return flat
        snapshots = glob.glob(os.path.join(self.hf_cache, "models--" + repo.replace("/", "--"), "snapshots", "*"))
        if snapshots:
            return sorted(snapshots)[-1]
        raise FileNotFoundError(f"text encoder {repo} not under {self.hf_cache}")

    def text_encoder_states(self) -> list[dict]:
        family = self.family
        if family in ("ideogram4", "krea2", "qwen-image"):
            source = self._hf_snapshot(self.spec.hf_text_encoder) if family != "qwen-image" else os.path.join(self.model_dir, "text_encoder")
            state = load_shards(_shard_files(source))
            if self.fp8_text_encoder:
                try:
                    state = evaluator_fp8_text_encoder(state, family)
                    self.text_encoder_mode = "fp8-rederived"
                except Exception as error:  # the evaluator's own bf16-loading path is the champion's behaviour; never worse than that
                    self.text_encoder_mode = f"bf16 (fp8 rebuild failed: {type(error).__name__})"
            return [state]
        if family == "z-image":
            return [load_shards(_shard_files(os.path.join(self.model_dir, "text_encoder")))]
        if self._root_has("text_encoders.clip_l.transformer.") and self._root_has("text_encoders.t5xxl.transformer."):
            bundle = self._bundle()
            return [prefixed(bundle, "text_encoders.clip_l.transformer."), prefixed(bundle, "text_encoders.t5xxl.transformer.")]
        clip_l = load_shards(_shard_files(os.path.join(self.model_dir, "text_encoder")))
        t5 = load_shards(_shard_files(os.path.join(self.model_dir, "text_encoder_2")))
        # The evaluator loads t5xxl_fp16 and ComfyUI runs T5 in the file's dtype; the snapshot's T5 is bf16.
        t5 = {key: (value.to(torch.float16) if value.is_floating_point() else value) for key, value in t5.items()}
        return [clip_l, t5]

    def vae(self):
        """('comfy', state_dict) or ('diffusers', directory)."""
        family = self.family
        if family == "z-image":
            # The evaluator's own z-image path (and the champion's): diffusers AutoencoderKL from the snapshot's vae/,
            # float32, posterior mode (boot.LatentEncoder "diffusers"). With it the replica reproduces the evaluator per
            # sample (median |d| 0.010 %, every band within 0.02 %); the ComfyUI ae.safetensors latents (1.0-2.1 % rms apart,
            # almost all high-frequency) shift the replica by -0.1 % at sigma < 0.16 and +0.03 % at 0.4-0.9. Training on them
            # (24 Sep: zctl/zdiet2/zdiet3, b1z) cost ~0.006 % of test score, not the high-sigma gap (results/image-0925-zprobe).
            local = os.path.join(self.model_dir, "vae")
            if os.path.exists(os.path.join(local, "config.json")):
                return "diffusers", local
            baked = self._baked_vae()
            if baked:
                return "comfy", load_file(baked, device="cpu")
            return "diffusers", local
        if family == "flux":
            baked = self._baked_vae()
            if baked:
                return "comfy", load_file(baked, device="cpu")
            local = os.path.join(self.model_dir, "ae.safetensors")
            if os.path.exists(local):
                return "comfy", load_file(local, device="cpu")
            if self._root_has("vae."):
                return "comfy", prefixed(self._bundle(), "vae.")
            return "diffusers", os.path.join(self.model_dir, "vae")
        path = os.path.join(self.baked_dir, self.spec.baked_vae)
        if not os.path.exists(path):
            raise FileNotFoundError(f"baked VAE missing: {path}")
        return "comfy", load_file(path, device="cpu")

    def release(self) -> None:
        self._checkpoint = None
