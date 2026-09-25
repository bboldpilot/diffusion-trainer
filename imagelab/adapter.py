"""Trainable low-rank deltas injected through ComfyUI weight wrappers, exported as ComfyUI-native LoRA.

A wrapper is a callable ComfyUI applies to a weight inside its own cast path at forward time, so the
delta takes part in autograd while the base weights stay frozen and untouched. The exported file uses the
key layout ComfyUI's LoRA loader reads for any model (`diffusion_model.<module>.lora_up.weight`,
`.lora_down.weight`, `.alpha`), which is exactly what the evaluator applies with LoraLoader at strength 1.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Callable

import torch
from safetensors.torch import save_file


def linear_targets(diffusion_model: torch.nn.Module, include: str | None = None, exclude: str | None = None) -> list[tuple[str, tuple[int, int]]]:
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    out = []
    for name, module in diffusion_model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(module, torch.nn.Linear) or weight is None or weight.dim() != 2:
            continue
        if inc is not None and not inc.search(name):
            continue
        if exc is not None and exc.search(name):
            continue
        out.append((name, (int(weight.shape[0]), int(weight.shape[1]))))
    return out


class Slot:
    """One Linear layer's delta: up (out x r, zero-initialised) @ down (r x in) * scale."""

    def __init__(self, name: str, shape: tuple[int, int], rank: int, scale: float, device, generator: torch.Generator):
        self.name = name
        self.out_features, self.in_features = shape
        self.rank, self.scale = rank, scale
        self.down = torch.nn.Parameter((torch.randn(rank, self.in_features, generator=generator) / math.sqrt(self.in_features)).to(device))
        self.up = torch.nn.Parameter(torch.zeros(self.out_features, rank, device=device))
        self.enabled = True
        self.quantizer: Callable | None = None      # optional hook: (patched_float_weight, original_weight) -> tensor
        self._override: tuple[torch.Tensor, torch.Tensor, float] | None = None

    # ComfyUI weight-wrapper protocol
    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return weight
        up, down, scale = (self._override if self._override is not None else (self.up, self.down, self.scale))
        delta = (up.float() @ down.float()) * scale
        patched = weight.float() + delta.reshape(weight.shape)
        if self.quantizer is not None:
            patched = self.quantizer(patched, weight)
        return patched.to(weight.dtype)

    def to(self, device):
        return self


class AdapterBank:
    def __init__(self, targets: list[tuple[str, tuple[int, int]]], rank: int, alpha: float, device, seed: int):
        if not targets:
            raise RuntimeError("no LoRA targets matched")
        self.rank, self.alpha = rank, float(alpha)
        self.scale = self.alpha / rank
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.slots = [Slot(name, shape, rank, self.scale, device, generator) for name, shape in targets]
        self.by_name = {slot.name: slot for slot in self.slots}

    def reinit(self, seed: int) -> None:
        """Fresh factors in place (wrappers stay attached): down ~ N(0, 1/in), up = 0."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for slot in self.slots:
                slot.down.copy_((torch.randn(slot.rank, slot.in_features, generator=generator) / math.sqrt(slot.in_features)).to(slot.down.device))
                slot.up.zero_()

    # ------------------------------------------------------------------ parameters
    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [p for slot in self.slots for p in (slot.up, slot.down)]

    @property
    def up_parameters(self) -> list[torch.nn.Parameter]:
        return [slot.up for slot in self.slots]

    @property
    def down_parameters(self) -> list[torch.nn.Parameter]:
        return [slot.down for slot in self.slots]

    def count(self) -> int:
        return sum(p.numel() for p in self.parameters)

    # ------------------------------------------------------------------ ComfyUI hookup
    def attach(self, patcher) -> None:
        for slot in self.slots:
            patcher.add_weight_wrapper(f"diffusion_model.{slot.name}.weight", slot)

    def set_enabled(self, flag: bool) -> None:
        for slot in self.slots:
            slot.enabled = flag

    def set_quantizer(self, quantizer: Callable | None) -> None:
        for slot in self.slots:
            slot.quantizer = quantizer

    # ------------------------------------------------------------------ snapshots
    def snapshot(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {slot.name: (slot.up.detach().clone(), slot.down.detach().clone()) for slot in self.slots}

    def identity_snapshot(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {slot.name: (torch.zeros_like(slot.up), slot.down.detach().clone()) for slot in self.slots}

    def load_snapshot(self, snap: dict) -> None:
        with torch.no_grad():
            for name, (up, down) in snap.items():
                slot = self.by_name[name]
                slot.up.copy_(up)
                slot.down.copy_(down)

    def snapshot_from_flat(self, tensors: list[torch.Tensor]) -> dict:
        snap, position = {}, 0
        for slot in self.slots:
            snap[slot.name] = (tensors[position], tensors[position + 1])
            position += 2
        return snap

    def override(self, snap: dict | None, scale: float | None = None) -> None:
        """Evaluate a candidate's tensors without copying them into the trainable parameters."""
        for slot in self.slots:
            if snap is None:
                slot._override = None
            else:
                up, down = snap[slot.name]
                slot._override = (up, down, self.scale if scale is None else scale)

    # ------------------------------------------------------------------ export
    def export(self, snap: dict, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """ComfyUI-native keys. The loader multiplies up @ down by alpha / rank_of_down."""
        out = {}
        for name, (up, down) in snap.items():
            key = f"diffusion_model.{name}"
            out[f"{key}.lora_up.weight"] = up.detach().to("cpu", dtype).contiguous()
            out[f"{key}.lora_down.weight"] = down.detach().to("cpu", dtype).contiguous()
            out[f"{key}.alpha"] = torch.tensor(float(self.scale * down.shape[0]))
        return out


def merge_snapshots(snaps: list[dict], weights: list[float] | None = None) -> dict:
    """Exact weighted average of the deltas by factor concatenation (rank grows with the count)."""
    count = len(snaps)
    weights = weights or [1.0 / count] * count
    merged = {}
    for name in snaps[0]:
        ups = [snap[name][0] * weight for snap, weight in zip(snaps, weights)]
        downs = [snap[name][1] for snap in snaps]
        merged[name] = (torch.cat(ups, 1), torch.cat(downs, 0))
    return merged


def save_adapter(path: str, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=metadata)
    os.replace(tmp, path)


def checkpoint_blocks(diffusion_model: torch.nn.Module, min_len: int = 4, stride: int = 1) -> int:
    """Wrap every block of the model's long ModuleLists in activation checkpointing (training only)."""
    import torch.utils.checkpoint as checkpoint

    wrapped = 0
    for _, module in diffusion_model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= min_len:
            for index, block in enumerate(module):
                if index % stride or isinstance(block, torch.nn.ModuleList) or getattr(block, "_imagelab_ckpt", False):
                    continue
                original = block.forward

                def forward(*args, _original=original, **kwargs):
                    if torch.is_grad_enabled():
                        return checkpoint.checkpoint(_original, *args, use_reentrant=False, **kwargs)
                    return _original(*args, **kwargs)

                block.forward = forward
                block._imagelab_ckpt = True
                wrapped += 1
    return wrapped


# ------------------------------------------------------------------------------------------------------------
# fp8-aware export (opt-in: `imagelab.train --export fp8aware`; the plain export above is unchanged).
#
# Why. The evaluator does not apply a LoRA to ideogram4's fp8 base in float. ComfyUI 694815f4 merges the delta
# in fp16 and writes the result into a FRESH per-tensor fp8 e4m3 grid with stochastic rounding:
#   model_patcher.py:856-875  temp = dequantize(W8) in fp16; weight += fp16(alpha * up @ down); set_weight(seed=crc32(key))
#   ops.py:1307-1309          requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed)
#   quant_ops.py:89-105       scale = amax(|w|) / 448; w *= fp16(1 / scale); comfy.float.stochastic_rounding(w, e4m3)
#   float.py:72-77            uint8 dither from a generator seeded with the key; comfy-kitchen floor(m + dither / 256)
# The new scale is the merged tensor's amax / 448, almost never a power of two, so every element of every patched
# layer (the untouched base included) lands on a misaligned grid and is re-rounded: zero-mean noise several times
# larger than a small trained delta (results/image-0924-margin/fp8-absorption.md). When the merged amax is exactly
# 1.75 * 2**e the scale is exactly 2**(e - 8): the base (stored at scale 1.0) already sits on the new grid, passes
# the rounding unchanged, and only the delta itself is dithered.
#
# What. Per layer the export picks a target T = 1.75 * 2**e and moves a few base elements so that the merged
# amax is exactly T: either T is the next such value above every possible merged magnitude and ONE element (the
# largest whose evaluator value is known exactly) moves up to it, or T is the value below and the few elements
# above it are clipped down to it; whichever moves the base less (sum of squared moves, at most `max_rows` rows).
# The moves are appended to the LoRA as rank-1 terms, two per touched row: an fp16-exact step and a term that
# cancels the trained delta at the moved elements. The trained factors are carried over unchanged (up is
# multiplied by alpha / rank and alpha set to the new rank, so the loader's scale is exactly 1). Inputs: the
# adapter and the public base weights, nothing else; training never sees this, and the trainer keeps the plain
# export next to it as last.plain.safetensors.
# ------------------------------------------------------------------------------------------------------------
E4M3_MAX = 448.0
_POW2_MANTISSA = 1.75                 # 448 = 1.75 * 2**8
_INTERVAL_EPS = 2.0 ** -20            # fp32 slack when bracketing the row-scaled quantiser's cells
_FP16_MARGIN = 1.0 + 2.0 ** -9        # the two fp16 roundings of the merge (delta cast, add) stay inside this


def _e4m3_from_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.uint8).view(torch.float8_e4m3fn).float()


def _rtn_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Round to nearest (ties to even) onto the unit-scale e4m3 grid, as torch's cast does."""
    return values.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def _fp16_exact(value: float) -> bool:
    return float(torch.tensor(value, dtype=torch.float64).to(torch.float16)) == value


@dataclass
class Fp8Plan:
    """What the export needs to know about one fp8 base layer, in evaluator-grid values (scale 1.0)."""
    shape: tuple[int, int]
    top_index: torch.Tensor     # flat indices whose |evaluator value| may exceed rest_bound (long, cpu, sorted)
    top_bound: torch.Tensor     # upper bound of |evaluator value| there (float32)
    top_value: torch.Tensor     # signed evaluator value there when certain, else the signed upper candidate
    top_certain: torch.Tensor   # bool
    rest_bound: float           # upper bound of |evaluator value| at every other index
    anchor: int                 # flat index of the largest-magnitude element whose evaluator value is certain
    anchor_value: float         # that value, signed
    mode: str                   # "exact" (the evaluator's own file) | "interval" (the trainer's row-scaled shard)
    certain_frac: float


class _PlanAccumulator:
    """Row-chunked reduction to a plan; the result does not depend on the chunking."""

    def __init__(self, shape, mode: str, top: int):
        self.shape, self.mode, self.top = (int(shape[0]), int(shape[1])), mode, top
        self.parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.anchor: tuple[float, int, float] | None = None     # (|value|, -flat index, value)
        self.certain = 0
        self.count = 0

    def add(self, offset: int, bound: torch.Tensor, certain: torch.Tensor, value: torch.Tensor) -> None:
        flat_bound, flat_certain, flat_value = bound.reshape(-1), certain.reshape(-1), value.reshape(-1)
        top, index = torch.topk(flat_bound, min(self.top + 1, flat_bound.numel()))
        self.parts.append((top.float().cpu(), index.cpu() + offset, flat_certain[index].cpu(), flat_value[index].float().cpu()))
        self.certain += int(flat_certain.sum())
        self.count += flat_certain.numel()
        if bool(flat_certain.any()):
            magnitude = torch.where(flat_certain, flat_value.abs(), torch.full_like(flat_bound, -1.0))
            best = float(magnitude.max())
            first = int(torch.nonzero(magnitude == best)[0])
            candidate = (best, -(offset + first), float(flat_value[first]))
            if self.anchor is None or candidate[:2] > self.anchor[:2]:
                self.anchor = candidate

    def finish(self) -> Fp8Plan | None:
        if self.anchor is None:
            return None
        bound, index, certain, value = (torch.cat(column) for column in zip(*self.parts))
        kth = float(torch.topk(bound, min(self.top + 1, bound.numel())).values[-1])
        chosen = bound > kth               # every element above the (top+1)-th bound was collected by its chunk
        order = torch.argsort(index[chosen])
        return Fp8Plan(self.shape, index[chosen][order], bound[chosen][order], value[chosen][order],
                       certain[chosen][order], kth, -self.anchor[1], self.anchor[2], self.mode,
                       self.certain / max(1, self.count))


def fp8_plan_exact(weight: torch.Tensor, top: int = 64, chunk_rows: int = 4096, device=None) -> Fp8Plan | None:
    """The evaluator's base file: e4m3 at per-tensor scale 1.0, so every evaluator value is known."""
    plan = _PlanAccumulator(weight.shape, "exact", top)
    for start in range(0, weight.shape[0], chunk_rows):
        values = weight[start:start + chunk_rows].to(device).float()
        magnitude = values.abs()
        plan.add(start * weight.shape[1], magnitude, torch.ones_like(magnitude, dtype=torch.bool), values)
    return plan.finish()


def fp8_plan_rowscaled(weight: torch.Tensor, row_scale: torch.Tensor, top: int = 64, chunk_rows: int = 4096,
                       device=None) -> Fp8Plan | None:
    """The trainer's shard: q (e4m3) times a per-row scale s. The evaluator's file holds the same weights on the
    unit-scale grid, so each evaluator value lies in RTN(rounding cell of q, times s); where both ends of that cell
    round to the same grid value the evaluator value is certain. Checked against the published evaluator file on
    30 ideogram4 layers (24 Sep): 100 % of evaluator values inside the bounds, 100 % of the certain ones (45-63 %
    of the elements) exact."""
    plan = _PlanAccumulator(weight.shape, "interval", top)
    scales = row_scale.float().reshape(-1)
    for start in range(0, weight.shape[0], chunk_rows):
        q = weight[start:start + chunk_rows].to(device)
        low_value, high_value, sign = _evaluator_bounds(q, scales[start:start + chunk_rows].to(q.device).reshape(-1, 1))
        plan.add(start * weight.shape[1], high_value, low_value == high_value, sign * high_value)
    return plan.finish()


def _evaluator_bounds(q: torch.Tensor, s: torch.Tensor):
    """Magnitude bounds (low, high) and sign of the evaluator's unit-grid value behind row-scaled e4m3 q * s."""
    raw = q.view(torch.uint8)
    bits = (raw & 0x7F).to(torch.int16)
    magnitude = _e4m3_from_bits(bits)
    below = _e4m3_from_bits((bits - 1).clamp(min=0))
    above = _e4m3_from_bits((bits + 1).clamp(max=0x7E))
    low = torch.where(bits > 0, (magnitude + below) * 0.5, torch.zeros_like(magnitude)) * s
    high = torch.where(bits < 0x7E, (magnitude + above) * 0.5, magnitude) * s
    sign = torch.where((raw & 0x80) != 0, -1.0, 1.0).to(magnitude.device)
    return _rtn_e4m3(low * (1.0 - _INTERVAL_EPS)), _rtn_e4m3(high * (1.0 + _INTERVAL_EPS)), sign


def fp8_plans_from_table(table: dict, names: list[str], shard_paths: list[str] | None = None) -> tuple[dict[str, Fp8Plan], dict]:
    """Exact plans from the evaluator-base table (harness/image-gate/fp8aware-table.py: per layer the K largest
    |values| of the evaluator's own fp8 file and a bound for the rest). With the trainer's shard every table value
    is checked against the bounds of its element; a layer that disagrees (the published file changed) is left out
    and reported, so the caller falls back to the shard's interval plan for it."""
    from safetensors import safe_open

    plans: dict[str, Fp8Plan] = {}
    rejected: dict[str, str] = {}
    handles = [safe_open(path, framework="pt", device="cpu") for path in (shard_paths or [])]
    layers = table.get("layers", {})
    for name in names:
        entry = layers.get(name)
        if not entry or not entry.get("index"):
            continue
        shape = (int(entry["shape"][0]), int(entry["shape"][1]))
        index = torch.tensor(entry["index"], dtype=torch.long)
        value = torch.tensor(entry["value"], dtype=torch.float32)
        handle = next((h for h in handles if f"{name}.weight" in h.keys()), None)
        if handle is not None:
            weight, scale = handle.get_slice(f"{name}.weight"), handle.get_slice(f"{name}.weight_scale")
            if tuple(weight.get_shape()) != shape:
                rejected[name] = f"shape {tuple(weight.get_shape())} != table {shape}"
                continue
            bad = 0
            for row in sorted(set((index // shape[1]).tolist())):
                in_row = (index // shape[1]) == row
                columns = index[in_row] % shape[1]
                low, high, sign = _evaluator_bounds(weight[row:row + 1].view(torch.uint8)[0, columns], scale[row:row + 1].float())
                expected = value[in_row]
                ok = (expected.abs() >= low) & (expected.abs() <= high) & ((expected == 0) | (torch.sign(expected) == sign) | (high == 0))
                bad += int((~ok).sum())
            if bad:
                rejected[name] = f"{bad} table value(s) outside the shard's bounds"
                continue
        order = torch.argsort(index)
        index, value = index[order], value[order]
        anchor = int(torch.nonzero(value.abs() == value.abs().max())[0])
        plans[name] = Fp8Plan(shape, index, value.abs(), value, torch.ones_like(value, dtype=torch.bool),
                              float(entry["rest_bound"]), int(index[anchor]), float(value[anchor]), "table", 1.0)
    return plans, rejected


def fp8_plans_from_files(paths: list[str], names: list[str], device=None, top: int = 64) -> dict[str, Fp8Plan]:
    """Plans for the named Linear layers (`<name>.weight`) from safetensors files holding fp8 weights. A scalar
    weight_scale of 1.0 (the evaluator's file) gives exact plans, a per-row weight_scale (the trainer's shard)
    interval plans; bf16 bases and other layouts get no plan and keep the plain export."""
    from safetensors import safe_open

    plans: dict[str, Fp8Plan] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for name in sorted(set(names) - set(plans)):
                if f"{name}.weight" not in keys:
                    continue
                weight = handle.get_tensor(f"{name}.weight")
                if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
                    continue
                scale_key = next((k for k in (f"{name}.weight_scale", f"{name}.scale_weight") if k in keys), None)
                scale = handle.get_tensor(scale_key) if scale_key else None
                plan = None
                if scale is None or (scale.numel() == 1 and float(scale) == 1.0):
                    plan = fp8_plan_exact(weight, top=top, device=device)
                elif scale.numel() == weight.shape[0]:
                    plan = fp8_plan_rowscaled(weight, scale, top=top, device=device)
                if plan is not None:
                    plans[name] = plan
    return plans


def _choose_moves(plan: Fp8Plan, flat_delta: torch.Tensor, max_rows: int) -> dict | None:
    """The cheapest target (next 1.75 * 2**e above the reach, or the one below it) and the base moves it needs."""
    columns = plan.shape[1]
    top_delta = flat_delta[plan.top_index.to(flat_delta.device)].abs().cpu()
    top_reach = plan.top_bound + top_delta                                   # largest possible merged magnitude
    untracked = plan.rest_bound + float(flat_delta.abs().max())
    anchor_delta = float(flat_delta[plan.anchor])
    reach = max([untracked, abs(plan.anchor_value) + abs(anchor_delta)] + top_reach.tolist()) * _FP16_MARGIN
    if not reach > 0:
        return None
    upper = _POW2_MANTISSA * 2.0 ** math.ceil(math.log2(reach / _POW2_MANTISSA))
    best = None
    for kind, target in (("up", upper), ("down", upper / 2)):
        if untracked * _FP16_MARGIN > target or not 2.0 ** -14 <= E4M3_MAX / target <= 65504.0:
            continue                                  # quant_ops.py:101 needs fp16(1 / scale) exact
        moves = {int(plan.top_index[k]): (float(plan.top_value[k]), bool(plan.top_certain[k]))
                 for k in torch.nonzero(top_reach * _FP16_MARGIN > target).flatten().tolist()}
        if not any(certain for _, certain in moves.values()):
            moves[plan.anchor] = (plan.anchor_value, True)       # one element must land exactly on the target
        steps = {index: (-target if value < 0 else target) - value for index, (value, _) in moves.items()}
        rows = sorted({index // columns for index in steps})
        if len(rows) > max_rows or not all(_fp16_exact(step) for step in steps.values()):
            continue
        cost = sum(step * step for step in steps.values())
        if best is None or (cost, len(rows)) < (best["cost"], len(best["rows"])):
            best = {"kind": kind, "target": target, "reach": reach, "steps": steps, "rows": rows, "cost": cost}
    return best


def fp8aware_export(tensors: dict[str, torch.Tensor], plans: dict[str, Fp8Plan], prefix: str = "diffusion_model.",
                    device=None, max_rows: int = 8) -> tuple[dict[str, torch.Tensor], dict]:
    """Pure function of (plain exported LoRA tensors, base plans): returns new tensors and a report. Layers
    without a plan, or where the construction cannot be made exact, pass through untouched."""
    out = dict(tensors)
    report: dict = {"modes": sorted({plan.mode for plan in plans.values()}), "applied": 0, "extra_rank_max": 0,
                    "skipped": {}, "layers": {}}
    for up_key in sorted(key for key in tensors if key.endswith(".lora_up.weight")):
        base = up_key[: -len(".lora_up.weight")]
        name = base[len(prefix):] if base.startswith(prefix) else base
        plan = plans.get(name)
        if plan is None:
            report["skipped"][base] = "no fp8 plan"
            continue
        down_key, alpha_key = f"{base}.lora_down.weight", f"{base}.alpha"
        up = tensors[up_key].to(device=device, dtype=torch.float32)
        down = tensors[down_key].to(device=up.device, dtype=torch.float32)
        rank = down.shape[0]
        # comfy/weight_adapter/lora.py:246-249: the loader scales up @ down by alpha / rank (1.0 without alpha)
        up = up * (float(tensors[alpha_key].item()) / rank if alpha_key in tensors else 1.0)
        delta = up @ down
        if tuple(delta.shape) != plan.shape:
            report["skipped"][base] = f"shape {tuple(delta.shape)} != base {plan.shape}"
            continue
        flat = delta.reshape(-1)
        choice = _choose_moves(plan, flat, max_rows)
        if choice is None:
            report["skipped"][base] = "no exact target within the row budget"
            continue
        columns = plan.shape[1]
        extra_up = torch.zeros(plan.shape[0], 2 * len(choice["rows"]), device=up.device)
        extra_down = torch.zeros(2 * len(choice["rows"]), columns, device=up.device)
        slot = {row: 2 * k for k, row in enumerate(choice["rows"])}
        for index, step in choice["steps"].items():
            row, column = divmod(index, columns)
            extra_up[row, slot[row]] = extra_up[row, slot[row] + 1] = 1.0
            extra_down[slot[row], column] = step                  # fp16-exact: fp16(W8) + fp16(step) lands on +-T
            extra_down[slot[row] + 1, column] = -flat[index]      # cancels the trained delta there (to ~1e-10)
        out[up_key] = torch.cat([up, extra_up], 1).cpu().contiguous()
        out[down_key] = torch.cat([down, extra_down], 0).cpu().contiguous()
        out[alpha_key] = torch.tensor(float(rank + extra_up.shape[1]))
        report["applied"] += 1
        report["extra_rank_max"] = max(report["extra_rank_max"], extra_up.shape[1])
        moved = sorted(choice["steps"].items())
        report["layers"][base] = {"mode": plan.mode, "kind": choice["kind"], "target": choice["target"],
                                  "reach": choice["reach"], "rows": len(choice["rows"]), "moves": len(moved),
                                  "move_abs_max": max(abs(step) for _, step in moved), "move_sq": choice["cost"],
                                  "moved": [[index // columns, index % columns, step] for index, step in moved]}
    moves = sorted(entry["move_abs_max"] for entry in report["layers"].values())
    report["move_abs_max_median"] = moves[len(moves) // 2] if moves else None
    report["move_abs_max_max"] = moves[-1] if moves else None
    return out, report
