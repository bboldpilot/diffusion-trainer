"""F5 (SN56_V5_PACK): throughput for the v5 lane (transformers-5 side venv).

The lane used to run with sdpa, no packing, per-device batch <= 8 and every
sample padded to max_length. This module picks one of two modes per task:

  v5_fa2  flash_attention_2 + padding-free packing. Packed rows carry no
          attention mask; position_ids restart at 0 for each example, which
          transformers 5 turns into FA2 varlen cu_seqlens (no cross-example
          attention). Short-conv layers (LFM2) get a seq_idx derived from the
          same position_ids so their causal conv is reset at every example
          boundary as well. Normal batch buckets.
  v5_dyn  sdpa, no packing, length-grouped batches padded only to the longest
          sample in the batch (not max_length), normal batch buckets.

auto picks v5_fa2 only when (a) the model type is one whose every sequence
mixer is isolated under padding-free packing (PACK_SAFE_MODEL_TYPES) and (b) a
real flash-attn varlen forward/backward succeeds on this GPU in a throwaway
subprocess (so a broken wheel can never take the training run down); v5_dyn
otherwise. SN56_V5_PACK=1 forces v5_fa2, SN56_V5_PACK=0 restores the old lane
config (instruct_config keeps that block verbatim).
"""

import json
import os
import subprocess
import sys

# Model types whose sequence mixers are all isolated under padding-free packing:
# attention layers through transformers' attention interface (position_ids ->
# cu_seqlens / block mask) and LFM2 short-conv layers through seq_idx (native in
# the causal_conv1d fast path, patched below for the torch path). Hybrid
# linear-attention/SSM types (qwen3_5 GatedDeltaNet, ...) are NOT listed: their
# recurrent state would leak across examples.
PACK_SAFE_MODEL_TYPES = {"lfm2", "lfm2_moe"}
SEQ_IDX_MODEL_TYPES = {"lfm2", "lfm2_moe"}

MODES = ("v5_fa2", "v5_dyn")


def model_type_of(model_path: str):
    try:
        with open(os.path.join(model_path, "config.json"), "r") as f:
            cfg = json.load(f)
    except Exception:
        return None
    model_type = cfg.get("model_type")
    if not model_type and isinstance(cfg.get("text_config"), dict):
        model_type = cfg["text_config"].get("model_type")
    return model_type


_FA2_PROBE = r"""
import json, sys
res = {"ok": False}
try:
    import torch
    res["torch"] = torch.__version__
    if not torch.cuda.is_available():
        raise RuntimeError("cuda unavailable")
    import transformers
    from transformers.utils import is_flash_attn_2_available
    res["transformers"] = transformers.__version__
    if not is_flash_attn_2_available():
        raise RuntimeError("transformers: flash_attn_2 not available")
    import flash_attn
    from flash_attn import flash_attn_varlen_func
    res["flash_attn"] = getattr(flash_attn, "__version__", "?")
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    cfg = AutoConfig.from_pretrained(sys.argv[1])
    cls = getattr(transformers, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[cfg.model_type])
    if not getattr(cls, "_supports_flash_attn", False):
        raise RuntimeError(cls.__name__ + " does not support flash_attention_2")
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    res["head_dim"] = hd
    torch.manual_seed(0)
    mk = lambda: torch.randn(10, 2, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q, k, v = mk(), mk(), mk()
    cu = torch.tensor([0, 4, 10], dtype=torch.int32, device="cuda")
    out = flash_attn_varlen_func(q, k, v, cu, cu, 6, 6, causal=True)
    out.float().square().sum().backward()
    # Isolation: the second example alone must give the same output.
    cu2 = torch.tensor([0, 6], dtype=torch.int32, device="cuda")
    alone = flash_attn_varlen_func(q[4:], k[4:], v[4:], cu2, cu2, 6, 6, causal=True)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all() or q.grad is None or not torch.isfinite(q.grad).all():
        raise RuntimeError("non-finite varlen output/grad")
    err = (out[4:].float() - alone.float()).abs().max().item()
    res["isolation_err"] = err
    if err > 1e-2:
        raise RuntimeError("varlen output differs from per-example output")
    res["ok"] = True
except Exception as e:
    res["error"] = type(e).__name__ + ": " + str(e)[:300]
print("SN56_FA2_PROBE " + json.dumps(res), flush=True)
"""


def fa2_probe(model_path: str, timeout: float = 300.0):
    """(ok, detail) from a flash-attn varlen fwd+bwd in a child interpreter (the
    lane's python), so the CUDA context and any crash stay out of this process."""
    env = dict(os.environ)
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _FA2_PROBE, model_path],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except Exception as e:  # timeout, spawn failure
        return False, f"probe did not run ({type(e).__name__}: {e})"
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SN56_FA2_PROBE "):
            try:
                res = json.loads(line[len("SN56_FA2_PROBE "):])
            except ValueError:
                break
            detail = ", ".join(f"{k}={res[k]}" for k in
                               ("flash_attn", "torch", "transformers", "head_dim", "isolation_err", "error")
                               if k in res)
            return bool(res.get("ok")), detail
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
    return False, f"probe rc={proc.returncode}: {tail[0][:200]}"


def choose_v5_mode(model_path: str, setting: str):
    """(mode, reason) for SN56_V5_PACK in {auto, 1}. setting 0 never gets here."""
    model_type = model_type_of(model_path)
    if setting == "1":
        return "v5_fa2", f"forced (SN56_V5_PACK=1), model_type={model_type}"
    if model_type not in PACK_SAFE_MODEL_TYPES:
        return "v5_dyn", f"model_type={model_type} not verified for padding-free packing"
    ok, detail = fa2_probe(model_path)
    if ok:
        return "v5_fa2", f"flash-attn varlen probe ok ({detail})"
    return "v5_dyn", f"flash-attn unusable ({detail})"


# ── train-side pieces (imported by train_instruct only in a v5 mode) ─────────

class NoPadDataset:
    """MyDataset without the pad-to-max_length: items are truncated (prompt side,
    like MyDataset) but left at their own length; V5Collator pads per batch.
    Exposes the same attributes (eval_dataset, tokenizer, max_length) that the
    dedup / PLW / Gaussian-subsample / PackedDataset code reads."""

    def __init__(self, tokenizer, data_path: str, max_length: int) -> None:
        with open(data_path, "r") as f:
            self.eval_dataset = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.eval_dataset)

    def __getitem__(self, idx):
        import torch
        from utility import smart_truncate

        dp = self.eval_dataset[idx]
        if self.max_length and self.max_length > 0 and len(dp["input_ids"]) > self.max_length:
            dp = smart_truncate(dict(dp), self.max_length)
        return {k: torch.tensor(dp[k]) for k in ("input_ids", "attention_mask", "labels")}


class V5Collator:
    """Batches for both v5 modes.

    Padding-free packed items (position_ids, no attention_mask) are flattened
    into ONE row (1, total_tokens): transformers 5 detects packing from
    non-monotonic position_ids only at batch size 1. seq_idx (int32, one id per
    example, derived from position_ids) is added for short-conv models.
    Unpacked items are right-padded to the longest item in the batch."""

    def __init__(self, pad_token_id: int, pass_seq_idx: bool = False):
        self.pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
        self.pass_seq_idx = pass_seq_idx

    def __call__(self, features):
        if "position_ids" in features[0] and "attention_mask" not in features[0]:
            return self._flatten(features)
        return self._pad(features)

    @staticmethod
    def _t(x):
        import torch
        return x.long() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

    def _flatten(self, features):
        import torch

        input_ids = torch.cat([self._t(f["input_ids"]) for f in features]).unsqueeze(0)
        labels = torch.cat([self._t(f["labels"]) for f in features]).unsqueeze(0)
        position_ids = torch.cat([self._t(f["position_ids"]) for f in features]).unsqueeze(0)
        # use_cache=False: with a KV cache transformers skips packed-sequence
        # detection (no block mask) and the conv takes its cache path.
        batch = {"input_ids": input_ids, "labels": labels, "position_ids": position_ids,
                 "use_cache": False}
        if self.pass_seq_idx:
            batch["seq_idx"] = seq_idx_from_position_ids(position_ids)
        return batch

    def _pad(self, features):
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            ids = self._t(f["input_ids"])
            n = ids.numel()
            pad = longest - n
            mask = self._t(f["attention_mask"]) if "attention_mask" in f else torch.ones(n, dtype=torch.long)
            out["input_ids"].append(torch.cat([ids, ids.new_full((pad,), self.pad_token_id)]))
            out["labels"].append(torch.cat([self._t(f["labels"]), ids.new_full((pad,), -100)]))
            out["attention_mask"].append(torch.cat([mask, ids.new_zeros((pad,))]))
        return {k: torch.stack(v) for k, v in out.items()}


def seq_idx_from_position_ids(position_ids):
    """(B, T) int32 example ids: a new example starts wherever position_ids do not
    continue +1 (same rule as transformers' find_packed_sequence_indices)."""
    import torch

    first = position_ids[:, :1] - 1
    diff = torch.diff(position_ids, prepend=first, dim=-1)
    return (diff != 1).cumsum(-1).to(torch.int32)


def _masked_causal_depthwise_conv(x, conv, seq_idx):
    """conv(x) of a causal depthwise Conv1d (padding=K-1, taking the first L
    outputs) with every term that reads across an example boundary removed:
    out[t] = sum_j w[K-1-j] * x[t-j] * [seq_idx[t-j] == seq_idx[t]]."""
    import torch.nn.functional as F

    L = x.shape[-1]
    out = conv(x)[..., :L]
    w = conv.weight[:, 0, :]  # (C, K)
    K = w.shape[-1]
    for j in range(1, min(K, L)):
        cross = (seq_idx[:, j:] != seq_idx[:, :-j]).unsqueeze(1).to(x.dtype)  # (B, 1, L-j)
        leak = x[..., :-j] * w[:, K - 1 - j].unsqueeze(-1) * cross
        out = out - F.pad(leak, (j, 0))
    return out


def patch_short_conv_seq_idx():
    """Make LFM2 short-conv layers honour seq_idx on the torch (slow) path.

    transformers 5 resets the conv at packed-example boundaries only inside the
    causal_conv1d CUDA kernel; without that package the torch path ignores
    seq_idx and the first K-1 tokens of each example read the tail of the
    previous one. Only the no-cache training/eval forward with a seq_idx is
    rerouted; everything else calls the original forward."""
    patched = []
    targets = (
        ("transformers.models.lfm2.modeling_lfm2", "Lfm2ShortConv"),
        ("transformers.models.lfm2_moe.modeling_lfm2_moe", "Lfm2MoeShortConv"),
    )
    import importlib

    for mod_name, cls_name in targets:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        if getattr(cls, "_sn56_seq_idx_patch", False):
            patched.append(cls_name)
            continue
        orig_forward = cls.forward

        def forward(self, hidden_states, past_key_values=None, attention_mask=None, seq_idx=None,
                    _orig=orig_forward, _mod=mod):
            fast = (
                getattr(_mod, "is_fast_path_available", False)
                and "cuda" in hidden_states.device.type
                and not getattr(_mod, "is_torchdynamo_compiling", lambda: False)()
            )
            if seq_idx is None or fast or past_key_values is not None:
                return _orig(self, hidden_states, past_key_values=past_key_values,
                             attention_mask=attention_mask, seq_idx=seq_idx)
            x = _mod.apply_mask_to_padding_states(hidden_states, attention_mask)
            BCx = self.in_proj(x).transpose(-1, -2)
            B, C, x = BCx.chunk(3, dim=-2)
            Bx = B * x
            conv_out = _masked_causal_depthwise_conv(Bx, self.conv, seq_idx)
            y = C * conv_out
            y = y.transpose(-1, -2).contiguous()
            return self.out_proj(y)

        cls.forward = forward
        cls._sn56_seq_idx_patch = True
        patched.append(cls_name)
    return patched
