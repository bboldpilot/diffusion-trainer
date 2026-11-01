"""F6 (SN56_PER_EXAMPLE_LOSS=1, default off): evaluator-native loss weighting.

The validator scores test_loss as the MEAN OVER EXAMPLES of each example's
per-token cross-entropy on its response; the stock causal-LM loss is a mean
over all response tokens in the batch, so long answers dominate. Here every
example's response tokens are weighted 1/len_i and the batch loss is the mean
of per-example means.

Example boundaries, whatever the batch layout:
  position_ids present  -> a new example wherever positions do not continue +1
                           (naive packing, v5 padding-free packing)
  attention_mask > 1    -> the seq-id mask of the FA packing path (0 = pad)
  otherwise             -> one example per row (unpacked batches)

With gradient accumulation the step loss is the mean of the micro-batch
example-means (model_accepts_loss_kwargs is turned off so the Trainer divides
by the accumulation count) — exact whenever grad_accum == 1.

Memory (P-G, 25 Sep): calling the model without labels made Liger's forward
materialise the full (B, T, V) logits (65.9 GiB at bs 140 on Qwen2) and the
OOM escaped to text_trainer's retry ladder, which halved the batch and
re-centred the LR four times. The loss no longer materialises the full
logits: the model runs with logits_to_keep=1 (only the last position's
logits, through the DDP wrapper as usual), a forward hook on the decoder
captures the final hidden states, and the head + cross-entropy run over the
response tokens only, in chunks under torch.utils.checkpoint so the backward
recomputes each chunk's logits instead of keeping them (the fused-linear-CE
memory profile, in stock torch). A model without that path (no decoder /
lm_head / logits_to_keep, or a head the check below cannot reproduce) runs in
"logits" mode, for which train_instruct caps the per-device batch so the full
logits fit. An OOM inside this loss, whichever path, switches the trainer to
the stock loss for the rest of the run instead of escaping: F6 never reaches
the retry ladder and never changes the LR.

SN56_F6_GLOBAL (default off): under DDP each rank used to normalise by its OWN
denominator (n_r examples, or sum n_i^(1-alpha)) before DDP averaged the
gradients, i.e. mean_r(N_r/D_r) instead of the single-GPU objective sum N / sum D.
With the flag the denominators are all-reduced first (from the labels only, before
any forward, on every rank and every training call whatever the F6 mode, so the
collective can never be skipped by one rank) and each rank returns R*N_r/D, whose
DDP average is exactly sum N / sum D. One process: unchanged. The accumulation
window is still normalised per micro-batch (grad_accum is 1 on these runs).

SN56_PER_EXAMPLE_ALPHA (K1-b, default 1.0): example i's tokens are weighted
n_i^-alpha and the batch loss is sum_i n_i^-alpha * S_i / sum_i n_i^(1-alpha)
(S_i = summed response CE). alpha=1 is the per-example mean above (that code
path is unchanged); alpha=0 is the stock token mean; 0.5 sits between.
"""

import gc
import inspect
import sys

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHUNK_ROWS = 2048                       # rows of logits alive at a time (bf16 + fp32 CE)
LOGITS_BUDGET_BYTES = 6 * 1024 ** 3     # "logits" mode: bf16 (B, T, V) + one fp32 CE chunk
HEAD_CHECK_REL_TOL = 2e-2               # hidden-path self-check on the last position


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _oom_types():
    out = []
    for t in (getattr(torch, "OutOfMemoryError", None),
              getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)):
        if isinstance(t, type) and t not in out:
            out.append(t)
    return tuple(out)


def is_oom(err) -> bool:
    if isinstance(err, _oom_types()):
        return True
    msg = str(err).lower()
    return "out of memory" in msg


def example_ids(labels, position_ids=None, attention_mask=None):
    """(B, T) int64 ids, unique across rows, one per example."""
    B, T = labels.shape
    if position_ids is not None:
        pos = position_ids
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        if pos.shape[0] != B:
            pos = pos.expand(B, -1)
        first = pos[:, :1] - 1
        seg = (torch.diff(pos, prepend=first, dim=-1) != 1).long().cumsum(-1)
    elif attention_mask is not None and attention_mask.dim() == 2 and int(attention_mask.max()) > 1:
        seg = attention_mask.long()
    else:
        seg = torch.zeros_like(labels, dtype=torch.long)
    row = torch.arange(B, device=labels.device).unsqueeze(1) * (T + 2)
    return seg.to(labels.device) + row


def _response_positions(labels, position_ids=None, attention_mask=None):
    """Flat indices of the response targets after the causal shift (logits[t]
    predicts labels[t+1]; a target belongs to the example of its own position),
    with their targets and example ids."""
    seg = example_ids(labels, position_ids, attention_mask)[:, 1:].reshape(-1)
    tgt = labels[:, 1:].reshape(-1)
    idx = (tgt != -100).nonzero(as_tuple=True)[0]
    return idx, tgt.index_select(0, idx), seg.index_select(0, idx)


def _per_example_from_tokens(ce, seg_sel, alpha: float = 1.0):
    """(sum over examples of the per-example mean token CE, number of examples);
    for alpha != 1, (sum_i n_i^-alpha * S_i, sum_i n_i^(1-alpha)) instead."""
    uniq, inv = torch.unique(seg_sel, return_inverse=True)
    sums = torch.zeros(uniq.numel(), dtype=ce.dtype, device=ce.device).index_add_(0, inv, ce)
    counts = torch.zeros(uniq.numel(), dtype=ce.dtype, device=ce.device).index_add_(
        0, inv, torch.ones_like(ce))
    if alpha == 1.0:
        return (sums / counts).sum(), int(uniq.numel())
    w = counts.detach().double().pow(-alpha)
    return (sums * w.to(sums.dtype)).sum(), float((counts.detach().double() * w).sum())


def local_denominator(labels, position_ids=None, attention_mask=None, alpha: float = 1.0):
    """The batch's F6 denominator from the labels alone (float64 scalar on labels' device):
    number of examples with response tokens (alpha=1) or sum_i n_i^(1-alpha)."""
    idx, _tgt, seg = _response_positions(labels, position_ids, attention_mask)
    if idx.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=labels.device)
    uniq, inv = torch.unique(seg, return_inverse=True)
    if alpha == 1.0:
        return torch.tensor(float(uniq.numel()), dtype=torch.float64, device=labels.device)
    counts = torch.zeros(uniq.numel(), dtype=torch.float64, device=labels.device).index_add_(
        0, inv, torch.ones(inv.numel(), dtype=torch.float64, device=labels.device))
    return (counts * counts.pow(-alpha)).sum()


def global_denominator(labels, position_ids=None, attention_mask=None, alpha: float = 1.0):
    """(sum of local_denominator over the DDP ranks, world size). Every rank must call it."""
    import torch.distributed as dist
    d = local_denominator(labels, position_ids, attention_mask, alpha)
    world = dist.get_world_size()
    dist.all_reduce(d, op=dist.ReduceOp.SUM)
    return float(d), world


def per_example_ce(logits, labels, position_ids=None, attention_mask=None, chunk_tokens: int = 8192,
                   alpha: float = 1.0):
    """(sum over examples of per-example mean token CE, number of examples)
    from materialised logits. CE in float32, response tokens only, in chunks
    to bound the float32 copy."""
    idx, tgt, seg = _response_positions(labels, position_ids, attention_mask)
    if idx.numel() == 0:
        return logits.sum() * 0.0, 0
    lg = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    ce_parts = []
    for start in range(0, idx.numel(), chunk_tokens):
        sl = idx[start:start + chunk_tokens]
        ce_parts.append(F.cross_entropy(lg.index_select(0, sl).float(),
                                        tgt[start:start + chunk_tokens], reduction="none"))
    return _per_example_from_tokens(torch.cat(ce_parts), seg, alpha)


def _chunk_ce(h, weight, bias, tgt, softcap):
    logits = F.linear(h.to(weight.dtype), weight, bias).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits, tgt, reduction="none")


def linear_ce_chunked(hidden_rows, weight, bias, targets, chunk_rows: int = CHUNK_ROWS, softcap=None):
    """Per-token CE of lm_head(hidden_rows) against targets without holding
    every logit: a chunk's logits exist only inside torch.utils.checkpoint
    (recomputed in the backward), so memory is one (chunk_rows, V) block at a
    time, whatever the batch."""
    n = hidden_rows.shape[0]
    if n == 0:
        return hidden_rows.new_zeros(0, dtype=torch.float32)
    use_ckpt = torch.is_grad_enabled() and (hidden_rows.requires_grad or weight.requires_grad)
    parts = []
    for s in range(0, n, chunk_rows):
        h, t = hidden_rows[s:s + chunk_rows], targets[s:s + chunk_rows]
        if use_ckpt:
            parts.append(checkpoint(_chunk_ce, h, weight, bias, t, softcap, use_reentrant=False))
        else:
            parts.append(_chunk_ce(h, weight, bias, t, softcap))
    return torch.cat(parts)


def per_example_ce_from_hidden(hidden, weight, bias, labels, position_ids=None, attention_mask=None,
                               chunk_rows: int = CHUNK_ROWS, softcap=None, alpha: float = 1.0):
    """per_example_ce without the (B, T, V) logits: the head is applied to the
    response positions only, chunked and checkpointed."""
    idx, tgt, seg = _response_positions(labels, position_ids, attention_mask)
    if idx.numel() == 0:
        return hidden.sum() * 0.0, 0
    rows = hidden[:, :-1, :].reshape(-1, hidden.shape[-1]).index_select(0, idx)
    ce = linear_ce_chunked(rows, weight, bias, tgt, chunk_rows, softcap)
    return _per_example_from_tokens(ce, seg, alpha)


def cap_batch_for_logits(per_device_bs: int, grad_accum: int, max_length: int, vocab: int,
                         budget_bytes: int = LOGITS_BUDGET_BYTES):
    """"logits" mode: the largest per-device batch whose bf16 (B, T, V) logits
    plus one fp32 CE chunk fit budget_bytes, with the accumulation raised so
    the effective batch is kept."""
    if not vocab or not max_length or per_device_bs <= 1:
        return per_device_bs, grad_accum
    per_row = 2.0 * max_length * vocab
    fp32_chunk = 4.0 * min(8192, max_length) * vocab
    max_rows = max(1, int((budget_bytes - fp32_chunk) // per_row))
    if max_rows >= per_device_bs:
        return per_device_bs, grad_accum
    eff = per_device_bs * max(1, grad_accum)
    return max_rows, max(1, -(-eff // max_rows))


def _ddp_world() -> int:
    try:
        import torch.distributed as dist
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    except Exception:
        return 1


def make_per_example_trainer(base_cls, alpha: float = 1.0, global_norm: bool = False):
    """Subclass of the given Trainer whose loss is the per-example mean
    (alpha=1) or the n_i^-alpha weighted mean (module doc); global_norm =
    SN56_F6_GLOBAL (denominator summed over DDP ranks)."""
    _alpha = float(alpha)
    _global = bool(global_norm)

    class PerExampleLossTrainer(base_cls):
        f6_alpha = _alpha
        f6_global = _global

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # num_items_in_batch (a token count) does not apply to this loss; let
            # training_step divide by the accumulation count instead.
            self.model_accepts_loss_kwargs = False
            self._f6_mode = "hidden"      # hidden | logits | off
            self._f6_decoder = None
            self._f6_head = None
            self._f6_kw = None
            self._f6_hook = None
            self._f6_captured = None
            self._f6_checked = False
            self._f6_softcap = None
            self._f6_setup(self.model)

        # stderr: visible under the competition quiet mode, like tracebacks
        @staticmethod
        def _f6_log(msg):
            print(f"[sn56][perda] {msg}", file=sys.stderr, flush=True)

        def _f6_setup(self, model):
            m = _unwrap(model)
            decoder = None
            try:
                decoder = m.get_decoder() if hasattr(m, "get_decoder") else None
            except Exception:
                decoder = None
            if decoder is None:
                prefix = getattr(m, "base_model_prefix", "") or ""
                decoder = getattr(m, prefix, None) if prefix else None
            head = None
            try:
                head = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
            except Exception:
                head = None
            if head is None:
                head = getattr(m, "lm_head", None)
            try:
                params = inspect.signature(m.forward).parameters
            except (TypeError, ValueError):
                params = {}
            kw = ("logits_to_keep" if "logits_to_keep" in params
                  else "num_logits_to_keep" if "num_logits_to_keep" in params else None)
            ok = (isinstance(decoder, torch.nn.Module) and decoder is not m
                  and head is not None and getattr(head, "weight", None) is not None and kw is not None)
            if not ok:
                self._f6_mode = "logits"
                self._f6_log("sem caminho oculto (decoder/lm_head/logits_to_keep): modo logits, lote limitado")
                return
            self._f6_decoder, self._f6_head, self._f6_kw = decoder, head, kw
            cfg = getattr(m, "config", None)
            cap = getattr(cfg, "final_logit_softcapping", None) if cfg is not None else None
            self._f6_softcap = float(cap) if cap else None
            self._f6_hook = decoder.register_forward_hook(self._f6_capture)

        def _f6_capture(self, module, args, output):
            if isinstance(output, (tuple, list)):
                out = output[0] if output else None
            else:
                out = getattr(output, "last_hidden_state", None)
                if out is None:
                    try:
                        out = output[0]
                    except Exception:
                        out = None
            self._f6_captured = out if isinstance(out, torch.Tensor) else None

        def _f6_disable(self, why):
            self._f6_mode = "off"
            if self._f6_hook is not None:
                try:
                    self._f6_hook.remove()
                except Exception:
                    pass
                self._f6_hook = None
            self._f6_captured = None
            self._f6_log(f"{why}: F6 desligado, perda padrão daqui em diante (LR inalterado)")

        def _f6_head_check(self, outputs, hidden):
            """Once: the model's own last-position logits against the head on
            the captured hidden state. A mismatch means the head path is not
            lm_head(decoder(...)) and the loss must use the full logits."""
            head = self._f6_head
            logits1 = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            with torch.no_grad():
                ref = logits1[:, -1, :].float()
                mine = F.linear(hidden[:, -1, :].to(head.weight.dtype), head.weight,
                                getattr(head, "bias", None)).float()
                if self._f6_softcap:
                    mine = torch.tanh(mine / self._f6_softcap) * self._f6_softcap
                rel = (ref - mine).norm() / ref.norm().clamp_min(1e-6)
            return bool(torch.isfinite(rel)) and rel.item() <= HEAD_CHECK_REL_TOL, float(rel)

        def _f6_loss(self, model, inputs, return_outputs, gden=None):
            labels = inputs["labels"]
            fwd = {k: v for k, v in inputs.items() if k != "labels"}
            if self._f6_mode == "hidden":
                fwd[self._f6_kw] = 1
                self._f6_captured = None
                outputs = model(**fwd)
                hidden, self._f6_captured = self._f6_captured, None
                if hidden is None or hidden.dim() != 3 or tuple(hidden.shape[:2]) != tuple(labels.shape):
                    self._f6_mode = "logits"
                    self._f6_log("estado oculto não capturado: modo logits")
                    return self._f6_loss(model, inputs, return_outputs, gden)
                if not self._f6_checked:
                    self._f6_checked = True
                    ok, rel = self._f6_head_check(outputs, hidden)
                    if not ok:
                        self._f6_mode = "logits"
                        self._f6_log(f"cabeça não padrão (erro rel {rel:.3f}): modo logits")
                        return self._f6_loss(model, inputs, return_outputs, gden)
                head = self._f6_head
                total, n = per_example_ce_from_hidden(
                    hidden, head.weight, getattr(head, "bias", None), labels,
                    inputs.get("position_ids"), inputs.get("attention_mask"), softcap=self._f6_softcap,
                    alpha=self.f6_alpha)
            else:
                outputs = model(**fwd)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                total, n = per_example_ce(logits, labels, inputs.get("position_ids"),
                                          inputs.get("attention_mask"), alpha=self.f6_alpha)
            if gden is not None:
                # SN56_F6_GLOBAL: R * N_r / sum_s D_s, so DDP's average is sum N / sum D
                d_glob, world = gden
                loss = total * (world / d_glob) if d_glob > 0 else total * 0.0
            else:
                loss = total / max(1, n) if self.f6_alpha == 1.0 else total / (n if n > 0 else 1.0)
            return (loss, outputs) if return_outputs else loss

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            gden = None
            if (self.f6_global and "labels" in inputs and getattr(model, "training", False)
                    and _ddp_world() > 1):
                # collective first, from the labels only: every rank runs it on every
                # training call, whatever its F6 mode (an "off" rank still joins)
                gden = global_denominator(inputs["labels"], inputs.get("position_ids"),
                                          inputs.get("attention_mask"), self.f6_alpha)
            if "labels" not in inputs or self._f6_mode == "off":
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            try:
                return self._f6_loss(model, inputs, return_outputs, gden)
            except Exception as e:
                if not is_oom(e):
                    raise
                why = f"OOM dentro de F6 ({str(e)[:80]})"
            # the exception and its frames are released here; free, then the
            # stock loss for this and every later batch
            self._f6_captured = None
            self._f6_disable(why)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    PerExampleLossTrainer.__name__ = f"PerExampleLoss{base_cls.__name__}"
    return PerExampleLossTrainer
