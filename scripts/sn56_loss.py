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
"""

import torch
import torch.nn.functional as F


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


def per_example_ce(logits, labels, position_ids=None, attention_mask=None, chunk_tokens: int = 8192):
    """(sum over examples of per-example mean token CE, number of examples).

    Causal shift as in transformers: logits[t] predicts labels[t+1]; a target
    belongs to the example of its own position. CE in float32, computed only on
    response tokens, in chunks to bound the float32 copy."""
    seg = example_ids(labels, position_ids, attention_mask)[:, 1:].reshape(-1)
    tgt = labels[:, 1:].reshape(-1)
    lg = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    idx = (tgt != -100).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return logits.sum() * 0.0, 0
    ce_parts = []
    for start in range(0, idx.numel(), chunk_tokens):
        sl = idx[start:start + chunk_tokens]
        ce_parts.append(F.cross_entropy(lg.index_select(0, sl).float(), tgt.index_select(0, sl), reduction="none"))
    ce = torch.cat(ce_parts)
    uniq, inv = torch.unique(seg.index_select(0, idx), return_inverse=True)
    sums = torch.zeros(uniq.numel(), dtype=ce.dtype, device=ce.device).index_add_(0, inv, ce)
    counts = torch.zeros(uniq.numel(), dtype=ce.dtype, device=ce.device).index_add_(
        0, inv, torch.ones_like(ce))
    return (sums / counts).sum(), int(uniq.numel())


def make_per_example_trainer(base_cls):
    """Subclass of the given Trainer whose loss is the per-example mean."""

    class PerExampleLossTrainer(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # num_items_in_batch (a token count) does not apply to this loss; let
            # training_step divide by the accumulation count instead.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if "labels" not in inputs:
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            labels = inputs["labels"]
            fwd = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**fwd)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            total, n = per_example_ce(logits, labels, inputs.get("position_ids"),
                                      inputs.get("attention_mask"))
            loss = total / max(1, n)
            return (loss, outputs) if return_outputs else loss

    PerExampleLossTrainer.__name__ = f"PerExampleLoss{base_cls.__name__}"
    return PerExampleLossTrainer
