"""K1-a (SN56_DEV_METRIC=vali, default "stock"): the dev metric the validator ranks on.

What the validator ranks instruct / chat tasks on (G.O.D origin/main bfa699ad,
validator/evaluation/evaluators/instruct_text.py):
  eval_loss = Trainer.evaluate() with per_device_eval_batch_size pinned to 1
            = flat mean over examples of each example's token-mean CE over its
              response, denominator = the SHIFTED supervised count (:170-240,
              the identity the validator asserts in :403-424)
  KL tasks  : + kl_coef * KL(ft || base), token-weighted over ALL eval tokens at
              the UNSHIFTED label != -100 positions (:243-289, :369-385)

The stock dev loss is a token mean per (packed) row averaged over rows, so long
answers dominate it and it picks different checkpoints (3 of 4
replayed tasks). With this metric on, every dev row reports
  [sum of its examples' mean CE, number of examples, KL sum, KL token count]
as its "prediction"; Trainer.evaluation_loop gathers those like predictions
(gather_for_metrics drops the DDP padding duplicates), and eval_loss becomes
  sum(means) / n_examples + kl_coef * sum(KL) / sum(KL tokens)
on every rank. Everything that selects on eval_loss follows: best checkpoint and
loss.txt (CustomEvalSaveCallback), the greedy soup and the overfit early stop
(AdaptiveTrainingCallback). The stock value is kept as eval_loss_tokmean.

Examples never span rows (packing places whole examples), so per-row sums add up
to the exact global quantity whatever the packing, batch size or world size.
Example boundaries come from sn56_loss.example_ids (FA seq-id mask, v5
position_ids, or one example per row).

Numerics: CE and KL in fp32 from the model's logits (the validator's KL runs in
the model dtype; the difference is rounding, not definition). Known gap: dev rows
longer than the trainer's max_length are truncated by the dev dataset, while the
validator drops rows over its sequence_len (4096) and scores the rest whole;
train_instruct logs how many dev rows that touches.
"""

import sys

import torch
import torch.nn.functional as F

from sn56_loss import example_ids

N_STATS = 6          # [sum of per-example mean CE, n examples, KL sum, KL tokens, CE token sum, CE tokens]


def row_stats(logits, labels, position_ids=None, attention_mask=None, base_logits=None,
              chunk_tokens: int = 8192):
    """(B, 6) float64 per-row stats: the validator metric's four (module doc) plus the
    plain response-token CE sum and count (the stock token-mean bookkeeping).

    logits / base_logits: (B, T, V) from the fine-tuned / base model on the same
    batch; base_logits None = no KL term.
    """
    B, T = labels.shape
    out = torch.zeros(B, N_STATS, dtype=torch.float64, device=labels.device)
    if T < 2:
        return out
    V = logits.shape[-1]
    seg = example_ids(labels, position_ids, attention_mask)[:, 1:].reshape(-1)
    tgt = labels[:, 1:].reshape(-1)
    rows = torch.arange(B, device=labels.device).unsqueeze(1).expand(B, T - 1).reshape(-1)
    idx = (tgt != -100).nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        lg = logits[:, :-1, :].reshape(-1, V)
        tgt_sel = tgt.index_select(0, idx)
        ce = torch.cat([
            F.cross_entropy(lg.index_select(0, idx[s:s + chunk_tokens]).float(),
                            tgt_sel[s:s + chunk_tokens], reduction="none")
            for s in range(0, idx.numel(), chunk_tokens)
        ]).double()
        uniq, inv = torch.unique(seg.index_select(0, idx), return_inverse=True)
        sums = torch.zeros(uniq.numel(), dtype=torch.float64, device=ce.device).index_add_(0, inv, ce)
        cnts = torch.zeros(uniq.numel(), dtype=torch.float64, device=ce.device).index_add_(
            0, inv, torch.ones_like(ce))
        ex_row = torch.zeros(uniq.numel(), dtype=torch.long, device=ce.device).scatter_(
            0, inv, rows.index_select(0, idx))
        out[:, 0].index_add_(0, ex_row, sums / cnts)
        out[:, 1].index_add_(0, ex_row, torch.ones_like(sums))
        out[:, 4].index_add_(0, ex_row, sums)
        out[:, 5].index_add_(0, ex_row, cnts)
    if base_logits is not None:
        # validator: logits[t] against base[t] wherever labels[t] != -100 (no shift)
        kmask = (labels != -100).reshape(-1)
        kidx = kmask.nonzero(as_tuple=True)[0]
        if kidx.numel() > 0:
            ft = logits.reshape(-1, V)
            bs = base_logits.reshape(-1, base_logits.shape[-1])
            parts = []
            for s in range(0, kidx.numel(), chunk_tokens):
                sl = kidx[s:s + chunk_tokens]
                lp_ft = F.log_softmax(ft.index_select(0, sl).float(), dim=-1)
                lp_b = F.log_softmax(bs.index_select(0, sl).float(), dim=-1)
                parts.append((lp_ft.exp() * (lp_ft - lp_b)).sum(dim=-1))
            kl = torch.cat(parts).double()
            krow = torch.arange(B, device=labels.device).unsqueeze(1).expand(B, T).reshape(-1).index_select(0, kidx)
            out[:, 2].index_add_(0, krow, kl)
            out[:, 3].index_add_(0, krow, torch.ones_like(kl))
    return out


VALI_SEQUENCE_LEN = 4096   # validator/assets/test_axolotl.yml


def vali_sequence_len(max_position_embeddings) -> int:
    """The validator's eval sequence_len for a submission (validator/evaluation/common.py:546-549):
    4096, or ceil(mpe / 2) when the model's max_position_embeddings < 2 * 4096."""
    seq = VALI_SEQUENCE_LEN
    try:
        mpe = int(max_position_embeddings) if max_position_embeddings else 0
    except (TypeError, ValueError):
        mpe = 0
    if mpe and mpe < 2 * seq:
        seq = -(-mpe // 2)
    return seq


def scoreable_rows(rows, seq_len: int):
    """SN56_DEV_SCOREABLE: the dev rows the validator would score (axolotl drops samples longer than
    sequence_len instead of truncating them). Returns (kept, n_dropped); all rows when none would survive
    (the validator then retries with a longer sequence_len)."""
    kept = [dp for dp in rows if len(dp.get("input_ids", [])) <= seq_len]
    if not kept:
        return list(rows), 0
    return kept, len(rows) - len(kept)


def metric_from_stats(stats, kl_coef: float = 0.0) -> dict:
    """Global metric from gathered (N, 4) row stats (numpy or torch)."""
    s = torch.as_tensor(stats, dtype=torch.float64).reshape(-1, N_STATS).sum(dim=0)
    n = float(s[1])
    ce = float(s[0]) / n if n > 0 else float("nan")
    kl = float(s[2]) / float(s[3]) if float(s[3]) > 0 else 0.0
    tok = float(s[4]) / float(s[5]) if float(s[5]) > 0 else float("nan")
    return {"loss": ce + (kl_coef or 0.0) * kl, "ce_ex": ce, "kl_tok": kl, "n_examples": n,
            "kl_tokens": float(s[3]), "ce_tok": tok}


def make_vali_dev_trainer(base_cls, kl_coef: float = 0.0):
    """Subclass of base_cls whose eval_loss is the validator metric (module doc)."""

    class ValiDevMetricTrainer(base_cls):
        sn56_dev_kl_coef = float(kl_coef or 0.0)
        _sn56_dev_logged = False

        def _sn56_dev_base_logits(self, fwd):
            """Base-model logits for the KL term, same source as KLRegularizedTrainer."""
            if not (getattr(self, "kl_coef", 0) and self.sn56_dev_kl_coef):
                return None
            if getattr(self, "use_lora_base", False):
                with self.model.disable_adapter():
                    return self.model(**fwd).logits
            base = getattr(self, "base_model", None)
            if base is None:
                return None
            ref = fwd.get("input_ids")
            if ref is not None and base.device != ref.device:
                base.to(ref.device)
            return base(**fwd).logits

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if "labels" not in inputs:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs = self._prepare_inputs(inputs)
            labels = inputs["labels"]
            fwd = {k: v for k, v in inputs.items() if k != "labels"}
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    outputs = model(**fwd)
                    base_logits = self._sn56_dev_base_logits(fwd)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                stats = row_stats(logits, labels, inputs.get("position_ids"), inputs.get("attention_mask"),
                                  base_logits=base_logits)
                # the stock bookkeeping value: token mean over this batch's responses (no extra fp32 logits copy)
                tok_mean = (stats[:, 4].sum() / stats[:, 5].sum().clamp_min(1)).float()
            del outputs, logits, base_logits
            return tok_mean.detach(), stats, None

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                            metric_key_prefix="eval"):
            out = super().evaluation_loop(dataloader, description, prediction_loss_only=False,
                                          ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            stats = out.predictions
            key = f"{metric_key_prefix}_loss"
            try:
                m = metric_from_stats(stats, self.sn56_dev_kl_coef)
            except Exception as e:  # never lose an eval over the metric
                print(f"[sn56][devmetric] stats inválidos ({e}); eval_loss padrão mantido", file=sys.stderr, flush=True)
                return out
            if not m["n_examples"] > 0:
                return out
            out.metrics[f"{metric_key_prefix}_loss_tokmean"] = out.metrics.get(key)   # stock: mean of batch token means
            out.metrics[f"{metric_key_prefix}_ce_tok"] = m["ce_tok"]                   # global token mean
            out.metrics[key] = m["loss"]
            out.metrics[f"{metric_key_prefix}_ce_ex"] = m["ce_ex"]
            if self.sn56_dev_kl_coef:
                out.metrics[f"{metric_key_prefix}_kl_tok"] = m["kl_tok"]
            out.metrics[f"{metric_key_prefix}_n_examples"] = m["n_examples"]
            if not ValiDevMetricTrainer._sn56_dev_logged:
                ValiDevMetricTrainer._sn56_dev_logged = True
                print(f"[sn56][devmetric] vali: {m['n_examples']:.0f} exemplos, ce_ex={m['ce_ex']:.5f} "
                      f"kl={m['kl_tok']:.5f}x{self.sn56_dev_kl_coef:g} -> {m['loss']:.5f} "
                      f"(tokmean {out.metrics[f'{metric_key_prefix}_loss_tokmean']})", file=sys.stderr, flush=True)
            return out

    ValiDevMetricTrainer.__name__ = f"ValiDev{base_cls.__name__}"
    return ValiDevMetricTrainer
