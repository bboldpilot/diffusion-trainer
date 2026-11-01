"""SN56_PER_EXAMPLE_ALPHA (K1-b): F6 weighting n_i^-alpha.

alpha=1 is the existing per-example mean (same code path), alpha=0 the stock
token mean, alpha=0.5 the manual formula sum_i n_i^-a S_i / sum_i n_i^(1-a);
hidden path == logits path for alpha != 1, and the Trainer uses it.

Run: python scripts/tests/test_sn56_loss_alpha.py
"""
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sn56_loss import make_per_example_trainer, per_example_ce, per_example_ce_from_hidden  # noqa: E402

V = 11


def _layout():
    torch.manual_seed(5)
    lens = [3, 12, 6]
    mask = torch.tensor([[1] * 3 + [2] * 12 + [3] * 6 + [0] * 3])
    labels = torch.full((1, 24), -100)
    st = 0
    for n in lens:
        labels[0, st + 1:st + n] = torch.randint(0, V, (n - 1,))
        st += n
    return labels, mask


def _manual(logits, labels, mask, alpha):
    ce = F.cross_entropy(logits[0, :-1].float(), labels[0, 1:], ignore_index=-100, reduction="none").double()
    seg = mask[0, 1:]
    num = den = 0.0
    for k in seg.unique().tolist():
        sel = (seg == k) & (labels[0, 1:] != -100)
        if sel.sum() == 0:
            continue
        n = float(sel.sum())
        num += float(ce[sel].sum()) * n ** -alpha
        den += n ** (1 - alpha)
    return num / den


def test_alpha_values():
    labels, mask = _layout()
    logits = torch.randn(1, 24, V, dtype=torch.float64)
    t1, n1 = per_example_ce(logits, labels, attention_mask=mask)
    assert isinstance(n1, int) and n1 == 3
    for a in (1.0, 0.5, 0.0, 1.5):
        t, n = per_example_ce(logits, labels, attention_mask=mask, alpha=a)
        got = float(t) / (n if n else 1)
        assert abs(got - _manual(logits, labels, mask, a)) < 1e-6, (a, got, _manual(logits, labels, mask, a))
    tok = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
    t0, n0 = per_example_ce(logits, labels, attention_mask=mask, alpha=0.0)
    assert abs(float(t0) / n0 - float(tok)) < 1e-6
    assert abs(float(t1) / n1 - _manual(logits, labels, mask, 1.0)) < 1e-6
    # hidden path, alpha 0.5, with grads
    D = 6
    hidden = torch.randn(1, 24, D, dtype=torch.float64, requires_grad=True)
    w = torch.randn(V, D, dtype=torch.float64, requires_grad=True)
    th, nh = per_example_ce_from_hidden(hidden, w, None, labels, attention_mask=mask, chunk_rows=5, alpha=0.5)
    tl, nl = per_example_ce(F.linear(hidden, w), labels, attention_mask=mask, alpha=0.5)
    assert abs(nh - nl) < 1e-12 and torch.allclose(th, tl, atol=1e-9)
    (th / nh).backward()
    print("[ok] alpha 1 / 0.5 / 0 / 1.5 == manual n_i^-alpha weighting; alpha 0 == token mean; hidden == logits")


def test_trainer_alpha():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
    except Exception as e:  # pragma: no cover
        print(f"[skip] trainer ({e})")
        return
    cfg = LlamaConfig(vocab_size=V, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0], [7, 8, 9, 1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 3, 4, 5, 6, -100, -100], [-100, -100, -100, -100, -100, -100, 4, 5]]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(output_dir=tmp, report_to=[], per_device_train_batch_size=2)
        try:
            args = TrainingArguments(use_cpu=True, **kw)
        except TypeError:
            args = TrainingArguments(no_cuda=True, **kw)
        tr = make_per_example_trainer(Trainer, alpha=0.5)(model=model, args=args)
        loss = tr.compute_loss(model, dict(batch))
        full = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        t, n = per_example_ce(full, batch["labels"], attention_mask=batch["attention_mask"], alpha=0.5)
        assert torch.allclose(loss, t / n, atol=1e-6) and tr.f6_alpha == 0.5
        t1 = make_per_example_trainer(Trainer)(model=model, args=args)
        assert t1.f6_alpha == 1.0
    print("[ok] PerExampleLossTrainer(alpha=0.5) loss == manual; default alpha 1.0")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 LOSS ALPHA TESTS PASSED")
