"""SN56_PER_EXAMPLE_LOSS (F6): mean over examples of per-token response CE,
for every batch layout train_instruct produces.

Run: python scripts/tests/test_sn56_loss.py   (torch; the Trainer test needs transformers)
"""
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sn56_loss import example_ids, per_example_ce  # noqa: E402

V = 11


def _example_mean(logits_1d, labels_1d):
    """Per-token CE mean of ONE example laid out alone (causal shift)."""
    lg, tg = logits_1d[:-1], labels_1d[1:]
    m = tg != -100
    return F.cross_entropy(lg[m].double(), tg[m], reduction="mean")


def test_example_ids_layouts():
    labels = torch.zeros(1, 8, dtype=torch.long)
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 0, 1]])
    assert example_ids(labels, position_ids=pos)[0].tolist() == [0, 0, 0, 1, 1, 2, 3, 3]
    fa_mask = torch.tensor([[1, 1, 2, 2, 2, 3, 0, 0]])
    assert example_ids(labels, attention_mask=fa_mask)[0].tolist() == [1, 1, 2, 2, 2, 3, 0, 0]
    plain = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    ids = example_ids(torch.zeros(2, 4, dtype=torch.long), attention_mask=plain)
    assert len(set(ids[0].tolist())) == 1 and len(set(ids[1].tolist())) == 1 and ids[0, 0] != ids[1, 0]
    print("[ok] example ids: position_ids / FA seq-id mask / one per row")


def test_packed_position_ids_matches_per_example():
    torch.manual_seed(0)
    lens = [4, 9, 3]
    ex_labels = [torch.randint(0, V, (n,)) for n in lens]
    for lab in ex_labels:
        lab[0] = -100
    ex_labels[1][1:4] = -100                      # a prompt span
    logits = torch.randn(1, sum(lens), V, dtype=torch.float64)
    labels = torch.cat(ex_labels).unsqueeze(0)
    pos = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0)
    total, n = per_example_ce(logits, labels, position_ids=pos)
    ref, st = [], 0
    for k, L in enumerate(lens):
        ref.append(_example_mean(logits[0, st:st + L], ex_labels[k]))
        st += L
    assert n == 3 and torch.allclose(total.double(), torch.stack(ref).sum(), atol=1e-5), (total, ref)
    # token-mean (the stock loss) weights the long example more
    tok = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
    assert not torch.allclose(total.double() / n, tok, atol=1e-5)
    print("[ok] packed (position_ids): sum of per-example means")


def test_fa_mask_and_unpacked_rows():
    torch.manual_seed(1)
    # FA packing layout: seq ids 1..n, 0 = padding (labels -100 there)
    mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 0, 0]])
    labels = torch.tensor([[-100, 3, 4, -100, 5, 6, 7, -100, -100]])
    logits = torch.randn(1, 9, V, dtype=torch.float64)
    total, n = per_example_ce(logits, labels, attention_mask=mask)
    ref = _example_mean(logits[0, :3], labels[0, :3]) + _example_mean(logits[0, 3:7], labels[0, 3:7])
    assert n == 2 and torch.allclose(total.double(), ref, atol=1e-5)
    # unpacked padded rows: one example per row
    labels2 = torch.tensor([[-100, 1, 2, 3, -100], [-100, 4, -100, -100, -100]])
    mask2 = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    logits2 = torch.randn(2, 5, V, dtype=torch.float64)
    total2, n2 = per_example_ce(logits2, labels2, attention_mask=mask2)
    ref2 = _example_mean(logits2[0], labels2[0]) + _example_mean(logits2[1], labels2[1])
    assert n2 == 2 and torch.allclose(total2.double(), ref2, atol=1e-5)
    # equal-length responses: per-example mean == token mean
    labels3 = torch.tensor([[-100, 1, 2], [-100, 3, 4]])
    logits3 = torch.randn(2, 3, V, dtype=torch.float64)
    t3, n3 = per_example_ce(logits3, labels3)
    tok3 = F.cross_entropy(logits3[:, :-1].reshape(-1, V), labels3[:, 1:].reshape(-1))
    assert torch.allclose(t3.double() / n3, tok3, atol=1e-5)
    # no response token at all: zero loss that still has a graph
    z, nz = per_example_ce(logits3.requires_grad_(), torch.full((2, 3), -100))
    assert nz == 0 and z.item() == 0.0 and z.requires_grad
    print("[ok] FA seq-id mask, unpacked rows, equal lengths, empty batch")


def test_chunking_is_exact():
    torch.manual_seed(2)
    logits = torch.randn(2, 40, V, dtype=torch.float64)
    labels = torch.randint(0, V, (2, 40))
    labels[:, :5] = -100
    a, na = per_example_ce(logits, labels, chunk_tokens=7)
    b, nb = per_example_ce(logits, labels, chunk_tokens=10 ** 6)
    assert na == nb and torch.allclose(a, b, atol=1e-6)
    print("[ok] chunked CE is exact")


def test_trainer_compute_loss():
    try:
        import transformers
        from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
    except Exception as e:  # pragma: no cover
        print(f"[skip] trainer test ({e})")
        return
    from sn56_loss import make_per_example_trainer

    cfg = LlamaConfig(vocab_size=V, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(output_dir=tmp, report_to=[], per_device_train_batch_size=2)
        try:
            args = TrainingArguments(use_cpu=True, **kw)
        except TypeError:
            args = TrainingArguments(no_cuda=True, **kw)
        cls = make_per_example_trainer(Trainer)
        trainer = cls(model=model, args=args)
        assert trainer.model_accepts_loss_kwargs is False
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0], [7, 8, 9, 1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 3, 4, 5, 6, -100, -100], [-100, -100, -100, -100, -100, -100, 4, 5]]),
        }
        loss, out = trainer.compute_loss(model, dict(batch), return_outputs=True)
        total, n = per_example_ce(out.logits, batch["labels"], attention_mask=batch["attention_mask"])
        assert n == 2 and torch.allclose(loss, total / 2)
        stock = model(**batch).loss
        assert not torch.allclose(loss, stock)            # stock = token mean, 6 vs 2 tokens
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
    print(f"[ok] PerExampleLoss{Trainer.__name__}.compute_loss (transformers {transformers.__version__})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 LOSS TESTS PASSED")
