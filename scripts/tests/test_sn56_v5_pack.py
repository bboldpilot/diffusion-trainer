"""SN56_V5_PACK (F5): padding-free packing is isolated per example.

Needs torch; the LFM2 tests need transformers >= 5 (the v5 lane's stack) and
are skipped otherwise. CPU only: attention runs on sdpa, where transformers
turns the restarting position_ids into a block-diagonal mask (FA2 turns the
same position_ids into varlen cu_seqlens). The short-conv layers are what this
file really checks: without the seq_idx patch the first tokens of every packed
example read the previous example.

Run: python scripts/tests/test_sn56_v5_pack.py
"""
import os
import sys
import types

import torch

try:  # utility.py (smart_truncate) imports wandb; stub it off the image
    import wandb  # noqa: F401
except ImportError:
    import importlib.machinery

    _stub = types.ModuleType("wandb")
    _stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    sys.modules["wandb"] = _stub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v5_pack import (  # noqa: E402
    V5Collator,
    _masked_causal_depthwise_conv,
    seq_idx_from_position_ids,
)


def _tf_major():
    try:
        import transformers
        return int(transformers.__version__.split(".")[0])
    except Exception:
        return 0


def test_masked_conv_equals_per_example_conv():
    torch.manual_seed(0)
    C, K = 6, 3
    conv = torch.nn.Conv1d(C, C, kernel_size=K, groups=C, padding=K - 1, bias=True).double()
    lengths = [5, 1, 7, 2]
    x = torch.randn(1, C, sum(lengths), dtype=torch.float64)
    seq_idx = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]).unsqueeze(0)
    packed = _masked_causal_depthwise_conv(x, conv, seq_idx)
    start = 0
    for n in lengths:
        alone = conv(x[..., start:start + n])[..., :n]
        assert torch.allclose(packed[..., start:start + n], alone, atol=1e-12), (start, n)
        start += n
    naive = conv(x)[..., : x.shape[-1]]
    assert not torch.allclose(naive, packed), "plain conv should leak across boundaries"
    print("[ok] masked causal conv == per-example conv (plain conv leaks)")


def test_seq_idx_from_position_ids():
    pos = torch.tensor([[0, 1, 2, 0, 0, 1, 0, 1, 2, 3]])
    assert seq_idx_from_position_ids(pos).tolist() == [[0, 0, 0, 1, 2, 2, 3, 3, 3, 3]]
    assert seq_idx_from_position_ids(pos).dtype == torch.int32
    print("[ok] seq_idx from position_ids")


def test_padding_free_pack_and_collator():
    from monkeypatch import pack_data_points_padding_free

    items = [
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "attention_mask": [1, 1, 1]},
        {"input_ids": [8, 9], "labels": [8, 9], "attention_mask": [1, 1]},
    ]
    row = pack_data_points_padding_free(items, None, 16)
    assert set(row) == {"input_ids", "labels", "position_ids"}
    assert row["input_ids"].tolist() == [5, 6, 7, 8, 9]
    assert row["labels"].tolist() == [-100, 6, 7, -100, 9]       # first label of each example masked
    assert row["position_ids"].tolist() == [0, 1, 2, 0, 1]
    row2 = pack_data_points_padding_free(items[1:], None, 16)
    batch = V5Collator(pad_token_id=0, pass_seq_idx=True)([row, row2])
    assert batch["input_ids"].shape == (1, 7) and "attention_mask" not in batch
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 1]]
    assert batch["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 2, 2]]
    assert batch["use_cache"] is False   # a KV cache would disable packed-sequence detection
    assert "seq_idx" not in V5Collator(pad_token_id=0)([row])
    print("[ok] padding-free pack + flattening collator")


def test_dynamic_padding_collator():
    feats = [
        {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]),
         "labels": torch.tensor([-100, 2, 3])},
        {"input_ids": torch.tensor([4]), "attention_mask": torch.tensor([1]), "labels": torch.tensor([4])},
    ]
    b = V5Collator(pad_token_id=99)(feats)
    assert b["input_ids"].tolist() == [[1, 2, 3], [4, 99, 99]]      # padded to the longest, not max_length
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    print("[ok] dynamic right padding")


def test_packed_dataset_padding_free_roundtrip():
    from monkeypatch import PackedDataset

    class _DS:
        eval_dataset = [
            {"input_ids": list(range(1, n + 1)), "labels": [-100] + list(range(2, n + 1)),
             "attention_mask": [1] * n}
            for n in (5, 9, 3, 12, 4, 7)
        ]

    ds = PackedDataset(_DS(), None, max_input_length=16, padding_free=True)
    seen = 0
    for i in range(len(ds)):
        item = ds[i]
        n = item["input_ids"].numel()
        assert n <= 16 and "attention_mask" not in item
        seen += int((item["position_ids"] == 0).sum())
    assert seen == len(_DS.eval_dataset)
    print(f"[ok] PackedDataset(padding_free) -> {len(ds)} rows, every example once")


def _tiny_lfm2():
    from transformers import Lfm2Config, Lfm2ForCausalLM

    cfg = Lfm2Config(
        vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        layer_types=["conv", "full_attention", "conv", "full_attention"], conv_L_cache=3,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_cache=False,  # train_instruct sets config.use_cache = False
    )
    torch.manual_seed(0)
    model = Lfm2ForCausalLM._from_config(cfg, attn_implementation="sdpa").double().eval()
    return model


def test_lfm2_packed_equals_unpacked():
    if _tf_major() < 5:
        print("[skip] LFM2 packing test needs transformers >= 5")
        return
    import transformers.models.lfm2.modeling_lfm2 as m
    from monkeypatch import pack_data_points_padding_free
    from v5_pack import patch_short_conv_seq_idx

    model = _tiny_lfm2()
    torch.manual_seed(1)
    seqs = [torch.randint(3, 96, (n,)).tolist() for n in (6, 9, 4)]
    items = [{"input_ids": s, "labels": list(s)} for s in seqs]
    batch = V5Collator(pad_token_id=0, pass_seq_idx=True)([pack_data_points_padding_free(items, None, 64)])
    with torch.no_grad():
        alone = [model(input_ids=torch.tensor([s])).logits[0] for s in seqs]
        unpatched = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                          seq_idx=batch["seq_idx"]).logits[0]
    orig_forward = m.Lfm2ShortConv.forward
    try:
        assert "Lfm2ShortConv" in patch_short_conv_seq_idx()
        with torch.no_grad():
            patched = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                            seq_idx=batch["seq_idx"]).logits[0]
        start, worst_patched, worst_unpatched = 0, 0.0, 0.0
        for s, ref in zip(seqs, alone):
            n = len(s)
            worst_patched = max(worst_patched, (patched[start:start + n] - ref).abs().max().item())
            worst_unpatched = max(worst_unpatched, (unpatched[start:start + n] - ref).abs().max().item())
            start += n
        assert worst_patched < 1e-9, worst_patched
        assert worst_unpatched > 1e-6, "expected the torch conv path to leak without the patch"
        # gradients flow through the patched path; loss never crosses examples
        model.train()
        out = model(**batch)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"[ok] LFM2 packed == per-example (max |diff| {worst_patched:.1e}; "
              f"unpatched conv leaks {worst_unpatched:.1e}); backward ok")
    finally:
        m.Lfm2ShortConv.forward = orig_forward
        m.Lfm2ShortConv._sn56_seq_idx_patch = False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 V5-PACK TESTS PASSED")
