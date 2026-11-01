"""SN56_F6_GLOBAL: the per-example (alpha) loss under 2-rank DDP (gloo, CPU) gives the single-process gradient of
the whole batch; the old per-rank normalisation does not (uneven example counts / lengths across ranks).

Run: python scripts/tests/test_sn56_f6_ddp.py   (torch, transformers 4.51.3)
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

import sn56_loss as L  # noqa: E402

T = 12


class _Base:
    """Stand-in for transformers.Trainer: only what PerExampleLossTrainer touches."""

    def __init__(self, model):
        self.model = model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        raise AssertionError("stock loss must not be reached")


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=4)).float()


def _batches():
    """Rank 0: three short examples; rank 1: one long one (one example per row)."""
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 48, (4, T), generator=g)
    labels = ids.clone()
    labels[0, :9] = -100            # 3 response tokens
    labels[1, :7] = -100            # 5
    labels[2, :10] = -100           # 2
    labels[3, :2] = -100            # 10 (rank 1 alone)
    am = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": am, "labels": labels}


def _split(b, rank):
    sl = slice(0, 3) if rank == 0 else slice(3, 4)
    return {k: v[sl] for k, v in b.items()}


def _grads(model):
    m = model.module if hasattr(model, "module") else model
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def _worker(rank, world, path, alpha, global_norm, out):
    os.environ.update({"MASTER_ADDR": "127.0.0.1"})
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=world)
    model = torch.nn.parallel.DistributedDataParallel(_model())
    model.train()
    tr = L.make_per_example_trainer(_Base, alpha=alpha, global_norm=global_norm)(model)
    loss = tr.compute_loss(model, _split(_batches(), rank))
    loss.backward()
    if rank == 0:
        torch.save({"grads": _grads(model), "mode": tr._f6_mode, "loss": float(loss)}, out)
    dist.barrier()
    dist.destroy_process_group()


def _ddp(alpha, global_norm):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "g.pt")
        mp.spawn(_worker, args=(2, os.path.join(d, "init"), alpha, global_norm, out), nprocs=2, join=True)
        return torch.load(out)


def _single(alpha):
    model = _model().train()
    tr = L.make_per_example_trainer(_Base, alpha=alpha, global_norm=True)(model)
    loss = tr.compute_loss(model, _batches())     # world 1: global == local
    loss.backward()
    return _grads(model), tr._f6_mode, float(loss)


def _rel(a, b):
    num = sum(float((a[k] - b[k]).pow(2).sum()) for k in a)
    den = sum(float(b[k].pow(2).sum()) for k in b)
    return (num / max(den, 1e-30)) ** 0.5


def test_denominators():
    b = _batches()
    for alpha, want in ((1.0, 4.0), (0.5, 3 ** 0.5 + 5 ** 0.5 + 2 ** 0.5 + 10 ** 0.5), (0.0, 20.0)):
        got = float(L.local_denominator(b["labels"], None, b["attention_mask"], alpha))
        assert abs(got - want) < 1e-9, (alpha, got, want)
    print("[ok] f6 ddp: local_denominator = #examples (a=1), sum n^(1-a) (a=0.5), #tokens (a=0)")


def test_ddp_matches_single():
    for alpha in (0.5, 1.0):
        g1, mode1, _ = _single(alpha)
        r_glob = _ddp(alpha, True)
        r_loc = _ddp(alpha, False)
        e_glob, e_loc = _rel(r_glob["grads"], g1), _rel(r_loc["grads"], g1)
        assert mode1 == r_glob["mode"] == "hidden", (mode1, r_glob["mode"])
        assert e_glob < 1e-5, (alpha, e_glob)
        assert e_loc > 1e-2, (alpha, e_loc)
        print(f"[ok] f6 ddp alpha={alpha}: 2-rank global grad == 1-process grad (rel {e_glob:.1e}); "
              f"per-rank normalisation off by {e_loc:.2f}")


if __name__ == "__main__":
    test_denominators()
    test_ddp_matches_single()
    print("ALL 2 SN56 F6 DDP TESTS PASSED")
