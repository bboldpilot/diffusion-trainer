"""SN56_DEV_METRIC=vali (K1-a): the dev metric equals the validator's eval_loss.

Three levels:
  1. row_stats on synthetic logits == brute force, for every batch layout
     (FA seq-id mask, v5 position_ids, unpacked rows), with and without KL.
  2. Trainer.evaluate() of the wrapped trainer on a tiny Llama == the validator's
     OWN functions (_compute_per_example_losses / _calculate_instruct_kl_divergence,
     extracted verbatim from the upstream checkout, bs=1) for plain and KL tasks.
  3. The 23 Oct T0 per-example vectors (validator-faithful scorer, G0 +-1e-4):
     each vector is turned into per-token CE through a lookup "model", packed by
     the trainer's real PackedDataset (FA layout) and evaluated by the real
     HF evaluation loop; the metric must return each file's eval_loss, and the
     ranking is compared with the stock token-mean dev metric.

Run: python scripts/tests/test_sn56_devmetric.py
Env: SN56_WS = workspace root (default: three levels above scripts/), for the
     upstream checkout and results/text-1023-t0 (levels 2/3 skip without them).
"""
import ast
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:  # utility.py (smart_truncate, via PackedDataset) imports wandb; stub it off the image
    import wandb  # noqa: F401
except ImportError:
    import importlib.machinery
    import types
    _stub = types.ModuleType("wandb")
    _stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    sys.modules["wandb"] = _stub

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sn56_devmetric import metric_from_stats, row_stats  # noqa: E402

WS = Path(os.environ.get("SN56_WS", HERE.parents[3]))
V = 13


def _bf_example(logits_rows, labels_rows):
    """token-mean CE of one example given its own logits/labels (shifted)."""
    lg, lb = logits_rows[:-1], labels_rows[1:]
    m = lb != -100
    return F.cross_entropy(lg[m].float(), lb[m], reduction="mean").double()


def _bf_kl(ft, base, labels):
    m = labels != -100
    lp_f, lp_b = F.log_softmax(ft[m].float(), -1), F.log_softmax(base[m].float(), -1)
    return (lp_f.exp() * (lp_f - lp_b)).sum(-1).double()


def test_row_stats_layouts():
    torch.manual_seed(0)
    # FA packing: two rows, seq ids 1..k, 0 = pad
    lens = [[4, 7, 3], [9, 2]]
    T = 16
    labels = torch.full((2, T), -100)
    mask = torch.zeros(2, T, dtype=torch.long)
    ref_means, ref_kl, ref_kln = [], [], 0
    logits = torch.randn(2, T, V, dtype=torch.float64)
    base = torch.randn(2, T, V, dtype=torch.float64)
    for r, row in enumerate(lens):
        st = 0
        for k, n in enumerate(row):
            lab = torch.randint(0, V, (n,))
            lab[0] = -100
            if n > 3:
                lab[1] = -100                      # a prompt token
            labels[r, st:st + n] = lab
            mask[r, st:st + n] = k + 1
            ref_means.append(_bf_example(logits[r, st:st + n], lab))
            st += n
        kl = _bf_kl(logits[r], base[r], labels[r])
        ref_kl.append(kl.sum())
        ref_kln += kl.numel()
    s = row_stats(logits, labels, attention_mask=mask, base_logits=base, chunk_tokens=5)
    assert s.shape == (2, 6)
    # columns 4/5: plain response-token CE sum and count (the stock token mean)
    tok_ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1),
                             ignore_index=-100, reduction="sum").double()
    assert torch.allclose(s[:, 4].sum(), tok_ce, atol=1e-6) and s[:, 5].sum() == (labels[:, 1:] != -100).sum()
    assert torch.allclose(s[:, 0].sum(), torch.stack(ref_means).sum(), atol=1e-7)
    assert s[:, 1].tolist() == [3.0, 2.0]
    assert torch.allclose(s[:, 2].sum(), torch.stack(ref_kl).sum(), atol=1e-7) and s[:, 3].sum() == ref_kln
    m = metric_from_stats(s, kl_coef=0.7)
    want = torch.stack(ref_means).mean() + 0.7 * torch.stack(ref_kl).sum() / ref_kln
    assert abs(m["loss"] - float(want)) < 1e-7, (m, float(want))   # fp32 CE, summation order
    # the same examples as v5 padding-free rows (position_ids restart per example)
    pos = torch.zeros(2, T, dtype=torch.long)
    for r, row in enumerate(lens):
        st = 0
        for n in row:
            pos[r, st:st + n] = torch.arange(n)
            st += n
        pos[r, st:] = torch.arange(T - st)          # tail: labels are -100 there
    s2 = row_stats(logits, labels, position_ids=pos)
    assert torch.allclose(s2[:, :2], s[:, :2], atol=1e-7)
    # unpacked rows: one example per row
    lab_u = torch.tensor([[-100, -100, 3, 4, 5, -100], [-100, 6, -100, -100, -100, -100]])
    lg_u = torch.randn(2, 6, V, dtype=torch.float64)
    s3 = row_stats(lg_u, lab_u, attention_mask=(lab_u != -100).long() | 1)
    ref3 = _bf_example(lg_u[0], lab_u[0]) + _bf_example(lg_u[1], lab_u[1])
    assert torch.allclose(s3[:, 0].sum(), ref3, atol=1e-7) and s3[:, 1].sum() == 2
    # token mean differs from the example mean when lengths differ
    tok = F.cross_entropy(lg_u[:, :-1].reshape(-1, V).float(), lab_u[:, 1:].reshape(-1), ignore_index=-100)
    m3 = float(s3[:, 0].sum() / 2)
    assert abs(float(tok) - m3) > 1e-4
    print("[ok] row_stats: FA mask / position_ids / unpacked rows, KL at unshifted positions")


# ── validator functions, verbatim from the upstream checkout ──
def _upstream_src(path):
    up = WS / "upstream"
    if not (up / ".git").exists() and not (up / ".git").is_file():
        return None
    try:
        return subprocess.run(["git", "-C", str(up), "show", f"origin/main:{path}"],
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return None


def _validator_fns():
    src = _upstream_src("validator/evaluation/evaluators/instruct_text.py")
    csrc = _upstream_src("validator/evaluation/constants.py")
    if src is None or csrc is None:
        return None
    chunk = None
    for line in csrc.splitlines():
        if line.startswith("PER_EXAMPLE_LOSS_CE_CHUNK_SIZE"):
            chunk = int(eval(line.split("=", 1)[1].split("#")[0]))
    from torch.nn.utils.rnn import pad_sequence
    ns = {"torch": torch, "F": F, "pad_sequence": pad_sequence,
          "eval_cst": SimpleNamespace(PER_EXAMPLE_LOSS_CE_CHUNK_SIZE=chunk or 512),
          "logger": SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
          "AutoTokenizer": object, "AutoModelForCausalLM": object}
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "_collate_evaluation_batch", "_compute_per_example_losses", "_calculate_instruct_kl_divergence"):
            exec(compile(ast.get_source_segment(src, node), f"upstream:instruct_text.py:{node.name}", "exec"), ns)
    return ns


def test_trainer_matches_validator():
    try:
        import transformers
        from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
    except Exception as e:  # pragma: no cover
        print(f"[skip] trainer test ({e})")
        return
    ns = _validator_fns()
    if ns is None:
        print("[skip] validator functions (no upstream checkout at SN56_WS/upstream)")
        return
    from sn56_devmetric import make_vali_dev_trainer

    cfg = LlamaConfig(vocab_size=V, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    torch.manual_seed(0)
    ft = LlamaForCausalLM(cfg).eval()
    base = LlamaForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    ds = []
    for i in range(23):                        # response 1..40 tokens, prompt 3..12
        p = int(torch.randint(3, 13, (1,), generator=g))
        r = 1 + (i * 7) % 40
        ids = torch.randint(1, V, (p + r,), generator=g).tolist()
        ds.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * p + ids[p:]})
    tok = SimpleNamespace(pad_token_id=0)
    # validator: bs=1 per-example CE (the eval_loss identity) and its KL term
    real_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self           # CPU run of the verbatim code
    try:
        vec = ns["_compute_per_example_losses"](ft, ds, tok, batch_size=1)
        kl = ns["_calculate_instruct_kl_divergence"](base, ft, ds, tok, batch_size=4)
    finally:
        torch.Tensor.cuda = real_cuda
    want_plain = sum(vec) / len(vec)

    def collate(feats):
        return ns["_collate_evaluation_batch"](feats, tok)

    class KLStandIn(Trainer):          # the attributes KLRegularizedTrainer exposes
        def __init__(self, *a, kl_coef=0.0, base_model=None, **k):
            super().__init__(*a, **k)
            self.kl_coef, self.base_model, self.use_lora_base = kl_coef, base_model, False

    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(output_dir=tmp, report_to=[], per_device_eval_batch_size=4)
        try:
            args = TrainingArguments(use_cpu=True, **kw)
        except TypeError:
            args = TrainingArguments(no_cuda=True, **kw)
        stock = Trainer(model=ft, args=args, eval_dataset=ds, data_collator=collate).evaluate()["eval_loss"]
        tr = make_vali_dev_trainer(Trainer)(model=ft, args=args, eval_dataset=ds, data_collator=collate)
        m = tr.evaluate()
        assert abs(m["eval_loss"] - want_plain) < 1e-6, (m["eval_loss"], want_plain)
        assert abs(m["eval_loss_tokmean"] - stock) < 1e-6 and abs(stock - want_plain) > 1e-3, (stock, want_plain)
        assert m["eval_n_examples"] == len(ds)
        c = 0.37
        trk = make_vali_dev_trainer(KLStandIn, kl_coef=c)(model=ft, args=args, eval_dataset=ds,
                                                          data_collator=collate, kl_coef=c, base_model=base)
        mk = trk.evaluate()
        want_kl = want_plain + c * kl
        assert abs(mk["eval_loss"] - want_kl) < 1e-6, (mk["eval_loss"], want_kl)
        assert abs(mk["eval_kl_tok"] - kl) < 1e-6
    print(f"[ok] Trainer.evaluate (vali) == validator functions: plain {want_plain:.6f} "
          f"(stock token mean {stock:.6f}), KL {want_kl:.6f} = {want_plain:.6f} + {c} x {kl:.6f} "
          f"(transformers {transformers.__version__})")


# ── T0 vectors through the real packing + evaluation loop ──
TASKS = {
    "7923f3a2": ("r1-7923f3a2/results/T0-7923f3a2-fix/7923f3a2",
                 "tourn_21d7f15f1e6bf934_20261019/tasks/7923f3a2-4520-48c2-9169-d6f715516e56"),
    "7d3d80aa": ("ab-7d3d80aa/results/AB-7d3d80aa-fix/7d3d80aa",
                 "tourn_21d7f15f1e6bf934_20261019/tasks/7d3d80aa-6bfd-4592-9f9a-87e75855d892"),
}


class _LookupLM(torch.nn.Module):
    """logits = table[input_ids]: per-token CE fixed in advance (test double)."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, **kw):
        logits = self.table[input_ids]
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                          labels[:, 1:].reshape(-1), ignore_index=-100)
        return out


def _build(per_ex, n_tok):
    """Examples whose response tokens carry CE values with mean per_ex[i] exactly.
    Two-class logits [a, 0], target class 1: CE = log(1 + e^a) -> a = log(expm1(ce))."""
    rows, table = [], [[0.0, 0.0]]                  # id 0 = pad
    for i, (L, n) in enumerate(zip(per_ex, n_tok)):
        w = torch.tensor([1.0 + 0.9 * math.sin(1.7 * t + i) for t in range(n)], dtype=torch.float64)
        ce = (L * w / w.mean()).clamp_min(1e-12)
        ids = list(range(len(table), len(table) + n + 1))     # prompt token + n response tokens
        for t in range(n):                                    # position t predicts response token t+1
            table.append([math.log(math.expm1(float(ce[t]))), 0.0])
        table.append([0.0, 0.0])                              # last position: next example (masked)
        rows.append({"input_ids": ids, "attention_mask": [1] * (n + 1), "labels": [-100] + [1] * n})
    return rows, torch.tensor(table, dtype=torch.float64)


def test_t0_vectors():
    t0 = WS / "results" / "text-1023-t0" / "pods"
    data = WS / "harness" / "gradientswatch" / "data"
    if not t0.exists() or not data.exists():
        print(f"[skip] T0 vectors (not found under {WS})")
        return
    try:
        from transformers import Trainer, TrainingArguments
        from monkeypatch import PackedDataset
    except Exception as e:  # pragma: no cover
        print(f"[skip] T0 test ({e})")
        return
    from sn56_devmetric import make_vali_dev_trainer

    report = {}
    for task, (sub, tdir) in TASKS.items():
        files = sorted(glob.glob(str(t0 / sub / "losses-*.json")))
        test_rows = json.load(open(data / tdir / "test_data.json"))
        audit = json.load(open(data / tdir / "audit.json"))
        published = {h["hotkey"][:8]: h.get("test_loss") for h in audit.get("hotkey_details", [])}
        res = {}
        for f in files:
            d = json.load(open(f))
            vec = [float(x) for x in d["per_example_losses"]]
            # response length proxy (chars/3.5 + EOS), as levers-1028-text/pe.py
            n_tok = [max(1, round(len(str(test_rows[i].get("output", ""))) / 3.5)) + 1 for i in d["row_index"]]
            n_tok = [min(n, 600) for n in n_tok]
            rows, table = _build(vec, n_tok)
            dev = PackedDataset(SimpleNamespace(eval_dataset=rows),
                                SimpleNamespace(pad_token_id=0, padding_side="right"),
                                max_input_length=1024, use_fa=True)
            model = _LookupLM(table)
            with tempfile.TemporaryDirectory() as tmp:
                kw = dict(output_dir=tmp, report_to=[], per_device_eval_batch_size=1)
                try:
                    args = TrainingArguments(use_cpu=True, **kw)
                except TypeError:
                    args = TrainingArguments(no_cuda=True, **kw)
                m = make_vali_dev_trainer(Trainer)(model=model, args=args, eval_dataset=dev).evaluate()
            want = d["eval_loss"]
            assert abs(sum(vec) / len(vec) - want) < 1e-9
            assert abs(m["eval_loss"] - want) < 2e-6, (task, d["label"], m["eval_loss"], want)
            assert m["eval_n_examples"] == len(vec)
            tokw = sum(v * n for v, n in zip(vec, n_tok)) / sum(n_tok)
            res[d["label"]] = {"vali_dev": m["eval_loss"], "file_eval_loss": want,
                               "stock_dev_rows": m["eval_loss_tokmean"], "token_weighted": tokw,
                               "published": published.get(d["label"][:8]), "rows": len(dev)}
        rank = lambda k: [x for x, _ in sorted(res.items(), key=lambda kv: kv[1][k])]  # noqa: E731
        report[task] = {"models": res, "rank_vali": rank("vali_dev"), "rank_stock_rows": rank("stock_dev_rows"),
                        "rank_token_weighted": rank("token_weighted")}
        pub = {k: v["published"] for k, v in res.items() if v["published"] is not None}
        if len(pub) > 1:
            # T0 vectors are a replica (G0 +-1e-4), so near-tied published pairs may swap
            report[task]["rank_published"] = sorted(pub, key=pub.get)
            report[task]["published_order_kept"] = (
                [k for k in report[task]["rank_vali"] if k in pub] == report[task]["rank_published"])
        print(f"[ok] T0 {task}: {len(res)} vectors, vali dev == eval_loss (<2e-6) through PackedDataset(FA) + "
              f"evaluation_loop; ours rank vali {report[task]['rank_vali'].index('ours') + 1} vs stock-rows "
              f"{report[task]['rank_stock_rows'].index('ours') + 1} vs token-weighted "
              f"{report[task]['rank_token_weighted'].index('ours') + 1} of {len(res)}")
    out = os.environ.get("SN56_DEVMETRIC_REPORT")
    if out:
        Path(out).write_text(json.dumps(report, indent=1))
        print(f"     report -> {out}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 DEV METRIC TESTS PASSED")
