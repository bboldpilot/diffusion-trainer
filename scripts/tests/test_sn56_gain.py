"""SN56_GAIN (K4): fp32 per-tensor gains on augmentation-eligible weights.

  1. eligibility = the validator's rule; hooks == W <- g*W exactly (Linear with
     bias, LayerNorm-like ln_f); fold_ keeps the function and resets g; remove()
     leaves a plain model.
  2. bf16 recovery: a student whose eligible weights were partly scaled by
     s=1.4 (the weight_scaling augmentation) distils back to the teacher. Pure
     bf16 Adam at a realistic LR barely moves the scale; with gains it does.
  3. HF Trainer integration on CPU: gain group in the optimizer, fold before
     every eval/save, no extra state_dict keys, unhook + sn56_gain.json at the end.

Run: python scripts/tests/test_sn56_gain.py
"""
import copy
import json
import math
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sn56_gain import GainCallback, GainSet, eligible, make_gain_trainer  # noqa: E402

V = 64


def _llama(tied=False, dtype=torch.float32, seed=0, layers=2, hidden=32):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=V, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=tied, attention_bias=True)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).to(dtype).eval()


def _scaled_copy(model, gains: dict):
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n in gains:
                p.copy_((p.float() * gains[n]).to(p.dtype))
    return ref


def test_eligibility_hooks_fold():
    m = _llama()
    gs = GainSet(m)
    names = set(gs.names)
    assert names and all(eligible(n) for n in names)
    assert not any("norm" in n or "embed" in n or n.startswith("lm_head") for n in names), names
    want = {n for n, p in m.named_parameters() if eligible(n) and not n.startswith("lm_head")}
    assert names == want, (want - names, names - want)
    torch.manual_seed(1)
    g = {n: float(0.7 + 0.6 * torch.rand(())) for n in gs.names}
    with torch.no_grad():
        for n, p in zip(gs.names, gs.params):
            p.fill_(g[n])
    x = torch.randint(0, V, (2, 11))
    ref = _scaled_copy(m, g)
    gs.install()
    out = m(x).logits
    assert torch.allclose(out, ref(x).logits, atol=1e-5), (out - ref(x).logits).abs().max()
    n = gs.fold_()
    assert n == len(gs) and all(float(p) == 1.0 for p in gs.params)
    assert torch.allclose(m(x).logits, out, atol=1e-5)
    assert all(abs(a - g[nm]) < 1e-6 for a, nm in zip(gs.applied, gs.names))
    gs.remove()
    assert not gs.installed and torch.allclose(m(x).logits, out, atol=1e-5)
    # bf16 fold: one rounding of the fp32 product, nothing else
    mb = _llama(dtype=torch.bfloat16)
    w0 = dict((k, v.detach().clone()) for k, v in mb.named_parameters())
    gb = GainSet(mb).install()
    with torch.no_grad():
        gb.params[0].fill_(0.8711)
    gb.fold_()
    name0 = gb.names[0]
    w_new = dict(mb.named_parameters())[name0].detach()
    assert torch.equal(w_new, (w0[name0].float() * 0.8711).to(torch.bfloat16))
    print(f"[ok] {len(gs)} eligible gains (validator rule, lm_head skipped); hooks == g*W; fold keeps the "
          f"function, g->1; bf16 fold = one rounding")


def test_layernorm_like_out_hook():
    from transformers import BloomConfig, BloomForCausalLM
    cfg = BloomConfig(vocab_size=V, hidden_size=32, n_layer=2, n_head=4)
    torch.manual_seed(0)
    m = BloomForCausalLM(cfg).eval()
    with torch.no_grad():                     # non-trivial LayerNorm bias
        m.transformer.ln_f.bias.normal_(0, 0.3)
    gs = GainSet(m)
    kinds = dict(zip(gs.names, gs.kinds))
    assert kinds.get("transformer.ln_f.weight") == "out", kinds
    assert not any("layernorm" in n for n in gs.names)          # input/post_attention_layernorm: "norm"
    g = {n: (1.0 / 1.427 if i % 2 == 0 else 1.0) for i, n in enumerate(gs.names)}
    g["transformer.ln_f.weight"] = 0.9
    with torch.no_grad():
        for n, p in zip(gs.names, gs.params):
            p.fill_(g[n])
    x = torch.randint(0, V, (2, 9))
    ref = _scaled_copy(m, g)
    gs.install()
    assert torch.allclose(m(x).logits, ref(x).logits, atol=1e-5)
    gs.remove()
    assert torch.allclose(m(x).logits, ref(x).logits, atol=1e-5)
    print(f"[ok] BLOOM: ln_f (eligible, no 'norm' in its name) via output hook (bias kept); {len(gs)} gains")


def _distil(student, teacher, steps, lr_w, lr_g=None, seed=0):
    """KL(teacher || student) on random tokens; bf16 student weights updated in
    place by torch Adam (the boss/our pure-bf16 regime)."""
    torch.manual_seed(seed)
    params = [p for p in student.parameters()]
    groups = [{"params": params, "lr": lr_w}]
    gs = None
    if lr_g:
        gs = GainSet(student).install()
        groups.append({"params": gs.params, "lr": lr_g, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    losses = []
    for step in range(steps):
        x = torch.randint(0, V, (8, 24))
        with torch.no_grad():
            t = F.log_softmax(teacher(x).logits.float(), -1)
        s = F.log_softmax(student(x).logits.float(), -1)
        loss = (t.exp() * (t - s)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    if gs is not None:
        gs.remove()
    return losses, gs


def test_bf16_recovery():
    teacher = _llama(seed=3, layers=2, hidden=64)
    base = copy.deepcopy(teacher).to(torch.bfloat16)
    eligible_names = [n for n, _ in base.named_parameters() if eligible(n) and not n.startswith("lm_head")]
    rng = torch.Generator().manual_seed(7)
    scaled = [n for n in eligible_names if torch.rand((), generator=rng) < 0.5]
    s = 1.4
    with torch.no_grad():
        for n, p in base.named_parameters():
            if n in scaled:
                p.copy_((p.float() * s).to(torch.bfloat16))
    w_true = {n: p.detach().float().clone() for n, p in teacher.named_parameters()}

    def eff_scale(model):
        out = {}
        for n, p in model.named_parameters():
            if n in eligible_names:
                out[n] = float((p.detach().float() * w_true[n]).sum() / (w_true[n] * w_true[n]).sum())
        return out

    steps, lr_w = 200, 3e-5
    a = copy.deepcopy(base)
    la, _ = _distil(a, teacher, steps, lr_w)
    b = copy.deepcopy(base)
    lb, gsb = _distil(b, teacher, steps, lr_w, lr_g=1e-2)
    ea, eb = eff_scale(a), eff_scale(b)
    err = lambda e: sorted(abs(math.log(e[n])) for n in scaled)  # noqa: E731
    med_a, med_b = err(ea)[len(scaled) // 2], err(eb)[len(scaled) // 2]
    unsc_b = max(abs(math.log(eb[n])) for n in eligible_names if n not in scaled)
    tail = lambda l: sum(l[-20:]) / 20  # noqa: E731
    print(f"     scaled {len(scaled)}/{len(eligible_names)} x{s}; bf16 Adam lr {lr_w:g}, {steps} steps")
    print(f"     |log eff scale| median on scaled tensors: bf16 only {med_a:.4f} (start {math.log(s):.4f}), "
          f"with gains {med_b:.4f}; max on unscaled with gains {unsc_b:.4f}")
    print(f"     KL(teacher||student): start {la[0]:.4f} -> bf16 only {tail(la):.4f}, with gains {tail(lb):.4f}")
    assert med_a > 0.9 * math.log(s), med_a          # bf16 Adam alone cannot undo the scale
    assert med_b < 0.35 * math.log(s), med_b         # the gains do
    assert tail(lb) < 0.5 * tail(la), (tail(lb), tail(la))
    print("[ok] bf16 recovery: gains undo the x1.4 scaling that pure bf16 Adam keeps")


def test_trainer_integration():
    try:
        from transformers import Trainer, TrainerCallback, TrainingArguments
    except Exception as e:  # pragma: no cover
        print(f"[skip] trainer test ({e})")
        return
    m = _llama()
    data = [{"input_ids": torch.randint(0, V, (16,)).tolist()} for _ in range(24)]
    for d in data:
        d["labels"] = list(d["input_ids"])

    def collate(feats):
        x = torch.tensor([f["input_ids"] for f in feats])
        return {"input_ids": x, "labels": x.clone()}

    seen = {}

    class Probe(TrainerCallback):
        def on_evaluate(self, args, state, control, **kw):
            gs = trainer._sn56_gains
            seen.setdefault("eval_gains", []).append([float(p) for p in gs.params])

    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(output_dir=tmp, report_to=[], per_device_train_batch_size=4, per_device_eval_batch_size=4,
                  max_steps=6, eval_strategy="steps", eval_steps=3, save_strategy="steps", save_steps=3,
                  learning_rate=1e-4, optim="adamw_torch", logging_steps=1, save_only_model=True)
        try:
            args = TrainingArguments(use_cpu=True, **kw)
        except TypeError:
            args = TrainingArguments(no_cuda=True, **kw)
        cb = GainCallback()
        cls = make_gain_trainer(Trainer, gain_lr=5e-2)
        trainer = cls(model=m, args=args, train_dataset=data, eval_dataset=data[:8], data_collator=collate,
                      callbacks=[cb, Probe()])
        cb.trainer = trainer
        order = [type(c).__name__ for c in trainer.callback_handler.callbacks]
        assert order.index("GainCallback") < order.index("Probe"), order   # fold before later callbacks
        keys0 = set(m.state_dict())
        trainer.train()
        gs = trainer._sn56_gains
        grp = [g for g in trainer.optimizer.param_groups if any(p is gs.params[0] for p in g["params"])]
        assert len(grp) == 1 and grp[0]["weight_decay"] == 0.0 and grp[0]["initial_lr"] == 5e-2
        assert len(seen["eval_gains"]) == 2 and all(v == 1.0 for row in seen["eval_gains"] for v in row)
        assert not gs.installed and gs.n_folds >= 3                 # eval x2, saves, train end
        moved = sum(1 for a in gs.applied if abs(a - 1.0) > 1e-4)
        assert moved > 0, gs.applied
        js = json.load(open(os.path.join(tmp, "sn56_gain.json")))
        assert js["n"] == len(gs) and len(js["scales"]) == len(gs)
        trainer.save_model(os.path.join(tmp, "final"))
        from safetensors import safe_open
        with safe_open(os.path.join(tmp, "final", "model.safetensors"), "pt") as f:
            saved = set(f.keys())
        assert saved <= keys0 and not any("gain" in k for k in saved), saved - keys0
    print(f"[ok] Trainer: gain group lr 5e-2 wd 0, g == 1 at every eval (folded), {moved}/{len(gs)} weights "
          f"rescaled, no extra keys saved, unhooked at train end, sn56_gain.json written")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 GAIN TESTS PASSED")
