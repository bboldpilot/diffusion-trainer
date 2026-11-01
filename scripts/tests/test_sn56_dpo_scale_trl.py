"""SN56_DPO_SCALE_CAL=1 (sn56_dpo_scale.py) on the REAL TRL DPOTrainer (the trainer image pins trl 0.18.0), CPU.

Tiny Llama + word-level tokenizer, 40 pairs, eval batch 5 (8 equal batches, so TRL's mean of batch means is the
pair mean):
  1. dev_pair_losses == evaluate()'s eval_loss, and at a = 0 (policy == reference) every pair is ln 2 exactly.
  2. calibrate() leaves the weights BIT-identical, runs no callback (evaluate() is not called), its a-curve is
     the dev loss of the explicitly scaled model (cross-check at a = 0.5 with evaluate()) and it picks the argmin.
  3. Over-confident policy: a few real DPO steps on clean pairs, the fine-tune direction extrapolated x4, scored on
     held-out pairs with 35% flipped labels -> dev loss at a = 1 is far above its minimum, calibrate picks a < 1
     and the gain is > min_gain.
  4. apply_and_save(): submission weights == ref + a*(theta - ref) (fp32 model here), config / loss.txt preserved,
     keep_dir holds the uncalibrated weights; a = 1 is a no-op (nothing written).
  4b. With the K5 dump subclass (all bench arms set SN56_DPO_DUMP_DIR): same numbers, calibrate writes no dump.
  5. supported(): world_size > 1 / no ref_model -> refused.

Run: python scripts/tests/test_sn56_dpo_scale_trl.py   (needs trl, datasets, tokenizers; skips otherwise)
"""
import copy
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup():
    import numpy as np
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = [f"w{i}" for i in range(40)]
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>")
    rng = np.random.default_rng(0)
    def sent(n): return " ".join(rng.choice(words, n))
    # learnable preference: chosen draws from the first half of the vocabulary, rejected from the second
    lo, hi = words[:20], words[20:]
    def pick(ws, n): return " " + " ".join(rng.choice(ws, n))
    def row():
        return {"prompt": sent(int(rng.integers(3, 8))), "chosen": pick(lo, int(rng.integers(2, 8))),
                "rejected": pick(hi, int(rng.integers(2, 8)))}
    ds = Dataset.from_list([row() for _ in range(40)])
    # held-out pairs with 35% of the labels flipped: the preference generalises, a confident model pays on the flips
    dev = []
    for i in range(40):
        r = row()
        if i % 20 < 7:
            r["chosen"], r["rejected"] = r["rejected"], r["chosen"]
        dev.append(r)
    dev_ds = Dataset.from_list(dev)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0)
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg)
    return tok, ds, dev_ds, base


def _trainer(tok, ds, policy, ref, tmp, eval_ds=None, **kw):
    from trl import DPOConfig, DPOTrainer
    args = DPOConfig(output_dir=tmp, report_to=[], per_device_eval_batch_size=5, per_device_train_batch_size=8,
                     max_length=64, beta=0.1, use_cpu=True, logging_steps=1, learning_rate=kw.get("lr", 5e-3),
                     max_steps=kw.get("steps", 0) or -1, save_strategy="no", eval_strategy="no")
    return DPOTrainer(model=policy, ref_model=ref, args=args, train_dataset=ds, eval_dataset=ds if eval_ds is None else eval_ds,
                      processing_class=tok)


def test_scale_calibration_real_trl():
    try:
        import numpy as np
        import torch
        import trl
        from safetensors.torch import load_file
    except Exception as e:  # pragma: no cover
        print(f"[skip] real TRL test ({e})")
        return
    try:
        tok, ds, dev_ds, base = _setup()
    except Exception as e:  # pragma: no cover
        print(f"[skip] real TRL test ({e})")
        return
    import sn56_dpo_scale as S

    with tempfile.TemporaryDirectory() as tmp:
        # ── 1/2: random-direction policy ──
        pol = copy.deepcopy(base)
        with torch.no_grad():
            for p in pol.parameters():
                p.add_(0.05 * torch.randn_like(p))
        tr = _trainer(tok, ds, pol, copy.deepcopy(base), tmp)
        ok, why = S.supported(tr, 1)
        assert ok, why
        m = tr.evaluate()
        v = S.dev_pair_losses(tr)
        assert len(v) == len(ds) and abs(v.mean() - m["eval_loss"]) < 1e-5, (v.mean(), m["eval_loss"])
        pairs = S.paired_params(tr.model, tr.ref_model)
        theta = S.snapshot(pairs)
        S.set_scaled(pairs, theta, 0.0)
        v0 = S.dev_pair_losses(tr)
        assert np.allclose(v0, math.log(2.0), atol=1e-6), v0[:5]
        S.set_scaled(pairs, theta, 0.5)
        m05 = tr.evaluate()["eval_loss"]
        with torch.no_grad():
            for (p, _), t in zip(pairs, theta):
                p.copy_(t)
        before = [p.detach().clone() for p in tr.model.parameters()]
        n_eval_calls = {"n": 0}
        orig_eval = tr.evaluate
        def _counting_eval(*a, **k):
            n_eval_calls["n"] += 1
            return orig_eval(*a, **k)
        tr.evaluate = _counting_eval
        rep = os.path.join(tmp, "dump", "scale-cal.json")
        out = S.calibrate(tr, [0.25, 0.5, 0.75, 1.0, 1.5], min_gain=0.001, report_path=rep)
        assert n_eval_calls["n"] == 0
        assert all(torch.equal(a, b) for a, b in zip(before, tr.model.parameters())), "weights not restored"
        assert abs(out["losses"]["0.5"] - m05) < 1e-5, (out["losses"]["0.5"], m05)
        curve = {float(k): v for k, v in out["losses"].items()}
        assert out["alpha"] == min(curve, key=curve.get) and curve[out["alpha"]] <= curve[1.0], out
        assert os.path.exists(rep)
        print(f"[ok] perturbed policy: dev == evaluate() ({m['eval_loss']:.5f}), a=0 -> ln 2 on every pair, "
              f"weights bit-identical after calibrate, a={out['alpha']:g} (curve {curve})")

        # ── 3: over-confident policy = real DPO fine-tune extrapolated x4 ──
        pol = copy.deepcopy(base)
        ft = _trainer(tok, ds, pol, copy.deepcopy(base), os.path.join(tmp, "ft"), lr=5e-3, steps=15)
        ft.train()
        d_ft = S.dev_pair_losses(ft).mean()
        with torch.no_grad():
            for p, b in zip(pol.parameters(), base.parameters()):
                p.copy_(b + 4.0 * (p - b))
        tr2 = _trainer(tok, ds, pol, copy.deepcopy(base), os.path.join(tmp, "cal"), eval_ds=dev_ds)
        out2 = S.calibrate(tr2, [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.25], min_gain=0.001)
        assert d_ft < math.log(2.0) - 0.01, d_ft
        assert out2["alpha"] < 1.0 and out2["loss_at_1"] - out2["loss_at_alpha"] > 0.001, out2
        print(f"[ok] over-confident (fine-tune x4): dev {out2['loss_at_1']:.4f} at a=1 -> {out2['loss_at_alpha']:.4f} "
              f"at a={out2['alpha']:g} (fine-tune itself {d_ft:.4f})")

        # ── 4: apply_and_save ──
        sub = os.path.join(tmp, "submission"); keep = os.path.join(tmp, "keep")
        tr2.model.save_pretrained(sub, safe_serialization=True)
        with open(os.path.join(sub, "loss.txt"), "w") as f:
            f.write("384,0.70")
        theta2 = {n: p.detach().clone() for n, p in tr2.model.named_parameters()}
        refp = dict(tr2.ref_model.named_parameters())
        a = out2["alpha"]
        assert S.apply_and_save(tr2, a, sub, keep_dir=keep) is True
        got = load_file(os.path.join(sub, "model.safetensors"))
        kept = load_file(os.path.join(keep, "model.safetensors"))
        for n, t in theta2.items():
            if n not in got:
                continue      # tied weights may be stored once
            want = refp[n].float() + a * (t.float() - refp[n].float())
            assert torch.allclose(got[n].float(), want, atol=1e-6), n
            assert torch.equal(kept[n], t), n
        assert open(os.path.join(sub, "loss.txt")).read() == "384,0.70"
        assert S.apply_and_save(tr2, 1.0, sub, keep_dir=os.path.join(tmp, "keep1")) is False
        assert not os.path.exists(os.path.join(tmp, "keep1"))
        print(f"[ok] apply_and_save: submission = ref + {a:g}*(theta-ref), loss.txt kept, keep_dir = theta; a=1 no-op")

        # ── 4b: the arms run with SN56_DPO_DUMP_DIR -> the GateDPOTrainer subclass; calibrate must not dump ──
        from trl import DPOConfig, DPOTrainer
        from sn56_dpo_select import make_gate_dpo_trainer
        gdump = os.path.join(tmp, "gdump")
        gcls = make_gate_dpo_trainer(DPOTrainer, dump_dir=gdump, select=False)
        gargs = DPOConfig(output_dir=os.path.join(tmp, "g"), report_to=[], per_device_eval_batch_size=5,
                          max_length=64, beta=0.1, use_cpu=True, save_strategy="no", eval_strategy="no")
        gtr = gcls(model=copy.deepcopy(tr2.model), ref_model=copy.deepcopy(base), args=gargs, train_dataset=ds,
                   eval_dataset=dev_ds, processing_class=tok)
        gout = S.calibrate(gtr, [0.25, 0.5, 1.0], min_gain=0.001)
        gm = gtr.evaluate()
        assert abs(gout["losses"]["1.0"] - gm["eval_loss"]) < 1e-5 and len(os.listdir(gdump)) == 1
        print(f"[ok] GateDPOTrainer (dump on): calibrate a=1 {gout['losses']['1.0']:.5f} == evaluate(), no dump written")

        # ── 5: supported() ──
        assert S.supported(tr2, 2)[0] is False
        tr2.ref_model = None
        assert S.supported(tr2, 1) == (False, "sem ref_model")
        print(f"[ok] supported(): DDP / no reference refused (trl {trl.__version__})")


if __name__ == "__main__":
    test_scale_calibration_real_trl()
    print("\nALL 1 SN56 DPO SCALE (TRL) TESTS PASSED")
