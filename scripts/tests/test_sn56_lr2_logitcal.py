"""SN56_LR_RETRY (sn56_retry.py) and SN56_LOGIT_CAL (sn56_logit_cal.py), both default off.

Run: python scripts/tests/test_sn56_lr2_logitcal.py   (torch, transformers 4.51.3, safetensors)
"""
import datetime
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import sn56_plan as P  # noqa: E402
import sn56_logit_cal as C  # noqa: E402
import sn56_retry as R  # noqa: E402


def _clear_env():
    for k in list(P.FLAGS) + [P.MASTER, "SN56_LR_FIXED"]:
        os.environ.pop(k, None)


# ── logit calibration ────────────────────────────────────────────────────────

def test_scaled_stats_brute_force():
    torch.manual_seed(0)
    B, T, V = 2, 12, 17
    logits = torch.randn(B, T, V) * 3
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = -100
    labels[0, 6:8] = -100          # prompt of the second packed example in row 0
    seq = torch.ones(B, T, dtype=torch.long)
    seq[0, 6:] = 2                  # FA seq-id mask: row 0 holds two examples
    grid = (0.7, 1.0, 1.3)
    got = C.scaled_example_stats(logits, labels, grid, attention_mask=seq)
    for i, a in enumerate(grid):
        tot, n = 0.0, 0
        for b in range(B):
            for sid in sorted(set(seq[b].tolist())):
                ce = []
                for t in range(T - 1):
                    if seq[b, t + 1] == sid and labels[b, t + 1] != -100:
                        z = a * logits[b, t].double()
                        ce.append(float(torch.logsumexp(z, 0) - z[labels[b, t + 1]]))
                if ce:
                    tot += sum(ce) / len(ce); n += 1
        assert abs(float(got[i, 0]) - tot) < 1e-6 and int(got[i, 1]) == n, (a, float(got[i, 0]), tot, n)
    print("[ok] logitcal: per-example CE of a*logits == brute force (packed rows, shift, prompt mask)")


def test_pick_recovers_temperature():
    torch.manual_seed(1)
    N, V = 4000, 40
    z = torch.randn(1, N + 1, V) * 2.5
    probs = torch.softmax(0.6 * z[0, :-1].double(), -1)
    y = torch.multinomial(probs, 1).squeeze(1)
    labels = torch.full((1, N + 1), -100)
    labels[0, 1:] = y
    grid = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2)
    st = C.scaled_example_stats(z, labels, grid)
    a, la, l1 = C.pick(grid, st, min_gain=0.001)
    assert a == 0.6 and la < l1 - 0.05, (a, la, l1)
    a2, la2, l12 = C.pick(grid, st, min_gain=10.0)
    assert a2 == 1.0 and la2 == l12
    print(f"[ok] logitcal: data drawn at T=1/0.6 -> a={a} (dev {l1:.4f} -> {la:.4f}); min_gain gate keeps a=1")


def _tiny(kind, tie=True):
    from transformers import (BloomConfig, BloomForCausalLM, GemmaConfig, GemmaForCausalLM, LlamaConfig,
                              LlamaForCausalLM)
    torch.manual_seed(2)
    if kind == "llama":
        return LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=tie))
    if kind == "bloom":
        return BloomForCausalLM(BloomConfig(vocab_size=50, hidden_size=32, n_layer=2, n_head=4))
    m = GemmaForCausalLM(GemmaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=4, head_dim=8))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(0.3)        # (1 + w): a non-zero w makes the weight fold non-linear
    return m


def test_fold_targets_and_check():
    fwd = {"input_ids": torch.tensor([[1, 5, 9, 13, 2, 7]]), "attention_mask": torch.ones(1, 6, dtype=torch.long)}
    cases = [("llama", True, ["model.norm.weight"]), ("llama", False, ["lm_head.weight"]),
             ("bloom", True, ["transformer.ln_f.weight", "transformer.ln_f.bias"])]
    for kind, tie, want in cases:
        m = _tiny(kind, tie).eval()
        t = C.fold_targets(m)
        assert sorted(n for n, _ in t) == sorted(want), (kind, tie, [n for n, _ in t])
        before = [p.detach().clone() for _, p in t]
        err = C.verify_fold(m, fwd, t, 0.9)
        assert err < 1e-5, (kind, tie, err)
        assert all(torch.equal(b, p) for b, (_, p) in zip(before, t)), "verify_fold must restore the weights"
    g = _tiny("gemma").eval()
    err = C.verify_fold(g, fwd, C.fold_targets(g), 0.9)
    assert err > C.FOLD_REL_TOL, err
    print(f"[ok] logitcal: fold target = norm (tied Llama), lm_head (untied), ln_f w+b (Bloom); exact; "
          f"Gemma (1+w) norm rejected (rel err {err:.3f})")


def test_apply_to_submission_and_fit():
    from transformers import BloomForCausalLM
    m = _tiny("bloom").eval()
    ids = torch.tensor([[3, 8, 11, 4, 9, 21, 30, 2]])
    with torch.no_grad():
        z0 = m(input_ids=ids).logits
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d, safe_serialization=True)
        names = [n for n, _ in C.fold_targets(m)]
        assert C.apply_to_submission(d, names, 0.9, log=lambda *_: None) == 2
        m2 = BloomForCausalLM.from_pretrained(d).eval()
        with torch.no_grad():
            z1 = m2(input_ids=ids).logits
        assert float((z1 - 0.9 * z0).abs().max()) < 1e-4
        assert C.apply_to_submission(d, names, 1.0, log=lambda *_: None) == 0

    class FakeTrainer:
        def __init__(self, model, batches):
            self.model, self.batches = model, batches

        def get_eval_dataloader(self):
            return list(self.batches)

        def _prepare_inputs(self, x):
            return x

    torch.manual_seed(3)
    batches = []
    for _ in range(4):
        x = torch.randint(0, 50, (2, 10))
        lab = x.clone(); lab[:, :3] = -100
        batches.append({"input_ids": x, "attention_mask": torch.ones_like(x), "labels": lab})
    m.train()
    res = C.fit_on_trainer(FakeTrainer(m, batches), grid=(0.5, 0.8, 1.0, 1.25), min_gain=0.0, log=lambda *_: None)
    assert m.training, "fit must restore train mode"
    assert res["a"] in (0.5, 0.8, 1.0, 1.25) and res["n_examples"] == 8
    i1 = res["grid"].index(1.0)
    assert abs(res["losses"][i1] - res["loss_1"]) < 1e-12 and res["loss_a"] <= res["loss_1"] + 1e-12
    if res["a"] != 1.0:
        assert res["names"] and res["fold_rel_err"] < C.FOLD_REL_TOL
    print(f"[ok] logitcal: safetensors fold reloads as a*logits; fit on a trainer -> a={res['a']} "
          f"({res['loss_1']:.4f} -> {res['loss_a']:.4f}), train mode restored")


# ── dev-selected second run ──────────────────────────────────────────────────

def _rec(**kw):
    end = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=kw.pop("left", 2000)))
    r = {"task_type": "InstructTextTask", "sn56_on": True, "use_kl": False, "dev_metric": "vali", "dev": 1.9117,
         "lr": 2.43e-5, "lr_fixed": None, "lr_argmin": 1.45e-5, "lr_search_s": 700.0, "t_start": 0.0,
         "t_train_end": 1565.0, "end_time": end.strftime("%Y-%m-%d %H:%M:%S")}
    r.update(kw)
    return r


def test_decide():
    go, lr2, why = R.decide(_rec())             # granite 30 Oct: 865 s post-search -> needs 1365 s
    assert go and abs(lr2 - 1.45e-5) < 1e-12, why
    for kw, frag in ((dict(left=1000), "left"), (dict(lr_argmin=2.2e-5), "argmin"), (dict(lr_argmin=None), "argmin"),
                     (dict(lr_fixed=1.45e-5), "second run"), (dict(use_kl=True), "scope"),
                     (dict(sn56_on=False), "scope"), (dict(dev_metric="stock"), "dev metric"),
                     (dict(dev=None), "dev metric"), (dict(dev=float("nan")), "dev metric")):
        go, _, why = R.decide(_rec(**kw))
        assert not go and frag in why, (kw, why)
    assert not R.decide(None)[0]
    assert R.better({"dev": 1.9117}, {"dev": 1.90}) and not R.better({"dev": 1.9117}, {"dev": 1.9110})
    assert not R.better({"dev": 1.9}, {"dev": float("nan")}) and not R.better({"dev": 1.9}, {})
    print("[ok] lr2: decide (granite case GO; time, ratio, scope, metric gates), better() margin 0.1%")


def test_second_run_flow():
    _clear_env()
    os.environ["SN56_R1_COOLDOWN"] = "0"; os.environ["SN56_R1_AVG2"] = "0"   # plain LR2 flow; COOLAVG: test_sn56_round4
    with tempfile.TemporaryDirectory() as d:
        out, sub = os.path.join(d, "out"), os.path.join(d, "sub")
        os.makedirs(os.path.join(out, "checkpoint-100")); os.makedirs(sub)
        for f, txt in (("config.json", "{}"), ("model.safetensors", "run1")):
            open(os.path.join(sub, f), "w").write(txt)
        req = os.path.join(d, "training_request_t.json")
        json.dump({"train_request": {"submission_dir": sub, "end_time": "x"}}, open(req, "w"))
        cmd = f"python train_instruct.py --output_dir {out} --request_path {req} --learning_rate 2e-5"
        calls = []

        def repl(c, name, val):
            parts = c.split(); i = parts.index(f"--{name}"); parts[i + 1] = val; return " ".join(parts)

        def fake_run(dev2):
            def run(c, logp, env_vars=None):
                calls.append((c, logp, env_vars))
                o2 = c.split("--output_dir ")[1].split()[0]
                r2 = json.load(open(c.split("--request_path ")[1].split()[0]))["train_request"]["submission_dir"]
                os.makedirs(o2, exist_ok=True)
                open(os.path.join(o2, "success.txt"), "w").write("Success")
                P.write_run_record(o2, _rec(dev=dev2, lr=float(env_vars["SN56_LR_FIXED"]), lr_fixed=1.45e-5))
                for f, txt in (("config.json", "{}"), ("model.safetensors", "run2")):
                    open(os.path.join(r2, f), "w").write(txt)
            return run

        P.write_run_record(out, _rec())
        os.environ["SN56_LR_RETRY"] = "0"
        assert R.maybe_second_run(cmd, os.path.join(d, "train.log"), out, req, sub, fake_run(1.80), repl,
                                  log=lambda *_: None) == "off"
        os.environ["SN56_LR_RETRY"] = "1"
        st = R.maybe_second_run(cmd, os.path.join(d, "train.log"), out, req, sub, fake_run(1.80), repl,
                                log=lambda *_: None)
        assert st == "swapped" and open(os.path.join(sub, "model.safetensors")).read() == "run2", st
        c, logp, env = calls[-1]
        assert env == {"SN56_LR_SEARCH": "0", "SN56_LR_FIXED": repr(1.45e-5)} and logp.endswith("train-lr2.log")
        assert "--output_dir " + out + "-lr2" in c and not os.path.exists(os.path.join(out, "checkpoint-100"))
        assert not os.path.exists(sub + "-lr2") and not os.path.exists(sub + ".lr1-old")
        # a worse second run leaves the first in place and removes its side directory
        open(os.path.join(sub, "model.safetensors"), "w").write("run1")
        P.write_run_record(out, _rec())
        st = R.maybe_second_run(cmd, os.path.join(d, "train.log"), out, req, sub, fake_run(1.95), repl,
                                log=lambda *_: None)
        assert st == "kept" and open(os.path.join(sub, "model.safetensors")).read() == "run1"
        assert not os.path.exists(sub + "-lr2")
    _clear_env()
    print("[ok] lr2: flag off -> nothing; better second run swapped in (env, side dirs, ckpts freed); worse kept")


def test_lr_fixed_and_hooks():
    _clear_env()
    assert P.lr_fixed() is None
    os.environ["SN56_LR_FIXED"] = "1.45e-05"; assert P.lr_fixed() == 1.45e-5
    for bad in ("0", "-1", "nan", "x"):
        os.environ["SN56_LR_FIXED"] = bad; assert P.lr_fixed() is None, bad
    _clear_env()
    assert P.flag_on("SN56_LR_RETRY") and not P.flag_on("SN56_LOGIT_CAL")   # 30 Oct defaults
    assert not P.sft_features("InstructTextTask", False, None, False)["logit_cal"]
    os.environ["SN56_LOGIT_CAL"] = "1"
    assert P.sft_features("InstructTextTask", False, None, False)["logit_cal"]
    assert P.sft_features("ChatTask", False, None, False)["logit_cal"]
    assert not P.sft_features("InstructTextTask", False, None, True)["logit_cal"]
    assert not P.sft_features("InstructTextTask", True, None, False)["logit_cal"]
    _clear_env()
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    for frag in ("sn56_plan.lr_fixed()", "sn56_plan.write_run_record(", "fit_on_trainer(trainer",
                 "apply_to_submission(train_request[\"submission_dir\"]"):
        assert frag in src, frag
    assert src.index("fit_on_trainer(trainer") < src.index("run_dev_pass(") < src.index("apply_to_submission(train_request")
    tt = open(os.path.join(HERE, "text_trainer.py")).read()
    assert "maybe_second_run(" in tt and "TaskType.INSTRUCTTEXTTASK.value" in tt
    import lr_search
    assert hasattr(lr_search, "LAST_POOL") and hasattr(lr_search, "LAST_ARGMIN")
    print("[ok] lr2/logitcal: SN56_LR_FIXED parse, flags default off, scope, call order fit < dev-pass < fold")


if __name__ == "__main__":
    test_scaled_stats_brute_force()
    test_pick_recovers_temperature()
    test_fold_targets_and_check()
    test_apply_to_submission_and_fit()
    test_decide()
    test_second_run_flow()
    test_lr_fixed_and_hooks()
    print("ALL 7 SN56 LR2/LOGITCAL TESTS PASSED")
