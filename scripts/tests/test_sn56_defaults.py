"""Shipped defaults (30 Oct): K1 (F6 alpha=0.5 + vali dev metric) and K4-repair on InstructText / ChatTask
full fine-tunes, GRPO expo on; the validator passes no SN56_* env to the container, so the defaults ARE the
submission's behaviour.

Run: python scripts/tests/test_sn56_defaults.py   (torch + safetensors for the expo writer test)
"""
import ast
import datetime
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import sn56_plan as P  # noqa: E402


def _clear_env():
    for k in list(P.FLAGS) + [P.MASTER]:
        os.environ.pop(k, None)


def test_flag_defaults():
    _clear_env()
    assert P.flag_on("SN56_PER_EXAMPLE_LOSS")
    assert P.flag_float("SN56_PER_EXAMPLE_ALPHA") == 0.5
    assert P.flag_str("SN56_DEV_METRIC") == "vali"
    assert P.flag_on("SN56_SCALE_REPAIR") and P.flag_on("SN56_K1_CHAT") and P.flag_on("SN56_GRPO_EXPO")
    # measured worse / no effect: stay off
    assert not P.flag_on("SN56_GAIN") and P.flag_str("SN56_DPO_SELECT") == "mean"
    # per-flag A/B still works
    os.environ["SN56_PER_EXAMPLE_ALPHA"] = "1"
    os.environ["SN56_DEV_METRIC"] = "stock"
    os.environ["SN56_GRPO_EXPO"] = "0"
    assert P.flag_float("SN56_PER_EXAMPLE_ALPHA") == 1.0 and P.flag_str("SN56_DEV_METRIC") == "stock"
    assert not P.flag_on("SN56_GRPO_EXPO")
    _clear_env()
    print("[ok] defaults: F6 on, alpha 0.5, dev vali, scale repair on, K1 chat on, GRPO expo on; gain off")


def test_feature_matrix():
    _clear_env()
    f = lambda *a: P.sft_features(*a)  # noqa: E731
    it = f("InstructTextTask", False, None, False)
    assert it == {"sn56_on": True, "sft_full": True, "per_example": True, "alpha": 0.5, "dev_vali": True,
                  "gain": False, "scale_repair": True, "logit_cal": False, "fp32_norms": False}, it
    ch = f("ChatTask", False, None, False)
    assert ch["sn56_on"] and ch["per_example"] and ch["scale_repair"] and ch["dev_vali"] and not ch["gain"], ch
    for lora, ds in ((True, None), (False, "ds_config/zero3.json")):
        for tt in ("InstructTextTask", "ChatTask"):
            x = f(tt, lora, ds, False)
            assert not x["sn56_on"] and not x["per_example"] and not x["scale_repair"] and x["dev_vali"], (tt, lora, ds, x)
    kl = f("InstructTextTask", False, None, True)
    assert kl["sn56_on"] and not kl["per_example"] and not kl["scale_repair"] and kl["dev_vali"], kl
    for tt in ("DpoTask", "GrpoTask"):
        x = f(tt, False, None, False)
        assert not any(x[k] for k in ("sn56_on", "per_example", "scale_repair", "dev_vali", "gain")), (tt, x)
    os.environ["SN56_K1_CHAT"] = "0"
    ch0 = f("ChatTask", False, None, False)
    assert not ch0["per_example"] and not ch0["scale_repair"] and ch0["dev_vali"], ch0
    assert f("InstructTextTask", False, None, False)["per_example"]
    _clear_env()
    os.environ["SN56_TEXT_FIX"] = "0"
    for tt in ("InstructTextTask", "ChatTask"):
        x = f(tt, False, None, False)
        assert not x["per_example"] and not x["scale_repair"] and not x["dev_vali"] and x["alpha"] == 1.0, (tt, x)
    _clear_env()
    print("[ok] matrix: InstructText+Chat full FT get F6/repair, LoRA/DS/KL do not, dev vali on every SFT run, "
          "SN56_K1_CHAT=0 and SN56_TEXT_FIX=0 roll back")


def test_trainers_use_the_gate():
    """The three entry points read the gates from sn56_plan, not from raw env."""
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    assert "sn56_plan.sft_features(" in src
    tree = ast.parse(src)
    assigned = {t.id: ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    for var, key in (("_f_per_example", "per_example"), ("_f6_alpha", "alpha"), ("_f_dev_vali", "dev_vali"),
                     ("_f_scale_repair", "scale_repair"), ("_f_gain", "gain"), ("_sn56_on", "sn56_on")):
        assert assigned.get(var) == f"_feat['{key}']", (var, assigned.get(var))
    for name in ("train_grpo.py", "grpo_config.py"):
        s = open(os.path.join(HERE, name)).read()
        assert 'flag_on("SN56_GRPO_EXPO")' in s and 'os.environ.get("SN56_GRPO_EXPO") == "1"' not in s, name
    print("[ok] train_instruct uses sft_features; GRPO expo gated by sn56_plan in train_grpo + grpo_config")


def test_expo_guards():
    import torch
    from safetensors.torch import load_file, save_file
    import grpo_expo as E
    now = datetime.datetime.now(datetime.timezone.utc)
    fmt = lambda d: (now + d).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    assert 590 < E._seconds_left({"end_time": fmt(datetime.timedelta(minutes=10))}) <= 600
    assert E._seconds_left({"end_time": fmt(-datetime.timedelta(minutes=1))}) < 0
    assert E._seconds_left({}) == float("inf") and E._seconds_left({"end_time": "garbage"}) == float("inf")
    with tempfile.TemporaryDirectory() as d:
        # sharded submission -> single model.safetensors, shards + index gone, config untouched
        a = {"w": torch.ones(3, 3)}
        save_file(a, os.path.join(d, "model-00001-of-00002.safetensors"))
        save_file({"v": torch.zeros(2)}, os.path.join(d, "model-00002-of-00002.safetensors"))
        open(os.path.join(d, "model.safetensors.index.json"), "w").write("{}")
        open(os.path.join(d, "config.json"), "w").write("{}")
        assert abs(E._weights_gb(d) - sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                                           if f.endswith(".safetensors")) / 1e9) < 1e-12
        E._write_winner(d, {"w": torch.full((3, 3), 2.0), "v": torch.ones(2)})
        assert sorted(os.listdir(d)) == ["config.json", "model.safetensors"], os.listdir(d)
        t = load_file(os.path.join(d, "model.safetensors"))
        assert float(t["w"][0, 0]) == 2.0 and float(t["v"][1]) == 1.0
    print("[ok] expo: end_time parse, size probe, atomic winner write (shards + index replaced)")


def test_expo_inputs_bloom():
    """grpo_expo feeds the model only input_ids + attention_mask: BloomForCausalLM raises on the token_type_ids a
    tokenizer without model_input_names emits (30 Oct, bloomz verify crash); generate works on the filtered dict."""
    import torch
    from transformers import BloomConfig, BloomForCausalLM
    import grpo_expo as E
    torch.manual_seed(0)
    m = BloomForCausalLM(BloomConfig(vocab_size=40, hidden_size=16, n_layer=2, n_head=2)).eval()
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    enc = {"input_ids": ids, "token_type_ids": torch.zeros_like(ids), "attention_mask": torch.ones_like(ids)}
    try:
        m(**enc)
        raise AssertionError("Bloom accepted token_type_ids: the regression this test guards is gone, re-check")
    except ValueError as e:
        assert "token_type_ids" in str(e), e
    clean = E._lm_inputs(enc)
    assert sorted(clean) == ["attention_mask", "input_ids"]
    with torch.no_grad():
        assert m(**clean).logits.shape == (1, 5, 40)
        out = m.generate(**clean, do_sample=False, max_new_tokens=3, pad_token_id=0)
    assert out.shape[1] == 8
    assert sorted(E._lm_inputs({"input_ids": ids})) == ["input_ids"]
    print("[ok] expo inputs: token_type_ids dropped, Bloom forward + generate run")


if __name__ == "__main__":
    test_flag_defaults()
    test_feature_matrix()
    test_trainers_use_the_gate()
    test_expo_guards()
    test_expo_inputs_bloom()
    print("ALL OK")
