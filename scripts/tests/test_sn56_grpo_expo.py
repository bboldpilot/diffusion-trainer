"""grpo_expo (SN56_GRPO_EXPO, default on) on CPU with tiny Qwen2 models under the trainer pins (transformers 4.51.3).

  * pair: conservative=False == the validator's normalize_rewards_and_compute_loss (verbatim upstream code through
    harness/pref-gate/boss_rule.py when the workspace is present); an exact tie is a loss in the conservative pair.
  * LoRA: an adapter submission is left untouched (before 31 Oct: loaded as base+adapter, every key ignored by
    load_state_dict(strict=False), identical sampling, tie -> "win", adapter_model.safetensors deleted).
  * unknown tensors: skipped instead of a silent no-op merge.
  * selection + write: stubbed _score -> chosen alpha, theta = B + alpha (T - B) bit-exact, KEEP_DIR, report, loss.txt;
    a tie-only grid writes nothing; end_time guard stops the grid.
  * parity sampler: model.generation_config (repetition_penalty, eos list) no longer leaks into sampling; restored after.
  * real _score inside run_expo end to end (no stub).

Run: python scripts/tests/test_sn56_grpo_expo.py   (torch, transformers==4.51.3, safetensors, tokenizers; peft for the
     LoRA demonstration, skipped without it)
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import grpo_expo as E  # noqa: E402

WORDS = [f"w{i}" for i in range(40)]
VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<x>": 3, **{w: i + 4 for i, w in enumerate(WORDS)}}
EOS, ALT_EOS = 2, 3
TASK = "t-expo"


def _tok():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="<pad>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>")


def _model(seed=0):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True, bos_token_id=1, eos_token_id=EOS, pad_token_id=0)
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


def _save(m, d):
    m.save_pretrained(d, safe_serialization=True); _tok().save_pretrained(d)
    return d


def _pair_dirs(tmp):
    """base + a full-FT 'submission' = base + noise (what the GRPO trainer leaves for < 2B models)."""
    import torch
    base = _model(0)
    ft = _model(0)
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        for p in ft.parameters():
            p.add_((0.05 * torch.randn(p.shape, generator=g)).to(p.dtype))
    return _save(base, os.path.join(tmp, "base")), _save(ft, os.path.join(tmp, "sub"))


def _dev_file(tmp, n=6):
    os.makedirs(os.path.join(tmp, "datasets"), exist_ok=True)
    rows = [{"prompt": " ".join(WORDS[(i * 3 + j) % 40] for j in range(5))} for i in range(n)]
    json.dump(rows, open(os.path.join(tmp, "datasets", f"grpo_dev_{TASK}.json"), "w"))


def _request(base, sub, end_in_s=None):
    import datetime
    r = {"submission_dir": sub, "model_path": base, "task_id": TASK, "dataset_type": {"field_prompt": "prompt"}}
    if end_in_s is not None:
        t = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=end_in_s)
        r["end_time"] = t.strftime("%Y-%m-%d %H:%M:%S")
    return r


def _digest(d):
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            h.update(f.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


class _Env:
    def __init__(self, **kv):
        self.kv = kv; self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Cwd:
    def __init__(self, d):
        self.d = d

    def __enter__(self):
        self.old = os.getcwd(); os.chdir(self.d)

    def __exit__(self, *a):
        os.chdir(self.old)


def _res(raw, kl, n=40, spread=0.01):
    return {"raw": list(raw), "kl": kl, "mean_words": 0.0,
            "per": [[r + (spread if i % 2 else -spread) for i in range(n)] for r in raw]}


def test_pair_semantics():
    w = [0.7, 0.2, 0.1]
    a = _res([-10.0, -3.0, 0.8], 0.010)
    # identical candidate: the validator's stable sort hands every tied rank reward to the repo listed second ...
    sa, sb = E._normalized_pair(a, a, w)
    assert abs(sb - sa - 0.5 * (0.7 + 0.2)) < 1e-12, (sa, sb)
    # ... the conservative pair (selection) counts the tie as lost: no gain over alpha=1 from an unchanged model
    sa, sb = E._normalized_pair(a, a, w, conservative=True)
    assert sb < sa and sb - (sa + abs(sa) * E.BOSS_MARGIN) < 0, (sa, sb)
    # significant rank win (> 2 SE) counts, a sub-2SE "win" does not
    win = _res([-9.0, -3.0, 0.8], 0.010)   # reward 0 won by 1.0 >> 2 SE, reward 1 tied -> lost
    sa, sb = E._normalized_pair(a, win, w, conservative=True)
    assert abs(sb - sa - (0.5 * 0.7 - 0.5 * 0.2)) < 1e-12, (sa, sb)
    tiny = _res([-9.999, -3.0, 0.8], 0.010, spread=5.0)
    sa, sb = E._normalized_pair(a, tiny, w, conservative=True)
    assert sb < sa, (sa, sb)
    # non-negative reward -> value/max, KL at 0.5
    nn = _res([-10.0, -3.0, 0.4], 0.030)
    sa2, sb2 = E._normalized_pair(a, nn, w)
    assert abs((sa2 - sb2) - ((0.25 * 0.7 + 0.25 * 0.2 + 0.1 * 1.0 - 0.005) - (0.75 * 0.7 + 0.75 * 0.2 + 0.1 * 0.5 - 0.015))) < 1e-12
    print("[ok] pair: tie -> validator gives it to the 2nd repo, the conservative pair counts it lost; 2SE rule; value/max")


def test_pair_matches_validator_verbatim():
    root = Path(__file__).resolve().parents[4]
    rule = root / "harness" / "pref-gate" / "boss_rule.py"
    if not rule.exists():
        print("[skip] verbatim validator check (no workspace harness here)")
        return
    sys.path.insert(0, str(rule.parent))
    try:
        import boss_rule as br
    except Exception as e:  # pragma: no cover
        print(f"[skip] verbatim validator check ({e!r})")
        return
    import random
    rnd = random.Random(7)
    n_cases = 0
    for _ in range(300):
        k = rnd.randint(1, 4)
        w = [rnd.random() for _ in range(k)]
        s = sum(w); w = [x / s for x in w]
        def side():
            return [rnd.choice([-rnd.uniform(0, 50), rnd.uniform(0, 1), 0.0, -5.0]) for _ in range(k)]
        ra, rb = side(), side()
        if rnd.random() < 0.2:
            rb = list(ra)  # exact ties
        ka, kb = rnd.uniform(0, 0.2), rnd.uniform(0, 0.2)
        names = [f"r{i}" for i in range(k)]
        ev = {"a": {"final_raw_rewards": dict(zip(names, ra)), "weights": dict(zip(names, w)), "kl_divergence": ka},
              "b": {"final_raw_rewards": dict(zip(names, rb)), "weights": dict(zip(names, w)), "kl_divergence": kb}}
        out = br.normalize_rewards_and_compute_loss(ev)
        sa, sb = E._normalized_pair({"raw": ra, "kl": ka}, {"raw": rb, "kl": kb}, w)
        assert abs(out["a"]["eval_loss"] - sa) < 1e-12 and abs(out["b"]["eval_loss"] - sb) < 1e-12, (ra, rb, w, out, sa, sb)
        n_cases += 1
    print(f"[ok] pair == upstream normalize_rewards_and_compute_loss (verbatim, {n_cases} random cases incl. ties)")


def test_eval_cap():
    with _Env(SN56_GRPO_EVAL_CAP=None):
        assert E.eval_cap(1000) == 1000
    for v, want in (("400", 400), ("5000", 1000), ("0", 1000), ("junk", 1000), ("-3", 1000)):
        with _Env(SN56_GRPO_EVAL_CAP=v):
            assert E.eval_cap(1000) == want, (v, want)
    src = (HERE / "train_grpo.py").read_text()
    assert "_cap = eval_cap(len(dev_ds))" in src and "dev_ds.select(range(_cap))" in src
    print("[ok] SN56_GRPO_EVAL_CAP: off by default, caps the trainer's dev set, junk -> off")


def test_kl_len_and_gen_config():
    from types import SimpleNamespace as NS
    m = lambda mpe: NS(config=NS(max_position_embeddings=mpe))  # noqa: E731
    assert [E._kl_max_len(m(x)) for x in (64, 513, 1023, 1024, 131072, None)] == [32, 257, 512, 512, 512, 512]
    tok = _tok()
    g = E._gen_config(tok)
    assert (g.max_new_tokens, g.do_sample, g.temperature, g.top_p, g.top_k, g.min_p, g.repetition_penalty) == \
        (256, True, 1.0, 1.0, 0, None, 1.0)
    assert (g.eos_token_id, g.pad_token_id, g.bos_token_id) == (EOS, 0, 1)
    assert E._cut_at_eos([[5, 6, EOS, 7, 0], [5, 6]], EOS) == [[5, 6, EOS], [5, 6]]
    assert E._cut_at_eos([[5, EOS]], None) == [[5, EOS]]
    print("[ok] KL length = validator rule (ceil(mpe/2) below 1024); sampler = trl 1.5.1 GRPOConfig defaults; EOS cut")


def test_parity_sampler_ignores_model_generation_config():
    import torch
    tok = _tok()
    base = _model(0)
    model = _model(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_((0.2 * torch.randn_like(p.float())).to(p.dtype))
    prompts = [" ".join(WORDS[i:i + 4]) for i in range(0, 24, 4)]
    sig = lambda cs: [float(sum(VOCAB.get(w, 0) for w in c.split())) for c in cs]  # noqa: E731
    lens = lambda cs: [float(len(c.split())) for c in cs]  # noqa: E731
    funcs = [sig, lens]

    def run(parity, rep_pen, eos):
        model.generation_config.repetition_penalty = rep_pen
        model.generation_config.eos_token_id = eos
        cfg_before = model.generation_config
        r = E._score(model, base, tok, prompts, funcs, seed=42, parity=parity)
        assert model.generation_config is cfg_before and model.generation_config.repetition_penalty == rep_pen
        return r
    on_a = run(True, 1.0, EOS)
    on_b = run(True, 5.0, [EOS, ALT_EOS])
    assert on_a["raw"] == on_b["raw"] and on_a["per"] == on_b["per"], "parity sampler still reads model.generation_config"
    off_a = run(False, 1.0, EOS)
    off_b = run(False, 5.0, [EOS, ALT_EOS])
    assert off_a["raw"] != off_b["raw"], "expected the 30 Oct sampler to pick up repetition_penalty/eos from the model"
    assert max(on_a["per"][1]) <= 256 and on_a["kl"] > 0
    same = E._score(base, base, tok, prompts, funcs, seed=42, parity=True)
    assert abs(same["kl"]) < 1e-6, same["kl"]
    print(f"[ok] parity sampler: model generation_config (rep 5.0, eos list) has no effect ({on_a['raw'][1]:.1f} words "
          f"both), restored after; 30 Oct sampler did change ({off_a['raw'][1]:.1f} vs {off_b['raw'][1]:.1f}); KL(base,base)=0")


def _stub_score(base_dir, sub_dir, best=2.0):
    """_score stand-in: reads the loaded model's alpha off one tensor; negative reward peaks at alpha=best,
    KL grows with alpha."""
    import torch
    key = "model.layers.0.mlp.down_proj.weight"
    B = E._tensors(base_dir)[key].float(); T = E._tensors(sub_dir)[key].float()
    d = T - B
    calls = []

    def score(model, base, tok, prompts, funcs, seed, parity=False):
        W = model.state_dict()[key].float().cpu()
        a = float(((W - B) * d).sum() / (d * d).sum())
        a = round(a * 4) / 4
        calls.append(a)
        return _res([-10.0 * abs(best - a), 0.5], 1e-4 * a * a)
    return score, calls


def test_run_expo_selection_and_write():
    import torch
    from safetensors.torch import load_file
    orig = E._score
    with tempfile.TemporaryDirectory() as tmp, _Cwd(tmp):
        base, sub = _pair_dirs(tmp)
        _dev_file(tmp)
        B, T = E._tensors(base), E._tensors(sub)
        keep, report = os.path.join(tmp, "keep"), os.path.join(tmp, "rep", "expo.json")
        logs = []
        E._score, calls = _stub_score(base, sub)
        try:
            with _Env(SN56_GRPO_EXPO_GRID="1.5,2,3", SN56_GRPO_EXPO_KEEP_DIR=keep, SN56_GRPO_EXPO_REPORT=report):
                E.run_expo(_request(base, sub, end_in_s=3600), [(lambda c: [0.0] * len(c), 0.8), (lambda c: [0.0] * len(c), 0.2)],
                           log=logs.append)
        finally:
            E._score = orig
        assert calls == [1.0, 1.5, 2.0, 3.0], calls
        # 1.5 and 2 both flip the rank reward, 1.5 at the lower KL; 3 ties alpha=1 -> lost
        rep = json.load(open(report))
        assert rep["chosen_alpha"] == 1.5 and rep["skipped"] is None and len(rep["alphas"]) == 4, rep
        assert rep["tensors"]["unknown"] == 0 and rep["tensors"]["moved"] == len(T), rep["tensors"]
        a3 = rep["alphas"][3]
        assert a3["pair_alpha"] < a3["pair_ref"], a3
        got = load_file(os.path.join(sub, "model.safetensors"))
        for k, t in T.items():
            want = (B[k].float() + 1.5 * (t.float() - B[k].float())).to(t.dtype)
            assert torch.equal(got[k], want), k
        kept = E._tensors(keep)
        assert all(torch.equal(kept[k], T[k]) for k in T)
        assert open(os.path.join(sub, "loss.txt")).read().endswith("expo_alpha=1.5")
        assert any("chosen alpha=1.5" in l for l in logs) and any(l.startswith("[sn56][expo] start:") for l in logs)
    print("[ok] run_expo: alpha 1.5 chosen (rank flip at the lowest KL; the 3.0 tie lost), theta = B + 1.5 (T - B) "
          "bit-exact, KEEP_DIR = alpha 1, report + loss.txt")


def test_tie_only_grid_and_time_guard_write_nothing():
    orig = E._score
    with tempfile.TemporaryDirectory() as tmp, _Cwd(tmp):
        base, sub = _pair_dirs(tmp)
        _dev_file(tmp)
        before = _digest(sub)
        report = os.path.join(tmp, "expo.json")
        rf = [(lambda c: [0.0] * len(c), 0.8), (lambda c: [0.0] * len(c), 0.2)]
        # alpha=3 samples exactly like alpha=1 on the rank reward (stub peak at 2): the 30 Oct pair took the tie
        E._score, calls = _stub_score(base, sub)
        try:
            with _Env(SN56_GRPO_EXPO_GRID="3", SN56_GRPO_EXPO_REPORT=report, SN56_GRPO_EXPO_KEEP_DIR=None):
                E.run_expo(_request(base, sub, end_in_s=3600), rf, log=lambda m: None)
            rep = json.load(open(report))
            assert calls == [1.0, 3.0] and rep["chosen_alpha"] == 1.0 and _digest(sub) == before, (calls, rep)
            # end_time 30 s away: alpha=1 is measured, the grid stops before 1.5, nothing written
            E._score, calls = _stub_score(base, sub)
            with _Env(SN56_GRPO_EXPO_GRID="1.5,2", SN56_GRPO_EXPO_REPORT=report):
                E.run_expo(_request(base, sub, end_in_s=30), rf, log=lambda m: None)
            rep = json.load(open(report))
            assert calls == [1.0] and rep["grid_stopped_before"] == 1.5 and _digest(sub) == before, (calls, rep)
        finally:
            E._score = orig
    print("[ok] tie-only grid writes nothing (30 Oct: candidate took the tie); end_time guard stops before 1.5")


def test_lora_and_unknown_tensors_are_skipped():
    import torch
    from safetensors.torch import save_file
    orig = E._score
    with tempfile.TemporaryDirectory() as tmp, _Cwd(tmp):
        base, full = _pair_dirs(tmp)
        _dev_file(tmp)
        report = os.path.join(tmp, "expo.json")
        rf = [(lambda c: [-1.0] * len(c), 1.0)]
        # (1) a real PEFT adapter (GRPO >= 2B trains LoRA)
        lora = os.path.join(tmp, "lora")
        try:
            from peft import LoraConfig, get_peft_model
            pm = get_peft_model(_model(0).float(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
            with torch.no_grad():
                for n_, p in pm.named_parameters():
                    if "lora_B" in n_:
                        p.add_(0.1)
            pm.save_pretrained(lora); _tok().save_pretrained(lora)
            cfg = json.load(open(os.path.join(lora, "adapter_config.json"))); cfg["base_model_name_or_path"] = base
            json.dump(cfg, open(os.path.join(lora, "adapter_config.json"), "w"))
            # the failure the guard removes: transformers loads base+adapter, load_state_dict ignores every file key
            from transformers import AutoModelForCausalLM
            try:
                m = AutoModelForCausalLM.from_pretrained(lora, torch_dtype=torch.bfloat16)
                T = E._tensors(lora)
                res = m.load_state_dict(T, strict=False)
                assert len(res.unexpected_keys) == len(T), (len(res.unexpected_keys), len(T))
                print(f"  [info] adapter dir loads as base+adapter; load_state_dict ignores {len(T)}/{len(T)} adapter keys "
                      "(the 30 Oct silent no-op)")
            except AssertionError:
                raise
            except Exception as e:
                print(f"  [info] from_pretrained(adapter dir) raised here ({type(e).__name__}); guard still checked")
        except ImportError:
            os.makedirs(lora, exist_ok=True); _tok().save_pretrained(lora)
            json.dump({"base_model_name_or_path": base, "peft_type": "LORA"}, open(os.path.join(lora, "adapter_config.json"), "w"))
            save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(4, 32)},
                      os.path.join(lora, "adapter_model.safetensors"))
            print("  [info] peft not installed: synthetic adapter dir")
        before = _digest(lora)
        E._score = lambda *a, **k: (_ for _ in ()).throw(AssertionError("_score must not run on a LoRA submission"))
        try:
            with _Env(SN56_GRPO_EXPO_REPORT=report, SN56_GRPO_EXPO_GRID="1.5,2,3"):
                E.run_expo(_request(base, lora, end_in_s=3600), rf, log=lambda m: None)
            rep = json.load(open(report))
            assert "LoRA" in (rep["skipped"] or "") and _digest(lora) == before
            assert os.path.exists(os.path.join(lora, "adapter_model.safetensors"))
            # (2) a loadable full model plus a stray tensor file whose keys the model does not have
            save_file({"base_model.model.lm_head.lora_A.weight": torch.ones(2, 2)}, os.path.join(full, "stray.safetensors"))
            before = _digest(full)
            with _Env(SN56_GRPO_EXPO_REPORT=report, SN56_GRPO_EXPO_GRID="1.5"):
                E.run_expo(_request(base, full, end_in_s=3600), rf, log=lambda m: None)
            rep = json.load(open(report))
            assert "do not map" in (rep["skipped"] or "") and rep["tensors"]["unknown"] == 1 and _digest(full) == before, rep
        finally:
            E._score = orig
    print("[ok] LoRA adapter submission untouched (adapter_model.safetensors kept); unknown tensor keys -> skipped")


def test_run_expo_real_score_end_to_end():
    """No stub: the real sampler + KL inside run_expo on CPU (both PARITY settings) — no exception, consistent files."""
    import torch
    from safetensors.torch import load_file
    for parity in ("1", "0"):
        with tempfile.TemporaryDirectory() as tmp, _Cwd(tmp):
            base, sub = _pair_dirs(tmp)
            _dev_file(tmp, n=4)
            B, T = E._tensors(base), E._tensors(sub)
            report = os.path.join(tmp, "expo.json")
            rf = [(lambda c: [-abs(30 - len(x.split())) for x in c], 0.8),
                  (lambda c: [len(set(x.split())) / max(1, len(x.split())) for x in c], 0.2)]
            with _Env(SN56_GRPO_EXPO_GRID="2", SN56_GRPO_EXPO_N="4", SN56_GRPO_EXPO_PARITY=parity,
                      SN56_GRPO_EXPO_REPORT=report, SN56_GRPO_EXPO_KEEP_DIR=None):
                E.run_expo(_request(base, sub, end_in_s=3600), rf, log=lambda m: None)
            rep = json.load(open(report))
            assert rep["parity"] == (parity == "1") and len(rep["alphas"]) == 2 and rep["skipped"] is None, rep
            got = load_file(os.path.join(sub, "model.safetensors"))
            k = "model.layers.1.self_attn.q_proj.weight"
            if rep["chosen_alpha"] == 1.0:
                assert torch.equal(got[k], T[k])
            else:
                assert torch.equal(got[k], (B[k].float() + 2.0 * (T[k].float() - B[k].float())).to(T[k].dtype))
            print(f"  [info] parity={parity}: alpha1 raw={['%.3f' % x for x in rep['alphas'][0]['raw']]} "
                  f"kl={rep['alphas'][0]['kl']:.4f}; alpha2 kl={rep['alphas'][1]['kl']:.4f}; chosen {rep['chosen_alpha']}")
    print("[ok] run_expo with the real sampler + KL on CPU (PARITY 1 and 0)")


if __name__ == "__main__":
    test_pair_semantics()
    test_pair_matches_validator_verbatim()
    test_eval_cap()
    test_kl_len_and_gen_config()
    test_parity_sampler_ignores_model_generation_config()
    test_run_expo_selection_and_write()
    test_tie_only_grid_and_time_guard_write_nothing()
    test_lora_and_unknown_tensors_are_skipped()
    test_run_expo_real_score_end_to_end()
    print("\nALL 9 SN56 GRPO EXPO TESTS PASSED")
