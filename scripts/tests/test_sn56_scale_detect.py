"""SN56_SCALE_REPAIR (K4-repair): blind lattice detection of weight_scaling + undo at load.

  1. Real cases (range-read samples cached by research/prep-1029-text/k4-detect, same sampling plan): 7d3d80aa
     (Qwen2.5-1.5B bf16) and bd68b438 (bloomz-560m f16) -> detected set == the seed-replicated set, s_hat ~ s;
     both clean bases -> nothing. Skipped without the cache (K4_CACHE) / ground truth (SN56_WS).
  2. Synthetic checkpoint on disk (tiny Llama, upstream selection rule + RNE(fp32(w)*fp32(s)) in bf16 and f16):
     repair_model finds the set and s from the safetensors alone, restores every weight to within one ulp,
     and leaves a clean checkpoint untouched.

Run: python scripts/tests/test_sn56_scale_detect.py
"""
import glob
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sn56_scale_detect as D  # noqa: E402

WS = Path(os.environ.get("SN56_WS", HERE.parents[3]))
CACHE = Path(os.environ.get("K4_CACHE", os.path.join(tempfile.gettempdir(), "k4", "cache")))


def _blind(raw, dt, wl):
    H = np.array([D.mant_hist(raw[n], dt) for n in wl])
    lat = D.lattice_detect(H, dt, wl)
    if lat["found"]:
        y = {n: D.logrms(raw[n], dt) for n in wl}
        lat["s_hat"] = D.resolve_alias([c["s"] for c in lat["candidates"] if c["keep"]], lat["detected"], y, wl)["s"]
    return lat


def test_real_cases_from_cache():
    gt_dir = WS / "research/prep-1029-text/k4-detect"
    cases = {"7d3d80aa": ("gradients-io-tournaments__augmented-437748741a51a240@", "Qwen__Qwen2.5-1.5B-Instruct@"),
             "bd68b438": ("gradients-io-tournaments__augmented-705e4129a1c14a7b@", "bigscience__bloomz-560m@")}
    if not CACHE.exists() or not gt_dir.exists():
        print(f"[skip] real cases (no cache at {CACHE} or ground truth at {gt_dir})")
        return
    for task, (aug_pfx, clean_pfx) in cases.items():
        gt = json.loads((gt_dir / f"gt-{task}.json").read_text())
        fa = glob.glob(str(CACHE / f"{aug_pfx}*.npz"))
        fc = glob.glob(str(CACHE / f"{clean_pfx}*.npz"))
        if not fa or not fc:
            print(f"[skip] {task} (cache files missing)")
            continue
        za, zc = np.load(fa[0]), np.load(fc[0])
        wl = [n for n in za.files if D.eligible(n)]
        t0 = time.time()
        lat = _blind({n: za[n] for n in wl}, gt["dtype"], wl)
        secs = time.time() - t0
        assert lat["found"] and set(lat["detected"]) == set(gt["selected"]), (task, len(lat["detected"]))
        assert abs(lat["s_hat"] / gt["intensity"] - 1) < 1e-4, (lat["s_hat"], gt["intensity"])
        wlc = [n for n in wl if n in zc.files]
        latc = _blind({n: zc[n] for n in wlc}, gt["dtype"], wlc)
        assert not latc["found"] or not latc["detected"], latc.get("detected")
        print(f"[ok] {task} ({gt['dtype']}): {len(lat['detected'])}/{len(wl)} == seed set, s_hat {lat['s_hat']:.7f} "
              f"(true {gt['intensity']:.7f}); clean base: nothing ({secs:.1f}s)")


def _tiny_dir(tmp, dtype, s=None, seed=0):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg).to(dtype)
    clean = {n: p.detach().clone() for n, p in m.named_parameters()}
    sel = []
    if s is not None:                          # upstream select_target_layers MULTI_LAYER + WEIGHT_SCALING
        wl = [n for n, _ in m.named_parameters() if D.eligible(n)]
        rng = random.Random(1234)
        frac = rng.uniform(0.25, 0.75)
        sel = rng.sample(wl, max(1, int(len(wl) * frac)))
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in sel:
                    p.copy_((p.float() * s).to(dtype))
    d = os.path.join(tmp, f"m-{dtype}-{s}")
    m.save_pretrained(d, safe_serialization=True)
    return d, m, clean, sel


def test_synthetic_checkpoint_repair():
    try:
        import torch
        from transformers import LlamaForCausalLM  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"[skip] synthetic ({e})")
        return
    msgs = []
    with tempfile.TemporaryDirectory() as tmp:
        for dtype, s in ((torch.bfloat16, 0.8317), (torch.float16, 1.2731)):
            d, m, clean, sel = _tiny_dir(tmp, dtype, s)
            r = D.repair_model(m, d, log=msgs.append)
            assert r["found"] and set(r["repaired"]) == set(sel), (dtype, sorted(set(sel) ^ set(r["repaired"])))
            assert abs(r["s_hat"] / s - 1) < 1e-3, (r["s_hat"], s)
            ulp = 2.0 ** -7 if dtype == torch.bfloat16 else 2.0 ** -10
            worst = 0.0
            for n, p in m.named_parameters():
                a, b = p.detach().float(), clean[n].float()
                nz = b != 0
                rel = ((a - b).abs()[nz] / b.abs()[nz]).max().item() if nz.any() else 0.0
                worst = max(worst, rel)
            assert worst <= ulp, (dtype, worst)
            # clean checkpoint of the same model: nothing detected, nothing touched
            dc, mc, cleanc, _ = _tiny_dir(tmp, dtype, None, seed=1)
            rc = D.repair_model(mc, dc, log=msgs.append)
            assert not rc["repaired"] and all(torch.equal(p, cleanc[n]) for n, p in mc.named_parameters())
            print(f"[ok] synthetic {str(dtype).split('.')[-1]} s={s}: {len(sel)} scaled found from the safetensors, "
                  f"s_hat {r['s_hat']:.5f}, restored within {worst:.2e} (<= 1 ulp {ulp:.1e}); clean -> untouched")
    assert any("weight_scaling desfeito" in x for x in msgs) and any("nada a reparar" in x for x in msgs)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 SCALE DETECT TESTS PASSED")
