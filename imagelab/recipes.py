"""Per-family training defaults. Hyperparameters, nothing else; every number here is a knob the
measurement harness varies, and the measured defaults get written back here.

ideogram4 = the 24 Sep B2 run on c7460d80 (0.75 h, 4-member cross-fit soup, seed 0). The other families'
values are starting points that have not been measured with this trainer yet."""
from __future__ import annotations

COMMON = {
    "holdout_frac": 0.1, "holdout_min": 3, "holdout_noises": 1, "holdout_noise_batch": 2,
    "eval_every": 250, "lr_final_frac": 0.1, "warmup_steps": 20, "ema": 0.99, "delta_average": 2,
    "prompt_mode": "balanced", "select": "mean", "band_min": 0, "band_max": 16, "band_weights": None,
    "band_repeat": None,   # sampling repeat, e.g. "0,1,14,15:3" (the champion's edge_band_repeat); unmeasured
    "ckpt": True, "ckpt_stride": 1, "up_lr_mult": 1.0, "weight_decay": 0.0, "grad_clip": 1.0,
    "fp8_te": False,   # evaluator-exact fp8 text encoder rebuild (bit-exact vs Comfy-Org files); on per family where measured
    "members": 1, "seed": 0,
}

FAMILY = {
    # measured 24 Sep 2026 (results/image-0924-r1field, results/image-0924-families): members = soup width that
    # won the evaluator's own scoring on the published task of each family; 1 = single member to the deadline.
    "ideogram4": {"rank": 16, "lr": 2e-4, "ckpt": False, "eval_every": 200, "members": 4, "export": "fp8aware", "fp8_te": True,
                  "include": r"layers\.\d+\.(attention|adaln_modulation|feed_forward)"},     # c7460d80: soup4 1st/18; fp8aware export −0.147% (A/B 24 Sep, exact=interval=table)
    "krea2": {"rank": 16, "lr": 2e-4, "include": None, "eval_every": 200, "members": 2, "fp8_te": True},   # 5c1de437: soup2 +0.09% of champion; 13e44b2d: soup2 +0.09%, + evaluator-exact fp8 TE −0.21% (beats champion)
    "z-image": {"rank": 16, "lr": 2e-4, "include": None, "eval_every": 500, "members": 1},      # 8fd01555: single +0.09% of winner, soup4 +0.66%; 244984ee: eval 500 −0.12% vs eval 200 (+20% steps), lr 3e-4 no effect
    "qwen-image": {"rank": 8, "lr": 1e-4, "members": 1, "eval_every": 300,                       # 3b6a0a8e (1.5 h): single +0.10% of champion (parity); champion's eval cadence
                   "include": r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out)"},
    "flux": {"rank": 16, "lr": 3e-4, "ckpt": False, "members": 1,                                # PixelWave 87f69ea0 full: −2.5% vs best published (best ema@500, overfits late; ship=best); peak VRAM 66 GB
             "include": r"(double_blocks|single_blocks)\.\d+\.(img_attn|txt_attn|img_mlp|txt_mlp|linear1|linear2)"},
}


def recipe(family: str, overrides: dict | None = None) -> dict:
    out = dict(COMMON)
    out.update(FAMILY[family])
    for key, value in (overrides or {}).items():
        if value is not None:
            out[key] = value
    return out


def recipe_argv(spec: dict) -> list[str]:
    """Turn a recipe dict into imagelab.train CLI flags."""
    argv: list[str] = []
    for key, value in spec.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if key == "ckpt":
                argv.append("--ckpt" if value else "--no-ckpt")
            elif value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            argv += [flag, ",".join(str(v) for v in value)]
            continue
        argv += [flag, str(value)]
    return argv
