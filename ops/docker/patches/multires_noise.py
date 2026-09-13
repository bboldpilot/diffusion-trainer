# ---------------------------------------------------------------------------
# ORIGIN: rivals/god-image-tourn-1dccbd24755b58a3-20260907-position-1/ops/docker/patches/multires_noise.py
#         (2026-09-07 image tournament winner, hotkey 5EhyCWPu). Copied VERBATIM on
#         2026-09-08 for the I6 winner-recipe port (PLAN-IMAGE-0908 v2); everything
#         below this header is the unmodified original.
# LICENSE: the source repo ships Apache-2.0 (LICENSE.md) plus the Gradients-on-Demand
#         NOTICE (Copyright 2025 Grads LLC; attribution to Gradients.io required) -
#         see this repo's LICENSE.md / NOTICE.
# GATE:   config key train.multires_noise_iterations (TrainConfig default 0 -> get_noise block skipped, see PROC_NEW `if _mrn_iters > 0`)
#         -> applying at build time changes nothing until IMG_RECIPE=winner sets it
#            (ops/docker/scripts/image_toolkit_entrypoint.py).
# ---------------------------------------------------------------------------
"""Build-time patch: multi-resolution ("pyramid") training noise for ai-toolkit.

WHY: the miner that placed rank-1 on ALL THREE round-1 krea2 tasks of the
2026-08-10 tournament (5HKEAZxF: 0.05513 / 0.05353 / 0.03343) published its
training config, and after a field-by-field diff against ours the ONLY training
mechanism it has that we lack is:
    multires_noise_iterations: 6
    multires_noise_discount:  0.3
Everything else already matches our template byte for byte (timestep_type
linear, loss_type mse, lr 1e-4, content_or_style balanced, use_ema false,
train_text_encoder false, do_differential_guidance true,
differential_guidance_scale 2, adamw8bit, weight_decay 1e-4, lora 32/32,
resolution buckets, save_every 200). Its lower step count (1333 @0.75h,
1833 @1.0h vs our 2000) is a CONSEQUENCE of this feature's per-step cost
(~2.0 s/step vs our ~1.33), not a separate lever.

WHAT: instead of pure i.i.d. Gaussian latent noise, add progressively
lower-resolution Gaussian components, each scaled by discount**i, then rescale
to unit variance. Low-frequency structure in the noise means the model sees --
and learns to remove -- correlated large-scale error, not only per-pixel error.

REFERENCE IMPLEMENTATION: taken verbatim (semantics-wise) from the kohya
`pyramid_noise_like` already vendored in THIS repo at
scripts/sd-script/library/custom_train_functions.py:458, which is the code the
`multires_noise_iterations` / `multires_noise_discount` option names come from.
Using the vendored reference removes any guesswork about what those two config
values mean.

INERT UNLESS SELECTED: `multires_noise_iterations` defaults to 0 and the hook
returns immediately at 0, so every existing recipe (i4, qwen, z, flux) trains
byte-identically. The BUILD fails loudly if either upstream anchor drifts.
"""

CFG = "/app/ai-toolkit/toolkit/config_modules.py"
PROC = "/app/ai-toolkit/jobs/process/BaseSDTrainProcess.py"

CFG_OLD = "        self.blended_blur_noise = kwargs.get('blended_blur_noise', False)"
CFG_NEW = """        self.blended_blur_noise = kwargs.get('blended_blur_noise', False)
        # SN56: multi-resolution (pyramid) training noise. 0 = off = upstream behaviour.
        self.multires_noise_iterations = kwargs.get('multires_noise_iterations', 0)
        self.multires_noise_discount = kwargs.get('multires_noise_discount', 0.3)"""

PROC_OLD = """        if self.train_config.blended_blur_noise:
            noise = get_blended_blur_noise(
                latents, noise, timestep
            )

        return noise"""

PROC_NEW = """        if self.train_config.blended_blur_noise:
            noise = get_blended_blur_noise(
                latents, noise, timestep
            )

        # SN56 multi-resolution (pyramid) noise. Semantics follow the kohya
        # `pyramid_noise_like` reference vendored in this repo at
        # scripts/sd-script/library/custom_train_functions.py.
        _mrn_iters = int(getattr(self.train_config, 'multires_noise_iterations', 0) or 0)
        if _mrn_iters > 0:
            _mrn_discount = float(getattr(self.train_config, 'multires_noise_discount', 0.3))
            _b, _c, _w, _h = noise.shape
            _up = torch.nn.Upsample(size=(_w, _h), mode='bilinear').to(noise.device)
            for _i in range(_mrn_iters):
                _r = random.random() * 2 + 2  # rather than always going 2x
                _wn, _hn = max(1, int(_w / (_r ** _i))), max(1, int(_h / (_r ** _i)))
                noise = noise + _up(
                    torch.randn(_b, _c, _wn, _hn, device=noise.device, dtype=noise.dtype)
                ) * (_mrn_discount ** _i)
                if _wn == 1 or _hn == 1:
                    break  # lowest resolution is 1x1
            noise = noise / noise.std()  # back to roughly unit variance

        return noise"""


def _patch(path, old, new, tag):
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if new in src:
        print("[mrn] %s already applied" % tag)
        return
    if old not in src:
        raise SystemExit(
            "[mrn] FATAL: expected upstream block not found in %s (%s) - "
            "upstream drifted, refusing to build unpatched image" % (path, tag)
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new, 1))
    print("[mrn] applied: %s" % tag)


_patch(CFG, CFG_OLD, CFG_NEW, "TrainConfig multires_noise_* fields")
_patch(PROC, PROC_OLD, PROC_NEW, "pyramid noise in get_noise")
