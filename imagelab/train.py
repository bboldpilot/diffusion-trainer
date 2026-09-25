"""imagelab.train — train one LoRA against the tournament objective on ComfyUI's stack.

  python -m imagelab.train --family ideogram4 --model-dir /cache/models/<repo> --hf-cache /cache/hf_cache \
      --baked-dir /opt/imagelab/assets --comfy-root /opt/ComfyUI --dataset /cache/datasets/<task>_tourn.zip \
      --out /tmp/imagelab-out --deadline-ts <epoch>   [recipe flags]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="imagelab.train")
    p.add_argument("--family", required=True, choices=["krea2", "ideogram4", "qwen-image", "z-image", "flux"])
    p.add_argument("--model-dir", required=True)
    p.add_argument("--hf-cache", default="/cache/hf_cache")
    p.add_argument("--baked-dir", default="/opt/imagelab/assets")
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--trigger-word", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--deadline-ts", type=float, default=None)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--save-reserve", type=float, default=90.0)
    # adapter
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="default = rank (scale 1)")
    p.add_argument("--include", default=None, help="regex on Linear module names under diffusion_model")
    p.add_argument("--exclude", default=None)
    p.add_argument("--no-fp8-roundtrip", action="store_true", help="ideogram4: keep exact bf16 dequantised weights")
    p.add_argument("--fp8-te", action="store_true", help="re-derive the evaluator's fp8-scaled text encoder from the bf16 weights (ideogram4/krea2/qwen-image)")
    # optimisation
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-final-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--up-lr-mult", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--band-min", type=int, default=0)
    p.add_argument("--band-max", type=int, default=16)
    p.add_argument("--band-weights", default=None, help="16 comma-separated multipliers (mean 1)")
    p.add_argument("--band-repeat", default=None,
                   help='sampling repeat, e.g. "0,1,14,15:3" (groups joined by ";"): those bands appear that many times per case cycle; loss unweighted')
    p.add_argument("--prompt-mode", default="balanced", choices=["balanced", "random"])
    p.add_argument("--empty-prompt-frac", type=float, default=0.5)
    # selection
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--holdout-min", type=int, default=3)
    p.add_argument("--holdout-noises", type=int, default=1)
    p.add_argument("--holdout-noise-batch", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--delta-average", type=int, default=2)
    p.add_argument("--select", default="mean", choices=["mean", "p75", "worst"])
    p.add_argument("--ship", default="best", choices=["best", "tail"])
    p.add_argument("--tail-frac", type=float, default=0.3)
    p.add_argument("--members", type=int, default=1, help=">1: cross-fit soup over rotating holdout folds")
    p.add_argument("--member-steps", type=int, default=0, help="0 = time-sliced members; else stop each at this step count")
    p.add_argument("--soup-rule", default="none", choices=["none", "plateau"],
                   help="plateau: member 0 probes on the single-run schedule, then soup (<= --members) or single")
    p.add_argument("--plateau-frac", type=float, default=0.7, help="plateau: best step <= this fraction of the steps so far")
    p.add_argument("--plateau-patience", type=int, default=2, help="plateau: evaluation rounds without a gain")
    p.add_argument("--plateau-tol", type=float, default=5e-4, help="plateau: relative gain that still counts as none")
    p.add_argument("--member-mult", type=float, default=1.0, help="plateau soup: members train member 0's best step * this")
    p.add_argument("--continue-data", default="fold", choices=["fold", "all"],
                   help="plateau single branch: keep member 0's holdout (fold) or add it to training and ship the tail (all)")
    # export
    p.add_argument("--export", default="plain", choices=["plain", "fp8aware"],
                   help="fp8aware: pin the evaluator's fp8 requantisation scale to a power of two (ideogram4; see adapter.py)")
    p.add_argument("--export-base", default=None,
                   help="fp8aware: evaluator base as a top-K table (.json from fp8aware-table.py) or the evaluator's own "
                        "per-tensor fp8 .safetensors; default = interval plans from the trainer's fp8 shard")
    p.add_argument("--export-max-rows", type=int, default=8, help="fp8aware: most rows moved per layer (2 extra ranks each)")
    # memory / misc
    p.add_argument("--ckpt", dest="ckpt", action="store_true")
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    p.set_defaults(ckpt=True)
    p.add_argument("--ckpt-stride", type=int, default=1)
    p.add_argument("--text-enc-dtype", default=None, choices=[None, "fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if not 0 <= args.band_min < args.band_max <= 16:
        raise SystemExit("require 0 <= band-min < band-max <= 16")
    if args.deadline_ts is None:
        if args.minutes is None:
            raise SystemExit("--deadline-ts or --minutes required")
        args.deadline_ts = time.time() + args.minutes * 60
    if args.alpha is None:
        args.alpha = float(args.rank)
    if args.band_weights:
        weights = [float(v) for v in args.band_weights.split(",")]
        if len(weights) != 16:
            raise SystemExit("--band-weights needs 16 values")
        args.band_weights = weights
    if args.band_repeat:
        args.band_repeat = parse_band_repeat(args.band_repeat)
    return args


def parse_band_repeat(text: str) -> dict[int, int]:
    """'0,1,14,15:3' -> {0: 3, 1: 3, 14: 3, 15: 3}; several groups joined by ';'."""
    out: dict[int, int] = {}
    for group in filter(None, (part.strip() for part in text.split(";"))):
        bands, sep, copies = group.partition(":")
        if not sep or not bands.strip():
            raise SystemExit(f"--band-repeat group {group!r} is not 'band,band,...:copies'")
        count = int(copies)
        if count < 1:
            raise SystemExit("--band-repeat copies must be >= 1")
        for band in (int(value) for value in bands.split(",")):
            if not 0 <= band < 16:
                raise SystemExit(f"--band-repeat band {band} outside 0..15")
            out[band] = count
    return out


def fp8aware_exporter(args, family: str, names: list[str], device, log):
    """Plans for the fp8-aware export (adapter.py), built once before training; None keeps the plain export."""
    import torch

    from imagelab.adapter import fp8_plans_from_files, fp8_plans_from_table, fp8aware_export
    from imagelab.families import _shard_files

    started = time.time()
    shards = _shard_files(os.path.join(args.model_dir, "transformer")) if family == "ideogram4" else []
    plans, notes = {}, {}
    try:
        if args.export_base and args.export_base.endswith(".json"):
            with open(args.export_base) as handle:
                plans, notes = fp8_plans_from_table(json.load(handle), names, shards)
        elif args.export_base:
            plans = fp8_plans_from_files([args.export_base], names, device=device)
        missing = [name for name in names if name not in plans]
        if missing and shards:
            plans.update(fp8_plans_from_files(shards, missing, device=device))
    except Exception as error:  # the plain export is always a valid artefact
        log(f"fp8aware export disabled: {type(error).__name__}: {error}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    modes = sorted({plan.mode for plan in plans.values()})
    log(f"fp8aware export: {len(plans)}/{len(names)} layers planned {modes} in {time.time() - started:.0f}s"
        + (f"; table rejected for {len(notes)} layer(s)" if notes else ""))
    if not plans:
        return None
    max_rows = args.export_max_rows
    return lambda tensors: fp8aware_export(tensors, plans, device=device, max_rows=max_rows)


def main(argv=None) -> int:
    args = parse(argv)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=1)

    import torch

    from imagelab import boot
    from imagelab.adapter import AdapterBank, checkpoint_blocks, linear_targets
    from imagelab.data import load_items, split_holdout
    from imagelab.families import FAMILIES, Assets
    from imagelab.loop import HijackSampler, LoopConfig, Trainer
    from imagelab.replica import HoldoutReplica

    family = args.family
    text_enc_dtype = args.text_enc_dtype or FAMILIES[family].text_enc_dtype
    mm = boot.boot(args.comfy_root, text_enc_dtype=text_enc_dtype)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True))
    log(f"family={family} deadline in {(args.deadline_ts - time.time()) / 60:.1f} min device={mm.get_torch_device()}")

    items = load_items(args.dataset, args.trigger_word)
    split_holdout(items, args.holdout_frac, args.holdout_min)
    holdout = [item for item in items if item.holdout]
    log(f"dataset: {len(items)} images, holdout {len(holdout)}")

    assets = Assets(family, args.model_dir, args.hf_cache, args.baked_dir, fp8_roundtrip=not args.no_fp8_roundtrip,
                    fp8_text_encoder=args.fp8_te)

    started = time.time()
    kind, payload = assets.vae()
    encoder = boot.LatentEncoder(kind, payload)
    for item in items:
        item.latent = encoder.encode(item.image)
        item.image = None
    encoder.free()
    log(f"latents encoded in {time.time() - started:.0f}s; shape {tuple(items[0].latent.shape)}; vae={kind}"
        + (f" {payload}" if isinstance(payload, str) else ""))

    started = time.time()
    clip = boot.load_clip(assets.text_encoder_states(), family)
    conds = {}
    for prompt in sorted({item.caption for item in items} | {""}):
        conds[prompt] = boot.encode_prompt(clip, prompt, family)
    del clip
    boot.free_gpu()
    log(f"{len(conds)} prompts encoded in {time.time() - started:.0f}s (text encoder: {assets.text_encoder_mode})")

    started = time.time()
    model = boot.load_diffusion(assets.diffusion_state(), family)
    assets.release()
    gc.collect()
    for item in items:
        item.scaled = model.model.process_latent_in(item.latent.clone()).float().cpu()
        item.latent = None
    params = sum(p.numel() for p in model.model.diffusion_model.parameters())
    log(f"diffusion model loaded in {time.time() - started:.0f}s: {type(model.model).__name__} {params / 1e9:.2f}B")

    targets = linear_targets(model.model.diffusion_model, args.include, args.exclude)
    bank = AdapterBank(targets, args.rank, args.alpha, model.load_device, args.seed)
    bank.attach(model)
    log(f"lora: {len(targets)} linear layers, rank {args.rank}, {bank.count() / 1e6:.1f}M params; "
        f"first={targets[0][0]} last={targets[-1][0]}")
    if args.ckpt:
        log("checkpointed blocks:", checkpoint_blocks(model.model.diffusion_model, stride=args.ckpt_stride))

    replica_items = holdout or items
    if not holdout:
        log("WARNING: no holdout image; selecting on the training images themselves")
    replica = HoldoutReplica(replica_items, noises=args.holdout_noises, noise_batch=args.holdout_noise_batch)
    cfg = LoopConfig(deadline_ts=args.deadline_ts, lr=args.lr, lr_final_frac=args.lr_final_frac,
                     warmup_steps=args.warmup_steps, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                     up_lr_mult=args.up_lr_mult, ema=args.ema, eval_every=args.eval_every,
                     delta_average=args.delta_average, band_min=args.band_min, band_max=args.band_max,
                     band_weights=args.band_weights, band_repeat=args.band_repeat, prompt_mode=args.prompt_mode,
                     empty_prompt_frac=args.empty_prompt_frac, select=args.select, ship=args.ship, tail_frac=args.tail_frac,
                     members=args.members, member_steps=args.member_steps, holdout_noises=args.holdout_noises,
                     holdout_noise_batch=args.holdout_noise_batch, soup_rule=args.soup_rule,
                     plateau_frac=args.plateau_frac, plateau_patience=args.plateau_patience,
                     plateau_tol=args.plateau_tol, member_mult=args.member_mult, continue_data=args.continue_data,
                     save_reserve_s=args.save_reserve, seed=args.seed, log=not args.quiet)
    trainer = Trainer(cfg, model, bank, items, conds, replica, args.out, family)
    if args.export == "fp8aware":
        trainer.export_transform = fp8aware_exporter(args, family, [name for name, _ in targets], model.load_device, log)

    guider = boot.make_guider(model, conds[""])
    first = items[0].scaled
    guider.sample(torch.zeros_like(first), first, HijackSampler(trainer), torch.tensor([0.5, 0.0]),
                  disable_pbar=True, seed=42)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
