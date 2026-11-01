"""SN56_DPO_SCALE_CAL=1: margin-scale calibration of a DPO fine-tune on the dev set.

The validator scores DPO by the held-out sigmoid loss -logsigmoid(0.1 * h) with h the
summed policy-vs-base log-ratio margin (reference = the base). The loss is unbounded
above, so an over-confident model (margins too large, wrong pairs confidently wrong)
pays in the tail even when its ranking is good. 29 Oct 198f2175 (dev dumps, test
vectors): our run's pair losses invert to margins z_i = beta*h_i exactly
(z = -log(expm1(loss))); mean softplus(-a*z) is minimised near a = 0.55 on dev
(0.6448 -> 0.6135) and a = 0.60 on test (0.6438 -> 0.6268), the boss sits at a = 0.90
(calibrated), the 26 Oct challenger at 0.75. Under the linear-margin assumption the
scale is a one-parameter fit.

This module does it in weight space, measured rather than assumed:
theta_a = ref + a * (theta - ref), every dev pair re-scored at each a with the trainer's
own DPO loss (beta, reference, sigmoid exactly as evaluate()), a = argmin of the dev
mean (1.0 always a candidate, applied only if it gains SN56_DPO_SCALE_MIN_GAIN nats).
calibrate() runs on the model the submission holds (after the soup, before the
dev-pass: the dev set is still unseen) and restores theta; apply_and_save() applies
the picked a to the final weights (after the dev-pass when it ran) and writes the
submission with dev_pass's atomic weights-only saver.

Scope: one process, full fine-tune, a reference model held by the trainer (TRL's
default deepcopy of the initial model). DDP / PEFT / precomputed reference log-probs
return None (logged) and leave the submission untouched.
"""
import json
import os
import shutil
import time

import numpy as np
import torch


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def supported(trainer, world_size: int = 1) -> tuple:
    """(ok, reason)."""
    if world_size and int(world_size) > 1:
        return False, f"world_size={world_size}"
    if torch.distributed.is_available() and torch.distributed.is_initialized() \
            and torch.distributed.get_world_size() > 1:
        return False, "distributed"
    if getattr(trainer, "is_peft_model", False) or getattr(trainer, "peft_config", None) is not None:
        return False, "peft"
    if getattr(trainer, "ref_model", None) is None:
        return False, "sem ref_model"
    if getattr(getattr(trainer, "args", None), "precompute_ref_log_probs", False):
        return False, "precompute_ref_log_probs"
    return True, ""


def paired_params(model, ref_model) -> list:
    """[(policy_param, ref_param)] matched by name and shape (tied weights appear once)."""
    ref = dict(_unwrap(ref_model).named_parameters())
    out = []
    for name, p in _unwrap(model).named_parameters():
        r = ref.get(name)
        if r is not None and r.shape == p.shape:
            out.append((p, r))
    return out


@torch.no_grad()
def snapshot(pairs) -> list:
    return [p.detach().to("cpu", copy=True) for p, _ in pairs]


@torch.no_grad()
def set_scaled(pairs, theta, alpha: float) -> None:
    """policy <- ref + alpha * (theta - ref), fp32 math, cast to the param dtype."""
    for (p, r), t in zip(pairs, theta):
        rf = r.detach().float()
        tf = t.to(device=p.device, dtype=torch.float32, non_blocking=True)
        p.copy_((rf + float(alpha) * (tf - rf)).to(p.dtype))


@torch.no_grad()
def scale_in_place(pairs, alpha: float) -> None:
    """policy <- ref + alpha * (policy - ref)."""
    for p, r in pairs:
        rf = r.detach().float()
        p.copy_((rf + float(alpha) * (p.detach().float() - rf)).to(p.dtype))


def dev_pair_losses(trainer, model=None, loader=None) -> np.ndarray:
    """Per-pair DPO losses on the trainer's eval (dev) set for the current weights.

    Captures dpo_loss()'s per-example output inside the trainer's own
    get_batch_loss_metrics (no evaluate(): no callbacks, no checkpoint, no metric store)."""
    model = trainer.model if model is None else model
    loader = trainer.get_eval_dataloader() if loader is None else loader
    captured = []
    orig = trainer.dpo_loss

    def _capture(*a, **kw):
        out = orig(*a, **kw)
        captured.append(out[0].detach().float().reshape(-1).cpu())
        return out

    trainer.dpo_loss = _capture        # instance attribute shadows the method
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = trainer._prepare_inputs(batch)
                with trainer.compute_loss_context_manager():
                    trainer.get_batch_loss_metrics(model, inputs, train_eval="eval")
    finally:
        del trainer.dpo_loss
        if was_training:
            model.train()
    return torch.cat(captured).numpy().astype(np.float64) if captured else np.zeros(0)


def pick_alpha(losses_by_alpha: dict, min_gain: float) -> float:
    """argmin of the dev mean; 1.0 unless the best beats it by more than min_gain (pure)."""
    finite = {a: v for a, v in losses_by_alpha.items() if v == v and np.isfinite(v)}
    if 1.0 not in finite:
        return 1.0
    best = min(finite, key=lambda a: (finite[a], abs(a - 1.0)))
    return float(best) if finite[1.0] - finite[best] > float(min_gain) else 1.0


def calibrate(trainer, grid, min_gain: float = 0.001, log=print, report_path: str = "",
              refine: bool = True, score_fn=None) -> dict:
    """Score dev at every alpha in grid (+ the two midpoints around the grid argmin), restore
    theta, return {"alpha", "losses", ...}. Never leaves the model scaled. score_fn(per-pair
    vector) -> lower-is-better scalar replaces the mean (SN56_DPO_SELECT=opp: the opponent-gate
    score on the gate split, so the scale is one more selection candidate, not a mean fit)."""
    model = trainer.model
    pairs = paired_params(model, trainer.ref_model)
    if not pairs:
        raise RuntimeError("no policy/reference parameter pairs")
    theta = snapshot(pairs)
    loader = trainer.get_eval_dataloader()
    t0 = time.perf_counter()
    res = {}
    stats = {}

    def _score(a):
        a = round(float(a), 4)
        if a in res:
            return
        set_scaled(pairs, theta, a)
        v = dev_pair_losses(trainer, model, loader)
        fin = v[np.isfinite(v)]
        if score_fn is not None:
            sc = float(score_fn(v))
            res[a] = sc if sc == sc else float("inf")
        else:
            res[a] = float(fin.mean()) if fin.size else float("inf")
        stats[a] = {"n": int(v.size), "p95": float(np.percentile(fin, 95)) if fin.size else float("nan"),
                    "win": float(np.mean(fin < np.log(2.0))) if fin.size else float("nan")}

    try:
        grid = sorted(set(float(a) for a in grid) | {1.0})
        for a in grid:
            _score(a)
        if refine:
            best = min(res, key=res.get)
            i = grid.index(best) if best in grid else None
            if i is not None:
                for j in (i - 1, i + 1):
                    if 0 <= j < len(grid):
                        _score((best + grid[j]) / 2.0)
    finally:
        # bit-exact restore (ref + 1.0*(theta-ref) need not round back to theta): copy theta
        with torch.no_grad():
            for (p, _), t in zip(pairs, theta):
                p.copy_(t.to(device=p.device, dtype=p.dtype))
    alpha = pick_alpha(res, min_gain)
    out = {"alpha": alpha, "loss_at_1": res.get(1.0), "loss_at_alpha": res.get(round(alpha, 4)),
           "losses": {str(k): v for k, v in sorted(res.items())}, "stats": {str(k): v for k, v in sorted(stats.items())},
           "min_gain": float(min_gain), "seconds": round(time.perf_counter() - t0, 1)}
    log(f"[sn56][escala] dev: a=1 {out['loss_at_1']:.5f} -> a={alpha:g} {out['loss_at_alpha']:.5f} "
        f"({len(res)} pontos, {out['seconds']:.0f}s) | "
        + " ".join(f"{k}:{v:.4f}" for k, v in sorted(res.items())))
    if report_path:
        try:
            os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(out, f, indent=1)
        except Exception as e:  # never lose a run over the report
            log(f"[sn56][escala] relatório falhou ({e})")
    return out


def apply_and_save(trainer, alpha: float, submission_dir: str, keep_dir: str = "", log=print) -> bool:
    """theta <- ref + alpha*(theta - ref) on the in-memory final model, then the atomic weights-only
    submission save (config/tokenizer/loss.txt kept). keep_dir: copy of the submission as it was."""
    if alpha == 1.0:
        return False
    if keep_dir:
        try:
            if os.path.exists(keep_dir):
                shutil.rmtree(keep_dir)
            shutil.copytree(submission_dir, keep_dir)
            log(f"[sn56][escala] cópia sem escala em {keep_dir}")
        except Exception as e:
            log(f"[sn56][escala] cópia sem escala falhou ({e})")
    pairs = paired_params(trainer.model, trainer.ref_model)
    scale_in_place(pairs, alpha)
    from dev_pass import _save_weights_only
    _save_weights_only(_unwrap(trainer.model), submission_dir, log)
    log(f"[sn56][escala] submissão = ref + {alpha:g}*(theta-ref)")
    return True
