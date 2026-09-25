"""InstructText full-fine-tune plan: feature flags and pure planning math.

Every behaviour change on the InstructText full-FT path sits behind an env flag
whose DEFAULT is the new behaviour; setting the flag to its old value restores
the inherited code path for A/B:

  flag                    default  old   what
  SN56_EVAL_SKIP_FIX      1        0     F1 eval-skip threshold from the real steps/epoch
  SN56_TIME_PLAN          1        0     F2 optimizer steps planned from the time budget
  SN56_MAX_EPOCHS         4        -     F2 upper safety cap on planned epochs
  SN56_LR_SEARCH          0        1     F3 0 = no in-process LR search, finder LR x mult
  SN56_LR_MULT            1.0      1.0   F3 multiplier on the finder estimate
  SN56_NEFTUNE            0        1     F4 NEFTune noise on the embeddings
  SN56_V5_PACK            auto     0     F5 v5-lane throughput mode (auto|0|1)
  SN56_PER_EXAMPLE_LOSS   0        0     F6 evaluator-native per-example loss (A/B)

SN56_TEXT_FIX=0 is a master switch: every flag above takes its old value.

Nothing heavy is imported here (no torch), so the planning math is testable on
any CPU (scripts/tests/test_sn56_plan.py).
"""

import datetime
import math
import os
from typing import Optional

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")

# name -> (default, old value). The old value is what SN56_TEXT_FIX=0 forces.
FLAGS = {
    "SN56_EVAL_SKIP_FIX": ("1", "0"),
    "SN56_TIME_PLAN": ("1", "0"),
    "SN56_MAX_EPOCHS": ("4", "4"),
    "SN56_LR_SEARCH": ("0", "1"),
    "SN56_LR_MULT": ("1.0", "1.0"),
    "SN56_NEFTUNE": ("0", "1"),
    "SN56_V5_PACK": ("auto", "0"),
    "SN56_PER_EXAMPLE_LOSS": ("0", "0"),
}
MASTER = "SN56_TEXT_FIX"


def _raw(name: str) -> str:
    default, old = FLAGS[name]
    if os.environ.get(MASTER, "1").strip().lower() in _FALSY:
        return old
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower()


def flag_on(name: str) -> bool:
    """Boolean flag; an unparseable value falls back to the default."""
    value = _raw(name)
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return FLAGS[name][0] in _TRUTHY


def flag_float(name: str) -> float:
    try:
        return float(_raw(name))
    except (TypeError, ValueError):
        return float(FLAGS[name][0])


def flag_str(name: str) -> str:
    return _raw(name)


def v5_pack_setting() -> str:
    """auto | 0 | 1 (anything else reads as auto)."""
    value = _raw("SN56_V5_PACK")
    if value in _FALSY:
        return "0"
    if value in _TRUTHY:
        return "1"
    return "auto"


def max_epochs() -> float:
    value = flag_float("SN56_MAX_EPOCHS")
    return value if value > 0 else float(FLAGS["SN56_MAX_EPOCHS"][0])


def flags_summary() -> str:
    parts = [f"{name[5:]}={_raw(name)}" for name in FLAGS]
    master = os.environ.get(MASTER)
    if master is not None:
        parts.append(f"TEXT_FIX={master}")
    return ",".join(parts)


# ── F1: eval-skip threshold ──────────────────────────────────────────────────

SKIP_EPOCHS = 0.75      # inherited: no evals before 0.75 epoch on multi-epoch runs
SKIP_MAX_FRACTION = 0.5  # new: never skip more than the first half of the plan


def legacy_eval_skip_threshold(total_steps_all_epochs: float, num_epochs: float) -> int:
    """The inherited rule (customized_trainer.on_step_end): steps/epoch derived
    from a total computed with the INITIAL epoch count (3) but divided by the
    re-planned one, so the threshold drifts to 0.75*3/E epochs."""
    if total_steps_all_epochs <= 0 or num_epochs <= 1:
        return 0
    steps_per_epoch = total_steps_all_epochs / max(1, num_epochs)
    return int(SKIP_EPOCHS * steps_per_epoch)


def eval_skip_threshold(steps_per_epoch: int, planned_total_steps: int, num_epochs: float) -> int:
    """F1: first step at which periodic evals are honoured (0 = none skipped).

    0.75 epoch of the CURRENT steps/epoch, capped at half the planned run so a
    short multi-epoch plan (E in (1, 1.5)) still gets evals over its second half.
    Sub-epoch plans skip nothing, as before."""
    if num_epochs <= 1 or steps_per_epoch <= 0:
        return 0
    threshold = int(SKIP_EPOCHS * steps_per_epoch)
    if planned_total_steps > 0:
        threshold = min(threshold, int(SKIP_MAX_FRACTION * planned_total_steps))
    return max(0, threshold)


# ── F2: time-budget plan ─────────────────────────────────────────────────────

def legacy_epoch_plan(achievable_steps: int, steps_per_epoch: int, n_train_items: int) -> float:
    """The inherited epoch rule (train_instruct time-aware epoch planning)."""
    cap = 4.0 if n_train_items < 10000 else 3.0
    return round(max(1.3, min(cap, 1.25 * achievable_steps / steps_per_epoch)), 2)


def eval_interval(steps_per_epoch: int, planned_steps: int) -> int:
    """Every 1/8 epoch (inherited), but at least ~8 evals over a sub-epoch plan."""
    return max(20, min(max(1, steps_per_epoch) // 8, max(1, planned_steps) // 8))


def eval_points(planned_steps: int, eval_steps: int, threshold: int,
                steps_per_epoch: int = -1, periodic_save_steps: int = -1) -> list:
    """Steps (< planned) at which an eval+save fires: the eval_steps grid plus the
    callback's epoch-boundary and periodic triggers, minus the skipped prefix.
    The end-time save is not counted — it happens after the planned stop."""
    points = set()
    for interval in (eval_steps, steps_per_epoch, periodic_save_steps):
        if interval is None or interval <= 1:
            continue
        for step in range(interval, planned_steps, interval):
            if step >= max(threshold, 2):
                points.add(step)
    return sorted(points)


def estimate_eval_cost_s(n_dev_rows_per_rank: int, t_per_step: float, rows_per_step: int,
                         n_params: int, save_bytes_per_s: float = 1.5e9) -> float:
    """Wall-clock of one eval+save. Eval runs at batch 1, forward only: ~half a
    training row's fwd+bwd(+recompute) cost per dev row, floored for launch
    overhead. Save: the step checkpoint plus the copy into the submission dir,
    bf16 weights at ~1.5 GB/s, plus a fixed 5 s."""
    micro = t_per_step / max(1, rows_per_step)
    forward = max(0, n_dev_rows_per_rank) * max(0.5 * micro, 0.01)
    save = 5.0 + 2 * (2.0 * max(0, n_params)) / save_bytes_per_s
    return forward + save


def devpass_reserve(n_dev_rows: int, world_size: int, t_per_step: float,
                    grad_accum: int, per_device_bs: int):
    """(seconds, extra minutes) — the dev-pass budget train_instruct reserves
    before end_time (same formula as its [devfit] block)."""
    micro = t_per_step / max(1, grad_accum * per_device_bs)
    secs = 2.0 * (n_dev_rows / max(1, world_size)) * micro + 120
    return secs, min(15.0, max(2.0, secs / 60))


def plan_time_budget(
    *,
    now_ts: float,
    stop_ts: float,
    t_per_step: float,
    steps_per_epoch: int,
    max_epochs: float = 4.0,
    warmup_cap: int = 200,
    eval_cost_s: float = 0.0,
    periodic_save_steps: int = -1,
    startup_s: float = 30.0,
) -> dict:
    """F2: size the run so the cosine reaches its floor at the real cutoff.

    stop_ts is the final-save trigger (end_time minus the save/dev-pass reserve):
    training stops there, so that is where the schedule should end. The plan is
    the largest step count P with P * t_per_step + evals(P) * eval_cost_s within
    (stop - now - startup), evals(P) being the eval/save points P itself implies
    (bisection: the overhead can dominate, so a fixed-point iteration can cycle).
    Epochs follow from steps (no floor), capped at max_epochs."""
    spe = max(1, int(steps_per_epoch))
    t_per_step = max(1e-6, float(t_per_step))
    max_epochs = max_epochs if max_epochs > 0 else 4.0
    cap = max(1, int(max_epochs * spe))
    available = max(0.0, stop_ts - now_ts - startup_s)

    def fits(steps: int) -> bool:
        interval_ = eval_interval(spe, steps)
        threshold_ = eval_skip_threshold(spe, steps, steps / spe)
        n = len(eval_points(steps, interval_, threshold_, spe, periodic_save_steps))
        return steps * t_per_step + n * eval_cost_s <= available

    hi = min(cap, int(available / t_per_step))
    if fits(hi):
        planned = hi
    else:
        lo = 0  # fits(0) always holds
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid
        planned = lo

    planned = max(planned, min(cap, 10))
    epochs = planned / spe
    interval = eval_interval(spe, planned)
    threshold = eval_skip_threshold(spe, planned, epochs)
    points = eval_points(planned, interval, threshold, spe, periodic_save_steps)
    warmup = min(int(warmup_cap), max(10, int(0.03 * planned)))
    return {
        "planned_steps": int(planned),
        "epochs": epochs,
        "steps_per_epoch": spe,
        "capped": planned >= cap,
        "cap_steps": cap,
        "warmup": int(warmup),
        "eval_steps": int(interval),
        "eval_skip_threshold": int(threshold),
        "first_eval_step": points[0] if points else None,
        "n_evals": len(points),
        "eval_overhead_s": len(points) * eval_cost_s,
        "available_s": available,
        "train_s": planned * t_per_step,
    }


# ── time helpers ─────────────────────────────────────────────────────────────

_FMT = "%Y-%m-%d %H:%M:%S"


def parse_utc(ts: str) -> float:
    return datetime.datetime.strptime(ts, _FMT).replace(tzinfo=datetime.timezone.utc).timestamp()


def fmt_utc(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(_FMT)


def lr_for_batch(lr_ref: float, eff_bs_ref: int, eff_bs_now: int, mult: float = 1.0) -> float:
    """F3: finder LR re-centred for the batch actually trained (sqrt rule, the
    same convention the cached-LR retry path and the search's OOM halving use)."""
    if eff_bs_ref and eff_bs_ref > 0 and eff_bs_now and eff_bs_now > 0:
        lr_ref = lr_ref * math.sqrt(eff_bs_now / eff_bs_ref)
    return lr_ref * mult


def tokens_correction(measured_tokens_per_step: Optional[float],
                      expected_tokens_per_step: Optional[float]) -> float:
    """Scale a step time measured on unrepresentative batches (length-grouped
    sampling serves the longest batches first) to the epoch average."""
    if not measured_tokens_per_step or not expected_tokens_per_step:
        return 1.0
    return max(0.25, min(1.0, expected_tokens_per_step / measured_tokens_per_step))
