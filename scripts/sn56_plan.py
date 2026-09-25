"""InstructText full-fine-tune plan: feature flags and pure planning math.

Every behaviour change on the InstructText full-FT path sits behind an env flag
whose DEFAULT is the new behaviour; setting the flag to its old value restores
the inherited code path for A/B:

  flag                    default  old   what
  SN56_EVAL_SKIP_FIX      1        0     F1 eval-skip threshold from the real steps/epoch
  SN56_TIME_PLAN          1        0     F2 optimizer steps planned from the time budget,
                                            tail (save/soup/dev-pass) reserved from measurements
  SN56_MAX_EPOCHS         4        -     F2 upper safety cap on planned epochs
  SN56_LR_SEARCH          1        1     F3 in-process LR search (0 = finder LR only)
  SN56_LR_MULT            1.0      1.0   F3 multiplier applied on top of the LR the run
                                            starts from: the search's pick, or the finder
                                            estimate when the search is off
  SN56_NEFTUNE            0        1     F4 NEFTune noise on the embeddings
  SN56_V5_PACK            auto     0     F5 v5-lane throughput mode (auto|0|1)
  SN56_PER_EXAMPLE_LOSS   0        0     F6 evaluator-native per-example loss (A/B)

SN56_TEXT_FIX=0 is a master switch: every flag above takes its old value.

F3 default (measured 21-25 Sep): the search is what repairs validator-augmented
bases (LFM: 1.146 / 11th with the search off -> 0.962 / 3rd with it on;
Qwen2-0.5B: search 1.25e-4 vs finder 7.8e-5, and finder x2 was the best fixed
value), so SN56_LR_SEARCH=1 is the default and SN56_LR_MULT still scales the
searched LR (SN56_LR_MULT=2 with the search off reproduces the fixed x2 arm).

Nothing heavy is imported here (no torch), so the planning math is testable on
any CPU (scripts/tests/test_sn56_plan.py).
"""

import datetime
import math
import os
import time
from typing import Optional

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")

# name -> (default, old value). The old value is what SN56_TEXT_FIX=0 forces.
FLAGS = {
    "SN56_EVAL_SKIP_FIX": ("1", "0"),
    "SN56_TIME_PLAN": ("1", "0"),
    "SN56_MAX_EPOCHS": ("4", "4"),
    "SN56_LR_SEARCH": ("1", "1"),
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


# ── F2: the tail — what runs after the final-save trigger ────────────────────
#
# Measured failure (3B, 1.5h, 25 Sep): the final-save trigger sat at T-5.8min
# (3 min inherited + the dev-pass estimate) while the tail behind it is the step
# in flight, the final eval, the checkpoint write, the copy into the submission
# dir, the greedy soup over up to 8 pooled snapshots (one eval + one merge each,
# then a weights-only save) and the dev-pass (+ its save). The soup was bounded
# only by "half of what is left" (2h task: 5 candidates, dev-pass skipped) and
# the reserve never scaled with the model. Here the tail is priced from the
# model size, the measured step time, a measured disk write rate and the dev
# size, with a safety margin, and the trigger is placed that far before
# end_time (text_trainer already sets end_time 3 min before the hours*3600
# kill). Each stage of the tail also carries its own deadline so an estimate
# that runs slow cuts the soup / dev-pass short instead of the final write.
# When the tail would eat more than TAIL_MAX_FRACTION of the time left it is
# shrunk: soup candidates first (down to none), then the dev-pass; the final
# eval + write + copy + safety is the floor.

TAIL_SAFETY_S = 90.0          # last write lands this long before end_time
TAIL_MAX_FRACTION = 0.25      # the tail never takes more than this of the time left
DEFAULT_WRITE_BPS = 0.7e9     # tournament pods: ~6 GB of bf16 weights in 7-9 s
WRITE_BPS_MIN, WRITE_BPS_MAX = 5e7, 3e9
SAVE_FIXED_S = 5.0            # config/tokenizer/index + fsync per save
COPY_FACTOR = 2.0             # rmtree + copytree of a checkpoint: read + write
MERGE_BPS = 2e9               # CPU fp32 accumulate over the trainable params
DEVPASS_OVERHEAD_S = 30.0     # optimizer build + loader + swap around the pass
LATE_ATTEMPT_MIN_S = 900.0    # fresh attempt with no measurements: needs 15 min
ATTEMPT_LOAD_S = 120.0        # model load + dataset packing on a retry
ATTEMPT_PROBE_STEPS = 12      # the timing probe's steps
ATTEMPT_MIN_TRAIN_STEPS = 20  # below this a retry is not worth a re-plan


def save_time_s(n_params: int, write_bps: float = DEFAULT_WRITE_BPS,
                bytes_per_param: float = 2.0) -> float:
    """One weights-only save (bf16) at the measured write rate."""
    return SAVE_FIXED_S + bytes_per_param * max(0, n_params) / max(WRITE_BPS_MIN, write_bps)


def eval_forward_s(n_dev_rows_per_rank: float, t_per_step: float, rows_per_step: int) -> float:
    """One pass over the dev set at eval batch 1, forward only (~half a training
    row's fwd+bwd(+recompute) per dev row, floored for launch overhead)."""
    micro = t_per_step / max(1, rows_per_step)
    return max(0.0, n_dev_rows_per_rank) * max(0.5 * micro, 0.01)


def soup_time_s(n_pool: int, eval_s: float, n_params: int, merge_bps: float = MERGE_BPS) -> float:
    """Greedy soup over n_pool snapshots: the seed eval, then one fp32 merge
    and one eval per further candidate."""
    if n_pool <= 1:
        return 0.0
    merge = max(1.0, 4.0 * max(0, n_params) / merge_bps)
    return eval_s + (n_pool - 1) * (merge + eval_s)


def devpass_time_s(n_dev_rows: int, world_size: int, t_per_step: float, grad_accum: int,
                   per_device_bs: int, save_s: float) -> float:
    """One low-LR epoch over dev at eval batch 1 (2x for the lost batch
    parallelism), then the weights-only save."""
    micro = t_per_step / max(1, grad_accum * per_device_bs)
    return 2.0 * (n_dev_rows / max(1, world_size)) * micro + save_s + DEVPASS_OVERHEAD_S


def tail_plan(
    *,
    now_ts: float,
    end_ts: float,
    t_per_step: float,
    n_params: int,
    n_dev_rows: int,
    world_size: int = 1,
    grad_accum: int = 1,
    per_device_bs: int = 1,
    write_bps: float = DEFAULT_WRITE_BPS,
    soup_pool: int = 8,
    want_soup: bool = True,
    want_devpass: bool = True,
    max_fraction: float = TAIL_MAX_FRACTION,
    safety_s: float = TAIL_SAFETY_S,
) -> dict:
    """Size the tail and place the final-save trigger.

    Returns the reserve (seconds before end_ts at which training stops), the
    soup pool size and dev-pass decision that fit, and the deadlines each stage
    must respect: soup_deadline_ts (soup incl. its save done), devpass_deadline_ts
    (dev-pass incl. its save done), final_write_ts (= end_ts - safety_s)."""
    t_per_step = max(1e-6, float(t_per_step))
    n_params = max(0, int(n_params))
    n_dev_rows = max(0, int(n_dev_rows))
    world_size = max(1, int(world_size))
    rows_per_step = max(1, int(grad_accum) * int(per_device_bs))
    save_s = save_time_s(n_params, write_bps)
    copy_s = COPY_FACTOR * save_s
    eval_s = eval_forward_s(n_dev_rows / world_size, t_per_step, rows_per_step) if n_dev_rows else 0.0
    # The RAM-mode callback persists its best weights at train end whether or
    # not a soup was combined, so that save is part of the tail whenever it runs.
    persist_s = save_s if want_soup else 0.0
    n_soup = max(1, int(soup_pool)) if (want_soup and n_dev_rows > 0) else 1
    devpass = bool(want_devpass and n_dev_rows > 0)
    available = max(0.0, end_ts - now_ts)
    cap = max_fraction * available
    floor = t_per_step + eval_s + save_s + copy_s + persist_s + safety_s

    def _parts(k: int, dp: bool):
        soup = soup_time_s(k, eval_s, n_params)
        dev = (devpass_time_s(n_dev_rows, world_size, t_per_step, grad_accum, per_device_bs, save_s)
               if dp else 0.0)
        return soup, dev

    soup_s, dev_s = _parts(n_soup, devpass)
    while floor + soup_s + dev_s > cap and n_soup > 1:
        n_soup -= 1
        soup_s, dev_s = _parts(n_soup, devpass)
    if floor + soup_s + dev_s > cap and devpass:
        devpass = False
        soup_s, dev_s = _parts(n_soup, devpass)
    reserve = floor + soup_s + dev_s
    final_write_ts = end_ts - safety_s
    devpass_deadline_ts = final_write_ts
    soup_deadline_ts = final_write_ts - dev_s
    return {
        "reserve_s": reserve,
        "trigger_ts": end_ts - reserve,
        "final_write_ts": final_write_ts,
        "soup_deadline_ts": soup_deadline_ts,
        "devpass_deadline_ts": devpass_deadline_ts,
        "n_soup": int(n_soup),
        "devpass": devpass,
        "step_s": t_per_step,
        "eval_s": eval_s,
        "save_s": save_s,
        "copy_s": copy_s,
        "persist_s": persist_s,
        "merge_s": max(1.0, 4.0 * n_params / MERGE_BPS),
        "soup_s": soup_s,
        "devpass_s": dev_s,
        "safety_s": safety_s,
        "write_bps": write_bps,
        "available_s": available,
        "cap_s": cap,
        "shrunk": (n_soup < max(1, int(soup_pool)) and want_soup and n_dev_rows > 0)
                  or (want_devpass and n_dev_rows > 0 and not devpass),
    }


def simulate_tail(tail: dict, end_ts: float, step_scale: float = 1.0,
                  save_scale: float = 1.0) -> dict:
    """Walk the tail from its trigger with the compute stages scaled by
    step_scale and the disk stages by save_scale (a slow disk, a long last
    step), honouring the in-tail deadlines the way the trainer does: a soup
    candidate is skipped once its merge + eval would pass the soup deadline,
    the dev-pass runs only if it can finish before its deadline. Returns the
    wall-clock of the last submission write and its margin to end_ts."""
    t = tail["trigger_ts"]
    t += tail["step_s"] * step_scale            # step in flight
    t += tail["eval_s"] * step_scale            # final eval
    t += tail["save_s"] * save_scale            # checkpoint write
    t += tail["copy_s"] * save_scale            # copy into the submission dir
    last_write = t
    if tail["n_soup"] > 1:
        per_eval = tail["eval_s"] * step_scale
        merge = tail["merge_s"] * step_scale
        soup_end = tail["soup_deadline_ts"] - tail["persist_s"] * save_scale
        t += per_eval                           # seed eval
        for _ in range(tail["n_soup"] - 1):
            if t + merge + per_eval > soup_end:
                break
            t += merge + per_eval
    if tail["persist_s"] > 0:
        t += tail["persist_s"] * save_scale
        last_write = t
    dev = tail["devpass_s"] * max(step_scale, save_scale)
    if tail["devpass"] and t + dev <= tail["devpass_deadline_ts"]:
        t += dev
        last_write = t
    return {"last_write_ts": last_write, "end_ts": end_ts, "margin_s": end_ts - last_write}


def late_attempt(remaining_s: float, has_submission: bool, t_per_step: Optional[float] = None,
                 reserve_s: Optional[float] = None, load_s: float = ATTEMPT_LOAD_S) -> bool:
    """True when a fresh attempt (a retry after a crash) could not load the
    model, probe the step time, train a meaningful number of steps and still
    write its submission before end_time. With a valid submission already on
    disk the attempt must then keep it and exit successfully instead of
    re-planning the last minutes (25 Sep: six such re-plans ran into the kill)."""
    if not has_submission:
        return False
    if t_per_step and reserve_s:
        need = load_s + (ATTEMPT_PROBE_STEPS + ATTEMPT_MIN_TRAIN_STEPS) * float(t_per_step) + float(reserve_s)
    else:
        need = LATE_ATTEMPT_MIN_S
    return remaining_s < need


def submission_is_valid(path: str) -> bool:
    """A loadable submission: config.json plus at least one weight file."""
    try:
        if not path or not os.path.isdir(path):
            return False
        names = os.listdir(path)
    except OSError:
        return False
    if "config.json" not in names:
        return False
    return any(n.endswith(".safetensors") or n.endswith(".bin") for n in names)


def measure_write_bps(directory: str, size_bytes: int = 256 * 1024 * 1024,
                      default: float = DEFAULT_WRITE_BPS) -> float:
    """Sequential write rate of the checkpoint disk (bytes/s), from one
    fsynced probe file; the default on any error, clamped either way."""
    path = os.path.join(directory, f".sn56_write_probe_{os.getpid()}")
    chunk = bytes(bytearray(os.urandom(1024)) * 1024)  # 1 MiB, incompressible
    written = 0
    try:
        os.makedirs(directory, exist_ok=True)
        t0 = time.perf_counter()
        with open(path, "wb") as f:
            while written < size_bytes:
                f.write(chunk)
                written += len(chunk)
            f.flush()
            os.fsync(f.fileno())
        dt = time.perf_counter() - t0
        bps = written / max(1e-3, dt)
    except OSError:
        bps = default
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return float(min(WRITE_BPS_MAX, max(WRITE_BPS_MIN, bps)))


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
