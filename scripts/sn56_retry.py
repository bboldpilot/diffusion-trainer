"""SN56_LR_RETRY (default off): a dev-selected second InstructText run at the LR search's argmin.

Why (30 Oct rehearsal of the 26 Oct R1): on granite-4.1-3b (0.72M tokens, 100 steps/epoch) the
edge-of-stability rule picked 2.43e-5 although the deepest search tier's argmin was 1.45e-5 (1.0%
lower probe loss after 84 steps, inside the first epoch). The run then peaked on dev at epoch 1.0
with the cosine (4-epoch horizon) still at 91% of its peak LR, overfit at the second pass and stopped
at 1565 s of a 3600 s budget: 9th of 14, +0.74% behind 1st, every length/difficulty slice equally
behind. The same edge rule is right elsewhere (19 Oct Llama: half the LR was 0.021 worse; MiniCPM
3rd of 14 at the edge pick), so the LR rule itself is not changed. Instead, when the first run stops
early with budget left, train once more from the base at the argmin LR (no search: SN56_LR_SEARCH=0
+ SN56_LR_FIXED, timed steps still plan the schedule) into a side directory, and keep whichever run
has the lower pre-dev-pass dev metric (the validator's per-example metric, SN56_DEV_METRIC=vali).
The first run's submission stays in place until the second one has finished and won; the swap is
two renames.

Called by text_trainer.py after a successful InstructText run; every failure keeps the first run.

SN56_R1_COOLDOWN (default on since 1 Nov): the second run cools over the useful horizon instead of the 4-epoch plan it
never reaches: SN56_MAX_EPOCHS = clamp(1.25 x the first run's best-dev epoch, 0.5, 4), min_lr_rate 0.1, evals
from 0.25 epoch. It also runs when the argmin is not below the first LR (same LR, cooled), as long as the first
run stopped early. SN56_R1_AVG2 (default on since 1 Nov): the first run keeps a copy of its pre-dev-pass weights; the second
run tries the 50:50 average with them (try_blend) and keeps it only if its dev metric improves.
"""
import datetime
import glob
import json
import math
import os
import shutil

import sn56_plan

RATIO_MIN = 1.2      # the argmin must sit at least 1.2x below the first run's LR
TIME_FACTOR = 1.3    # the second run needs 1.3x the first run's post-search wall-clock ...
TIME_FIXED_S = 240   # ... plus model load / save / swap
MARGIN_REL = 0.001   # the second run replaces the first only when its dev metric is lower by > 0.1 %
SWAP_MIN_LEFT_S = 30


def seconds_left(end_time: str, now: datetime.datetime = None) -> float:
    try:
        end = datetime.datetime.strptime(str(end_time), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError):
        return float("-inf")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return (end - now).total_seconds()


def decide(rec: dict, now: datetime.datetime = None):
    """(go, lr2, reason) from the first run's record (sn56_plan.write_run_record)."""
    if not rec:
        return False, None, "no run record"
    if rec.get("lr_fixed"):
        return False, None, "already a second run"
    if not rec.get("sn56_on") or rec.get("use_kl"):
        return False, None, "out of scope (InstructText full FT, no KL)"
    dev = rec.get("dev")
    if rec.get("dev_metric") != "vali" or dev is None or not math.isfinite(float(dev)):
        return False, None, "no comparable dev metric"
    lr, am = rec.get("lr"), rec.get("lr_argmin")
    lower = bool(lr and am and am > 0 and lr / am >= RATIO_MIN)
    cool = sn56_plan.flag_on("SN56_R1_COOLDOWN") and bool(rec.get("stopped_early")) and bool(lr)
    if not lower and not cool:
        return False, None, f"argmin {am} not below lr {lr} by {RATIO_MIN}x"
    lr2 = float(am) if lower else float(lr)
    t_run1 = float(rec["t_train_end"]) - float(rec["t_start"]) - float(rec.get("lr_search_s") or 0.0)
    need = TIME_FACTOR * max(t_run1, 0.0) + TIME_FIXED_S
    left = seconds_left(rec.get("end_time"), now)
    if left < need:
        return False, None, f"{left:.0f}s left < {need:.0f}s needed"
    return True, lr2, f"lr {lr:.3e} -> {lr2:.3e}{' (cooldown)' if cool else ''}, {left:.0f}s left >= {need:.0f}s"


def second_run_env(rec: dict, lr2: float) -> dict:
    """Env for the second run: fixed LR, no search; + cooldown / average plumbing when flagged."""
    env = {"SN56_LR_SEARCH": "0", "SN56_LR_FIXED": repr(lr2)}
    if sn56_plan.flag_on("SN56_R1_COOLDOWN"):
        bs, spe = rec.get("best_step"), rec.get("spe")
        if bs and spe:
            e = min(4.0, max(0.5, 1.25 * float(bs) / float(spe)))
            env["SN56_MAX_EPOCHS"] = f"{e:.3f}"
        env["SN56_MIN_LR_RATE"] = "0.1"
        env["SN56_EVAL_FROM"] = "0.25"
    pre = rec.get("pre_devpass_dir")
    if sn56_plan.flag_on("SN56_R1_AVG2") and pre and os.path.isdir(pre):
        env["SN56_AVG_WITH"] = pre
    return env


def blend_into(model, other: dict, w: float = 0.5) -> dict:
    """In place: p <- (1-w)*p + w*other[name] for every same-shape parameter; returns the CPU backups."""
    import torch
    backup = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            t = other.get(n)
            if t is None or tuple(t.shape) != tuple(p.shape):
                continue
            backup[n] = p.detach().to("cpu", copy=True)
            p.copy_(((1.0 - w) * p.float() + w * t.to(p.device).float()).to(p.dtype))
    return backup


def restore(model, backup: dict) -> None:
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in backup:
                p.copy_(backup[n].to(p.device))


def try_blend(trainer, other_dir: str, ckpt_avg, submission_dir: str, log=print, w: float = 0.5) -> bool:
    """SN56_R1_AVG2 in the second run: average with the first run's pre-dev-pass weights, score the dev set
    with the trainer's own (validator-form) evaluation loop, keep it only if it beats this run's best."""
    from safetensors.torch import load_file
    files = sorted(glob.glob(os.path.join(other_dir, "*.safetensors")))
    if not files:
        log(f"[sn56][avg2] sem pesos em {other_dir}")
        return False
    other = {}
    for f in files:
        other.update(load_file(f, device="cpu"))
    model = trainer.model
    while hasattr(model, "module"):
        model = model.module
    best = float(getattr(ckpt_avg, "best_loss", float("inf")))
    backup = blend_into(model, other, w)
    del other
    if not backup:
        log("[sn56][avg2] nenhum tensor em comum, pulando")
        return False
    was = model.training
    model.eval()
    out = trainer.evaluation_loop(trainer.get_eval_dataloader(), "avg2", metric_key_prefix="eval")
    model.train(was)
    dev = float(out.metrics.get("eval_loss", float("inf")))
    if math.isfinite(dev) and dev < best * (1.0 - 1e-4):
        ckpt_avg.best_loss = dev
        from dev_pass import _save_weights_only
        _save_weights_only(model, submission_dir, log)
        log(f"[sn56][avg2] media 50:50 vence: dev {best:.5f} -> {dev:.5f} ({len(backup)} tensores)")
        return True
    restore(model, backup)
    log(f"[sn56][avg2] media 50:50 nao vence: dev {best:.5f} vs {dev:.5f}; pesos da 2a corrida mantidos")
    return False


def better(rec1: dict, rec2: dict) -> bool:
    try:
        d1, d2 = float(rec1["dev"]), float(rec2["dev"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(d2) and d2 < d1 * (1.0 - MARGIN_REL)


def swap_in(sub: str, sub2: str) -> None:
    """sub2 replaces sub: two renames (sub is missing only between them), then the old run is removed."""
    old = sub.rstrip("/") + ".lr1-old"
    shutil.rmtree(old, ignore_errors=True)
    os.rename(sub, old)
    os.rename(sub2, sub)
    shutil.rmtree(old, ignore_errors=True)


def side(path: str) -> str:
    return path.rstrip("/") + "-lr2"


def maybe_second_run(train_cmd, log_path, output_dir, request_path, submission_dir,
                     run_cmd_with_log, replace_args_in_cmd, log=print) -> str:
    """Run the second pass when decide() says so; returns what happened (for the log)."""
    if not sn56_plan.flag_on("SN56_LR_RETRY"):
        return "off"
    rec1 = sn56_plan.read_run_record(output_dir)
    go, lr2, why = decide(rec1)
    log(f"[sn56][lr2] {'GO' if go else 'skip'}: {why}")
    if not go:
        if rec1 and rec1.get("pre_devpass_dir"):
            shutil.rmtree(rec1["pre_devpass_dir"], ignore_errors=True)
        return "skip"
    out2, sub2 = side(output_dir), side(submission_dir)
    for d in (out2, sub2):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(sub2, exist_ok=True)
    # the first run's checkpoints are no longer needed (its submission is separate): free the disk
    for c in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(c, ignore_errors=True)
    with open(request_path) as f:
        req = json.load(f)
    req["train_request"]["submission_dir"] = sub2
    req2_path = request_path[:-5] + "-lr2.json" if request_path.endswith(".json") else request_path + "-lr2"
    with open(req2_path, "w") as f:
        json.dump(req, f, indent=4, ensure_ascii=False)
    cmd2 = replace_args_in_cmd(replace_args_in_cmd(train_cmd, "output_dir", out2), "request_path", req2_path)
    log2 = log_path[:-4] + "-lr2.log" if log_path.endswith(".log") else log_path + "-lr2"
    env2 = second_run_env(rec1, lr2)
    log(f"[sn56][lr2] 2a corrida env {env2}")
    run_cmd_with_log(cmd2, log2, env_vars=env2)
    if rec1.get("pre_devpass_dir"):
        shutil.rmtree(rec1["pre_devpass_dir"], ignore_errors=True)
    rec2 = sn56_plan.read_run_record(out2)
    ok2 = os.path.exists(os.path.join(out2, "success.txt")) and sn56_plan.submission_is_valid(sub2) and rec2
    left = seconds_left(rec1.get("end_time"))
    if ok2 and better(rec1, rec2) and left > SWAP_MIN_LEFT_S:
        swap_in(submission_dir, sub2)
        log(f"[sn56][lr2] 2a corrida vence: dev {float(rec1['dev']):.5f} -> {float(rec2['dev']):.5f} "
            f"(lr {float(rec1['lr']):.3e} -> {lr2:.3e}); submissao trocada")
        return "swapped"
    shutil.rmtree(sub2, ignore_errors=True)
    log(f"[sn56][lr2] mantendo a 1a corrida: ok2={bool(ok2)} dev1={rec1.get('dev')} "
        f"dev2={(rec2 or {}).get('dev')} left={left:.0f}s")
    return "kept"
