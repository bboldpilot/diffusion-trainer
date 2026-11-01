"""DPO arm knobs (30 Oct, 198f2175 bench): env flags read by train_dpo.py.

Every DEFAULT is the inherited behaviour (= the frozen champion's DPO path), so
the shipped tree is unchanged until a default is flipped here:

  flag                      default  what
  SN56_DPO_LR               -        fixed LR; skips the in-process LR search
                                     (no step timing -> no time-based epoch plan:
                                     num_train_epochs stays at the command's 3
                                     unless SN56_DPO_EPOCHS sets it)
  SN56_DPO_LR_SCALE         1.0      multiplier on the LR the run starts from
                                     (search pick, cached pick or SN56_DPO_LR)
  SN56_DPO_EPOCHS           -        planned epochs (the cosine horizon) instead
                                     of the time rule
  SN56_DPO_EVAL_PER_EPOCH   -        evals per epoch instead of 5 (spe // 5)
  SN56_DPO_EVAL_FROM        -        earliest eval, in epochs, instead of 0.75
  SN56_DPO_DEVFIT           1        final low-LR pass over the dev set
  SN56_DPO_SCALE_CAL        0        margin-scale calibration on dev
                                     (sn56_dpo_scale.py): theta = ref + a*(theta-ref),
                                     a picked by the dev DPO loss
  SN56_DPO_SCALE_GRID       0.4,...  candidate a values (1.0 always added)
  SN56_DPO_SCALE_MIN_GAIN   0.001    dev nats a must gain over a=1 to be applied
  SN56_DPO_SCALE_KEEP_DIR   -        copy of the uncalibrated submission (bench)
  SN56_DPO_GATE_SPLIT       0        carve N training pairs (content-hash order) into a GATE split
                                     nobody trains, searches, soups or calibrates on; with
                                     SN56_DPO_SELECT=opp it is the selection set
  SN56_DPO_OPP_VEC          -        cached opponent per-pair losses on the gate split (json
                                     {ids, losses}); written here from SN56_DPO_OPP_MODEL if absent
  SN56_DPO_OPP_MODEL        -        opponent weights (a reproduction of the boss recipe trained on
                                     this task's training rows minus the gate split)
  SN56_DPO_INIT             -        policy starts from these weights; the DPO reference stays the
                                     task base (the validator's reference)
SN56_DPO_SELECT (sn56_plan registry, default mean) = mean | gate (K5) | opp (sn56_dpo_select.opp_gate_stats).

SN56_TEXT_FIX=0 (the sn56_plan master switch) forces every default.
"""
import os

_FALSY = ("0", "false", "no", "off")
_TRUTHY = ("1", "true", "yes", "on")
MASTER = "SN56_TEXT_FIX"

DEFAULT_SCALE_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25)

# name -> (key, kind, default)
KNOBS = {
    "SN56_DPO_LR": ("lr", "pos_float", None),
    "SN56_DPO_LR_SCALE": ("lr_scale", "pos_float", 1.0),
    "SN56_DPO_EPOCHS": ("epochs", "pos_float", None),
    "SN56_DPO_EVAL_PER_EPOCH": ("eval_per_epoch", "pos_int", None),
    "SN56_DPO_EVAL_FROM": ("eval_from", "nonneg_float", None),
    "SN56_DPO_DEVFIT": ("devfit", "bool", True),
    "SN56_DPO_SCALE_CAL": ("scale_cal", "bool", False),
    "SN56_DPO_SCALE_GRID": ("scale_grid", "grid", DEFAULT_SCALE_GRID),
    "SN56_DPO_SCALE_MIN_GAIN": ("scale_min_gain", "nonneg_float", 0.001),
    "SN56_DPO_SCALE_KEEP_DIR": ("scale_keep_dir", "str", ""),
    "SN56_DPO_GATE_SPLIT": ("gate_split", "nonneg_int", 0),
    "SN56_DPO_OPP_VEC": ("opp_vec", "str", ""),
    "SN56_DPO_OPP_MODEL": ("opp_model", "str", ""),
    "SN56_DPO_INIT": ("init", "str", ""),
}


def _parse(kind, raw, default):
    """Parsed value, or the default when raw is empty or unparseable."""
    if raw is None or not str(raw).strip():
        return default
    raw = str(raw).strip()
    try:
        if kind == "bool":
            low = raw.lower()
            if low in _TRUTHY:
                return True
            if low in _FALSY:
                return False
            return default
        if kind == "str":
            return raw
        if kind == "grid":
            vals = sorted({round(float(x), 4) for x in raw.split(",") if x.strip()})
            vals = [v for v in vals if 0.0 < v <= 3.0]
            return tuple(vals) if vals else default
        if kind == "pos_int":
            v = int(float(raw))
            return v if v > 0 else default
        if kind == "nonneg_int":
            v = int(float(raw))
            return v if v >= 0 else default
        v = float(raw)
        if v != v:  # NaN
            return default
        if kind == "pos_float":
            return v if v > 0 else default
        if kind == "nonneg_float":
            return v if v >= 0 else default
    except (TypeError, ValueError):
        return default
    return default


def read_knobs(environ=None) -> dict:
    """{key: value} for every knob; SN56_TEXT_FIX=0 forces the defaults."""
    env = os.environ if environ is None else environ
    forced = str(env.get(MASTER, "1")).strip().lower() in _FALSY
    out = {}
    for name, (key, kind, default) in KNOBS.items():
        out[key] = default if forced else _parse(kind, env.get(name), default)
    return out


def non_default(knobs: dict) -> dict:
    return {key: knobs[key] for _, (key, _, default) in KNOBS.items() if knobs.get(key) != default}


def summary(knobs: dict) -> str:
    nd = non_default(knobs)
    return ",".join(f"{k}={v}" for k, v in nd.items()) if nd else "padrão"


def scale_grid(knobs: dict) -> list:
    """Candidate scales, ascending, always containing 1.0 (the uncalibrated model)."""
    return sorted(set(knobs.get("scale_grid") or DEFAULT_SCALE_GRID) | {1.0})


def schedule(knobs: dict, steps_per_epoch: int, epochs: float, eval_steps, warmup_steps: int,
             is_final_run: bool) -> dict:
    """The schedule train_dpo.py runs after its own planning, with the knobs applied (pure).

    Returns epochs, warmup_steps, eval_steps (None = keep), eval_skip_threshold (None = the inherited rule
    in CustomEvalSaveCallback, which derives steps/epoch from total_steps_all_epochs / num_train_epochs and so
    breaks once the epoch count changes after the callback was built -> any schedule knob sets it explicitly),
    total_steps_all_epochs and changed (False = nothing to touch)."""
    spe = max(1, int(steps_per_epoch))
    out = {"epochs": float(epochs), "warmup_steps": int(warmup_steps), "eval_steps": None,
           "eval_skip_threshold": None, "total_steps_all_epochs": int(spe * float(epochs)), "changed": False}
    touched = False
    if knobs.get("epochs") is not None:
        out["epochs"] = float(knobs["epochs"])
        out["total_steps_all_epochs"] = int(spe * out["epochs"])
        # same rule train_dpo.py applies after its own planning: 3% of the new total, floor 10
        out["warmup_steps"] = min(int(warmup_steps), max(10, int(0.03 * out["total_steps_all_epochs"])))
        touched = True
    if is_final_run and knobs.get("eval_per_epoch") is not None:
        out["eval_steps"] = max(10, spe // int(knobs["eval_per_epoch"]))
        touched = True
    if is_final_run and knobs.get("eval_from") is not None:
        touched = True
    if touched and is_final_run:
        frm = knobs.get("eval_from")
        if frm is None:
            # the inherited rule: skip before 0.75 epoch only when training > 1 epoch
            frm = 0.75 if out["epochs"] > 1 else 0.0
        out["eval_skip_threshold"] = int(round(float(frm) * spe))
    out["changed"] = touched
    return out


def gate_split(ids, n: int, min_train: int = 100) -> tuple:
    """(gate_idx, train_idx): the distinct pairs with the lowest content hash form the gate split
    (about n rows), the rest train (both in original order). Depends only on row content, so every
    run of the task (opponent reproduction, candidates) carves the same gate split; duplicated pairs
    land on one side together (no gate pair leaks into training). n is capped so at least
    min_train rows still train; n <= 0 -> no split."""
    import hashlib
    total = len(ids)
    n = max(0, min(int(n), total - int(min_train)))
    if n <= 0:
        return [], list(range(total))
    rows_of = {}
    for i, k in enumerate(ids):
        rows_of.setdefault(str(k), []).append(i)
    chosen, count = set(), 0
    for k in sorted(rows_of, key=lambda k: hashlib.sha1(("gate:" + k).encode()).hexdigest()):
        if count >= n:
            break
        chosen.add(k)
        count += len(rows_of[k])
    gate = [i for i, k in enumerate(ids) if str(k) in chosen]
    return gate, [i for i, k in enumerate(ids) if str(k) not in chosen]
