"""SN56_LOGIT_CAL (default off): final logit-scale calibration on dev, folded into the weights.

The validator scores the mean over examples of each example's token-mean CE. A fine-tune that is
uniformly over- or under-confident gains from logits z -> a*z with one scalar a. The 26 Oct R1 bloomz
winner shipped its final LayerNorm weight AND bias x0.900, the 3rd x0.950 (its loss.txt says cal=0.95);
the 12 Oct Phi winner's lm_head was x0.898. Here a is fitted on the dev split with the validator's
metric (BEFORE the dev-pass trains on dev), kept only when it gains more than SN56_LOGIT_CAL_MIN_GAIN
nats, and written into the final submission files after the dev-pass:
  untied lm_head          -> lm_head.weight (and bias) x a
  tied embeddings         -> the last norm module's weight (and bias) x a
The fold is checked on a dev batch (new logits == a x old logits, relative error < 2e-2); a model where
that does not hold (e.g. a norm computing x*(1+w)) is left untouched. Single process, full fine-tune,
no KL term (the KL reference would move with a).

SN56_LOGIT_CAL_MODE (read only under SN56_LOGIT_CAL=1):
  dev (default)  as above: fit before the dev-pass, fold after it. The a is STALE: on bd68b438 the soup's dev said
                 a=0.88 (-0.020) and the dev-passed model's test got +0.025, its optimum back at ~1.00. The dev-pass
                 pulls the optimum to ~1 whatever it was, so the fold counts the same correction twice.
  nopass         fit as above, NO dev-pass: a is folded into the very weights it was measured on (the soup already
                 persisted in the submission). Dev rows are not trained on. Measures soup*a directly.
  gate           per-task choice on rows neither arm trained on. The dev-pass runs on one half of dev; the other half
                 (the jury) scores soup*a_pre against pass*a_post, each a fitted on its own weights. The pass stays
                 unless the soup arm is better by SN56_LOGIT_CAL_GATE_MARGIN nats (the soup was selected on dev, jury
                 included, so the margin leans to the pass). The pass is not uniformly good or bad: on rival
                 submissions and unseen rows, each arm at its own best a, it cost +0.009..+0.048 on bloomz and
                 gained -0.009 on SmolLM.
"""
import math
import os

import torch
import torch.nn.functional as F

from sn56_loss import example_ids

DEFAULT_GRID = (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.05, 1.10)
FOLD_REL_TOL = 2e-2
MODES = ("dev", "nopass", "gate")
GATE_PENDING = float("nan")     # fit's a in gate mode: the caller's "a != 1" sends it to apply_to_submission
GATE_MIN_ROWS = 100             # below this a half of dev decides nothing
_GATE = {}                      # gate mode: state between fit_on_trainer (before the pass) and apply_to_submission


class DevPassBlocked(RuntimeError):
    """Raised instead of building the dev loader: the caller's guard keeps the persisted submission."""


def mode_from_env() -> str:
    m = os.environ.get("SN56_LOGIT_CAL_MODE", "").strip().lower()
    return m if m in MODES else "dev"


def gate_margin_from_env() -> float:
    try:
        return max(0.0, float(os.environ.get("SN56_LOGIT_CAL_GATE_MARGIN", "0.002")))
    except ValueError:
        return 0.002


def grid_from_env():
    raw = os.environ.get("SN56_LOGIT_CAL_GRID", "").strip()
    try:
        g = tuple(float(x) for x in raw.split(",") if x.strip()) if raw else DEFAULT_GRID
    except ValueError:
        g = DEFAULT_GRID
    g = tuple(sorted({x for x in g if x > 0} | {1.0}))
    return g


def min_gain_from_env() -> float:
    try:
        return max(0.0, float(os.environ.get("SN56_LOGIT_CAL_MIN_GAIN", "0.001")))
    except ValueError:
        return 0.001


@torch.no_grad()
def scaled_example_stats(logits, labels, grid, position_ids=None, attention_mask=None, chunk_tokens: int = 4096):
    """(len(grid), 2) float64: per scale a, [sum over examples of the example's token-mean CE of a*logits,
    number of examples]. Causal shift and example boundaries as the validator (sn56_devmetric.row_stats)."""
    out = torch.zeros(len(grid), 2, dtype=torch.float64)
    B, T = labels.shape
    if T < 2:
        return out
    V = logits.shape[-1]
    seg = example_ids(labels, position_ids, attention_mask)[:, 1:].reshape(-1)
    tgt = labels[:, 1:].reshape(-1)
    idx = (tgt != -100).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return out
    lg = logits[:, :-1, :].reshape(-1, V)
    tsel = tgt.index_select(0, idx)
    ce = torch.zeros(len(grid), idx.numel(), dtype=torch.float64, device=logits.device)
    for s in range(0, idx.numel(), chunk_tokens):
        z = lg.index_select(0, idx[s:s + chunk_tokens]).float()
        zy = z.gather(1, tsel[s:s + chunk_tokens].unsqueeze(1)).squeeze(1)
        for i, a in enumerate(grid):
            ce[i, s:s + chunk_tokens] = (torch.logsumexp(a * z, dim=-1) - a * zy).double()
    uniq, inv = torch.unique(seg.index_select(0, idx), return_inverse=True)
    cnt = torch.zeros(uniq.numel(), dtype=torch.float64, device=ce.device).index_add_(0, inv, torch.ones_like(ce[0]))
    for i in range(len(grid)):
        sums = torch.zeros(uniq.numel(), dtype=torch.float64, device=ce.device).index_add_(0, inv, ce[i])
        out[i, 0] = float((sums / cnt).sum())
        out[i, 1] = float(uniq.numel())
    return out


def pick(grid, stats, min_gain: float):
    """(a, loss(a), loss(1)) from accumulated stats; a = 1 unless the best scale gains > min_gain."""
    losses = [float(stats[i, 0] / stats[i, 1]) if stats[i, 1] > 0 else float("nan") for i in range(len(grid))]
    i1 = list(grid).index(1.0)
    best = min(range(len(grid)), key=lambda i: losses[i] if math.isfinite(losses[i]) else float("inf"))
    if not math.isfinite(losses[best]) or losses[i1] - losses[best] <= min_gain:
        return 1.0, losses[i1], losses[i1]
    return float(grid[best]), losses[best], losses[i1]


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def fold_targets(model):
    """[(param name, tensor)] whose scaling by a scales the logits by a, or [] when unknown."""
    m = _unwrap(model)
    head = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
    emb = m.get_input_embeddings() if hasattr(m, "get_input_embeddings") else None
    if head is None:
        return []
    names = {id(p): n for n, p in m.named_parameters()}
    tied = emb is not None and getattr(head, "weight", None) is not None and head.weight is getattr(emb, "weight", None)
    if not tied:
        return [(names[id(p)], p) for p in head.parameters(recurse=False) if id(p) in names]
    norms = [mod for _, mod in m.named_modules()
             if "norm" in type(mod).__name__.lower() and getattr(mod, "weight", None) is not None]
    if not norms:
        return []
    last = norms[-1]
    return [(names[id(p)], p) for p in last.parameters(recurse=False) if id(p) in names]


@torch.no_grad()
def _scale_params(targets, a: float):
    for _, p in targets:
        p.mul_(a)


@torch.no_grad()
def verify_fold(model, fwd: dict, targets, a: float) -> float:
    """Relative error of logits(after scaling the targets by a) vs a*logits(before); restores the weights."""
    m = _unwrap(model)
    saved = [p.detach().clone() for _, p in targets]
    z0 = m(**fwd).logits.float()
    _scale_params(targets, a)
    z1 = m(**fwd).logits.float()
    for (_, p), s in zip(targets, saved):
        p.copy_(s)
    return float((z1 - a * z0).norm() / (a * z0).norm().clamp_min(1e-12))


def block_dev_pass(trainer, log=print) -> None:
    """Make the dev-pass that follows the fit a no-op: run_dev_pass asks the trainer for the dev loader before it
    touches a weight, and train_instruct's guard keeps the persisted submission on any exception. Only the
    dev-pass uses this loader after the fit (train_instruct: fit < dev-pass < fold)."""
    def _refuse(*_a, **_k):
        raise DevPassBlocked("SN56_LOGIT_CAL_MODE=nopass")
    trainer.get_eval_dataloader = _refuse
    log("[sn56][logitcal] modo nopass: dev-pass desligado, a dobra vai nos pesos medidos")


def fit_on_trainer(trainer, grid=None, min_gain=None, log=print) -> dict:
    """Fit a on the trainer's eval set (current in-memory weights = the selected checkpoint / soup)."""
    grid = tuple(grid or grid_from_env())
    min_gain = min_gain_from_env() if min_gain is None else min_gain
    mode = mode_from_env()
    model = _unwrap(trainer.model)
    was_training = model.training
    model.eval()
    try:
        if mode == "gate":
            return _gate_prepare(trainer, model, grid, min_gain, log)
        res = _fit(trainer, model, grid, min_gain, log)
    finally:
        model.train(was_training)
    res["mode"] = mode
    if mode == "nopass":
        block_dev_pass(trainer, log)
    return res


def _gate_prepare(trainer, model, grid, min_gain, log) -> dict:
    """Soup arm on the jury half; the dev-pass that follows sees only the other half."""
    from torch.utils.data import Subset
    _GATE.clear()
    full = trainer.eval_dataset
    n = len(full)
    if n < 2 * GATE_MIN_ROWS:
        log(f"[sn56][logitcal] gate: dev pequeno ({n} linhas), sem calibragem, dev-pass inteiro")
        return {"a": 1.0, "names": [], "mode": "gate", "why": "small dev"}
    pass_half, jury = Subset(full, list(range(0, n, 2))), Subset(full, list(range(1, n, 2)))
    trainer.eval_dataset = jury
    try:
        pre = _fit(trainer, model, grid, min_gain, log)
    except Exception:
        trainer.eval_dataset = full
        raise
    trainer.eval_dataset = pass_half
    _GATE.update(trainer=trainer, jury=jury, pre=pre, grid=grid, min_gain=min_gain,
                 snapshot={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
    return {"a": GATE_PENDING, "names": pre["names"], "mode": "gate", "pre": pre}


def _submission_differs(sub_dir: str, snapshot: dict) -> bool:
    """Did the dev-pass swap the submission? (a crashed or skipped pass leaves the soup on disk)"""
    from safetensors import safe_open
    for fn in sorted(f for f in os.listdir(sub_dir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(sub_dir, fn), "pt") as h:
            for k in h.keys():
                if k in snapshot and not torch.equal(h.get_tensor(k).to(snapshot[k].dtype), snapshot[k]):
                    return True
    return False


def _gate_finish(sub_dir: str, log) -> int:
    """Pick the arm on the jury, leave its weights in the submission, fold its a. Returns the tensors scaled."""
    st = dict(_GATE)
    _GATE.clear()
    if not st:
        return 0
    trainer, pre = st["trainer"], st["pre"]
    model = _unwrap(trainer.model)
    if not _submission_differs(sub_dir, st["snapshot"]):
        log(f"[sn56][logitcal] gate: dev-pass nao trocou a submissao; soup a={pre['a']:g}")
        return apply_to_submission(sub_dir, pre["names"], pre["a"], log=log)
    trainer.eval_dataset = st["jury"]
    was_training = model.training
    model.eval()
    try:
        post = _fit(trainer, model, st["grid"], st["min_gain"], log)
    finally:
        model.train(was_training)
    margin = gate_margin_from_env()
    keep_pass = not pre["loss_a"] < post["loss_a"] - margin
    log(f"[sn56][logitcal] gate: juri {post['n_examples']:.0f} exemplos; soup a={pre['a']:g} {pre['loss_a']:.5f} | "
        f"pass a={post['a']:g} {post['loss_a']:.5f} (margem {margin:g}) -> {'pass' if keep_pass else 'soup'}")
    if keep_pass:
        return apply_to_submission(sub_dir, post["names"], post["a"], log=log)
    from dev_pass import _save_weights_only
    model.load_state_dict(st["snapshot"])
    _save_weights_only(model, sub_dir, log)     # logs and keeps the old files on failure, never raises
    if _submission_differs(sub_dir, st["snapshot"]):
        log("[sn56][logitcal] gate: restauracao da soup falhou; fica o dev-pass sem dobra")
        return 0
    return apply_to_submission(sub_dir, pre["names"], pre["a"], log=log)


def _fit(trainer, model, grid, min_gain, log) -> dict:
    stats = torch.zeros(len(grid), 2, dtype=torch.float64)
    first = None
    for inputs in trainer.get_eval_dataloader():
        inputs = trainer._prepare_inputs(inputs)
        if "labels" not in inputs:
            continue
        fwd = {k: v for k, v in inputs.items() if k != "labels"}
        if first is None:
            first = fwd
        with torch.no_grad():
            logits = model(**fwd).logits
        stats += scaled_example_stats(logits, inputs["labels"], grid, inputs.get("position_ids"),
                                      inputs.get("attention_mask")).cpu()
        del logits
    a, la, l1 = pick(grid, stats, min_gain)
    res = {"a": a, "loss_a": la, "loss_1": l1, "grid": list(grid), "n_examples": float(stats[0, 1]),
           "losses": [float(stats[i, 0] / stats[i, 1]) if stats[i, 1] > 0 else None for i in range(len(grid))],
           "names": [], "fold_rel_err": None}
    if a != 1.0:
        targets = fold_targets(model)
        if not targets or first is None:
            res.update(a=1.0, why="no fold target")
        else:
            err = verify_fold(model, first, targets, a)
            res.update(names=[n for n, _ in targets], fold_rel_err=err)
            if not err < FOLD_REL_TOL:
                res.update(a=1.0, why=f"fold check failed (rel err {err:.3g})")
    log(f"[sn56][logitcal] a={res['a']:g} dev {res['loss_1']:.5f} -> {res['loss_a']:.5f} "
        f"({res['n_examples']:.0f} exemplos) alvos={res['names']} erro_dobra={res['fold_rel_err']}"
        + (f" ({res['why']})" if res.get("why") else ""))
    log("[sn56][logitcal] curva " + " ".join(f"{x:g}={v:.5f}" for x, v in zip(res["grid"], res["losses"]) if v is not None))
    return res


def apply_to_submission(sub_dir: str, names, a: float, log=print) -> int:
    """Multiply the named tensors by a in the submission's safetensors files (atomic per file).
    Returns the number of tensors scaled; 0 = nothing done."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    if a != a:                  # GATE_PENDING
        return _gate_finish(sub_dir, log)
    if a == 1.0 or not names:
        return 0
    want = set(names)
    files = sorted(f for f in os.listdir(sub_dir) if f.endswith(".safetensors"))
    done = 0
    for fn in files:
        path = os.path.join(sub_dir, fn)
        with safe_open(path, "pt") as h:
            keys = list(h.keys())
            if not want.intersection(keys):
                continue
            meta = h.metadata() or {"format": "pt"}
            tensors = {k: h.get_tensor(k) for k in keys}
        for k in want.intersection(keys):
            t = tensors[k]
            tensors[k] = (t.float() * a).to(t.dtype)
            done += 1
        tmp = path + ".logitcal.tmp"
        save_file(tensors, tmp, metadata=meta)
        os.replace(tmp, path)
    if done != len(want):
        log(f"[sn56][logitcal] AVISO: {done}/{len(want)} tensores encontrados em {sub_dir}")
    else:
        log(f"[sn56][logitcal] submissao: {done} tensores x{a:g}")
    return done
