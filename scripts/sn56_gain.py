"""K4 (SN56_GAIN=1, default off): fp32 per-tensor gain on augmentation-eligible weights.

Validator task prep (G.O.D trainer/model_prep/augmentation.py:18-52, 79-80) can
hand the trainer a base whose `weight_scaling` augmentation multiplied a random
25-75% of the eligible tensors ("weight" in the name, no "norm"/"embed") by ONE
factor s in [0.5, 1.5]. Undoing a 15-43% rescale through Adam on bf16 weights is
not possible at our LRs: the per-step update sits below half a bf16 ulp for most
weights and rounds away (bloomz 26 Oct: 13/13 submissions kept x1.39-1.43).

Here every eligible weight W gets a scalar fp32 gain g (init 1.0, or a given
value), trained jointly with the model in its own optimizer group at a high LR
(SN56_GAIN_LR, default 1e-2). The forward uses g*W exactly:
  Linear / Conv1D : a forward pre-hook scales the module input, x -> g*x, so
                    (g*x) W^T + b == x (g*W)^T + b (bias untouched, like the
                    augmentation)
  LayerNorm-like  : (eligible only when the name has no "norm", e.g. BLOOM/GPT-2
                    ln_f) a forward hook maps out -> (out - b) * g + b, i.e. gamma -> g*gamma
The gains live OUTSIDE the model (no extra keys in any state_dict or submission).
They are folded into the weights, W <- bf16(fp32(W) * g) and g <- 1, before every
evaluation and every save, and folded + unhooked at train end before the
checkpoint callbacks restore / soup, so every snapshot, checkpoint, soup member
and the submission is a plain model with the gain applied (export W <- g*W).

The output embedding (lm_head) is skipped: fused linear-CE paths (liger, the F6
hidden path) apply its weight without calling the module, so a hook there would
not see the training forward. DDP: the gains are not DDP parameters, so their
grads are averaged across ranks in on_pre_optimizer_step; model.zero_grad() does
not reach them, so they are cleared in on_optimizer_step. InstructText full-FT
only (no LoRA / DeepSpeed / FSDP), like every SN56_* fix.
"""

import json
import math
import os
import sys

import torch
from torch import nn

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - torch-only test environments
    TrainerCallback = object


def eligible(name: str) -> bool:
    """The validator's candidate rule (augmentation.py select_target_layers)."""
    return "weight" in name and "norm" not in name.lower() and "embed" not in name.lower()


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _linear_types():
    out = [nn.Linear]
    try:
        from transformers.pytorch_utils import Conv1D
        out.append(Conv1D)
    except Exception:
        pass
    return tuple(out)


def _log(msg):
    print(f"[sn56][ganho] {msg}", file=sys.stderr, flush=True)


class GainSet:
    """Scalar fp32 gains on the eligible weights of `model` (module doc)."""

    def __init__(self, model, init=None, skip_output_embeddings: bool = True):
        m = _unwrap(model)
        init = init or {}
        out_emb = None
        if skip_output_embeddings:
            try:
                out_emb = m.get_output_embeddings()
            except Exception:
                out_emb = None
        lin = _linear_types()
        self.names, self.modules, self.kinds, self.params = [], [], [], []
        for mname, mod in m.named_modules():
            w = getattr(mod, "weight", None)
            if not isinstance(w, nn.Parameter) or not w.requires_grad:
                continue
            pname = f"{mname}.weight" if mname else "weight"
            if not eligible(pname) or (out_emb is not None and mod is out_emb):
                continue
            if isinstance(mod, lin):
                kind = "in"
            elif isinstance(mod, nn.LayerNorm) or (w.dim() == 1 and "Norm" in type(mod).__name__):
                kind = "out"
            else:
                continue
            g = nn.Parameter(torch.tensor(float(init.get(pname, 1.0)), dtype=torch.float32, device=w.device))
            self.names.append(pname)
            self.modules.append(mod)
            self.kinds.append(kind)
            self.params.append(g)
        # cumulative scale actually applied to each weight (product of folds)
        self.applied = [1.0] * len(self.params)
        self._handles = []
        self.n_folds = 0

    def __len__(self):
        return len(self.params)

    # ── hooks ──
    @staticmethod
    def _pre_hook(g):
        def hook(module, args):
            if not args:
                return None
            x = args[0]
            return (x * g.to(x.dtype),) + tuple(args[1:])
        return hook

    @staticmethod
    def _out_hook(g):
        def hook(module, args, out):
            b = getattr(module, "bias", None)
            gg = g.to(out.dtype)
            return (out - b) * gg + b if b is not None else out * gg
        return hook

    def install(self):
        if self._handles:
            return self
        for mod, kind, g in zip(self.modules, self.kinds, self.params):
            if kind == "in":
                self._handles.append(mod.register_forward_pre_hook(self._pre_hook(g)))
            else:
                self._handles.append(mod.register_forward_hook(self._out_hook(g)))
        return self

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    @torch.no_grad()
    def fold_(self) -> int:
        """W <- bf16(fp32(W) * g), g <- 1 for every gain that moved. Returns how many."""
        n = 0
        for i, (mod, g) in enumerate(zip(self.modules, self.params)):
            gv = float(g.detach())
            if gv == 1.0 or not math.isfinite(gv):
                if not math.isfinite(gv):
                    g.fill_(1.0)
                continue
            w = mod.weight
            w.copy_((w.detach().float() * gv).to(w.dtype))
            self.applied[i] *= gv
            g.fill_(1.0)
            n += 1
        self.n_folds += 1
        return n

    def remove(self) -> int:
        n = self.fold_()
        for h in self._handles:
            h.remove()
        self._handles = []
        return n

    def sync_grads(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()) or not self.params:
            return
        flat = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= torch.distributed.get_world_size()
        for p, v in zip(self.params, flat):
            p.grad = v.clone()

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def summary(self, k: int = 8) -> dict:
        cur = [a * float(p.detach()) for a, p in zip(self.applied, self.params)]
        logs = sorted(((abs(math.log(c)) if c > 0 else float("inf"), n, c) for n, c in zip(self.names, cur)),
                      reverse=True)
        moved = sum(1 for d, _, _ in logs if d > math.log(1.01))
        return {"n": len(cur), "moved_gt_1pct": moved, "folds": self.n_folds,
                "top": [(n, round(c, 4)) for _, n, c in logs[:k]],
                "scales": dict(zip(self.names, [round(c, 6) for c in cur]))}


class GainCallback(TrainerCallback):
    """Grad sync / clear around the optimizer step; fold + unhook at train end.
    Must be the FIRST callback so the fold precedes the checkpoint callbacks."""

    def __init__(self):
        self.trainer = None

    def _gs(self):
        return getattr(self.trainer, "_sn56_gains", None) if self.trainer is not None else None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        gs = self._gs()
        if gs is not None and gs.installed:
            gs.sync_grads()

    def on_optimizer_step(self, args, state, control, **kwargs):
        gs = self._gs()
        if gs is not None:
            gs.zero_grad()

    def on_train_end(self, args, state, control, **kwargs):
        gs = self._gs()
        if gs is None or not gs.installed:
            return
        n = gs.remove()
        s = gs.summary()
        _log(f"fim: dobra final ({n} movidos), {s['moved_gt_1pct']}/{s['n']} ganhos >1% "
             f"em {s['folds']} dobras; top {s['top']}")
        # every tensor's applied scale on one line: the pod runners keep [sn56] lines
        _log("escalas " + json.dumps(s["scales"], separators=(",", ":")))
        out = getattr(args, "output_dir", None)
        rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        if out and rank == 0:
            try:
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "sn56_gain.json"), "w") as f:
                    json.dump(s, f)
            except Exception as e:
                _log(f"sn56_gain.json não salvo ({e})")


def make_gain_trainer(base_cls, gain_lr: float = 1e-2, init=None):
    """Subclass of base_cls that trains the gains (module doc)."""
    _lr, _init = float(gain_lr), dict(init or {})

    class GainTrainer(base_cls):
        sn56_gain_lr = _lr
        _sn56_gains = None

        def _sn56_gain_ensure(self):
            if self._sn56_gains is None:
                gs = GainSet(self.model, init=_init)
                gs.install()
                self._sn56_gains = gs
                moved = sum(1 for v in _init.values() if v != 1.0)
                _log(f"{len(gs)} ganhos fp32 (lr {self.sn56_gain_lr:g}, iniciais != 1: {moved})")
            return self._sn56_gains

        def create_optimizer(self):
            opt = super().create_optimizer()
            gs = self._sn56_gain_ensure()
            if gs.params:
                have = {id(p) for grp in opt.param_groups for p in grp["params"]}
                if id(gs.params[0]) not in have:
                    opt.add_param_group({"params": list(gs.params), "lr": self.sn56_gain_lr, "weight_decay": 0.0})
            return opt

        def _sn56_gain_fold(self, why):
            gs = self._sn56_gains
            if gs is not None and gs.installed:
                gs.fold_()

        def evaluate(self, *args, **kwargs):
            self._sn56_gain_fold("eval")
            return super().evaluate(*args, **kwargs)

        def _save_checkpoint(self, *args, **kwargs):
            self._sn56_gain_fold("save")
            return super()._save_checkpoint(*args, **kwargs)

        def save_model(self, *args, **kwargs):
            self._sn56_gain_fold("save")
            return super().save_model(*args, **kwargs)

    GainTrainer.__name__ = f"Gain{base_cls.__name__}"
    return GainTrainer
