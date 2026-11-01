"""SN56_DPO_SELECT=opp plumbing: the opponent's cached per-pair vector and the policy init.

The opponent is a task-local reproduction of the frozen boss recipe (our inherited DPO path is
byte-identical to it), trained on this task's training rows minus the gate split
(SN56_DPO_GATE_SPLIT). Its per-pair losses on the gate split are computed HERE, inside the
candidate's own trainer (same tokenizer, max_length / max_prompt_length truncation, collator,
beta 0.1 and task-base reference as the candidate's selection evals), never by another code
path: the selection compares like with like. The vector is cached as {ids, losses} keyed by the
content ID of each pair (sn56_dpo_select.pair_id) and aligned by ID, never by file order.

No test data and no public boss weights: the opponent directory must be our own reproduction.
"""
import glob
import os

import numpy as np
import torch


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def read_weights(d) -> dict:
    """Every tensor of the safetensors files in d (CPU)."""
    from safetensors.torch import load_file
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {d}")
    sd = {}
    for f in files:
        sd.update(load_file(f, device="cpu"))
    return sd


@torch.no_grad()
def load_weights_into(model, d) -> dict:
    """Copy d's weights into model's parameters by name. Every parameter must be covered except a
    tied output head (save_pretrained stores it once). Returns {"loaded", "tied_skipped"}."""
    m = _unwrap(model)
    sd = read_weights(d)
    params = dict(m.named_parameters())
    loaded, skipped = 0, []
    for name, p in params.items():
        t = sd.get(name)
        if t is None:
            skipped.append(name)
            continue
        if tuple(t.shape) != tuple(p.shape):
            raise ValueError(f"{name}: shape {tuple(t.shape)} != {tuple(p.shape)}")
        p.copy_(t.to(device=p.device, dtype=p.dtype))
        loaded += 1
    # a skipped name is acceptable only when it shares storage with a loaded parameter (tied head)
    loaded_ptrs = {p.data_ptr() for n, p in params.items() if n not in skipped}
    untied = [n for n in skipped if params[n].data_ptr() not in loaded_ptrs]
    if untied:
        raise ValueError(f"{len(untied)} parameters missing from {d} (first {untied[0]})")
    return {"loaded": loaded, "tied_skipped": skipped}


def score_weights(trainer, d, loader=None) -> np.ndarray:
    """Per-pair DPO losses of the weights in d on the trainer's eval set (default loader), with
    the trainer's own scoring (sn56_dpo_scale.dev_pair_losses); the policy is restored bit-exact."""
    import sn56_dpo_scale
    model = trainer.model
    pairs = [(n, p) for n, p in _unwrap(model).named_parameters()]
    snap = [p.detach().to("cpu", copy=True) for _, p in pairs]
    try:
        load_weights_into(model, d)
        return sn56_dpo_scale.dev_pair_losses(trainer, model, loader)
    finally:
        with torch.no_grad():
            for (_, p), t in zip(pairs, snap):
                p.copy_(t.to(device=p.device, dtype=p.dtype))


def opponent_vector(trainer, ids, vec_path="", model_dir="", log=print):
    """The opponent losses aligned to ids (the eval set's pair IDs, eval order): read vec_path
    when it exists, else score model_dir with this trainer and write vec_path. None (logged) when
    neither works."""
    from sn56_dpo_select import load_opp_vector, write_opp_vector
    if vec_path and os.path.exists(vec_path):
        try:
            v = load_opp_vector(vec_path, ids)
            log(f"[sn56][oponente] vetor {vec_path}: {v.size} pares, média {np.nanmean(v):.5f}")
            return v
        except Exception as e:
            log(f"[sn56][oponente] vetor {vec_path} inutilizável ({e})")
            if not model_dir:
                return None
    if not model_dir:
        log("[sn56][oponente] sem SN56_DPO_OPP_VEC nem SN56_DPO_OPP_MODEL")
        return None
    v = score_weights(trainer, model_dir)
    if v.size != len(ids):
        log(f"[sn56][oponente] {v.size} perdas para {len(ids)} pares — descartado")
        return None
    if vec_path:
        write_opp_vector(vec_path, ids, v, {"model_dir": os.path.abspath(model_dir), "n": int(v.size)})
    log(f"[sn56][oponente] {model_dir} pontuado no split do portão: {v.size} pares, média {np.nanmean(v):.5f}"
        + (f" -> {vec_path}" if vec_path else ""))
    return v
