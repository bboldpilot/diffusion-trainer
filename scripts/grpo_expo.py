"""Post-training weight extrapolation for GRPO, selected on the held-out dev prompts with the validator's own score.

Why (measured on the 21 Sep boss GRPO task): the validator scores a GRPO boss task by
normalising each reward ACROSS the repos evaluated together (validator/evaluation/result_processing.py:
non-negative rewards -> value/max, rewards with negatives and two repos -> 0.25 / 0.75 by rank), weighting, and
subtracting 0.5 * KL(base || model) measured on the PROMPT tokens only. With two repos the rank-based rewards are a
winner-takes-the-0.75 flip, so what decides the task is being better on each reward's mean — not how much better — while
the prompt-token KL of a GRPO fine-tune stays small. The trained policy's direction away from the base is the useful
signal; theta = base + alpha * (trained - base) with alpha > 1 pushes further along it at a KL price the score can
afford. alpha is chosen on the dev prompts (never trained on) by simulating exactly that two-repo score against the
un-extrapolated model (alpha = 1); alpha = 1 is kept unless a larger alpha clears the validator's 1% boss margin.

On by default (SN56_GRPO_EXPO, sn56_plan.py; =0 turns it off). Grid SN56_GRPO_EXPO_GRID, default "1.5,2,3"; dev
prompts SN56_GRPO_EXPO_N, default 200. Guards (30 Oct): skipped when the checkpoint is larger than SN56_GRPO_EXPO_MAX_GB
(default 40: base + candidate must fit next to the trainer on one GPU and twice in host RAM); the grid stops when the
next alpha would not finish before end_time; the winner is written to a temporary file and renamed over
model.safetensors before any old shard is removed, so a kill mid-write leaves a loadable submission.
Guards (31 Oct, CPU-tested only):
  * a LoRA submission (adapter_config.json; every GRPO bucket >= 2B trains LoRA, grpo_config.py) is skipped. Before,
    from_pretrained(adapter dir) loaded base + adapter, load_state_dict(strict=False) ignored every adapter key, the
    candidate sampled bit-identically to alpha=1, the exact tie went to the candidate (stable sort, below) and
    _write_winner deleted adapter_model.safetensors -> a broken submission.
  * any submission tensor that does not map onto the loaded model's state_dict -> skipped (no silent no-op merge).
  * conservative pair: an exact tie on a rank-based reward is a loss for the candidate (the validator's stable sort
    gives the tie to whichever repo it lists second).
SN56_GRPO_EXPO_PARITY (default 1; =0 restores the 30 Oct sampler): sample with exactly the evaluator's generation config
(_gen_config) and cut each completion at the first tokenizer EOS as trl does; KL length rule as the evaluator
(_kl_max_len). Harness-only, unset in production: SN56_GRPO_EXPO_KEEP_DIR (copy of the alpha=1 submission before the
winner overwrites it), SN56_GRPO_EXPO_REPORT (JSON: guards, per-alpha raw/KL/pair/time, chosen alpha, peak memory).
"""
import copy
import json
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F

GEN_TOKENS = 256          # trl GRPOConfig.max_completion_length default (validator eval uses the default)
NUM_GENERATIONS = 2       # validator/evaluation/constants.py GRPO_DEFAULT_NUM_GENERATIONS
KL_MAX_LEN = 512          # validator/evaluation/constants.py GRPO_KL_SEQUENCE_LENGTH
BETA_GRPO = 0.5           # validator/evaluation/constants.py BETA_GRPO
BOSS_MARGIN = 0.01        # validator/tournament/constants.py BOSS_ROUND_WIN_MARGIN
_FALSY = ("0", "false", "no", "off")


def _parity() -> bool:
    return os.environ.get("SN56_GRPO_EXPO_PARITY", "1").strip().lower() not in _FALSY


def eval_cap(n_dev: int) -> int:
    """SN56_GRPO_EVAL_CAP (default 0 = off, train_grpo.py): the trainer's evals sample only the first N dev prompts (dev
    order is shuffled). Every eval, including the end_time one that must finish before run_expo starts, samples each dev
    prompt twice: 21 Sep task, 0.5B, 725 prompts: 295 s of the 660 s reserve; a 1000-prompt 1.5B task may eat all of it."""
    try:
        cap = int(os.environ.get("SN56_GRPO_EVAL_CAP", "0"))
    except ValueError:
        return n_dev
    return min(cap, n_dev) if cap > 0 else n_dev


def _lm_inputs(enc):
    """input_ids + attention_mask only. Tokenizers saved without model_input_names also emit token_type_ids,
    which BloomForCausalLM (and other causal LMs) reject; a forward-signature filter cannot catch it because Bloom
    takes **deprecated_arguments."""
    return {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}


def _gen_config(tok):
    """The evaluator's sampling: trl 1.5.1 GRPOTrainer (use_vllm off) builds this GenerationConfig from the GRPOConfig
    defaults and the ORIGINAL model's tokenizer; under the evaluator's transformers 5 nothing from the model's
    generation_config.json is merged in. Our transformers 4.51.3 generate() fills every key left at its global default
    from model.generation_config (repetition_penalty, an eos_token_id list, ...), so _score swaps the model's
    generation_config for this one while sampling — what trl itself does below transformers 5."""
    from transformers import GenerationConfig
    return GenerationConfig(max_new_tokens=GEN_TOKENS, do_sample=True, pad_token_id=tok.pad_token_id,
                            bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, temperature=1.0, top_p=1.0,
                            top_k=0, min_p=None, repetition_penalty=1.0)


def _kl_max_len(model) -> int:
    """validator/evaluation/common.py calculate_kl_divergence: GRPO_KL_SEQUENCE_LENGTH, or ceil(max_position_embeddings
    / 2) when the window is shorter than twice that. (Its batch size is GRPO_KL_BATCH_SIZE = 1: one unpadded prompt per
    forward, as here.)"""
    mpe = getattr(model.config, "max_position_embeddings", None)
    return math.ceil(mpe / 2) if mpe and mpe < 2 * KL_MAX_LEN else KL_MAX_LEN


def _cut_at_eos(rows, eos_id):
    """trl masks everything after the first EOS before decoding; generate() pads finished rows instead."""
    if eos_id is None:
        return rows
    return [r[: r.index(eos_id) + 1] if eos_id in r else r for r in rows]


def _tensors(path):
    from safetensors import safe_open
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as h:
                for k in h.keys():
                    out[k] = h.get_tensor(k)
    return out


def _se_diff(x, y):
    def var(v):
        m = sum(v) / len(v)
        return sum((t - m) ** 2 for t in v) / max(1, len(v) - 1)
    return math.sqrt(var(x) / len(x) + var(y) / len(y))


def _normalized_pair(a: dict, b: dict, weights: list[float], conservative: bool = False) -> tuple[float, float]:
    """validator.evaluation.result_processing.normalize_rewards_and_compute_loss for two repos (a = REF, b = candidate).
    conservative=True: b wins a rank-based (negative-valued) reward only if its mean exceeds a's by > 2 SE on dev; an
    exact tie is a loss (the validator's stable sort would hand it to b only because b is listed second)."""
    sa = sb = 0.0
    for i, w in enumerate(weights):
        va, vb = a["raw"][i], b["raw"][i]
        if conservative and min(va, vb) < 0 and vb >= va and (vb - va) <= 2 * _se_diff(a["per"][i], b["per"][i]):
            vb = va - 1e-9   # not a significant win on dev -> count it as lost
        mn = min(va, vb)
        if mn < 0:
            # two repos with a negative -> 0.25 / 0.75 by (stable) rank; a is listed first
            na, nb = (0.25, 0.75) if sorted([("a", va - mn), ("b", vb - mn)], key=lambda x: x[1])[0][0] == "a" else (0.75, 0.25)
        else:
            mx = max(va, vb)
            na, nb = (va / mx, vb / mx) if mx > 0 else (1.0, 1.0)
        sa += w * na; sb += w * nb
    return sa - BETA_GRPO * a["kl"], sb - BETA_GRPO * b["kl"]


@torch.no_grad()
def _score(model, base, tok, prompts, funcs, seed, parity=False):
    model.eval()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok.padding_side = "left"
    comps = []
    bs = 32
    gen_cfg = _gen_config(tok) if parity else None
    saved_cfg = model.generation_config
    if gen_cfg is not None:
        model.generation_config = copy.deepcopy(gen_cfg)
    try:
        for i in range(0, len(prompts), bs):
            chunk = [p for p in prompts[i:i + bs] for _ in range(NUM_GENERATIONS)]
            enc = _lm_inputs(tok(chunk, return_tensors="pt", padding=True).to(model.device))
            if gen_cfg is not None:
                out = model.generate(**enc, generation_config=gen_cfg)
                rows = _cut_at_eos(out[:, enc["input_ids"].shape[1]:].tolist(), tok.eos_token_id)
                comps += tok.batch_decode(rows, skip_special_tokens=True)
            else:
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=GEN_TOKENS,
                                     pad_token_id=tok.pad_token_id)
                comps += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        model.generation_config = saved_cfg
    raw, per = [], []
    for f in funcs:
        r = [float(x) for x in f(comps)]
        per.append(r)
        raw.append(sum(r) / len(r))
    tok.padding_side = "right"
    kl_len = _kl_max_len(model) if parity else KL_MAX_LEN
    kl_sum, n = 0.0, 0
    for p in prompts:
        enc = _lm_inputs(tok([p], return_tensors="pt", truncation=True, max_length=kl_len).to(model.device))
        lo = base(**enc).logits.float(); lf = model(**enc).logits.float()
        kl = F.kl_div(F.log_softmax(lf, -1), F.softmax(lo, -1), reduction="none").sum(-1).mean(-1).sum().item()
        kl_sum += kl; n += 1
    return {"raw": raw, "per": per, "kl": kl_sum / max(n, 1), "mean_words": sum(len(c.split()) for c in comps) / max(1, len(comps))}


def _seconds_left(train_request: dict) -> float:
    """Seconds to the task's end_time ("%Y-%m-%d %H:%M:%S" UTC, text_trainer.py); inf when absent."""
    import datetime
    end = train_request.get("end_time")
    if not end:
        return float("inf")
    try:
        t_end = datetime.datetime.strptime(str(end), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return float("inf")
    return (t_end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()


def _weights_gb(path: str) -> float:
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors")) / 1e9


def _write_winner(sub: str, sd: dict) -> None:
    """Replace the submission's weights with sd, loadable at every instant: temp file, rename over
    model.safetensors (loaders take it before an index), then drop the old shards and index."""
    from safetensors.torch import save_file
    tmp = os.path.join(sub, ".expo-model.safetensors.tmp")
    save_file(sd, tmp, metadata={"format": "pt"})
    os.replace(tmp, os.path.join(sub, "model.safetensors"))
    for f in os.listdir(sub):
        if (f.endswith(".safetensors") and f != "model.safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(sub, f))


def _write_report(rep: dict, log) -> None:
    path = os.environ.get("SN56_GRPO_EXPO_REPORT")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".tmp", "w") as fh:
            json.dump(rep, fh, indent=1)
        os.replace(path + ".tmp", path)
    except Exception as e:  # never fail the submission over a harness report
        log(f"[sn56][expo] report not written ({e!r})")


def run_expo(train_request: dict, reward_functions: list, log=print) -> None:
    """reward_functions: list of (callable, weight) with RAW callables (unweighted), task order."""
    t0 = time.time()
    rep = {"parity": _parity(), "alphas": [], "chosen_alpha": 1.0, "skipped": None}
    try:
        _run(train_request, reward_functions, log, rep)
    except Exception as e:
        rep["error"] = repr(e)[:500]
        raise
    finally:
        rep["seconds"] = round(time.time() - t0, 1)
        _write_report(rep, log)


def _run(train_request: dict, reward_functions: list, log, rep: dict) -> None:
    t0 = time.time()

    def skip(reason):
        rep["skipped"] = reason
        log(f"[sn56][expo] skipped: {reason}")

    sub = train_request["submission_dir"]
    base_path = train_request["model_path"]
    if os.path.exists(os.path.join(sub, "adapter_config.json")):
        return skip("LoRA adapter submission (adapter_config.json): not extrapolated, adapter left untouched")
    max_gb = float(os.environ.get("SN56_GRPO_EXPO_MAX_GB", "40"))
    size_gb = max(_weights_gb(sub), _weights_gb(base_path))
    rep["size_gb"] = round(size_gb, 3)
    if size_gb > max_gb:
        return skip(f"{size_gb:.1f} GB > SN56_GRPO_EXPO_MAX_GB={max_gb:g}")
    dev_path = os.path.join("datasets", f"grpo_dev_{train_request['task_id']}.json")
    grid = [float(x) for x in os.environ.get("SN56_GRPO_EXPO_GRID", "1.5,2,3").split(",") if x.strip()]
    n = int(os.environ.get("SN56_GRPO_EXPO_N", "200"))
    parity = rep["parity"]
    field = train_request["dataset_type"].get("field_prompt") or "prompt"
    prompts = [str(r.get(field, r.get("prompt", ""))) for r in json.load(open(dev_path))][:n]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    free_gb = None
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        free_gb = round(torch.cuda.mem_get_info()[0] / 1e9, 1)
    rep.update({"grid": grid, "n_prompts": len(prompts), "free_gpu_gb_at_start": free_gb,
                "seconds_left_at_start": round(min(_seconds_left(train_request), 1e9))})
    log(f"[sn56][expo] start: {size_gb:.1f} GB (max {max_gb:g}), {_seconds_left(train_request):.0f}s to end_time, "
        f"grid={grid}, n={len(prompts)}, parity={int(parity)}, free_gpu={free_gb} GB")
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16).to(dev).eval()
    model = AutoModelForCausalLM.from_pretrained(sub, torch_dtype=torch.bfloat16).to(dev).eval()
    B, T = _tensors(base_path), _tensors(sub)
    names = set(model.state_dict())
    unknown = sorted(k for k in T if k not in names)
    moved = sum(1 for k, t in T.items() if k in B and B[k].shape == t.shape)
    rep["tensors"] = {"moved": moved, "copied": len(T) - moved, "unknown": len(unknown)}
    if unknown or not moved:
        return skip(f"submission tensors do not map onto the loaded model ({len(unknown)} unknown, e.g. {unknown[:2]}; "
                    f"{moved} extrapolable): a merge would be a silent no-op")
    funcs = [f for f, _ in reward_functions]; weights = [w for _, w in reward_functions]
    t_ref = time.time()
    ref = _score(model, base, tok, prompts, funcs, seed=42, parity=parity)
    t_alpha = time.time() - t_ref
    rep["alphas"].append({"alpha": 1.0, "raw": ref["raw"], "kl": ref["kl"], "words": ref["mean_words"],
                          "seconds": round(t_alpha, 1)})
    log(f"[sn56][expo] alpha=1.0 raw={['%.4f' % x for x in ref['raw']]} kl={ref['kl']:.5f} words={ref['mean_words']:.1f} "
        f"({t_alpha:.0f}s, {_seconds_left(train_request):.0f}s left)")
    best_alpha, best_gain, best_sd = 1.0, 0.0, None
    for a in grid:
        # one more alpha (score + merge) and the final write must fit before end_time
        if _seconds_left(train_request) < 1.5 * t_alpha + 60 + 10 * size_gb:
            rep["grid_stopped_before"] = a
            log(f"[sn56][expo] grid stopped before alpha={a}: {_seconds_left(train_request):.0f}s left")
            break
        ta = time.time()
        sd = {}
        for k, t in T.items():
            if k in B and B[k].shape == t.shape:
                b = B[k].to(dev).float(); sd[k] = (b + a * (t.to(dev).float() - b)).to(t.dtype)
            else:
                sd[k] = t.to(dev)
        res = model.load_state_dict(sd, strict=False)
        if res.unexpected_keys:
            return skip(f"load_state_dict ignored {len(res.unexpected_keys)} keys at alpha={a}")
        if getattr(model.config, "tie_word_embeddings", False):
            model.tie_weights()
        s = _score(model, base, tok, prompts, funcs, seed=42, parity=parity)
        s_ref, s_a = _normalized_pair(ref, s, weights, conservative=True)
        bar = s_ref + abs(s_ref) * BOSS_MARGIN
        gain = s_a - bar
        rep["alphas"].append({"alpha": a, "raw": s["raw"], "kl": s["kl"], "words": s["mean_words"], "pair_ref": s_ref,
                              "pair_alpha": s_a, "bar": bar, "seconds": round(time.time() - ta, 1)})
        log(f"[sn56][expo] alpha={a} raw={['%.4f' % x for x in s['raw']]} kl={s['kl']:.5f} words={s['mean_words']:.1f} "
            f"pair: ref={s_ref:.4f} alpha={s_a:.4f} bar={bar:.4f} ({time.time() - ta:.0f}s)")
        if gain > best_gain and math.isfinite(s["kl"]):
            best_alpha, best_gain, best_sd = a, gain, {k: v.cpu().contiguous() for k, v in sd.items()}
    if best_sd is not None:
        keep = os.environ.get("SN56_GRPO_EXPO_KEEP_DIR")
        if keep:
            shutil.rmtree(keep, ignore_errors=True); shutil.copytree(sub, keep)
        tw = time.time()
        _write_winner(sub, best_sd)
        rep["write_s"] = round(time.time() - tw, 1)
        with open(os.path.join(sub, "loss.txt"), "a") as fh:
            fh.write(f"\nexpo_alpha={best_alpha}")
    rep.update({"chosen_alpha": best_alpha, "gain_over_bar": best_gain})
    if dev == "cuda":
        rep["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    log(f"[sn56][expo] chosen alpha={best_alpha} gain_over_bar={best_gain:.4f} ({time.time() - t0:.0f}s)")
    del base, model
    torch.cuda.empty_cache()
