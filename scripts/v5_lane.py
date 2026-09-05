"""v5 lane: run a task on a side-by-side transformers 5 install when the pinned
main install (transformers 4.51.3) does not know the task model's architecture.

Why: the tournament's round-1 model pool now always includes one task on a 2026+
release (qwen3_5, gemma4, lfm2, lfm2_moe, ...). Those architectures exist only in
transformers >= 4.57 / 5.x. The validator's own evaluation stack runs transformers 5,
so such a task is evaluable — the only thing that cannot load the model is a 4.x
trainer. Rather than upgrade (and re-validate) the whole training stack, the image
carries a second venv (/workspace/v5_py, transformers 5 + peft, sharing the system
torch/flash-attn/bitsandbytes) and `text_trainer` re-execs itself into it only when
the model_type is unknown to the main install. Every other task runs exactly as
before.

Behavioural differences inside the lane (see instruct_config.get_training_json):
attention defaults to sdpa and sample packing is off — the Flash-Attention varlen
monkeypatch is validated only on the 4.x path. Both are overridable through
SN56_V5_ATTN / SN56_V5_PACKING for smoke tests.
"""

import json
import os
import sys

V5_PYTHON = "/workspace/v5_py/bin/python"
LANE_ENV = "SN56_V5_LANE"


def is_v5_lane() -> bool:
    return os.environ.get(LANE_ENV) == "1"


def transformers_major() -> int:
    import transformers

    try:
        return int(str(transformers.__version__).split(".")[0])
    except (ValueError, IndexError):
        return 4


def model_type_of(model_path: str):
    """model_type from a local model dir's config.json (composite configs nest it)."""
    try:
        with open(os.path.join(model_path, "config.json"), "r") as f:
            cfg = json.load(f)
    except Exception:
        return None
    model_type = cfg.get("model_type")
    if not model_type and isinstance(cfg.get("text_config"), dict):
        model_type = cfg["text_config"].get("model_type")
    return model_type


def needs_v5(model_path: str) -> bool:
    """True when the *current* transformers cannot build a config for this model."""
    model_type = model_type_of(model_path)
    if not model_type:
        return False
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except Exception:
        return False
    return model_type not in CONFIG_MAPPING_NAMES


def maybe_reexec(model_path: str, argv) -> None:
    """Replace this process with the v5 venv's interpreter when the model needs it.

    No-op when already in the lane, when the model loads on the main install, or
    when the image has no v5 venv (then the run proceeds — and fails — as before).
    """
    if is_v5_lane() or not needs_v5(model_path) or not os.path.exists(V5_PYTHON):
        return
    import transformers

    print(
        f"[sn56][v5-lane] model_type={model_type_of(model_path)!r} is unknown to "
        f"transformers {transformers.__version__}; re-exec under {V5_PYTHON}",
        flush=True,
    )
    sys.stdout.flush()
    os.environ[LANE_ENV] = "1"
    os.execv(V5_PYTHON, [V5_PYTHON, "-m", "text_trainer", *list(argv)[1:]])


def dtype_kw(dtype) -> dict:
    """from_pretrained keyword for the weights dtype: `dtype` on v5, `torch_dtype` on v4."""
    return {"dtype": dtype} if transformers_major() >= 5 else {"torch_dtype": dtype}


def trainer_tokenizer_kw(tokenizer) -> dict:
    """Trainer(...) keyword for the tokenizer: `processing_class` on v5, `tokenizer` on v4."""
    return {"processing_class": tokenizer} if transformers_major() >= 5 else {"tokenizer": tokenizer}


def launcher(gpu_nums: int, run_type: str) -> str:
    """Command prefix that starts train_instruct.py under the right interpreter.

    Outside the lane this is byte-identical to the historical behaviour
    (`python` / `torchrun` / `deepspeed` from PATH). Inside the lane the
    venv interpreter must be used explicitly, because PATH still resolves to
    the system python and its transformers 4.x.
    """
    if not is_v5_lane():
        if gpu_nums > 1 and run_type == "ddp":
            return f"torchrun --nproc_per_node={gpu_nums}"
        if run_type == "ds":
            return "deepspeed"
        return "python"
    py = sys.executable
    if gpu_nums > 1 and run_type == "ddp":
        return f"{py} -m torch.distributed.run --nproc_per_node={gpu_nums}"
    if run_type == "ds":
        return f"{py} -m deepspeed.launcher.runner"
    return py
