#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
import zipfile
from pathlib import Path

import yaml


CACHE_ROOT = Path("/cache")
CONFIG_TEMPLATE_ROOT = Path("/workspace/core/training_templates")
CONFIG_OUTPUT_ROOT = Path("/dataset/configs")
DATASET_ZIP_ROOT = CACHE_ROOT / "datasets"
IMAGE_DATASET_ROOT = Path("/dataset/images")
CHECKPOINTS_ROOT = Path("/app/checkpoints")
AI_TOOLKIT_ROOT = Path("/app/ai-toolkit")

TEMPLATE_BY_MODEL_TYPE = {
    "flux": "base_diffusion_flux.yaml",
    "z-image": "base_diffusion_zimage.yaml",
    "qwen-image": "base_diffusion_qwen_image.yaml",
    "ideogram4": "base_diffusion_ideogram4.yaml",
    "krea2": "base_diffusion_krea2.yaml",
}
IDEOGRAM4_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-4B-Instruct"

# The harness kills this container at hours_to_complete*3600 exactly and uploads
# whatever is on disk. Keep a slice of the budget back from training: krea2
# needs room for the post-train calibration, every other type just needs enough
# for the SIGTERM -> checkpoint-promote path below.
CALIBRATION_RESERVE_S = 420
PROMOTE_RESERVE_S = 120
TERM_GRACE_S = 30

NUMBERED_CKPT_RE = re.compile(r"[-_](\d+)\.safetensors$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--model-type", required=True, choices=sorted(TEMPLATE_BY_MODEL_TYPE))
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True)
    parser.add_argument("--trigger-word")
    return parser.parse_args()


def cached_model_path(model: str) -> Path:
    cache_name = model.replace("/", "--")
    return CACHE_ROOT / "models" / cache_name


def prepare_dataset(task_id: str, model_type: str) -> Path:
    zip_path = DATASET_ZIP_ROOT / f"{task_id}_tourn.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected cached image dataset zip at {zip_path}")

    if IMAGE_DATASET_ROOT.exists():
        shutil.rmtree(IMAGE_DATASET_ROOT)
    IMAGE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    extract_root = Path("/dataset/extracted") / task_id
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)

    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".txt"}
    copied = 0
    for source in extract_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in supported_suffixes:
            destination = IMAGE_DATASET_ROOT / source.name
            if destination.exists():
                destination = IMAGE_DATASET_ROOT / f"{source.parent.name}_{source.name}"
            shutil.copy2(source, destination)
            copied += 1

    if copied == 0:
        raise RuntimeError(f"No supported image/caption files found in {zip_path}")

    if model_type == "krea2":
        # Train the conditioning the eval actually scores. The krea2 eval weights
        # the blank-prompt branch 3x the text-guided branch, but the stock
        # template trains on the dataset captions ~95% of the time, so the model
        # the eval sees most is the one that trained least. Blanking the caption
        # files (the trigger word stays, it comes from the config) measured 3.9%
        # better composite than caption-trained on the same task, 5 seeds.
        blanked = 0
        for caption in IMAGE_DATASET_ROOT.glob("*.txt"):
            caption.write_text("")
            blanked += 1
        print(f"[krea2] blanked {blanked} caption files (blank-prompt branch is 75% of eval)", flush=True)

    return IMAGE_DATASET_ROOT


def prepare_config(args: argparse.Namespace, dataset_path: Path) -> Path:
    template_path = CONFIG_TEMPLATE_ROOT / TEMPLATE_BY_MODEL_TYPE[args.model_type]
    if not template_path.exists():
        raise FileNotFoundError(f"Missing ai-toolkit template for {args.model_type}: {template_path}")

    with template_path.open() as file:
        config = yaml.safe_load(file)

    config_body = config.setdefault("config", {})
    # "last", NOT expected_repo_name. ai-toolkit names the final checkpoint
    # "<config.name>.safetensors", and the evaluator picks the submission by
    # looking for a file named exactly "last.safetensors" -- its fallback is a
    # numbered-checkpoint regex the unnumbered final file does not match, so a
    # repo-named config silently submits the SECOND-to-last checkpoint and the
    # final save never leaves the container. (The HF repo name comes from the
    # uploader's EXPECTED_REPO_NAME env, not from this field.)
    config_body["name"] = "last"

    process = config_body.setdefault("process", [{}])[0]
    process["training_folder"] = str(CHECKPOINTS_ROOT / args.task_id)
    process["trigger_word"] = args.trigger_word

    datasets = process.setdefault("datasets", [{}])
    datasets[0]["folder_path"] = str(dataset_path)

    model_config = process.setdefault("model", {})
    model_path = cached_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected cached base model at {model_path}")
    # Trainer container has no internet; ai-toolkit must load every component from
    # the local cache dir. Its ideogram4 loader uses the local branch when
    # <name_or_path>/<subfolder> exists, so point it at the downloaded snapshot.
    model_config["name_or_path"] = str(model_path)

    if args.model_type == "ideogram4":
        text_encoder_path = IDEOGRAM4_TEXT_ENCODER_CACHE
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Expected cached Ideogram 4 text encoder at {text_encoder_path}"
            )
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)

    elif args.model_type == "krea2":
        text_encoder_path = KREA2_TEXT_ENCODER_CACHE
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Expected cached Krea 2 text encoder at {text_encoder_path}"
            )
        vae_path = model_path / "vae"
        if not vae_path.exists():
            raise FileNotFoundError(f"Expected cached Krea 2 VAE at {vae_path}")
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)
        # Krea2Model appends the "vae" subfolder when loading the VAE.
        model_kwargs["vae_path"] = str(model_path)

    CONFIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_OUTPUT_ROOT / f"{args.task_id}.yaml"
    with output_path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return output_path


def _calibration_model_kwargs(args: argparse.Namespace) -> dict:
    """The same component paths prepare_config resolves for krea2, so the
    calibration scorer loads the model from the identical local pieces the
    training run just used."""
    model_path = cached_model_path(args.model)
    return {
        "text_encoder_path": str(KREA2_TEXT_ENCODER_CACHE),
        "vae_path": str(model_path),
    }


def promote_latest_checkpoint(training_folder: Path) -> None:
    """Copy the highest-step numbered checkpoint to last.safetensors.

    ai-toolkit only writes the unnumbered final file when the run COMPLETES, and
    the evaluator's picker wants a file named exactly last.safetensors. A run
    stopped at the wall would otherwise ship nothing usable even with 90% of the
    training done. Skips files modified in the last 30s -- the trainer may have
    been mid-save when it was stopped, and a truncated safetensors scores zero.
    """
    settled: list[tuple[int, Path]] = []
    fresh: list[tuple[int, Path]] = []
    newest_ok = time.time() - 30
    for path in training_folder.rglob("*.safetensors"):
        if path.name == "last.safetensors":
            return  # completed naturally; the real final file wins
        match = NUMBERED_CKPT_RE.search(path.name)
        if match:
            bucket = settled if path.stat().st_mtime < newest_ok else fresh
            bucket.append((int(match.group(1)), path))
    # A fresh file is only worth the truncation risk when it is all there is.
    candidates = settled or fresh
    if not candidates:
        print("[wall] no promotable checkpoint found", flush=True)
        return
    step, source = max(candidates)
    target = source.parent / "last.safetensors"
    shutil.copy2(source, target)
    print(f"[wall] promoted step-{step} checkpoint to {target}", flush=True)


def run_training(config_path: Path, train_deadline: float) -> None:
    """run.py, but stopped in time for the calibration reserve.

    The harness SIGKILLs the whole container at the hour mark with no grace, so
    the entrypoint has to be the one that stops training early enough to leave
    the reserve intact. SIGTERM first so ai-toolkit can finish the step; SIGKILL
    only if it lingers.
    """
    proc = subprocess.Popen(["python3", "run.py", str(config_path)], cwd=AI_TOOLKIT_ROOT)
    while True:
        try:
            proc.wait(timeout=min(30.0, max(1.0, train_deadline - time.time())))
            return  # completed within budget (rc checked by caller via promote)
        except subprocess.TimeoutExpired:
            pass
        if time.time() >= train_deadline:
            print(f"[wall] training budget spent, stopping run.py (pid {proc.pid})", flush=True)
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            return


def main() -> None:
    started = time.time()
    args = parse_args()
    deadline = started + float(args.hours_to_complete) * 3600.0
    dataset_path = prepare_dataset(args.task_id, args.model_type)
    config_path = prepare_config(args, dataset_path)

    # Every image type gets the wall guard: the stock templates (2000 steps at
    # ~1.7-2s/step for flux) sit at or over a 1.0h budget on the harness GPU, so
    # an unguarded run regularly dies mid-save with nothing named for the
    # evaluator to pick up.
    training_folder = CHECKPOINTS_ROOT / args.task_id
    reserve = CALIBRATION_RESERVE_S if args.model_type == "krea2" else PROMOTE_RESERVE_S
    run_training(config_path, deadline - reserve)
    promote_latest_checkpoint(training_folder)

    if args.model_type == "krea2":
        # Post-training strength calibration (see lora_calibration.py). check=False
        # and a hard deadline inside: whatever calibration does or fails to do, the
        # artefact ai-toolkit saved is already on disk and stays the submission
        # unless calibration finds a measured improvement to fold in.
        subprocess.run(
            [
                "python3",
                "/usr/local/bin/lora_calibration.py",
                "--training-folder", str(CHECKPOINTS_ROOT / args.task_id),
                "--dataset-dir", str(dataset_path),
                "--base-model", str(cached_model_path(args.model)),
                "--model-kwargs", json.dumps(_calibration_model_kwargs(args)),
                "--deadline-epoch", str(deadline),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
