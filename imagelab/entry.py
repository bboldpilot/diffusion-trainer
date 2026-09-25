"""imagelab.entry — the process the validator's image-trainer container runs (the image ENTRYPOINT).

The validator passes the task as CLI flags only and sets no configuration environment (G.O.D @fb8cff26,
trainer/runtime.py run_trainer_container_image). What this wrapper is built around, all from that code:

* Clock. The container is killed hours_to_complete * 3600 s after it was started. After a kill, or after
  a clean exit 0, the output folder is uploaded; after a non-zero exit nothing is uploaded and the task
  counts as failed. So: any saved adapter is shipped and reported with exit 0, however training ended.
* Artefact. The uploader pushes /app/checkpoints/<task_id>/<expected_repo_name>/ into `checkpoints/` of the
  task's HF repo and the evaluator loads the single `last.safetensors` it finds there. A sub-folder holding a
  .safetensors file would be uploaded instead of the folder itself, so the folder holds exactly one file.
* Inputs. /cache is mounted read-only (models/<org>--<name>, hf_cache, datasets/<task_id>_tourn.zip); the
  container sits on an internal bridge without egress. The validator's HF credentials are present in the
  environment; they are not handed to the trainer, which runs fully offline.

Training runs in a child process (imagelab.train). Its saves are mirrored into the output folder as they
appear (staged next to it, renamed into place), so a crash or a kill after the first save still ships the
most recent one. Adapter metadata (selection tags, scores) is dropped from the shipped copy.

  python -m imagelab.entry --task-id ID --model ORG/NAME --dataset-zip URL --model-type ideogram4 \\
      --expected-repo-name NAME --hours-to-complete 0.75 [--trigger-word WORD]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time

LAUNCHED_AT = time.time()

TRAIN_END_BEFORE_KILL_S = 180.0   # trainer deadline; its loop keeps its own save reserve inside this
HARD_STOP_BEFORE_KILL_S = 60.0    # trainer still alive here -> stopped, last save shipped
POLL_S = 5.0
ARTEFACT = "last.safetensors"

_WITHHELD_ENV = ("HUGGINGFACE_TOKEN", "HUGGINGFACE_USERNAME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                 "WANDB_TOKEN", "WANDB_API_KEY")
_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


def _setting(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _say(message: str) -> None:
    print(f"[imagelab] {message}", flush=True)


def _task_flags() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="imagelab.entry", description="Gradients image-task trainer (validator contract).")
    cli.add_argument("--task-id", required=True)
    cli.add_argument("--model", required=True, help="HF repo id; the validator stages it under <cache>/models")
    cli.add_argument("--dataset-zip", required=True, help="dataset URL (unused offline) or a local zip for manual runs")
    cli.add_argument("--model-type", required=True)
    cli.add_argument("--expected-repo-name", required=True)
    cli.add_argument("--hours-to-complete", type=float, required=True)
    cli.add_argument("--trigger-word", default=None)
    return cli


def staged_model(cache_root: str, model: str) -> str:
    """The downloader's layout for image tasks (no anonymisation: image model prep is off)."""
    staged = os.path.join(cache_root, "models", model.replace("/", "--"))
    if os.path.isdir(staged):
        return staged
    if os.path.isdir(model):          # manual runs: --model may point at a local snapshot
        return model
    raise FileNotFoundError(f"model {model!r} is not staged at {staged}")


def staged_dataset(cache_root: str, task_id: str, given: str) -> str:
    staged = os.path.join(cache_root, "datasets", f"{task_id}_tourn.zip")
    if os.path.isfile(staged):
        return staged
    if os.path.isfile(given):         # manual runs: a local zip instead of the validator's URL
        return given
    raise FileNotFoundError(f"training zip is not staged at {staged}")


def trainer_environment(package_root: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _WITHHELD_ENV}
    env.update(_OFFLINE_ENV)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH", "")]))
    return env


def copy_without_metadata(source: str, target: str) -> int:
    """Copy a safetensors file with its `__metadata__` entry removed; tensor bytes are copied verbatim
    (header offsets are relative to the data section, so a shorter header leaves them valid)."""
    with open(source, "rb") as reader:
        (header_len,) = struct.unpack("<Q", reader.read(8))
        header = json.loads(reader.read(header_len))
        header.pop("__metadata__", None)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with open(target, "wb") as writer:
            writer.write(struct.pack("<Q", len(encoded)))
            writer.write(encoded)
            shutil.copyfileobj(reader, writer, 16 << 20)
            writer.flush()
            os.fsync(writer.fileno())
    return os.path.getsize(target)


class Shipper:
    """Mirrors the trainer's latest save into the upload folder; each publish is a rename, so the upload
    folder only ever holds a complete adapter."""

    def __init__(self, produced: str, upload_dir: str):
        self.produced = produced
        self.upload_dir = upload_dir
        self.staging_dir = os.path.join(os.path.dirname(upload_dir), "." + os.path.basename(upload_dir) + ".staging")
        self.shipped_signature: tuple | None = None
        self.shipped_bytes = 0

    def _signature(self) -> tuple | None:
        try:
            info = os.stat(self.produced)
        except FileNotFoundError:
            return None
        return info.st_ino, info.st_size, info.st_mtime_ns

    def sync(self) -> bool:
        signature = self._signature()
        if signature is None or signature == self.shipped_signature:
            return False
        staged = os.path.join(self.staging_dir, ARTEFACT)
        try:
            os.makedirs(self.staging_dir, exist_ok=True)
            os.makedirs(self.upload_dir, exist_ok=True)
            size = copy_without_metadata(self.produced, staged)
            os.replace(staged, os.path.join(self.upload_dir, ARTEFACT))
        except (OSError, ValueError, struct.error) as error:   # e.g. a save swapped mid-read: next poll retries
            _say(f"mirror retry: {type(error).__name__}: {error}")
            return False
        self.shipped_signature, self.shipped_bytes = signature, size
        return True

    def shipped(self) -> bool:
        return os.path.isfile(os.path.join(self.upload_dir, ARTEFACT))

    def close(self) -> None:
        shutil.rmtree(self.staging_dir, ignore_errors=True)


def stop_process_group(process: subprocess.Popen, grace_s: float = 15.0) -> None:
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue


def main(argv: list[str] | None = None) -> int:
    args = _task_flags().parse_args(argv)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from imagelab.families import normalize_family
    from imagelab.recipes import recipe, recipe_argv

    cache_root = _setting("IMAGELAB_CACHE", "/cache")
    output_root = _setting("IMAGELAB_CHECKPOINTS", "/app/checkpoints")
    scratch = _setting("IMAGELAB_WORK", "/tmp/imagelab-run")
    verbose = _setting("IMAGELAB_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}

    family = normalize_family(args.model_type)
    budget_s = args.hours_to_complete * 3600.0
    kill_at = LAUNCHED_AT + budget_s
    train_until = kill_at - min(TRAIN_END_BEFORE_KILL_S, 0.25 * budget_s)
    hard_stop = kill_at - min(HARD_STOP_BEFORE_KILL_S, 0.1 * budget_s)

    command = [sys.executable, "-m", "imagelab.train",
               "--family", family,
               "--model-dir", staged_model(cache_root, args.model),
               "--dataset", staged_dataset(cache_root, args.task_id, args.dataset_zip),
               "--hf-cache", os.path.join(cache_root, "hf_cache"),
               "--baked-dir", _setting("IMAGELAB_ASSETS", "/opt/imagelab/assets"),
               "--comfy-root", _setting("COMFY_ROOT", "/opt/ComfyUI"),
               "--out", scratch,
               "--deadline-ts", f"{train_until:.3f}"] + recipe_argv(recipe(family))
    if args.trigger_word:
        command += ["--trigger-word", args.trigger_word]
    if not verbose:
        command.append("--quiet")
    else:
        _say("launch: " + " ".join(command))

    shipper = Shipper(os.path.join(scratch, ARTEFACT), os.path.join(output_root, args.task_id, args.expected_repo_name))
    shutil.rmtree(scratch, ignore_errors=True)        # never mirror a save left over from an earlier run
    os.makedirs(scratch, exist_ok=True)
    child = subprocess.Popen(command, env=trainer_environment(package_root), start_new_session=True)

    def _forward(signum, _frame):     # manual runs / docker stop: end training, still ship
        stop_process_group(child)
    signal.signal(signal.SIGTERM, _forward)

    stopped_early = False
    try:
        while True:
            try:
                child.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if shipper.sync() and verbose:
                _say(f"mirrored {shipper.shipped_bytes / 1e6:.0f} MB at T+{time.time() - LAUNCHED_AT:.0f}s")
            if time.time() >= hard_stop:
                stopped_early = True
                stop_process_group(child)
                break
        shipper.sync()
    finally:
        shipper.close()

    rc = child.returncode
    if shipper.shipped():
        note = " (stopped at the hard deadline)" if stopped_early else ""
        _say(f"shipped {ARTEFACT} {shipper.shipped_bytes / 1e6:.0f} MB; trainer rc={rc}{note}")
        return 0
    _say(f"no adapter was saved; trainer rc={rc}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
