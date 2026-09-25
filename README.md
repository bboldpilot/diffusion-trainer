# SN56 tournament trainer

Trainers for the Gradients image and text tournaments. Each lane lists its
base and the mechanisms carried beyond that base.

## Text trainer

`dockerfiles/standalone-text-trainer.dockerfile` + `scripts/`. Base: the
published winning repository of tournament `tourn_20f928c9169811d7_20260810`
(tournament winners are published by design; the tree is vendored verbatim in
one commit so every change below is an attributable diff). Mechanisms on top,
all enabled by default on the instruct/chat training path:

- **Best-fit-decreasing sample packing**
  (`scripts/monkeypatch.py:pack_data_points_by_length`): the base packs with
  Next-Fit — one open bin, closed forever on the first item that does not fit.
  Replaced with best-fit-decreasing under the same contract (index groups,
  `max_size`, over-length singletons). Measured on real tournament data
  (kobprof/skolegpt-instruct, tinyllama tokenizer, 20k rows): 25.0% / 18.3% /
  10.5% fewer bins at max_length 1024/2048/4096, within 0.1-0.8% of the
  ceil(sum/max_length) floor. The budget is wall-clock, so fewer bins is
  directly more optimizer steps per task.

- **Packed-path truncation guard** (`scripts/monkeypatch.py:PackedDataset`):
  the base's OOM ladder halves `max_length` in the request without
  re-tokenizing; items tokenized under the old limit then trip the packing
  length assert on every remaining retry and the run falls through to the
  noise-model hedge. `PackedDataset` now prompt-side-truncates any over-length
  item at collection time. A no-op in normal runs.

- **Adaptive max_length ceiling + effective-batch repair**
  (`scripts/adaptive_max_length.py`, `scripts/instruct_config.py`): the
  adaptive context sizing used the model's full `max_position_embeddings` as
  its ceiling — a long-tailed dataset on a long-context model pushed
  max_length into the tens of thousands and collapsed the batch to 1. Capped
  at 8192. When the batch is rescaled for a grown context, gradient
  accumulation is now re-derived so the effective batch stays at the target
  of 64 instead of silently shrinking with the rescale.

- **Budget-aware gradient checkpointing** (`scripts/instruct_config.py`,
  `scripts/text_trainer.py`): the base hardcodes gradient checkpointing on
  for every instruct run; on sub-2B models with headroom to spare the
  recompute is pure step-time overhead. It is now disabled only when
  params < 2B, max_length ≤ 2048, no KL regularization (the frozen base
  replica doubles the footprint), and ≥ 70GB VRAM. On OOM the retry ladder
  re-enables it before touching the batch size.

- **v5 lane for 2026+ architectures** (`scripts/v5_lane.py`,
  `dockerfiles/standalone-text-trainer.dockerfile`): round 1 now always plays one
  task on a 2026 release (Qwen3.5, gemma-4, LFM2.5, granite-4.1, MiniCPM5). The
  validator evaluates with transformers 5, but the pinned training stack
  (transformers 4.51.3) cannot load qwen3_5 / gemma4 / lfm2 / lfm2_moe at all,
  so the task fails before training starts. The image carries a second venv
  with transformers 5.12.1 + peft 0.19.1 (system torch / flash-attn / bnb shared)
  and `text_trainer` re-execs into it only when the task model's `model_type` is
  unknown to the main install. Inside the lane attention is sdpa and packing is
  off (the Flash-Attention varlen packing patch is validated on 4.x only); the
  4.x path is byte-for-byte unchanged. Also normalises tokenizer configs written
  by transformers 5 (`extra_special_tokens` list, `TokenizersBackend` class) for
  the 4.x tokenization step, which covers the `llama`/`granite`-typed 2026 models
  that do load on 4.x.

## Image trainer — `imagelab/`

Both image dockerfiles the validator builds (`ops/docker/standalone-image-toolkit-trainer.dockerfile`
for ideogram4 / krea2 / qwen-image / z-image, `ops/docker/standalone-image-trainer.dockerfile` for flux)
are identical and produce one image: ComfyUI at the evaluator's revision, torch 2.9.1 / cu128 and the
evaluator's library pins, plus the `imagelab` package. `ENTRYPOINT python -m imagelab.entry` receives the
validator's task flags. No ai-toolkit, no kohya.

imagelab trains a LoRA directly against the tournament's image metric (latent flow-velocity MSE over
16 sigma strata, captioned and empty prompt weighted 0.5 / 0.5) inside ComfyUI's own model-loading and
conditioning path, so training, internal selection and the evaluator share one numerical stack.

- **Metric-identical objective** (`imagelab/objective.py`, `imagelab/data.py`): the evaluator's image
  preparation (1024 long side, 16-px crop, posterior-mean VAE latents), its sigma grid, velocity target and
  per-sample error; the base model is loaded the way the evaluator loads it (ideogram4: fp8 weights kept on
  the fp8 grid).
- **ComfyUI-native adapter** (`imagelab/adapter.py`): low-rank deltas applied through ComfyUI weight
  wrappers (base weights frozen and untouched), exported with the key layout ComfyUI's LoRA loader applies.
- **Holdout replica for checkpoint selection** (`imagelab/replica.py`, `imagelab/loop.py`): held-out
  training images, chosen by a hash-ordered rule, are scored with the metric's structure; raw, EMA and
  delta-averaged states compete, and the best is saved atomically from the first evaluation on.
- **Cross-fit soup** (`--members K`, `imagelab/loop.py`; on for ideogram4, K = 4): the wall is split into
  K slices; member k trains from a fresh initialisation with holdout fold k (rotating windows over the hash
  order), every image is trained on by K-1 members, and the deltas of the members that beat their own base
  are averaged exactly. The soup ships only if it beats the base on fold 0, otherwise the best single member.
- **Budget and delivery** (`imagelab/entry.py`): training runs in a child process with a deadline 180 s
  before the validator's kill; every save is mirrored into
  `/app/checkpoints/<task_id>/<expected_repo_name>/last.safetensors` by rename, so a crash or a kill after
  the first save still ships the latest adapter, and the container exits 0 whenever an adapter exists. The
  trainer runs offline and never sees the validator's credentials.

Per-family hyperparameters live in `imagelab/recipes.py`. The VAE files the evaluator decodes with are
baked at build time by `ops/docker/imagelab_assets.py` (fixed revisions, SHA-256 checked).

### ai-toolkit fallback (not built by the validator)

`ops/docker/aitk-fallback.dockerfile` keeps the previous image trainer — the pinned ai-toolkit stack with
`ops/docker/scripts/image_toolkit_entrypoint.py`, the krea2 prompt-prefix fix, krea2 tail-checkpoint selection,
post-training LoRA strength calibration, the wall-clock guard and `IMG_RECIPE=winner`, which applies four build-time patches ported
verbatim from the 2026-09-07 image winner (`ops/docker/patches/`, origin header in each) — so the lane can
be switched back by copying it over the two image dockerfile names. None of it is part of the image built
from this commit.

## Integrity

All mechanisms in both lanes operate only on the job's own inputs and outputs. The training
container is given a base model and a dataset by the validator and has no network egress;
nothing in this repository fetches an external dataset, benchmark or evaluation data, or an
external adapter, and nothing touches the evaluation pipeline. The only weights added to the
image trainer's image are the evaluator's own VAE files (the ones
`validator/evaluation/image_flow_adapter.py` loads), fetched at build time at fixed revisions and
verified by SHA-256; they encode the task's own training images the way the evaluator encodes its
test images. Checkpoint selection and the cross-fit soup score held-out images from the task's
training set only. Output weights derive solely from the validator-provided base model and dataset.
