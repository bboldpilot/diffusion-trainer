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

- **krea2 checkpoint selection on the eval objective** (`ops/docker/scripts/
  lora_calibration.py`, `image_toolkit_entrypoint.py`): the final save is not the
  best save (15-checkpoint sweep: optimum at step 2400 of 3000, 6.2% better than
  the shipped final). The post-training pass now scores the tail checkpoints
  (newest first, 4 images x 1 seed, eval-exact img2img settings) inside a reserve
  scaled to the task budget and ships the minimum when it beats as-trained by
  >1%; strength calibration follows with the remaining budget. Fail-open, atomic
  rename, training images only, no network.

## Image trainer

Diffusion LoRA trainer for the Gradients image tournament, built on the
ai-toolkit training stack. Mechanisms beyond the baseline:

- **krea2 prompt-encoding fix** (`ops/docker/patches/krea2_dynamic_prefix.py`):
  the pinned trainer slices the tokenized system prefix with a fixed constant
  that is only valid for one tokenizer serialization; the patch measures the
  real prefix length at encode time so conditioning stays aligned (and the
  empty-prompt unconditional pass stops producing zero tokens) whatever
  Qwen3-VL snapshot is cached.

- **Post-training LoRA strength calibration**
  (`ops/docker/scripts/lora_calibration.py`): after training, the LoRA's two
  module groups (image-stream `blocks.*`, text-conditioning `txtfusion.*`) are
  re-scored at a small grid of per-group strengths under the evaluation's own
  img2img conditions, and the winning strengths are folded into the saved
  weights. Budget-aware and fail-open: with no time left, or no measured
  improvement, the artefact ships exactly as trained.

- **Wall-clock guard** (`image_toolkit_entrypoint.py`): the harness kills the
  training container at `hours_to_complete` exactly and uploads whatever is on
  disk, but the trainer only writes the evaluator-preferred `last.safetensors`
  on natural completion. The entrypoint stops training shortly before the wall
  and promotes the highest settled checkpoint to that name, so a budget-bound
  run still ships its best work instead of nothing.

- **Blank-prompt conditioning for krea2** (`image_toolkit_entrypoint.py`): the
  krea2 evaluation weights the unconditioned branch 3:1 over the text-guided
  branch, while caption-file training conditions on text ~95% of the time.
  Captions are blanked at dataset prep so the training distribution matches the
  scored distribution.

Layout follows the trainer contract: `ops/docker/standalone-image-toolkit-trainer.dockerfile`
builds the toolkit-era model types, `ops/docker/standalone-image-trainer.dockerfile`
routes flux tasks through the same container and entrypoint, and
`core/training_templates/` carries the per-architecture training configs.

## Integrity

All mechanisms in both lanes operate only on the job's own inputs and outputs. The training
container is given a base model and a dataset by the validator and has no network
egress; nothing in this repository fetches an external dataset, benchmark or
evaluation data, or an external pretrained model or adapter, and nothing touches
the evaluation pipeline. The calibration step is model selection over the adapter
this job just trained, scored on the training images with the code in this repo.
Output weights derive solely from the validator-provided base model and dataset.
