# SN56 image tournament trainer

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

Both mechanisms operate only on this job's own inputs and outputs. The training
container is given a base model and a dataset by the validator and has no network
egress; nothing in this repository fetches an external dataset, benchmark or
evaluation data, or an external pretrained model or adapter, and nothing touches
the evaluation pipeline. The calibration step is model selection over the adapter
this job just trained, scored on the training images with the code in this repo.
Output weights derive solely from the validator-provided base model and dataset.
