"""imagelab — an evaluator-native LoRA trainer for the Gradients image tournaments.

The tournament scores a candidate LoRA with a deterministic latent-space flow-velocity MSE: 16 sigma
strata x 16 seeded noises per test image, once with the caption and once with an empty prompt, on the
same ComfyUI stack the evaluator runs. This package trains directly against that objective, inside
ComfyUI's own model-loading and conditioning lifecycle, and selects checkpoints with an internal replica
of the metric on held-out training images. Nothing here reads test data; the container has no network.
"""

__version__ = "0.1.0"
