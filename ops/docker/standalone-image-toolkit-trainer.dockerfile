FROM diagonalge/ai-toolkit:latest

WORKDIR /app/ai-toolkit

RUN git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# krea2 prompt encoding: replace the fixed system-prefix slice with one measured
# under the tokenizer actually loaded (see the patch header for the failure it
# closes). The build fails loudly if the pinned source drifts.
COPY ops/docker/patches/krea2_dynamic_prefix.py /tmp/krea2_dynamic_prefix.py
RUN python3 /tmp/krea2_dynamic_prefix.py

COPY core/training_templates /workspace/core/training_templates
COPY ops/docker/scripts/image_toolkit_entrypoint.py /usr/local/bin/image_toolkit_entrypoint.py

# INTEGRITY DECLARATION (challenger code review): lora_calibration.py runs AFTER
# training and is a model-selection step over the miner's own trained adapter.
# Its only inputs are the validator-provided base model and the validator-provided
# training images already staged at /dataset/images. It consumes NO external
# dataset, NO benchmark or evaluation data, and NO external pretrained model or
# adapter; the container has no network egress at training time. It does not read,
# write, or influence any part of the evaluation pipeline -- it re-scores the
# adapter this job just trained, on the training images, and rescales that
# adapter's own weights. Output weights derive solely from the validator-provided
# base model and the validator-provided dataset.
COPY ops/docker/scripts/lora_calibration.py /usr/local/bin/lora_calibration.py

RUN chmod +x /usr/local/bin/image_toolkit_entrypoint.py /usr/local/bin/lora_calibration.py

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "/usr/local/bin/image_toolkit_entrypoint.py"]
