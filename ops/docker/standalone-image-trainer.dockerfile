# Gradients image tasks -> imagelab, on the image evaluator's own numerical stack.
#
# How the validator uses this file (G.O.D trainer/runtime.py): it builds it with the repository root as
# context, without layer cache and with a 30-minute limit, through the Docker Engine /build API (classic
# builder: no BuildKit-only syntax here), then runs it on one H100 with the task as CLI flags only, /cache
# read-only and no network egress. FLUX tasks build ops/docker/standalone-image-trainer.dockerfile, every
# other family this file; the two files are identical so all families run the same image.
#
# For reviewers: the only weights added at build time are the VAE files the validator's image evaluator
# itself loads (validator/evaluation/image_flow_adapter.py, FAMILIES), pinned by revision and checked by
# SHA-256 in ops/docker/imagelab_assets.py. They turn the task's own training images into latents exactly
# as the evaluator turns its test images into latents. No dataset, adapter or other model enters the image,
# and nothing is downloaded while training.
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9

ARG COMFYUI_REVISION=694815f498295080a0e15a1502edc9dba841b110
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=10
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/imagelab/lib \
    COMFY_ROOT=/opt/ComfyUI \
    IMAGELAB_ASSETS=/opt/imagelab/assets \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ComfyUI source at the evaluator's revision, fetched the way the evaluator image fetches it.
RUN mkdir -p /opt/ComfyUI \
 && git -C /opt/ComfyUI init -q \
 && git -C /opt/ComfyUI fetch -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_REVISION}" \
 && git -C /opt/ComfyUI checkout -q FETCH_HEAD \
 && rm -rf /opt/ComfyUI/.git

# Evaluator pins: torch 2.9.1 / cu128, then ComfyUI's requirements + the evaluator's library pins.
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
COPY ops/docker/requirements/imagelab.txt ops/docker/requirements/imagelab-lock.txt /opt/imagelab/requirements/
RUN pip install -c /opt/imagelab/requirements/imagelab-lock.txt \
        -r /opt/ComfyUI/requirements.txt -r /opt/imagelab/requirements/imagelab.txt

COPY ops/docker/imagelab_assets.py /opt/imagelab/imagelab_assets.py
RUN python /opt/imagelab/imagelab_assets.py /opt/imagelab/assets

COPY imagelab /opt/imagelab/lib/imagelab
RUN python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.9.1') else 'torch ' + torch.__version__)" \
 && python -m compileall -q /opt/imagelab/lib \
 && python -m imagelab.entry --help > /dev/null \
 && mkdir -p /app/checkpoints

WORKDIR /opt/imagelab
ENTRYPOINT ["python", "-m", "imagelab.entry"]
