FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    vim \
    zip \
    tmux \
    iotop \
    nvtop \
    bmon \
    wget \
    nano \
    zsh \
    htop \
    && rm -rf /var/lib/apt/lists/*
# Default dir
RUN mkdir -p /workspace
RUN mkdir -p /cache
RUN mkdir -p /workspace/scripts/datasets
RUN mkdir -p /app/checkpoints
WORKDIR /workspace/scripts
# Copy current folder to /workspace/auto_ml
COPY scripts /workspace/scripts
# Make entrypoint script executable
RUN chmod +x /workspace/scripts/entrypoint.sh
# Pytorch (Auto-selects backend https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)

# Create a virtual environment for data generation
RUN python -m venv /workspace/axo_py
RUN bash -c "source /workspace/axo_py/bin/activate && \
    pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install --no-build-isolation axolotl==0.9.1 && \
    pip install requests==2.32.3 && \
    deactivate"


# install the main dependencies
RUN pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install -r /workspace/scripts/training_requirements.txt --system && \
    pip install hf_transfer==0.1.9 && \
    pip install tenacity==9.1.2 && \
    pip install tiktoken==0.9.0 && \
    pip install flash-attn==v2.7.4.post1 --no-build-isolation && \
    uv pip install vllm==0.8.3 --system && \
    pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0"

# v5 lane (scripts/v5_lane.py): a side venv with transformers 5 for architectures the
# pinned 4.51.3 cannot load (qwen3_5, gemma4, lfm2, lfm2_moe, ...). It shares the system
# torch / flash-attn / bitsandbytes / deepspeed through --system-site-packages; only the
# HF stack is overridden. text_trainer re-execs into it solely when the task model's
# model_type is unknown to the main install; every other task is untouched.
# Failure-tolerant on purpose: if this layer cannot be built the venv is removed and the
# image still builds; text_trainer then finds no /workspace/v5_py and behaves as before.
# The venv's own pip, NOT uv: uv resolves without the system site-packages, so it pulled a
# fresh torch (2.14) next to the system 2.4.1 and the system torchvision/torchaudio/flash-attn
# then failed to load inside the lane (transformers import died -> venv silently removed ->
# a 2026+ round-1 task would have errored). pip sees the system torch and leaves it alone.
# The import check covers torchvision because transformers imports it when present, and
# bitsandbytes: transformers 5 gates bnb optimizers on bitsandbytes >= its minimum (the system
# 0.45.4 is below it -> 'You need to install bitsandbytes' inside the lane, 10 Sep TR preflight).
RUN (python -m venv --system-site-packages /workspace/v5_py && \
     /workspace/v5_py/bin/python -m pip install --no-cache-dir \
        "transformers==5.12.1" "peft==0.19.1" "tokenizers==0.22.2" \
        "huggingface_hub>=1.5.0,<2.0" "accelerate==1.14.0" "datasets>=4.7,<6" "bitsandbytes>=0.46" && \
     /workspace/v5_py/bin/python -c "import torch, torchvision, transformers, peft; from transformers.utils.import_utils import is_bitsandbytes_available as b; assert b(); print('v5 lane:', transformers.__version__, peft.__version__, 'torch', torch.__version__, 'bnb ok')") \
    || (echo "v5 lane venv build FAILED - continuing without it" && rm -rf /workspace/v5_py)

ENTRYPOINT ["./entrypoint.sh"]