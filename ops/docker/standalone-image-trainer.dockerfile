# FLUX tasks resolve to this dockerfile (trainer get_dockerfile_path routes
# ImageModelType.FLUX away from the toolkit path), so a repo without it fails
# every flux task with FileNotFoundError before training starts. We run flux
# through the same ai-toolkit container and entrypoint as the other four image
# model types: the toolkit supports the flux arch and the shared entrypoint
# already routes model-type -> template. One training path, one set of fixes.

FROM diagonalge/ai-toolkit:latest

WORKDIR /app/ai-toolkit

RUN git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

COPY core/training_templates /workspace/core/training_templates
COPY ops/docker/scripts/image_toolkit_entrypoint.py /usr/local/bin/image_toolkit_entrypoint.py
COPY ops/docker/scripts/lora_calibration.py /usr/local/bin/lora_calibration.py

RUN chmod +x /usr/local/bin/image_toolkit_entrypoint.py /usr/local/bin/lora_calibration.py

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "/usr/local/bin/image_toolkit_entrypoint.py"]
