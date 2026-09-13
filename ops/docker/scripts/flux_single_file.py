"""Load a complete validator-provided FLUX checkpoint without network access.

Only architecture configs/tokenizers are baked into the image. All model weights
come from the task checkpoint, including its CLIP, T5 and VAE components.
"""
import json
import struct
from pathlib import Path

CONFIG_ROOT = Path('/opt/flux-single-file-config')
COMPONENT_PREFIXES = {
    'transformer': 'model.diffusion_model.',
    'clip': 'text_encoders.clip_l.transformer.',
    't5': 'text_encoders.t5xxl.transformer.',
    'vae': 'vae.',
}


def resolve_checkpoint(model_path):
    path = Path(model_path)
    if path.is_dir() and (path / 'transformer/config.json').is_file():
        return None  # Keep the existing Diffusers-folder loader.
    candidates = [path] if path.is_file() else sorted(path.glob('*.safetensors'))
    if len(candidates) != 1:
        raise ValueError(f'FLUX needs a Diffusers folder or exactly one root checkpoint: {path}')
    checkpoint = candidates[0]
    with checkpoint.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f'Truncated FLUX checkpoint: {checkpoint}')
        length = struct.unpack('<Q', raw)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, checkpoint.stat().st_size - 8):
            raise ValueError(f'Invalid FLUX checkpoint header: {checkpoint}')
        header = json.loads(stream.read(length))
    keys = set(header) - {'__metadata__'}
    missing = [name for name, prefix in COMPONENT_PREFIXES.items()
               if not any(key.startswith(prefix) for key in keys)]
    if missing:
        raise ValueError(f'FLUX standalone checkpoint lacks {missing}; refusing external-weight fallback')
    print(f'[flux] complete single-file checkpoint: {checkpoint} ({len(keys)} tensors)', flush=True)
    return checkpoint


def load_pipeline(checkpoint, dtype):
    checkpoint = resolve_checkpoint(checkpoint)
    if checkpoint is None:
        raise ValueError('load_pipeline requires a standalone checkpoint')
    if not (CONFIG_ROOT / 'model_index.json').is_file():
        raise FileNotFoundError(f'Build-time FLUX configs missing: {CONFIG_ROOT}')
    from diffusers import FluxPipeline
    pipeline = FluxPipeline.from_single_file(
        str(checkpoint), config=str(CONFIG_ROOT), torch_dtype=dtype,
        local_files_only=True,
    )
    return finish_pipeline(pipeline)


def finish_pipeline(pipeline):
    # The checkpoint stores T5.shared once. The single-file loader materializes
    # it but leaves the tied encoder.embed_tokens alias on meta in the pinned
    # stack. Re-establish the model's declared sharing, never invent weights.
    pipeline.text_encoder_2.tie_weights()
    missing = []
    for component in ('transformer', 'text_encoder', 'text_encoder_2', 'vae'):
        model = getattr(pipeline, component)
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if tensor.is_meta:
                missing.append(f'{component}.{name}')
    if missing:
        raise RuntimeError(f'FLUX checkpoint left unmaterialized tensors: {missing[:20]}')
    return pipeline
