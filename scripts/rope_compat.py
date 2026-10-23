"""Translate v5 RoPE serialization before the pinned v4 stack reads a base.

Keep both representations consistent: v5 accepts rope_scaling as well, and
full-model checkpoints carry the normalized config into validator evaluation.
No model weights, remote code or global transformers monkeypatches are involved.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile


def requires_v5_rope(config: dict) -> bool:
    """Reject layouts/semantics that cannot be represented faithfully on 4.51.3."""
    if not config.get('rope_parameters'):
        return False
    params = config.get('rope_scaling') or config['rope_parameters']
    if not isinstance(params, dict) or any(isinstance(v, dict) for v in params.values()):
        return True
    if 'rope_theta' not in params and 'rope_theta' not in config:
        return True
    kind = params.get('rope_type', params.get('type', 'default'))
    if kind not in {'default', 'linear', 'dynamic', 'yarn', 'longrope', 'llama3'}:
        return True
    if config.get('model_type') == 'phi3' and kind == 'longrope':
        allowed = {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor',
                   'original_max_position_embeddings', 'short_factor', 'long_factor'}
        if set(params) - allowed or params.get('partial_rotary_factor', 1.0) != 1.0:
            return True
    # v4 YaRN derives factor from these lengths; v5 uses explicit factor.
    if kind == 'yarn':
        original = config.get('original_max_position_embeddings',
                              params.get('original_max_position_embeddings', config.get('max_position_embeddings')))
        if not original or params.get('truncate', True) is False:
            return True
        if params.get('factor') != config.get('max_position_embeddings', 0) / original:
            return True
    return False


def normalize_rope_config(config: dict) -> dict:
    """Pure, idempotent conversion; legacy-only configs remain byte-for-byte usable."""
    result = deepcopy(config)
    if not config.get('rope_parameters'):
        return result
    if requires_v5_rope(config):
        raise ValueError('RoPE layout requires the transformers v5 lane')
    # Match v5 convert_rope_params_to_dict precedence, including mixed configs.
    params = deepcopy(config.get('rope_scaling') or config['rope_parameters'])
    kind = params.get('rope_type', params.get('type', 'default'))
    params['rope_type'] = kind
    theta = params.get('rope_theta', config.get('rope_theta', 10000.0))
    params['rope_theta'] = theta
    result['rope_theta'] = theta
    if 'partial_rotary_factor' in config:
        params['partial_rotary_factor'] = config['partial_rotary_factor']
    elif 'partial_rotary_factor' in params:
        result['partial_rotary_factor'] = params['partial_rotary_factor']
    if kind in {'llama3', 'yarn', 'longrope'}:
        params['original_max_position_embeddings'] = config.get(
            'original_max_position_embeddings',
            params.get('original_max_position_embeddings', config['max_position_embeddings']))
    if kind != 'default':
        params['type'] = kind
    result['rope_parameters'] = params
    scaling = {k: v for k, v in params.items() if k not in ('rope_theta', 'partial_rotary_factor')}
    # Phi3's v4 validator requires the old `type` spelling as well.
    if kind != 'default':
        scaling['type'] = kind
    if config.get('model_type') == 'phi3' and kind == 'longrope':
        # Phi3 4.51.3 accepts exactly these three keys, unlike generic RoPE.
        result['original_max_position_embeddings'] = params['original_max_position_embeddings']
        scaling = {k: scaling[k] for k in ('type', 'short_factor', 'long_factor')}
    result['rope_scaling'] = None if kind == 'default' else scaling
    return result


def _model_view(model_path: str) -> Path:
    """Keep the basename and link every entry without writing to the source.

    The view lives until container teardown: tokenizer/trainer subprocesses and
    saved adapter base paths may still reference it after the caller returns.
    """
    source = Path(model_path).absolute()
    parent = Path(tempfile.mkdtemp(prefix="sn56-model-"))
    view = parent / source.name
    try:
        view.mkdir()
        for entry in source.iterdir():
            (view / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    except Exception:
        shutil.rmtree(parent)
        raise
    return view


def prepare_rope_config(model_path: str) -> str:
    """Return a writable config view if needed; never modify the cached base.

    Unsupported layouts stay untouched for maybe_reexec's v5 decision. A v5
    process likewise uses the original path and native config interpretation.
    """
    from v5_lane import transformers_major
    if transformers_major() >= 5:
        return model_path
    original = json.loads((Path(model_path) / 'config.json').read_text())
    if requires_v5_rope(original):
        return model_path
    normalized = normalize_rope_config(original)
    if normalized == original:
        return model_path
    view = _model_view(model_path)
    try:
        config = view / 'config.json'
        config.unlink()  # detach the link before writing the private copy
        config.write_text(json.dumps(normalized, indent=2) + '\n')
    except Exception:
        shutil.rmtree(view.parent)
        raise
    return str(view)


def prepare_tokenizer_config(model_path: str, fixer) -> str:
    """Run the existing tokenizer fixer on a private copy, never through a link.

    Even a RoPE view links tokenizer_config.json into the read-only cache. Only
    when the fixer changes it do we publish a second view with a private copy.
    """
    source = Path(model_path) / 'tokenizer_config.json'
    if not source.is_file():
        return model_path
    with tempfile.TemporaryDirectory(prefix='sn56-tokenizer-') as temporary:
        candidate = Path(temporary) / source.name
        candidate.write_bytes(source.read_bytes())
        if not fixer(temporary):
            return model_path
        view = _model_view(model_path)
        try:
            target = view / source.name
            target.unlink()
            target.write_bytes(candidate.read_bytes())
        except Exception:
            shutil.rmtree(view.parent)
            raise
    return str(view)
