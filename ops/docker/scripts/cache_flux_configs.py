"""Build-time config/tokenizer cache; downloads no pretrained model weights.

Source: the Diffusers team's FLUX.1-dev packaging. Strip its BNB quantization
settings: runtime weights are the task's full checkpoint, not the BNB repo.
"""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'diffusers/FLUX.1-dev-bnb-4bit'
REVISION = '27114c0933fb58aea21f707e113c25e8c6558487'
ROOT = Path('/opt/flux-single-file-config')
FILES = [
    'model_index.json', 'scheduler/scheduler_config.json',
    'transformer/config.json', 'vae/config.json',
    'text_encoder/config.json', 'text_encoder_2/config.json',
    'tokenizer/*', 'tokenizer_2/*',
]


def main():
    snapshot_download(REPO, revision=REVISION, local_dir=str(ROOT), allow_patterns=FILES)
    for component in ('transformer', 'text_encoder', 'text_encoder_2'):
        path = ROOT / component / 'config.json'
        config = json.loads(path.read_text())
        config.pop('quantization_config', None)
        config.pop('_pre_quantization_dtype', None)
        path.write_text(json.dumps(config, indent=2) + '\n')
    (ROOT / 'source.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION,
                                               'weights_downloaded': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
