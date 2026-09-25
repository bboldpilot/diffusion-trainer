"""Build step: bake the VAE files the tournament's image evaluator decodes with.

The evaluator loads these exact Comfy-Org files (G.O.D validator/evaluation/image_flow_adapter.py, FAMILIES);
the trainer encodes the task's own training images with them so its latents match the evaluator's. Each file
is fetched at a fixed revision and must match its published size and SHA-256, otherwise the build fails.
Krea-2's qwen_image_vae.safetensors is byte-identical to Comfy-Org/Qwen-Image_ComfyUI's (qwen-image), and
z_image_turbo's ae.safetensors to rayonlabs/FLUX.1-dev's (flux), so three files cover all five families.
Nothing here runs at training time (the training container has no network).

  python imagelab_assets.py /opt/imagelab/assets          # fetch + verify
  python imagelab_assets.py /opt/imagelab/assets --check  # verify only
"""
import hashlib
import os
import sys
import time
import urllib.request

PINNED = (
    # file name (what imagelab.families expects), repo, revision, path in repo, bytes, sha256
    ("flux2-vae.safetensors", "Comfy-Org/Ideogram-4", "2aa6c75ce6d5fabded0ca4d0f76abbfaf8edc87d",
     "vae/flux2-vae.safetensors", 336211292, "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"),
    ("qwen_image_vae.safetensors", "Comfy-Org/Krea-2", "eb1eddd3983a54678545a9b2c178c5853b30f7be",
     "vae/qwen_image_vae.safetensors", 253806246, "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
    ("ae.safetensors", "Comfy-Org/z_image_turbo", "6fc90a3b1b653e935a0d175e260736de25b84df5",
     "split_files/vae/ae.safetensors", 335304388, "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"),
)


def digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def verified(path: str, size: int, sha256: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) == size and digest(path) == sha256


def fetch(url: str, target: str, attempts: int = 5) -> None:
    partial = target + ".part"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
                while block := response.read(8 << 20):
                    out.write(block)
            os.replace(partial, target)
            return
        except OSError as error:
            print(f"fetch attempt {attempt}/{attempts} failed: {error}", flush=True)
            time.sleep(5 * attempt)
    raise SystemExit(f"could not fetch {url}")


def main() -> int:
    directory, check_only = sys.argv[1], "--check" in sys.argv[2:]
    os.makedirs(directory, exist_ok=True)
    failures = 0
    for name, repo, revision, path, size, sha256 in PINNED:
        target = os.path.join(directory, name)
        if not check_only and not verified(target, size, sha256):
            fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{path}", target)
        ok = verified(target, size, sha256)
        failures += not ok
        print(f"{name}: {'ok' if ok else 'MISMATCH'} ({repo}@{revision[:8]} {path})", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
