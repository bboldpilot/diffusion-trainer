"""Training-set loading with the same image preparation the evaluator applies to test images.

Long side resized to 1024 (LANCZOS), centre-cropped to multiples of 16, RGB; the image hash is the
SHA-256 of the prepared pixels plus the size string. Captions come from the sibling .txt; an image
without one gets the task's trigger word, or an empty caption.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def prepare_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    if min(w, h) <= 0:
        raise ValueError("invalid image dimensions")
    size = (1024, int(h / w * 1024)) if w > h else (int(w / h * 1024), 1024)
    image = image.resize(size, Image.Resampling.LANCZOS)
    cw, ch = (v // 16 * 16 for v in size)
    left, top = (size[0] - cw) // 2, (size[1] - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).convert("RGB")


def image_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()


@dataclass
class Item:
    name: str
    caption: str
    sha256: str
    image: Image.Image | None = None
    holdout: bool = False
    latent: object = None     # unscaled VAE latent (cpu float32) until scaled
    scaled: object = None     # model-native latent (process_latent_in applied)
    extra: dict = field(default_factory=dict)


def read_rows(source) -> list[tuple[str, bytes, str | None]]:
    source = Path(source)
    rows: list[tuple[str, bytes, str | None]] = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                caption = path.with_suffix(".txt")
                rows.append((str(path.relative_to(source)), path.read_bytes(),
                             caption.read_text(encoding="utf-8") if caption.exists() else None))
    else:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            for name in sorted(names):
                if name.lower().endswith(EXTENSIONS) and not Path(name).name.startswith("."):
                    caption = str(Path(name).with_suffix(".txt"))
                    rows.append((name, archive.read(name),
                                 archive.read(caption).decode("utf-8") if caption in names else None))
    if not rows:
        raise ValueError("empty image dataset")
    return rows


def load_items(source, trigger_word: str | None = None) -> list[Item]:
    items = []
    for name, raw, caption in read_rows(source):
        text = caption if caption is not None and caption.strip() else (trigger_word or "")
        image = prepare_image(Image.open(io.BytesIO(raw)))
        items.append(Item(name=name, caption=text, sha256=image_hash(image), image=image))
    return items


def split_holdout(items: list[Item], fraction: float, minimum: int) -> None:
    """Deterministic holdout: order images by content hash and take evenly spaced positions, so the
    held-out set spreads over the hash order rather than clustering. `fraction` 0 marks nothing."""
    count = 0 if fraction <= 0 else min(len(items) - 1, max(minimum, round(fraction * len(items))))
    chosen: set[int] = set()
    if count > 0:
        order = sorted(range(len(items)), key=lambda index: items[index].sha256)
        for position in range(count):
            chosen.add(order[round(position * (len(order) - 1) / max(1, count - 1))] if count > 1 else order[0])
    for index, item in enumerate(items):
        item.holdout = index in chosen
