#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import time
import zipfile
from pathlib import Path

import yaml


CACHE_ROOT = Path("/cache")
CONFIG_TEMPLATE_ROOT = Path("/workspace/core/training_templates")
CONFIG_OUTPUT_ROOT = Path("/dataset/configs")
DATASET_ZIP_ROOT = CACHE_ROOT / "datasets"
IMAGE_DATASET_ROOT = Path("/dataset/images")
CHECKPOINTS_ROOT = Path("/app/checkpoints")
AI_TOOLKIT_ROOT = Path("/app/ai-toolkit")

TEMPLATE_BY_MODEL_TYPE = {
    "flux": "base_diffusion_flux.yaml",
    "z-image": "base_diffusion_zimage.yaml",
    "qwen-image": "base_diffusion_qwen_image.yaml",
    "ideogram4": "base_diffusion_ideogram4.yaml",
    "krea2": "base_diffusion_krea2.yaml",
}
IDEOGRAM4_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-4B-Instruct"

# The harness kills this container at hours_to_complete*3600 exactly and uploads
# whatever is on disk. Keep a slice of the budget back from training: krea2
# needs room for the post-train calibration, every other type just needs enough
# for the SIGTERM -> checkpoint-promote path below.
CALIBRATION_RESERVE_S = 420
# The krea2 post-training pass (checkpoint selection + strength calibration) gets a
# reserve that scales with the task: 20% of the budget, never below the 420 s the
# calibration-only pass was measured in, never above 12 min. At 1.0 h that is 720 s:
# with SELECT_IMAGES x SELECT_SEEDS = 4 gens (~75 s on H100) it fits the as-trained
# control plus 4-5 tail checkpoints; the 600 training steps it costs are the ones the
# 15-checkpoint sweep showed to be past the optimum (best step 2400 of 3000).
SELECT_RESERVE_MAX_S = 720
SELECT_RESERVE_FRACTION = 0.20


def krea2_reserve_s(hours_to_complete: float) -> int:
    budget = hours_to_complete * 3600.0
    return int(min(SELECT_RESERVE_MAX_S, max(CALIBRATION_RESERVE_S, SELECT_RESERVE_FRACTION * budget)))

# WHICH post-training pass krea2 gets. "calibrate" is what is measured today
# (dropping the text-fusion group won all six paired comparisons, 1.0-4.0%).
# "select" is the newer pass: score the checkpoint grid on the eval's own
# objective and ship the minimum instead of whatever the wall happened to land
# on. Both share one model load, and selection runs first because a strength
# measured on one checkpoint says nothing about another.
#
# Shipped as "calibrate" while selection did not fit the 420 s reserve (measured:
# the control score alone took 151 s at 4 images x 2 seeds). With the tail-window
# candidates, 1-seed scoring and the scaled reserve (krea2_reserve_s) selection runs
# first and calibration takes whatever budget is left; both stop on their own
# deadline checks and fail open to the as-trained artefact.
POST_TRAIN_MODE = "select+calibrate"
PROMOTE_RESERVE_S = 120
TERM_GRACE_S = 30

NUMBERED_CKPT_RE = re.compile(r"[-_](\d+)\.safetensors$")


# =============================================================================
# I6 port (PLAN-IMAGE-0908 v2): opt-in "winner recipe".
#
# IMG_RECIPE=winner switches krea2 and ideogram4 to the EFFECTIVE recipe of the
# 2026-09-07 image-tournament winner (hotkey 5EhyCWPu), reconstructed in
# rivals-config/winner-0907-{krea2,ideogram4}.yaml from
# rivals/god-image-tourn-1dccbd24755b58a3-20260907-position-1 (scripts/fine-tune/
# krea.py, ideogram.py; ops/docker/patches/*). IMG_RECIPE unset or "ours" = the
# flow in this file exactly as before. Every winner-only step is FAIL-OPEN: a
# failure prints a "[recipe] WARN" line and the run continues on OUR recipe
# (captions re-blanked, our reserve, select+calibrate).
#
# The four ai-toolkit patches in ops/docker/patches/ are inert until a config key
# or env toggle selects them (checked in their source), so building them in
# changes nothing for IMG_RECIPE=ours:
#   multires_noise.py    train.multires_noise_iterations  (default 0   -> off)
#   deadline_anneal.py   train.deadline_ts                (default 0.0 -> off)
#   tail_save_densify.py train.tailsave_deadline_ts       (default 0.0 -> off)
#   mod_lora.py          env KREA2_MODLORA                (default "0" -> off)
# =============================================================================
RECIPE_ENV = "IMG_RECIPE"
# krea2 winner path trains to the wall (its watchdog promotes at T-300 s, its LR
# anneal ends at T-120 s). We keep our SIGTERM -> promote flow but stop at
# wall-120 s instead of holding the 20% select+calibrate reserve.
WINNER_WALL_MARGIN_S = 120
WINNER_KREA2_STEPS = 2000                # winner template steps; krea.py:12-18 leaves them unless KREA2_STEPS>0
WINNER_KREA2_ANNEAL_LEAD_S = 120.0       # krea.py:53   deadline_ts = wall - 120
WINNER_TAILSAVE_LEAD_S = 300.0           # krea.py:66-69 / ideogram.py:201-205  tailsave target = wall - 300
# ideogram4 CONTENT school sizing (ideogram.py:37-43, :93-101)
WINNER_I4_SPS = 2.1
WINNER_I4_CAP = 1700
WINNER_I4_FLOOR = 120
WINNER_I4_EPOCH_GUARD = 25
WINNER_I4_UPLOAD_RESERVE_S = 420.0
WINNER_I4_SAVE_EVERY = 100
IDEOGRAM4_UNCOND_LORA = CACHE_ROOT / "hf_cache" / "ideogram_4_unconditional_lora_r16.safetensors"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}   # winner imgcommon.IMG_EXTS
# recipe_dryrun.py flips this to build configs on a machine without /cache or /app/ai-toolkit.
SKIP_PATH_CHECKS = False
# substrings that prove a build-time patch actually landed in the ai-toolkit tree
_PATCH_MARKERS = {
    "multires_noise": ("toolkit/config_modules.py", "kwargs.get('multires_noise_iterations'"),
    "deadline_anneal": ("toolkit/config_modules.py", "kwargs.get('deadline_ts'"),
    "tail_save_densify": ("toolkit/config_modules.py", "kwargs.get('tailsave_deadline_ts'"),
    "mod_lora": ("extensions_built_in/diffusion_models/krea2/krea2.py", "MODULATION_LORA_TARGETS"),
}


def recipe():
    raw = os.environ.get(RECIPE_ENV, "").strip().lower()
    if raw in ("", "ours"):
        return "ours"
    if raw == "winner":
        return "winner"
    if raw in ("winner+select", "winner-select", "winner_select"):
        return "winner+select"
    if raw in ("winner+holdout", "winner-holdout", "winner_holdout"):
        return "winner+holdout"
    print(f"[recipe] WARN unknown {RECIPE_ENV}={raw!r} -> using ours", flush=True)
    return "ours"


# P2 (10 Eyl, ölçülmüş): eğitimden k görüntü AYRILIR, duvarın içine birden çok aday sığdırılır ve
# ayrılan görüntülerde en iyisi seçilir. Dayanağı: gerçek iç-holdout sıralaması validator'ın kendi
# saklı-küme sıralamasıyla birebir çıktı (rho = 1,000), aynı config'in iki koşusu arasında %19 fark
# vardı, ve sabit adım sayısı görevden göreve tutmadı (n>=22'de 260 adım az kaldı) -> adım sayısı
# aday ekseni olur. Eğitim-İÇİ görüntülerle seçim ÇALIŞMAZ (rho = 0,086), bu yüzden ayırma şart.
HOLDOUT_ROOT = Path("/dataset/holdout")
HOLDOUT_MIN_TRAIN = 10        # altında ayırmak eğitimi veriden eder; tek koşuya düşülür
HOLDOUT_MAX_K = 8
# Aday adım çarpanları: reçetenin kendi seçtiği adım sayısı 1,0; ikinci aday daha kısa. 9-10 Eyl
# ölçümleri kısa koşunun küçük n'de kazandığını, büyük n'de kaybettiğini gösterdi -> ikisini de koş.
HOLDOUT_STEP_FACTORS = (1.0, 0.6)
# Duvar payını adaylara BÖLERKEN eşit bölmek ekseni yok ediyor: 10 Eyl P2-verify'da 0,75 h
# bütçesinde iki aday da 1012 s aldı, ikisi de aynı 600. adımda duvara çarptı -- "2000 adım"
# ile "1200 adım" AYNI uzunlukta iki artefakt üretti (yalnız LR takvimleri farklıydı).
# Eksen gerçekte EĞİTİM UZUNLUĞU; duvar altında uzunluk = süre. O yüzden pay eşit değil, ve
# ikinci aday ölçülen hızdan hesaplanan, SLOTUNA SIĞAN bir adım hedefiyle koşar: takvimi
# tamamlanır, yani gerçekten "daha kısa ama tam annelenmiş" bir aday olur.
HOLDOUT_SLOT_SHARES = (0.58, 0.42)
HOLDOUT_RATE_MARGIN = 0.95    # ölçülen hızın bu kadarını kullan (checkpoint adımı gerçeğin altında)
# Puanlama rezervi: model yükleme + aday başına örnekleme. Bütçenin %25'i, 600-1500 s arası.
def holdout_reserve_s(hours_to_complete: float) -> int:
    return int(max(600.0, min(1500.0, 0.25 * float(hours_to_complete) * 3600.0)))


def recipe_holdout() -> bool:
    """P2: ideogram4 winner reçetesi + iç-holdout ile çok-aday seçimi."""
    return recipe() == "winner+holdout"


def holdout_k(n_images: int) -> int:
    """Kaç görüntü ayrılacak: n//3, en çok 8; eğitimde en az HOLDOUT_MIN_TRAIN kalmalı."""
    if n_images < HOLDOUT_MIN_TRAIN + 4:
        return 0
    return max(0, min(HOLDOUT_MAX_K, n_images // 3, n_images - HOLDOUT_MIN_TRAIN))


def split_holdout(dataset_root: Path) -> int:
    """Deterministik olarak k çifti HOLDOUT_ROOT'a TAŞI (eğitim onları görmemeli).

    Seçim sıralı listede eşit aralıklarla: veri seti çoğu görevde konu/poz sırasına göre
    yazılmış oluyor, baştan ya da sondan almak holdout'u tek bir moda kilitliyor.
    """
    stems = sorted({p.stem for p in dataset_root.glob("*") if p.suffix.lower() != ".txt"})
    k = holdout_k(len(stems))
    if k <= 0:
        print(f"[holdout] n={len(stems)} çok küçük — ayırma YOK, tek koşu", flush=True)
        return 0
    if HOLDOUT_ROOT.exists():
        shutil.rmtree(HOLDOUT_ROOT)
    HOLDOUT_ROOT.mkdir(parents=True, exist_ok=True)
    step = len(stems) / float(k)
    picked = sorted({stems[min(len(stems) - 1, int(i * step))] for i in range(k)})
    for stem in picked:
        for src in dataset_root.glob(f"{stem}.*"):
            shutil.move(str(src), str(HOLDOUT_ROOT / src.name))
    left = len({p.stem for p in dataset_root.glob("*") if p.suffix.lower() != ".txt"})
    print(f"[holdout] {len(picked)} görüntü ayrıldı ({', '.join(picked)}); eğitimde {left} kaldı",
          flush=True)
    return len(picked)


def recipe_select() -> bool:
    """I7: winner recipe for training, OUR post-train checkpoint selection on top (krea2 only).
    The winner's tail-save densification supplies the candidate grid; the anneal and tail-save
    anchors move from the wall to wall-reserve so the LR floor is reached where we actually stop."""
    return recipe() == "winner+select"


def _require_exists(path, message):
    if not SKIP_PATH_CHECKS and not Path(path).exists():
        raise FileNotFoundError(message)


def _path_exists(path):
    return True if SKIP_PATH_CHECKS else Path(path).exists()


def patch_present(name):
    """True/False read from the ai-toolkit source; None when it cannot be read (dry-run
    on a machine without /app/ai-toolkit) -- treated as present so the keys are emitted."""
    rel, marker = _PATCH_MARKERS[name]
    path = AI_TOOLKIT_ROOT / rel
    if SKIP_PATH_CHECKS or not path.exists():
        return None
    try:
        return marker in path.read_text(encoding="utf-8")
    except OSError:
        return None


def _mechanism_ok(name):
    """Fail-open gate: a patch missing from the built image drops only its own keys."""
    if patch_present(name) is False:
        print(f"[recipe] WARN patch {name} NOT applied in this image -> its keys skipped "
              f"(our behaviour for that mechanism)", flush=True)
        return False
    return True


def report_patch_status():
    words = {True: "applied", False: "MISSING", None: "unverifiable (no ai-toolkit tree here)"}
    for name in _PATCH_MARKERS:
        print(f"[recipe] patch {name}: {words[patch_present(name)]}", flush=True)


def _count_images(dataset_path):
    n = 0
    try:
        for p in Path(dataset_path).rglob("*"):
            if p.name.startswith("."):
                continue
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                n += 1
    except OSError:
        return 0
    return n


def _envf(key, default=None):
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _envi(key, default=None):
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _blank_krea2_captions(dataset_root):
    """Our krea2 recipe: train the blank-prompt branch the eval weights 3x (see prepare_dataset)."""
    blanked = 0
    for caption in Path(dataset_root).glob("*.txt"):
        caption.write_text("")
        blanked += 1
    print(f"[krea2] blanked {blanked} caption files (blank-prompt branch is 75% of eval)", flush=True)


def _winner_krea2(process, args):
    """krea.py:5-78 replicated: template steps/lr untouched, the KREA2_* knobs layered on top.
    Defaults ARE the shipped recipe (no env set in a tournament)."""
    train = process.setdefault("train", {})
    ds = process.setdefault("datasets", [{}])[0]
    hours = float(args.hours_to_complete)
    started = float(args.started_ts)
    try:                                                    # krea.py:12-18
        st = int(os.environ.get("KREA2_STEPS", "0"))
    except ValueError:
        st = 0
    train["steps"] = st if st > 0 else WINNER_KREA2_STEPS
    try:                                                    # krea.py:19-28 (native ai-toolkit option, no patch)
        npair = int(os.environ.get("KREA2_NPAIR", "4"))
        if npair > 1:
            train["optimal_noise_pairing_samples"] = npair
    except ValueError:
        print("[recipe] WARN KREA2_NPAIR unparsable -> noise pairing off", flush=True)
    if _mechanism_ok("multires_noise"):                     # krea.py:29-37
        try:
            mrn = int(os.environ.get("KREA2_MRN", "6"))
            if mrn > 0:
                train["multires_noise_iterations"] = mrn
                train["multires_noise_discount"] = float(os.environ.get("KREA2_MRN_DISC", "0.3"))
        except ValueError:
            print("[recipe] WARN KREA2_MRN/KREA2_MRN_DISC unparsable -> multires off", flush=True)
    try:                                                    # krea.py:38-44
        ds["caption_dropout_rate"] = float(os.environ.get("KREA2_CAPDROP", "0.20"))
    except ValueError:
        print("[recipe] WARN KREA2_CAPDROP unparsable -> template dropout kept", flush=True)
    if _mechanism_ok("deadline_anneal") and \
            os.environ.get("KREA2_ANNEAL", "1") not in ("0", "", "false", "False"):   # krea.py:45-62
        try:
            train["deadline_ts"] = started + hours * 3600.0 - WINNER_KREA2_ANNEAL_LEAD_S
            train["deadline_anneal_frac"] = float(os.environ.get("KREA2_ANNEAL_FRAC", "0.30"))
            train["deadline_anneal_floor"] = float(os.environ.get("KREA2_ANNEAL_FLOOR", "0.05"))
        except ValueError:
            for k in ("deadline_ts", "deadline_anneal_frac", "deadline_anneal_floor"):
                train.pop(k, None)
            print("[recipe] WARN KREA2_ANNEAL_* unparsable -> anneal off", flush=True)
    if _mechanism_ok("tail_save_densify"):                  # krea.py:63-71
        train["tailsave_deadline_ts"] = started + hours * 3600.0 - WINNER_TAILSAVE_LEAD_S
    if getattr(args, "recipe_select", False):               # I7: training stops at wall-reserve, not wall-120
        reserve = krea2_reserve_s(hours)
        stop_ts = started + hours * 3600.0 - reserve
        if "deadline_ts" in train:
            train["deadline_ts"] = stop_ts
        if "tailsave_deadline_ts" in train:
            train["tailsave_deadline_ts"] = stop_ts - (WINNER_TAILSAVE_LEAD_S - WINNER_KREA2_ANNEAL_LEAD_S)
        print(f"[recipe] winner+select: anneal end / tail-save anchors moved to wall-{reserve}s "
              f"(select+calibrate reserve); post-train selection ON", flush=True)
    if _mechanism_ok("mod_lora"):                           # krea.py:72-75; read by the patched krea2.py inside run.py
        os.environ.setdefault("KREA2_MODLORA", "1")
        print(f"[recipe] KREA2_MODLORA={os.environ['KREA2_MODLORA']} "
              f"(1 = LoRA also on tproj + last.linear)", flush=True)


_I4_CATEGORY_SETS = {   # ideogram.py:57-64; LOG-ONLY here (this port always runs CONTENT)
    "logo": ("logo", "brandmark", "wordmark", "monogram", "emblem", "brand mark"),
    "line": ("line art", "line-art", "lineart", "linework", "schematic", "blueprint"),
    "watercolor": ("watercolor", "gouache", "aquarelle"),
    "social": ("infographic", "headline", "banner", "social"),
    "design": ("ui", "interface", "mobile", "dashboard", "mockup", "light mode"),
    "product": ("product", "packshot", "studio shot", "macro", "lifestyle"),
}


def _winner_i4_category(dataset_path):
    """ideogram.py:46-90 keyword-argmax vote (>=25% of captions), to log which school the winner would run."""
    caps = []
    for p in Path(dataset_path).rglob("*.txt"):
        try:
            caps.append(p.read_text(encoding="utf-8", errors="ignore").lower())
        except OSError:
            pass
    if not caps:
        return "default"
    counts = {}
    for c in caps:
        for cat, kws in _I4_CATEGORY_SETS.items():
            if any(re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", c) for kw in kws):
                counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return "default"
    cat, hits = max(counts.items(), key=lambda kv: kv[1])
    return cat if hits >= 0.25 * len(caps) else "default"


def _winner_ideogram4(process, args, dataset_path):
    """ideogram.py CONTENT school (:118-139) + size-to-complete (:93-101) + IMG_* arms (:142-173)
    + tail-save (:201-205). The DETAIL school (line art) is NOT ported: logged only."""
    train = process.setdefault("train", {})
    save = process.setdefault("save", {})
    ds = process.setdefault("datasets", [{}])[0]
    hours = float(args.hours_to_complete)
    started = float(args.started_ts)
    n = _count_images(dataset_path)
    cat = _winner_i4_category(dataset_path)
    would = "DETAIL" if cat in ("line", "sketch", "lineart") else "CONTENT"
    print(f"[recipe] ideogram4 category vote={cat} (winner would run {would}); this port runs CONTENT", flush=True)
    # CONTENT school (ideogram.py:118-132)
    train["min_denoising_steps"] = 250
    train["lr"] = 4e-4
    train["unet_lr"] = 4e-4
    train["content_or_style"] = "balanced"
    train["timestep_type"] = "linear"
    train["ema_config"] = {"use_ema": True, "ema_decay": 0.995}
    train["train_text_encoder"] = False
    train["cache_text_embeddings"] = True
    for k in ("do_cfg", "cfg_scale", "text_encoder_lr"):
        train.pop(k, None)
    ds["caption_dropout_rate"] = _envf("IMG_CAPTION_DROPOUT", 0.05)      # ideogram.py:136
    ds["cache_latents_to_disk"] = True                                    # ideogram.py:137
    # size-to-complete (ideogram.py:93-101, imgcommon.remaining_secs reserve 420)
    remain = max(0.0, hours * 3600.0 - (time.time() - started) - WINNER_I4_UPLOAD_RESERVE_S)
    fit = int(remain / WINNER_I4_SPS)
    epoch_cap = WINNER_I4_EPOCH_GUARD * n if n > 0 else WINNER_I4_CAP
    train["steps"] = max(WINNER_I4_FLOOR, min(WINNER_I4_CAP, fit, epoch_cap))
    print(f"[recipe] ideogram4 steps={train['steps']} = max({WINNER_I4_FLOOR}, min({WINNER_I4_CAP}, "
          f"fit={fit} [remain {int(remain)} s / {WINNER_I4_SPS} s/step], "
          f"epoch_guard={epoch_cap} [{WINNER_I4_EPOCH_GUARD} x n={n}]))", flush=True)
    save["save_every"] = WINNER_I4_SAVE_EVERY                             # ideogram.py:191
    save["max_step_saves_to_keep"] = 100                                  # ideogram.py:192
    na = _envi("IMG_NETWORK_ALPHA")                                       # ideogram.py:193-195
    if na:
        process.setdefault("network", {})["linear_alpha"] = na
    # _apply_arms (ideogram.py:142-173): explicit IMG_* always win over the school
    lr = _envf("IMG_LR")
    if lr:
        train["lr"] = train["unet_lr"] = lr
    steps = _envi("IMG_STEPS")
    if steps:
        train["steps"] = steps
    ema = os.environ.get("IMG_EMA", "").strip().lower()
    if ema in ("0", "off", "none"):
        train["ema_config"] = {"use_ema": False, "ema_decay": 0.995}
    elif ema and ema not in ("1", "on", "true"):
        try:
            train["ema_config"] = {"use_ema": True, "ema_decay": float(ema)}
        except ValueError:
            pass
    md = _envi("IMG_MIN_DENOISE")
    if md and md > 0:
        train["min_denoising_steps"] = md
    xd = _envi("IMG_MAX_DENOISE")
    if xd and 0 < xd < 1000:
        train["max_denoising_steps"] = xd
    cs = os.environ.get("IMG_CONTENT_STYLE", "").strip().lower()
    if cs in ("content", "style", "balanced"):
        train["content_or_style"] = cs
    ts = os.environ.get("IMG_TIMESTEP", "").strip().lower()
    if ts in ("linear", "sigmoid", "shift", "lognorm_blend", "weighted"):
        train["timestep_type"] = ts
    te = os.environ.get("IMG_TRAIN_TE", "").strip().lower()
    if te and te not in ("0", "off", "none"):
        train["train_text_encoder"] = True
        train["text_encoder_lr"] = _envf("IMG_TE_LR", 1e-7)
    if _mechanism_ok("tail_save_densify"):                                # ideogram.py:201-205
        train["tailsave_deadline_ts"] = started + hours * 3600.0 - WINNER_TAILSAVE_LEAD_S
    # unconditional LoRA only when the validator actually staged it (fail-fast otherwise in the loader)
    model_config = process.setdefault("model", {})
    if _path_exists(IDEOGRAM4_UNCOND_LORA):
        model_config["unconditional_lora_path"] = str(IDEOGRAM4_UNCOND_LORA)
        print(f"[recipe] ideogram4 unconditional LoRA present -> {IDEOGRAM4_UNCOND_LORA}", flush=True)
    else:
        model_config.pop("unconditional_lora_path", None)
        print(f"[recipe] WARN ideogram4 unconditional LoRA ABSENT at {IDEOGRAM4_UNCOND_LORA} -> key removed", flush=True)


def apply_winner_recipe(args, config, dataset_path):
    """Mutate config in place for krea2 / ideogram4. Returns True when a port applied."""
    process = config["config"]["process"][0]
    if args.model_type == "krea2":
        _winner_krea2(process, args)
    elif args.model_type == "ideogram4":
        _winner_ideogram4(process, args, dataset_path)
    else:
        print(f"[recipe] winner recipe not ported for {args.model_type} -> ours", flush=True)
        return False
    return True


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}{i}."))
        return out
    return {prefix[:-1]: obj}


def config_diff_lines(before, after):
    """Key-level diff of two config dicts (dotted paths), for the startup log and the dry-run."""
    a, b = _flatten(before), _flatten(after)
    lines = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            lines.append(f"  + {k} = {b[k]!r}")
        elif k not in b:
            lines.append(f"  - {k} (was {a[k]!r})")
        elif a[k] != b[k]:
            lines.append(f"  ~ {k}: {a[k]!r} -> {b[k]!r}")
    return lines
# ============================================================ end I6 recipe block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--model-type", required=True, choices=sorted(TEMPLATE_BY_MODEL_TYPE))
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True)
    parser.add_argument("--trigger-word")
    return parser.parse_args()


def cached_model_path(model: str) -> Path:
    cache_name = model.replace("/", "--")
    return CACHE_ROOT / "models" / cache_name


def prepare_dataset(task_id: str, model_type: str) -> Path:
    zip_path = DATASET_ZIP_ROOT / f"{task_id}_tourn.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected cached image dataset zip at {zip_path}")

    if IMAGE_DATASET_ROOT.exists():
        shutil.rmtree(IMAGE_DATASET_ROOT)
    IMAGE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    extract_root = Path("/dataset/extracted") / task_id
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)

    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".txt"}
    copied = 0
    for source in extract_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in supported_suffixes:
            destination = IMAGE_DATASET_ROOT / source.name
            if destination.exists():
                destination = IMAGE_DATASET_ROOT / f"{source.parent.name}_{source.name}"
            shutil.copy2(source, destination)
            copied += 1

    if copied == 0:
        raise RuntimeError(f"No supported image/caption files found in {zip_path}")

    if model_type == "krea2" and recipe() in ("winner", "winner+holdout"):
        # I6 winner recipe: captions stay verbatim; caption_dropout_rate 0.20 (set in
        # prepare_config) is the winner's blank-prompt exposure instead of blanking.
        print("[recipe] winner: krea2 captions kept verbatim (no blanking; dropout 0.20 instead)", flush=True)
    elif model_type == "krea2":
        # Train the conditioning the eval actually scores. The krea2 eval weights
        # the blank-prompt branch 3x the text-guided branch, but the stock
        # template trains on the dataset captions ~95% of the time, so the model
        # the eval sees most is the one that trained least. Blanking the caption
        # files (the trigger word stays, it comes from the config) measured 3.9%
        # better composite than caption-trained on the same task, 5 seeds.
        _blank_krea2_captions(IMAGE_DATASET_ROOT)

    if recipe_holdout() and model_type == "ideogram4":
        split_holdout(IMAGE_DATASET_ROOT)

    return IMAGE_DATASET_ROOT


def prepare_config(args: argparse.Namespace, dataset_path: Path) -> Path:
    template_path = CONFIG_TEMPLATE_ROOT / TEMPLATE_BY_MODEL_TYPE[args.model_type]
    if not template_path.exists():
        raise FileNotFoundError(f"Missing ai-toolkit template for {args.model_type}: {template_path}")

    with template_path.open() as file:
        config = yaml.safe_load(file)

    config_body = config.setdefault("config", {})
    # expected_repo_name, and it is NOT cosmetic: ai-toolkit writes to
    # <training_folder>/<config.name>, and the uploader reads exactly
    # /app/checkpoints/<task_id>/<expected_repo_name>. Renaming this to "last" (to make the
    # evaluator's picker find the final save) cost us all three tasks of the 2026-08-17 image
    # round -- trained weights on disk, uploader looking at a path that did not exist,
    # submission_id None. The evaluator's need is met in promote_latest_checkpoint instead,
    # by putting a `last.safetensors` COPY inside this same folder.
    config_body["name"] = args.expected_repo_name

    process = config_body.setdefault("process", [{}])[0]
    process["training_folder"] = str(CHECKPOINTS_ROOT / args.task_id)
    process["trigger_word"] = args.trigger_word

    datasets = process.setdefault("datasets", [{}])
    datasets[0]["folder_path"] = str(dataset_path)

    model_config = process.setdefault("model", {})
    model_path = cached_model_path(args.model)
    _require_exists(model_path, f"Expected cached base model at {model_path}")
    # Trainer container has no internet; ai-toolkit must load every component from
    # the local cache dir. Its ideogram4 loader uses the local branch when
    # <name_or_path>/<subfolder> exists, so point it at the downloaded snapshot.
    model_config["name_or_path"] = str(model_path)
    if args.model_type == "flux" and model_path.exists():
        from flux_single_file import resolve_checkpoint
        checkpoint = resolve_checkpoint(model_path)
        if checkpoint is not None:
            model_config["name_or_path"] = str(checkpoint)

    if args.model_type == "ideogram4":
        text_encoder_path = IDEOGRAM4_TEXT_ENCODER_CACHE
        _require_exists(text_encoder_path, f"Expected cached Ideogram 4 text encoder at {text_encoder_path}")
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)

    elif args.model_type == "krea2":
        text_encoder_path = KREA2_TEXT_ENCODER_CACHE
        _require_exists(text_encoder_path, f"Expected cached Krea 2 text encoder at {text_encoder_path}")
        vae_path = model_path / "vae"
        _require_exists(vae_path, f"Expected cached Krea 2 VAE at {vae_path}")
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)
        # Krea2Model appends the "vae" subfolder when loading the VAE.
        model_kwargs["vae_path"] = str(model_path)

    # I6: opt-in winner recipe, applied on top of the fully resolved OUR config; fail-open.
    args.recipe_effective = "ours"
    args.recipe_select = False
    # Holdout changes candidate training/selection, not the underlying recipe.
    # On krea2 it uses the single-run winner recipe; only ideogram4 splits data.
    if recipe() in ("winner", "winner+select", "winner+holdout"):
        args.recipe_select = recipe_select()
        before = copy.deepcopy(config)
        had_modlora = "KREA2_MODLORA" in os.environ
        try:
            applied = apply_winner_recipe(args, config, dataset_path)
        except Exception as exc:  # noqa: BLE001 -- fail-open to our recipe
            print(f"[recipe] WARN winner recipe failed ({type(exc).__name__}: {exc}) -> OUR config", flush=True)
            config = before
            if not had_modlora:
                os.environ.pop("KREA2_MODLORA", None)
            if args.model_type == "krea2":
                _blank_krea2_captions(dataset_path)   # restore our exact recipe
        else:
            if applied:
                args.recipe_effective = "winner"
                lines = config_diff_lines(before, config)
                print(f"[recipe] effective config diff vs our template ({len(lines)} key(s)):", flush=True)
                for line in lines:
                    print(line, flush=True)

    CONFIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_OUTPUT_ROOT / f"{args.task_id}.yaml"
    with output_path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return output_path


def _calibration_model_kwargs(args: argparse.Namespace) -> dict:
    """The same component paths prepare_config resolves for THIS model_type, so an
    out-of-process scorer loads the model from the identical local pieces the training
    run just used.

    10 Sep P2-verify: this used to hardcode the krea2 (4B) text encoder because its only
    caller was the krea2 calibration pass. holdout_select (ideogram4) reused it and died
    with `Repo id must be in the form 'repo_name'...: '/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct'`
    -- the 4B snapshot is never downloaded on an ideogram4 pod, so transformers fell back to
    treating the path as a hub id. Selection was skipped and candidate[0] shipped unselected.
    """
    model_path = cached_model_path(args.model)
    if args.model_type == "ideogram4":
        return {"text_encoder_path": str(IDEOGRAM4_TEXT_ENCODER_CACHE)}
    return {
        "text_encoder_path": str(KREA2_TEXT_ENCODER_CACHE),
        "vae_path": str(model_path),
    }


def _last_checkpoint_step(output_folder: Path) -> int:
    """En yüksek numaralı checkpoint'in adımı (yoksa 0). Aday hızını ölçmek için: gerçek
    adım sayısının ALTINDA kalır (son kayıttan sonra atılan adımları saymaz), bu da hedefi
    muhafazakâr tutar -- fazla tahmin ikinci adayı yine duvara çarptırırdı."""
    steps = [int(m.group(1)) for p in output_folder.rglob("*.safetensors")
             if (m := NUMBERED_CKPT_RE.search(p.name))]
    return max(steps, default=0)


def promote_latest_checkpoint(output_folder: Path, config_name: str) -> None:
    """Guarantee a file named exactly last.safetensors INSIDE the folder the uploader reads.

    Two contracts have to hold at once, and 2026-08-17 was the week we honoured one by
    breaking the other -- three tasks trained fine and shipped nothing:

      * the UPLOADER reads `/app/checkpoints/<task_id>/<expected_repo_name>` verbatim
        (`trainer/training_paths.get_checkpoints_output_path`). ai-toolkit writes to
        `<training_folder>/<config.name>`, so config.name IS that contract -- rename it and
        the uploader's folder does not exist, the container dies on FileNotFoundError, and
        the task is recorded with no submission at all, whatever sits on disk.
      * the EVALUATOR picks a submission by scanning for a file ending in `last.safetensors`
        (`validator/evaluation/constants.py`), falling back to a numbered-checkpoint regex
        that the unnumbered final save does NOT match -- so without this copy the run ships
        its second-to-last checkpoint.

    COPY, never rename: the numbered series has to stay intact for the fallback picker, and
    the unnumbered final save is what ai-toolkit itself considers complete. Files touched in
    the last 30s are skipped -- the trainer may have been mid-save when the wall stopped it,
    and a truncated safetensors scores zero rather than slightly worse.
    """
    target = output_folder / "last.safetensors"
    if target.exists():
        print(f"[wall] last.safetensors already in place ({target})", flush=True)
        return

    # A completed run writes `<config.name>.safetensors`; that is the real final weight.
    final = output_folder / f"{config_name}.safetensors"
    if final.exists():
        shutil.copy2(final, target)
        print(f"[wall] promoted completed run's final save to {target}", flush=True)
        return

    settled: list[tuple[int, Path]] = []
    fresh: list[tuple[int, Path]] = []
    newest_ok = time.time() - 30
    for path in output_folder.rglob("*.safetensors"):
        match = NUMBERED_CKPT_RE.search(path.name)
        if match:
            bucket = settled if path.stat().st_mtime < newest_ok else fresh
            bucket.append((int(match.group(1)), path))
    # A fresh file is only worth the truncation risk when it is all there is.
    candidates = settled or fresh
    if not candidates:
        print(f"[wall] NO PROMOTABLE CHECKPOINT in {output_folder} -- this run ships nothing",
              flush=True)
        return
    step, source = max(candidates)
    shutil.copy2(source, target)
    print(f"[wall] promoted step-{step} checkpoint to {target}", flush=True)


def run_training(config_path: Path, train_deadline: float) -> int:
    """run.py, but stopped in time for the calibration reserve.

    The harness SIGKILLs the whole container at the hour mark with no grace, so
    the entrypoint has to be the one that stops training early enough to leave
    the reserve intact. SIGTERM first so ai-toolkit can finish the step; SIGKILL
    only if it lingers.
    """
    proc = subprocess.Popen(["python3", "run.py", str(config_path)], cwd=AI_TOOLKIT_ROOT)
    while True:
        try:
            rc = proc.wait(timeout=min(30.0, max(1.0, train_deadline - time.time())))
            print(f"[wall] run.py exited rc={rc}", flush=True)
            return rc  # A saved checkpoint can still be shipped after a late failure.
        except subprocess.TimeoutExpired:
            pass
        if time.time() >= train_deadline:
            print(f"[wall] training budget spent, stopping run.py (pid {proc.pid})", flush=True)
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            return proc.returncode


def run_holdout_candidates(args, dataset_path: Path, config_path: Path, training_folder: Path,
                           output_folder: Path, train_deadline: float, wall_deadline: float) -> None:
    """Duvarın eğitim kısmını adaylara böl, her adayın artefaktını sakla, holdout ile seç.

    Aday ekseni ADIM SAYISI: reçetenin kendi seçtiği değer ve onun %60'ı. Her aday kendi
    checkpoint klasörüne yazar; kazanan `output_folder`'a promote edilir. Hiçbir adım
    teslimatı riske atmaz: holdout boşsa ya da puanlama düşerse ilk adayın artefaktı durur.
    """
    import yaml as _yaml

    with open(config_path) as fh:
        base_cfg = _yaml.safe_load(fh)
    process = base_cfg["config"]["process"][0]
    planned = int(process["train"]["steps"])
    n_hold = len(list(HOLDOUT_ROOT.glob("*.png"))) if HOLDOUT_ROOT.exists() else 0
    factors = HOLDOUT_STEP_FACTORS if n_hold > 0 else (1.0,)
    window = max(0.0, train_deadline - time.time())
    shares = HOLDOUT_SLOT_SHARES[:len(factors)] or (1.0,)
    shares = tuple(x / sum(shares) for x in shares)
    print(f"[holdout] {len(factors)} aday, pay {['%.0f%%' % (100 * x) for x in shares]} "
          f"({int(window)} s), taban adım {planned}, holdout {n_hold} görüntü", flush=True)

    candidates: list[tuple[str, Path]] = []
    rate: float | None = None        # ölçülen adım/s, bir önceki adaydan
    for i, factor in enumerate(factors):
        slot = window * shares[i]
        if rate is not None:
            # Slotuna sığan hedef: takvim duvardan önce biter, aday gerçekten daha kısa olur.
            steps = max(WINNER_I4_FLOOR, min(int(planned * factor), int(rate * slot * HOLDOUT_RATE_MARGIN)))
        else:
            steps = max(WINNER_I4_FLOOR, int(planned * factor))
        name = f"c{i}-{steps}"
        cfg = copy.deepcopy(base_cfg)
        cand_proc = cfg["config"]["process"][0]
        cand_proc["train"]["steps"] = steps
        cand_proc["training_folder"] = str(training_folder / "holdout")
        cfg["config"]["name"] = f"{args.expected_repo_name}-{name}"
        cand_path = config_path.parent / f"{config_path.stem}-{name}.yaml"
        with open(cand_path, "w") as fh:
            _yaml.safe_dump(cfg, fh, sort_keys=False)
        cand_deadline = min(train_deadline, time.time() + slot)
        print(f"[holdout] aday {name}: steps={steps}, {int(cand_deadline - time.time())} s"
              + (f" (ölçülen {rate:.3f} adım/s)" if rate is not None else ""), flush=True)
        started = time.time()
        run_training(cand_path, cand_deadline)
        elapsed = max(1.0, time.time() - started)
        saved = training_folder / "holdout" / f"{args.expected_repo_name}-{name}"
        done = _last_checkpoint_step(saved)
        if rate is None and done:
            rate = done / elapsed
            print(f"[holdout] hız ölçüldü: {done} adım / {int(elapsed)} s = {rate:.3f} adım/s", flush=True)
        # promote_latest_checkpoint'in kendisi: tamamlanmış kayıt varsa onu, yoksa en yüksek
        # numaralı (ve yarım kalmamış) checkpoint'i last.safetensors yapar.
        promote_latest_checkpoint(saved, f"{args.expected_repo_name}-{name}")
        latest = saved / "last.safetensors"
        if not latest.is_file():
            print(f"[holdout] aday {name}: artefakt yok, atlandı", flush=True)
            continue
        keep = training_folder / "holdout" / f"{name}.safetensors"
        shutil.copy2(latest, keep)
        candidates.append((name, keep))
        print(f"[holdout] aday {name}: {keep.name} ({keep.stat().st_size / 1e6:.0f} MB)", flush=True)

    if not candidates:
        print("[holdout] hiç aday üretilemedi", flush=True)
        return
    output_folder.mkdir(parents=True, exist_ok=True)
    winner = candidates[0][1]        # teslimat garantisi: seçim düşse bile ilk aday gider
    if n_hold > 0 and len(candidates) > 1:
        out_file = training_folder / "holdout" / "winner.txt"
        cmd = ["python3", str(Path(__file__).with_name("holdout_select.py")),
               "--base-model", str(cached_model_path(args.model)),
               "--holdout-dir", str(HOLDOUT_ROOT),
               "--model-kwargs", json.dumps(_calibration_model_kwargs(args)),
               "--deadline-epoch", str(wall_deadline - PROMOTE_RESERVE_S),
               "--out", str(out_file),
               "--report", str(training_folder / "holdout" / "ranking.json")]
        for name, path in candidates:
            cmd += ["--candidate", f"{name}={path}"]
        subprocess.run(cmd, check=False)
        if out_file.exists():
            chosen = Path(out_file.read_text().strip())
            if chosen.is_file():
                winner = chosen
    elif n_hold == 0:
        print("[holdout] holdout yok — seçim atlandı, tek aday gidiyor", flush=True)
    shutil.copy2(winner, output_folder / "last.safetensors")
    print(f"[holdout] TESLİM {winner.name} -> {output_folder / 'last.safetensors'}", flush=True)


def _report_submission(output_folder: Path) -> None:
    """State, in the container log, exactly what this run is handing to the uploader.

    The uploader refuses an absent or metadata-only folder, and that refusal is the whole
    difference between a scored submission and a silent zero. When it happens the only trace
    left anywhere is a `submission_id: None` row in the public audit days later, so the fact
    is printed here while the container is still alive to be read. No pickable,
    nonempty checkpoint is a failed run, even if run.py returned zero.
    """
    if not output_folder.is_dir():
        print(f"[submit] FOLDER MISSING: {output_folder} -- uploader will find nothing",
              flush=True)
        raise RuntimeError(f"Training produced no submission directory: {output_folder}")
    weights = sorted(p for p in output_folder.glob("*.safetensors") if p.is_file())
    total = sum(p.stat().st_size for p in weights)
    names = ", ".join(p.name for p in weights) or "(yok)"
    print(f"[submit] {output_folder}: {len(weights)} safetensors, {total / 1e6:.0f} MB -> {names}",
          flush=True)
    if any(p.name.endswith("last.safetensors") and p.stat().st_size == 0 for p in weights):
        raise RuntimeError(f"Evaluator-preferred checkpoint is empty: {output_folder}")
    pickable = [p for p in weights if p.stat().st_size > 0 and
                (p.name.endswith("last.safetensors") or NUMBERED_CKPT_RE.search(p.name))]
    if not pickable:
        raise RuntimeError(f"Training produced no nonempty, evaluator-pickable checkpoint: {output_folder}")
    if not (output_folder / "last.safetensors").is_file():
        print("[submit] WARNING: last.safetensors yok — evaluator numaralı checkpoint'e düşecek",
              flush=True)


def main() -> None:
    started = time.time()
    args = parse_args()
    args.started_ts = started
    deadline = started + float(args.hours_to_complete) * 3600.0
    print(f"[recipe] {RECIPE_ENV}={os.environ.get(RECIPE_ENV, '') or '<unset>'} -> {recipe()}", flush=True)
    if recipe() in ("winner", "winner+select", "winner+holdout"):
        report_patch_status()
    dataset_path = prepare_dataset(args.task_id, args.model_type)
    config_path = prepare_config(args, dataset_path)

    # Every image type gets the wall guard: the stock templates (2000 steps at
    # ~1.7-2s/step for flux) sit at or over a 1.0h budget on the harness GPU, so
    # an unguarded run regularly dies mid-save with nothing named for the
    # evaluator to pick up.
    training_folder = CHECKPOINTS_ROOT / args.task_id
    output_folder = training_folder / args.expected_repo_name
    reserve = krea2_reserve_s(float(args.hours_to_complete)) if args.model_type == "krea2" else PROMOTE_RESERVE_S
    if args.model_type == "krea2" and getattr(args, "recipe_effective", "ours") == "winner" \
            and not getattr(args, "recipe_select", False):
        reserve = WINNER_WALL_MARGIN_S   # winner trains to the wall; no select+calibrate pass to fund
        print(f"[recipe] winner: krea2 trains to wall-{reserve}s, post-train select+calibrate disabled", flush=True)
    if recipe_holdout() and args.model_type == "ideogram4":
        reserve = holdout_reserve_s(float(args.hours_to_complete))
        print(f"[wall] budget {float(args.hours_to_complete):.2f}h, holdout-select reserve {reserve}s", flush=True)
        run_holdout_candidates(args, dataset_path, config_path, training_folder, output_folder,
                               deadline - reserve, deadline)
    else:
        print(f"[wall] budget {float(args.hours_to_complete):.2f}h, post-training reserve {reserve}s", flush=True)
        run_training(config_path, deadline - reserve)
    # Post-training steps must never be the reason a trained artefact fails to ship. The
    # harness marks a task FAILED and uploads NOTHING when this container exits non-zero,
    # while a container killed at the wall is treated as success and whatever is on disk goes
    # up (trainer/runtime.py). An exception here would therefore throw away a finished run.
    try:
        promote_latest_checkpoint(output_folder, args.expected_repo_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[wall] promote failed ({type(exc).__name__}: {exc})", flush=True)
    _report_submission(output_folder)

    if args.model_type == "krea2" and getattr(args, "recipe_effective", "ours") == "winner" \
            and not getattr(args, "recipe_select", False):
        print("[recipe] winner: post-train select+calibrate SKIPPED (POST_TRAIN_MODE=none for this recipe)", flush=True)
    elif args.model_type == "krea2":
        # Post-training pass (see lora_calibration.py, POST_TRAIN_MODE). check=False
        # and a hard deadline inside: whatever calibration does or fails to do, the
        # artefact ai-toolkit saved is already on disk and stays the submission
        # unless calibration finds a measured improvement to fold in.
        subprocess.run(
            [
                "python3",
                "/usr/local/bin/lora_calibration.py",
                "--mode", POST_TRAIN_MODE,
                "--training-folder", str(CHECKPOINTS_ROOT / args.task_id),
                "--dataset-dir", str(dataset_path),
                "--base-model", str(cached_model_path(args.model)),
                "--model-kwargs", json.dumps(_calibration_model_kwargs(args)),
                "--deadline-epoch", str(deadline),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
