#!/usr/bin/env python3
"""Dry-run the image config build for IMG_RECIPE=ours vs IMG_RECIPE=winner -- no GPU, no /cache,
no /app/ai-toolkit needed (I6 port, PLAN-IMAGE-0908 v2).

    python3 ops/docker/scripts/recipe_dryrun.py                      # krea2 + ideogram4, n=45, 1.0h
    python3 ops/docker/scripts/recipe_dryrun.py --hours 0.75 --n 30
    python3 ops/docker/scripts/recipe_dryrun.py --arch krea2 --print-yaml

What it does: imports image_toolkit_entrypoint, sets SKIP_PATH_CHECKS (cache-path checks pass,
patch markers count as "unverifiable" = present), points the template root at this repo's
core/training_templates, builds a temp dataset of n empty image/caption pairs, and calls
prepare_config() twice per arch. The entrypoint itself prints the "[recipe] effective config
diff" block exactly as it would on the pod; this script adds the ours->winner key diff with
wall-relative timestamps and the env the winner path exports (KREA2_MODLORA).
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

import image_toolkit_entrypoint as entry  # noqa: E402

MODEL_BY_TYPE = {
    "krea2": "krea/Krea-2-Raw",
    "ideogram4": "gradients-io-tournaments/ideogram-4-fp8",
    "z-image": "gradients-io-tournaments/Z-Image-Turbo",
    "qwen-image": "gradients-io-tournaments/Qwen-Image-Jib-Mix",
    "flux": "dataautogpt3/FLUX-MonochromeManga",
}
_ENV_CLEAR = ("IMG_RECIPE", "KREA2_MODLORA")


def make_dataset(root, n, caption):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"img_{i:03d}.png").write_bytes(b"")
        (root / f"img_{i:03d}.txt").write_text(caption)
    return root


def build(arch, rec, n_hours, dataset, out_root, started):
    for k in _ENV_CLEAR:
        os.environ.pop(k, None)
    if rec != "ours":
        os.environ["IMG_RECIPE"] = rec
    entry.CONFIG_OUTPUT_ROOT = out_root / rec
    args = argparse.Namespace(
        task_id=f"dryrun-{arch}", model=MODEL_BY_TYPE[arch], dataset_zip="-", model_type=arch,
        expected_repo_name="dryrun-repo", hours_to_complete=str(n_hours), trigger_word=None,
        started_ts=started,
    )
    print(f"--- build {arch} / IMG_RECIPE={rec} ---")
    path = entry.prepare_config(args, dataset)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    exported = {k: os.environ[k] for k in ("KREA2_MODLORA",) if k in os.environ}
    return cfg, getattr(args, "recipe_effective", "ours"), exported


def humanise(line, started, hours):
    """Show *_ts values as wall-relative offsets so the diff is readable."""
    import re
    def _sub(m):
        val = float(m.group(0))
        return f"{val:.0f} (= wall{val - (started + hours * 3600.0):+.0f}s)"
    if "_ts" in line:
        return re.sub(r"\d{9,}\.\d+", _sub, line)
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="krea2,ideogram4", help="comma list of model types")
    ap.add_argument("--n", type=int, default=45, help="image/caption pairs in the fake dataset")
    ap.add_argument("--hours", type=float, default=1.0, help="hours_to_complete")
    ap.add_argument("--caption", default="a photo of a thing on a table",
                    help="caption written to every .txt (drives the log-only ideogram4 category vote)")
    ap.add_argument("--templates", default=str(REPO / "core" / "training_templates"))
    ap.add_argument("--print-yaml", action="store_true", help="also dump the full winner yaml per arch")
    ap.add_argument("--no-skip-checks", action="store_true", help="keep the real /cache path checks (pod only)")
    a = ap.parse_args()

    entry.SKIP_PATH_CHECKS = not a.no_skip_checks
    entry.CONFIG_TEMPLATE_ROOT = Path(a.templates)
    print(f"[dryrun] templates={entry.CONFIG_TEMPLATE_ROOT} skip_checks={entry.SKIP_PATH_CHECKS} "
          f"n={a.n} hours={a.hours}")
    entry.report_patch_status()

    with tempfile.TemporaryDirectory(prefix="recipe-dryrun-") as tmp:
        tmp = Path(tmp)
        dataset = make_dataset(tmp / "images", a.n, a.caption)
        started = time.time()
        for arch in [x.strip() for x in a.arch.split(",") if x.strip()]:
            ours, eff_o, env_o = build(arch, "ours", a.hours, dataset, tmp / "configs", started)
            winner, eff_w, env_w = build(arch, "winner", a.hours, dataset, tmp / "configs", started)
            lines = entry.config_diff_lines(ours, winner)
            print(f"=== {arch}: ours -> winner  (effective: ours={eff_o}, winner={eff_w}; "
                  f"{len(lines)} key(s) differ; env exported by winner path: {env_w or '{}'})")
            for line in lines:
                print(humanise(line, started, a.hours))
            if a.print_yaml:
                print(f"--- full winner yaml ({arch}) ---")
                print(yaml.safe_dump(winner, sort_keys=False))
    for k in _ENV_CLEAR:
        os.environ.pop(k, None)


if __name__ == "__main__":
    main()
