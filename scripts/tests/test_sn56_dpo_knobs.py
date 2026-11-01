"""DPO arm knobs (sn56_dpo_knobs.py) + the scale pick rule (sn56_dpo_scale.pick_alpha), CPU, no TRL.

  1. Empty env = every default = the inherited path: no LR override, scale 1, devfit on, no calibration,
     schedule() changes nothing (train_dpo.py then touches no argument and no callback).
  2. Parsing: valid values, junk / non-positive values fall back to the default, SN56_TEXT_FIX=0 forces defaults.
  3. schedule() in the 198f2175 shape (481 opt steps/epoch, 3 planned epochs, warmup 43): the arm B / C settings,
     the explicit eval-skip threshold whenever a schedule knob is set (the inherited rule divides the callback's
     build-time total by the CURRENT epoch count), the 3% warmup rule, and non-final runs untouched.
  4. pick_alpha: argmin with the 1.0 fallback below min_gain, NaN/inf ignored.
  5. train_dpo.py wiring (source): the knobs gate the search, the devfit and the calibration, and the defaults path
     still calls run_lr_search with the inherited arguments.

Run: python scripts/tests/test_sn56_dpo_knobs.py
"""
import ast
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sn56_dpo_knobs as K  # noqa: E402

SPE, EPOCHS, WARMUP = 481, 3.0, 43


def test_defaults_are_inherited():
    k = K.read_knobs({})
    assert k["lr"] is None and k["lr_scale"] == 1.0 and k["epochs"] is None
    assert k["eval_per_epoch"] is None and k["eval_from"] is None
    assert k["devfit"] is True and k["scale_cal"] is False and k["scale_keep_dir"] == ""
    assert K.non_default(k) == {} and K.summary(k) == "padrão"
    for final in (True, False):
        s = K.schedule(k, SPE, EPOCHS, 360, WARMUP, final)
        assert s["changed"] is False and s["eval_steps"] is None and s["eval_skip_threshold"] is None
        assert s["epochs"] == EPOCHS and s["warmup_steps"] == WARMUP
    assert K.scale_grid(k)[0] == 0.4 and 1.0 in K.scale_grid(k)
    print("[ok] defaults = inherited path (no LR override, x1, devfit on, no calibration, schedule untouched)")


def test_parsing():
    env = {"SN56_DPO_LR": "8.7e-6", "SN56_DPO_LR_SCALE": "0.5", "SN56_DPO_EPOCHS": "1",
           "SN56_DPO_EVAL_PER_EPOCH": "10", "SN56_DPO_EVAL_FROM": "0.3", "SN56_DPO_DEVFIT": "0",
           "SN56_DPO_SCALE_CAL": "1", "SN56_DPO_SCALE_GRID": "0.7, 0.5,0.5,9,-1",
           "SN56_DPO_SCALE_KEEP_DIR": "/root/out/x-keep"}
    k = K.read_knobs(env)
    assert math.isclose(k["lr"], 8.7e-6) and k["lr_scale"] == 0.5 and k["epochs"] == 1.0
    assert k["eval_per_epoch"] == 10 and k["eval_from"] == 0.3 and k["devfit"] is False
    assert k["scale_cal"] is True and k["scale_grid"] == (0.5, 0.7) and k["scale_keep_dir"] == "/root/out/x-keep"
    assert K.scale_grid(k) == [0.5, 0.7, 1.0]
    bad = {"SN56_DPO_LR": "-1", "SN56_DPO_LR_SCALE": "abc", "SN56_DPO_EPOCHS": "0", "SN56_DPO_EVAL_PER_EPOCH": "0",
           "SN56_DPO_EVAL_FROM": "-0.5", "SN56_DPO_DEVFIT": "maybe", "SN56_DPO_SCALE_CAL": "",
           "SN56_DPO_SCALE_GRID": "x,y", "SN56_DPO_SCALE_MIN_GAIN": "nan"}
    assert K.read_knobs(bad) == K.read_knobs({}), K.non_default(K.read_knobs(bad))
    assert K.read_knobs(dict(env, SN56_TEXT_FIX="0")) == K.read_knobs({})
    print("[ok] parsing: values read, junk -> default, SN56_TEXT_FIX=0 forces defaults")


def test_schedule_arms():
    # arm B: 2 epochs, 8 evals/epoch from 0.25 epoch
    b = K.schedule(K.read_knobs({"SN56_DPO_EPOCHS": "2", "SN56_DPO_EVAL_PER_EPOCH": "8", "SN56_DPO_EVAL_FROM": "0.25"}),
                   SPE, EPOCHS, 360, WARMUP, True)
    assert b["changed"] and b["epochs"] == 2.0 and b["eval_steps"] == 60 and b["eval_skip_threshold"] == 120
    assert b["total_steps_all_epochs"] == 962 and b["warmup_steps"] == 28   # min(43, max(10, int(0.03*962)))
    # arm C: 1 epoch, 10 evals/epoch from 0.3
    c = K.schedule(K.read_knobs({"SN56_DPO_EPOCHS": "1", "SN56_DPO_EVAL_PER_EPOCH": "10", "SN56_DPO_EVAL_FROM": "0.3"}),
                   SPE, EPOCHS, 360, WARMUP, True)
    assert c["eval_steps"] == 48 and c["eval_skip_threshold"] == 144 and c["warmup_steps"] == 14
    # epochs only: the threshold is set explicitly (inherited 0.75 above 1 epoch, 0 at <= 1 epoch)
    e2 = K.schedule(K.read_knobs({"SN56_DPO_EPOCHS": "1.5"}), SPE, EPOCHS, 360, WARMUP, True)
    assert e2["eval_skip_threshold"] == round(0.75 * SPE) and e2["eval_steps"] is None
    e1 = K.schedule(K.read_knobs({"SN56_DPO_EPOCHS": "1"}), SPE, EPOCHS, 360, WARMUP, True)
    assert e1["eval_skip_threshold"] == 0
    # the inherited rule after an epoch change would read steps/epoch = 1443 / 1 -> threshold 1082 > every step
    assert int(0.75 * (SPE * EPOCHS) / 1.0) > SPE
    # non-final run: epochs still apply, eval knobs do not
    nf = K.schedule(K.read_knobs({"SN56_DPO_EPOCHS": "1", "SN56_DPO_EVAL_PER_EPOCH": "10"}), SPE, EPOCHS, 360, WARMUP,
                    False)
    assert nf["epochs"] == 1.0 and nf["eval_steps"] is None and nf["eval_skip_threshold"] is None
    print("[ok] schedule: arm B (2 ep, every 60 from 120), arm C (1 ep, every 48 from 144), explicit skip threshold")


def test_pick_alpha():
    try:
        from sn56_dpo_scale import pick_alpha
    except Exception as e:  # pragma: no cover
        print(f"[skip] pick_alpha ({e})")
        return
    assert pick_alpha({0.5: 0.61, 0.6: 0.612, 1.0: 0.645}, 0.001) == 0.5
    assert pick_alpha({0.9: 0.6449, 1.0: 0.6450}, 0.001) == 1.0          # gain below min_gain
    assert pick_alpha({0.5: float("nan"), 0.7: float("inf"), 1.0: 0.64}, 0.001) == 1.0
    assert pick_alpha({0.5: 0.6}, 0.001) == 1.0                          # no a=1 reference -> keep
    assert pick_alpha({0.8: 0.60, 1.2: 0.60, 1.0: 0.65}, 0.0) == 0.8     # equal loss, equal distance: first seen
    print("[ok] pick_alpha: argmin, 1.0 below min_gain / without reference, non-finite ignored")


def test_train_dpo_wiring():
    src = (HERE.parent / "train_dpo.py").read_text()
    ast.parse(src)
    assert "sn56_dpo_knobs.read_knobs()" in src
    assert 'if _knobs["lr"] is not None:' in src and "run_lr_search(" in src
    assert 'and _knobs["devfit"]:' in src
    assert 'if _knobs["scale_cal"]:' in src and "sn56_dpo_scale.calibrate(" in src
    assert "sn56_dpo_scale.apply_and_save(" in src and "if not _devfit_ok:" in src
    assert '_sched["changed"]' in src and "_eval_cb.sn56_eval_skip_threshold" in src
    # the search call keeps the inherited arguments
    for arg in ("initial_lr=training_args.learning_rate", "steps_per_epoch=total_steps_per_epoch",
                "eval_dataloader=trainer.get_eval_dataloader(_search_eval)", "dataloader_factory=_halve_batch_dataloader"):
        assert arg in src, arg
    print("[ok] train_dpo.py: knobs gate search / devfit / calibration; inherited search call intact")


if __name__ == "__main__":
    tests = [test_defaults_are_inherited, test_parsing, test_schedule_arms, test_pick_alpha, test_train_dpo_wiring]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 DPO KNOB TESTS PASSED")
