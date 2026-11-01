"""SN56_BATCH_DIV and SN56_PLW (31 Oct): both default to the inherited behaviour.

Run: python scripts/tests/test_sn56_batch_plw.py   (CPU, no torch)
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import sn56_plan  # noqa: E402


def _env(**kv):
    for k in ("SN56_BATCH_DIV", "SN56_PLW", "SN56_TEXT_FIX", "SN56_R1_SMALLBATCH"):
        os.environ.pop(k, None)
    os.environ.update(kv)


def test_registered_defaults():
    assert sn56_plan.FLAGS["SN56_BATCH_DIV"] == ("1", "1")
    assert sn56_plan.FLAGS["SN56_PLW"] == ("1", "1")
    assert sn56_plan.FLAGS["SN56_MIN_STEPS_MULT"] == ("1", "1")
    _env()
    assert sn56_plan.flag_on("SN56_PLW") is True
    assert sn56_plan.batch_div(3, 14) == (3, 14)


def test_batch_div():
    _env(SN56_BATCH_DIV="3")
    assert sn56_plan.batch_div(3, 14) == (1, 14)     # the 31 Oct R2 7.6B config: accumulation goes first
    assert sn56_plan.batch_div(1, 100) == (1, 33)    # small models with accumulation 1: per-device batch shrinks
    assert sn56_plan.batch_div(2, 7) == (1, 5)
    assert sn56_plan.batch_div(1, 1) == (1, 1)
    _env(SN56_BATCH_DIV="2")
    assert sn56_plan.batch_div(4, 8) == (2, 8)
    _env(SN56_BATCH_DIV="0.5")                       # never grows the batch
    assert sn56_plan.batch_div(3, 14) == (3, 14)
    _env(SN56_BATCH_DIV="junk")
    assert sn56_plan.batch_div(3, 14) == (3, 14)


def test_r1_smallbatch():
    assert sn56_plan.FLAGS["SN56_R1_SMALLBATCH"] == ("1", "0")
    _env()
    assert sn56_plan.flag_on("SN56_R1_SMALLBATCH") is True
    _env(SN56_TEXT_FIX="0")                              # master rollback turns it off
    assert sn56_plan.flag_on("SN56_R1_SMALLBATCH") is False
    _env()
    assert sn56_plan.batch_div(1, 100, 3.0) == (1, 33)   # the R1 divisor reaches batch_div through floor_div
    assert sn56_plan.batch_div(8, 8, 3.0) == (3, 7)      # with accumulation, accumulation shrinks first
    _env(SN56_BATCH_DIV="4")                             # a larger explicit divisor wins
    assert sn56_plan.batch_div(1, 100, 3.0) == (1, 25)
    _env(SN56_BATCH_DIV="2")
    assert sn56_plan.batch_div(1, 100, 3.0) == (1, 33)
    cfg = open(os.path.join(HERE, "instruct_config.py")).read()
    assert ('_r1sb = (3.0 if sn56_plan.flag_on("SN56_R1_SMALLBATCH") and 1_000_000_000 <= param_nums < 4_000_000_000'
            in cfg)
    assert 'run_config["batch_size"], _r1sb)' in cfg
    assert '_msm = max(sn56_plan.flag_float("SN56_MIN_STEPS_MULT"), _r1sb)' in cfg
    assert cfg.index("_r1sb = (3.0") < cfg.index("batch_div(run_config")


def test_plw_flag_and_rollback():
    _env(SN56_PLW="0")
    assert sn56_plan.flag_on("SN56_PLW") is False
    _env(SN56_PLW="0", SN56_BATCH_DIV="3", SN56_TEXT_FIX="0")   # master rollback restores the inherited values
    assert sn56_plan.flag_on("SN56_PLW") is True
    assert sn56_plan.batch_div(3, 14) == (3, 14)


def test_wiring():
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    assert 'not sn56_plan.flag_on("SN56_PLW")' in src and "_plw = 0.05 / max(1.0, _pc_ratio / 5.0)" in src
    cfg = open(os.path.join(HERE, "instruct_config.py")).read()
    assert "batch_div(run_config[\"gradient_accumulation_steps\"], run_config[\"batch_size\"], _r1sb)" in cfg
    assert cfg.index("batch_div(") < cfg.index("run_cmd = get_run_cmd(run_config")
    assert 'flag_float("SN56_MIN_STEPS_MULT")' in cfg
    assert cfg.index('flag_float("SN56_MIN_STEPS_MULT")') > cfg.index('train_request["min_steps"] = max(')
    assert sn56_plan.FLAGS["SN56_GA_RESCALE"][1] == "1"                     # rollback keeps the 83ed96f re-derivation
    i = cfg.index('_keep_ga = (not sn56_plan.flag_on("SN56_GA_RESCALE") and packing_enabled')
    assert '== "InstructTextTask")' in cfg[i:i + 200] and "if not _keep_ga:" in cfg[i:i + 300]
    assert cfg.index("max_length > default_max") < i < cfg.index("-(-64 // data_per_step)")


if __name__ == "__main__":
    test_registered_defaults()
    test_batch_div()
    test_r1_smallbatch()
    test_plw_flag_and_rollback()
    test_wiring()
    _env()
    print("5/5 passed")
