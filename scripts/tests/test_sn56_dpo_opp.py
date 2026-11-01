"""SN56_DPO_SELECT=opp (sn56_dpo_select.opp_gate_stats & co, sn56_dpo_knobs.gate_split), CPU, no TRL.

  1. Verbatim: opp_gate_stats' gate fields and verdict equal the UPSTREAM functions (exec'd from the
     upstream checkout, not our copy) on three synthetic opponents (pass / win-rate fail / gap fail);
     sn56_paired_gate.py stays byte-identical (test_sn56_dpo_select checks the source).
  2. Ranking: every crowned candidate scores <= opp_mean < every failure; among failures the larger
     WORSE standardized slack ranks first (a one-sided winner does not beat a balanced near-miss);
     rank_candidates(rule="opp") orders the same way.
  3. ID alignment: written vector read back in another order, duplicates consistent, a missing id or a
     conflicting duplicate raises.
  4. gate_split: deterministic, independent of row order, duplicates on one side, size ~n, capped.
  5. [measured, if the 29 Oct round-2 dumps are on disk] D1 falsifier: against the task-local reproduction
     A@480 (dev, 722 pairs) NO checkpoint of arms A/B/C passes both bounds; mean-gap LB <= 0 everywhere.

Run: python scripts/tests/test_sn56_dpo_opp.py
"""
import ast
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sn56_dpo_knobs as K  # noqa: E402
from sn56_dpo_select import (GAP_SCALE, WR_SCALE, load_opp_vector, opp_gate_stats, pair_id,  # noqa: E402
                             rank_candidates, write_opp_vector)

WS = Path(os.environ.get("SN56_WS", HERE.parents[3]))


def _upstream_gate():
    """compare_paired_losses + challenger_takes_paired_task exec'd from upstream origin/main."""
    try:
        src = subprocess.run(["git", "-C", str(WS / "upstream"), "show", "origin/main:validator/tournament/thresholds.py"],
                             capture_output=True, text=True, check=True).stdout
        csrc = subprocess.run(["git", "-C", str(WS / "upstream"), "show", "origin/main:validator/tournament/constants.py"],
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    consts = {}
    for ln in csrc.splitlines():
        if ln.startswith("BOSS_ROUND_") and "=" in ln:
            k, v = ln.split("=", 1)
            try:
                consts[k.strip().split(":")[0].strip()] = eval(v.split("#")[0].strip())
            except Exception:
                pass             # non-literal constants (dates etc.) are not used by the gate

    class PLC(SimpleNamespace):
        def model_dump(self):
            return dict(self.__dict__)

    ns = {"np": np, "t_cst": SimpleNamespace(**consts), "PairedLossComparison": PLC}
    names = ("compare_paired_losses", "challenger_takes_paired_task", "_bootstrap_lower_bounds")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.get_source_segment(src, node), f"upstream:{node.name}", "exec"), ns)
    return ns


def _opponent_and_candidates(n=1000, seed=7):
    rng = np.random.default_rng(seed)
    opp = np.clip(0.61 + rng.normal(0, 0.35, n), 1e-3, None)
    better = np.clip(opp - (0.03 + rng.normal(0, 0.04, n)), 1e-4, None)          # wins most, mean gap >> 0.01
    wr_fail = opp - np.where(rng.random(n) < 0.3, 0.12, -0.02)                       # few big wins, many small losses
    gap_fail = np.clip(opp - (0.004 + rng.normal(0, 0.01, n)), 1e-4, None)           # wins majority, gap too small
    return opp, {"better": better, "wr_fail": wr_fail, "gap_fail": gap_fail}


def test_verbatim_fields():
    up = _upstream_gate()
    opp, cands = _opponent_and_candidates()
    for name, v in cands.items():
        st = opp_gate_stats(v, opp)
        if up is not None:
            comp = up["compare_paired_losses"](list(opp), list(v))
            want = comp.model_dump()
            for k in ("n_examples", "n_decided", "win_rate", "win_rate_lower_bound", "mean_gap_nats",
                      "mean_gap_lower_bound", "required_mean_gap_nats", "challenger_won", "is_draw"):
                assert st[k] == want[k], (name, k, st[k], want[k])
            takes, _ = up["challenger_takes_paired_task"](comp, float(np.mean(opp)), float(np.mean(v)))
            assert st["takes_task"] == takes, name
    assert opp_gate_stats(cands["better"], opp)["takes_task"]
    assert not opp_gate_stats(cands["wr_fail"], opp)["takes_task"]
    assert not opp_gate_stats(cands["gap_fail"], opp)["takes_task"]
    print("[ok] opp_gate_stats == upstream compare_paired_losses + challenger_takes_paired_task"
          + (" (exec'd from upstream origin/main)" if up is not None else " [upstream checkout missing: own copy only]"))


def test_ranking():
    opp, cands = _opponent_and_candidates()
    st = {k: opp_gate_stats(v, opp) for k, v in cands.items()}
    om = st["better"]["opp_mean"]
    assert st["better"]["score"] <= om < st["wr_fail"]["score"] and om < st["gap_fail"]["score"]
    worst = {k: min(s["slack_wr"], s["slack_gap"]) for k, s in st.items()}
    for k, s in st.items():
        assert abs(s["score"] - (max(om - GAP_SCALE * worst[k], om + 1e-6) if not s["takes_task"]
                                 else om - GAP_SCALE * worst[k])) < 1e-12
    order = [r[0] for r in rank_candidates(cands, rule="opp", proxy_losses=opp)]
    assert order[0] == "better"
    assert order[1:] == sorted(["wr_fail", "gap_fail"], key=lambda k: -worst[k])
    # balanced near-miss beats a one-sided candidate (great win rate, bad gap)
    a = {"slack_wr": 3.0, "slack_gap": -2.0}; b = {"slack_wr": -0.5, "slack_gap": -0.4}
    assert min(b.values()) > min(a.values())
    # units: one slack unit = WR_SCALE of win-rate LB or GAP_SCALE nats of gap LB
    assert (WR_SCALE, GAP_SCALE) == (0.05, 0.01)
    # all-NaN candidate is never chosen
    assert math.isinf(opp_gate_stats([float("nan")] * len(opp), opp)["score"])
    print(f"[ok] ranking: crowned {st['better']['score']:.4f} <= opp mean {om:.4f} < failures "
          f"(gap_fail {st['gap_fail']['score']:.4f}, wr_fail {st['wr_fail']['score']:.4f}); worst-slack max-min")


def test_id_alignment():
    rows = [("p%d" % i, "c%d" % i, "r%d" % i) for i in range(50)] + [("p3", "c3", "r3")]   # one duplicate
    ids = [pair_id(*r) for r in rows]
    assert ids[3] == ids[-1] and len(set(ids)) == 50
    losses = [0.1 * i for i in range(50)]
    losses.append(losses[3])                                       # the duplicate scores identically
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "opp.json")
        perm = list(reversed(range(len(rows))))
        write_opp_vector(path, [ids[i] for i in perm], [losses[i] for i in perm])
        got = load_opp_vector(path, ids)
        assert np.allclose(got, losses)
        try:
            load_opp_vector(path, ids + [pair_id("x", "y", "z")]); raise AssertionError("missing id accepted")
        except ValueError:
            pass
        write_opp_vector(path, ids, losses[:-1] + [losses[3] + 0.01])   # duplicate with a different loss
        try:
            load_opp_vector(path, ids); raise AssertionError("conflicting duplicate accepted")
        except ValueError:
            pass
    print("[ok] ID alignment: order-free, duplicates consistent, missing / conflicting ids raise")


def test_gate_split():
    rows = [f"id{i % 900}" for i in range(1000)]                   # 100 duplicated ids
    g, t = K.gate_split(rows, 100)
    assert set(g).isdisjoint(t) and len(g) + len(t) == 1000 and 100 <= len(g) <= 102
    assert {rows[i] for i in g}.isdisjoint({rows[i] for i in t})   # no id on both sides
    g2, _ = K.gate_split(list(reversed(rows)), 100)
    assert {rows[::-1][i] for i in g2} == {rows[i] for i in g}      # content-defined, not order-defined
    assert K.gate_split(rows, 0) == ([], list(range(1000)))
    gc, tc = K.gate_split(rows, 5000, min_train=100)
    assert len(tc) >= 99 and len(gc) <= 901
    k = K.read_knobs({})
    assert k["gate_split"] == 0 and k["opp_vec"] == "" and k["opp_model"] == "" and k["init"] == ""
    k = K.read_knobs({"SN56_DPO_GATE_SPLIT": "1000", "SN56_DPO_OPP_VEC": "/x.json", "SN56_DPO_INIT": "/m"})
    assert k["gate_split"] == 1000 and k["opp_vec"] == "/x.json" and k["init"] == "/m"
    assert K.read_knobs({"SN56_DPO_GATE_SPLIT": "-3"})["gate_split"] == 0
    print(f"[ok] gate_split: {len(g)} rows, content-defined, duplicates together, defaults off")


def test_round2_falsifier():
    R = WS / "results/text-1030-k/pods/m3dpo-198f2175/results"
    files = sorted(glob.glob(str(R / "dev-*/dev-step*.json")))
    if not files:
        print("[skip] round-2 dumps not on disk")
        return
    vecs = {}
    for p in files:
        d = json.loads(Path(p).read_text())
        vecs[f"{Path(p).parent.name[4:]}@{d['step']}e{d['eval_index']}"] = np.array(d["per_pair"])
    opp_key = next(k for k in vecs if k.startswith("A@480e"))
    opp = vecs.pop(opp_key)
    rows = rank_candidates(vecs, rule="opp", proxy_losses=opp)
    passed = [r for r in rows if r[2]["takes_task"]]
    best = rows[0]
    gap_lb_max = max(r[2]["mean_gap_lower_bound"] for r in rows)
    assert not passed and gap_lb_max <= 0.0, (len(passed), gap_lb_max)
    print(f"[ok] D1 falsifier [measured]: 0/{len(rows)} round-2 checkpoints pass vs reproduction {opp_key}; "
          f"best {best[0]} wr_lb {best[2]['win_rate_lower_bound']:.3f} gap_lb {best[2]['mean_gap_lower_bound']:+.4f}; "
          f"max gap_lb {gap_lb_max:+.4f}")


if __name__ == "__main__":
    tests = [test_verbatim_fields, test_ranking, test_id_alignment, test_gate_split, test_round2_falsifier]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 DPO OPP TESTS PASSED")
