"""SN56_DPO_SELECT=gate (K5): checkpoint selection by the validator's paired gate.

  1. sn56_paired_gate.py is byte-identical to the upstream functions and constants
     (re-extracted from SN56_WS/upstream origin/main; skipped without a checkout).
  2. Base proxy semantics: policy == reference gives ln 2 on every pair -> draw;
     a uniform improvement d gives score ln 2 - d.
  3. Synthetic pairs in the 198f2175 shape: candidate A has the better mean but a
     heavy tail, B is slightly worse on the mean but tight. The mean rule picks A,
     the gate rule picks B; against a synthetic boss the verbatim gate fails A on
     the bound (mean gap > 0.01, bound < 0.01) and passes B.
  4. Trainer glue: a DPO-shaped Trainer (dpo_loss -> per-pair losses) wrapped by
     make_gate_dpo_trainer: eval_loss == gate score, mean kept, dump written;
     select=False records only.

Run: python scripts/tests/test_sn56_dpo_select.py
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

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sn56_paired_gate as G  # noqa: E402
from sn56_dpo_select import BASE_PROXY_LOSS, gate_vs_proxy, make_gate_dpo_trainer, rank_candidates  # noqa: E402

WS = Path(os.environ.get("SN56_WS", HERE.parents[3]))
LN2 = math.log(2.0)


def _upstream(path):
    try:
        return subprocess.run(["git", "-C", str(WS / "upstream"), "show", f"origin/main:{path}"],
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return None


def test_verbatim_against_upstream():
    src = _upstream("validator/tournament/thresholds.py")
    csrc = _upstream("validator/tournament/constants.py")
    if src is None or csrc is None:
        print("[skip] verbatim check (no upstream checkout)")
        return
    mine = Path(G.__file__).read_text()
    names = ("compare_paired_losses", "challenger_takes_paired_task", "_bootstrap_lower_bounds")
    up = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name in names}
    me = {n.name: ast.get_source_segment(mine, n) for n in ast.parse(mine).body
          if isinstance(n, ast.FunctionDef) and n.name in names}
    for n in names:
        assert up[n] == me[n], f"{n} differs from upstream"
    for k, v in vars(G.t_cst).items():
        line = next(ln for ln in csrc.splitlines() if ln.startswith(k + " ="))
        assert eval(line.split("=", 1)[1].split("#")[0].strip()) == v, (k, line, v)
    print(f"[ok] sn56_paired_gate.py == upstream origin/main ({', '.join(names)}; {len(vars(G.t_cst))} constants)")


def test_base_proxy():
    flat = gate_vs_proxy([LN2] * 500)
    assert flat["is_draw"] and flat["n_decided"] == 0 and abs(flat["score"] - LN2) < 1e-12
    d = 0.05
    up = gate_vs_proxy([LN2 - d] * 500)
    assert up["challenger_won"] and abs(up["mean_gap_lower_bound"] - d) < 1e-12 and abs(up["score"] - (LN2 - d)) < 1e-12
    # a NaN pair is dropped from both sides, as the validator does
    nanv = gate_vs_proxy([LN2 - d] * 499 + [float("nan")])
    assert nanv["n_examples"] == 499
    print("[ok] base proxy: ln 2 vector = draw / score ln 2; uniform -d -> score ln 2 - d; NaN dropped")


def _synthetic(n=1000, seed=20261026):
    rng = np.random.default_rng(seed)
    boss = np.clip(LN2 - 0.07 + rng.normal(0, 0.15, n), 1e-4, None)
    tight = boss - (0.012 + rng.normal(0, 0.025, n))                 # B: gap ~ N(0.012, 0.025)
    gap_a = 0.06 + rng.normal(0, 0.05, n)
    cat = rng.random(n) < 0.03                                         # 3% catastrophic pairs
    gap_a[cat] -= 1.5
    heavy = boss - gap_a                                               # A: better mean, heavy tail
    return boss, heavy, tight


def test_gate_rule_vs_mean_rule():
    boss, heavy, tight = _synthetic()
    assert heavy.mean() < tight.mean()                       # A is better on the mean
    by_mean = rank_candidates({"A_heavy": heavy, "B_tight": tight}, rule="mean")
    by_gate = rank_candidates({"A_heavy": heavy, "B_tight": tight}, rule="gate")
    assert by_mean[0][0] == "A_heavy" and by_gate[0][0] == "B_tight", (by_mean, by_gate)
    va = G.compare_paired_losses(boss, heavy)
    vb = G.compare_paired_losses(boss, tight)
    wa, ra = G.challenger_takes_paired_task(va, float(boss.mean()), float(heavy.mean()))
    wb, rb = G.challenger_takes_paired_task(vb, float(boss.mean()), float(tight.mean()))
    assert va.mean_gap_nats > 0.01 and va.mean_gap_lower_bound < 0.01 and not wa, ra
    assert wb, rb
    sa, sb = by_gate[1][2], by_gate[0][2]
    print(f"     A heavy: dev mean {heavy.mean():.4f} score {sa['score']:.4f} | vs boss: {ra}")
    print(f"     B tight: dev mean {tight.mean():.4f} score {sb['score']:.4f} | vs boss: {rb}")
    print("[ok] mean rule -> A (loses the gate on the bound, the 198f2175 shape); gate rule -> B (passes)")


def test_trainer_glue():
    try:
        from transformers import Trainer, TrainingArguments
    except Exception as e:  # pragma: no cover
        print(f"[skip] trainer glue ({e})")
        return
    boss, heavy, tight = _synthetic(n=96)
    rng = np.random.default_rng(1)
    # per pair: policy/ref chosen/rejected log-prob sums giving loss -logsigmoid(0.1 delta) == heavy[i]
    delta = -10.0 * np.log(np.expm1(heavy))                 # inverse of softplus(-0.1 delta)
    ref_c, ref_r = rng.normal(-80, 10, heavy.size), rng.normal(-90, 10, heavy.size)
    pol_c, pol_r = ref_c + delta / 2, ref_r - delta / 2

    class PairModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("pc", torch.tensor(pol_c))
            self.register_buffer("pr", torch.tensor(pol_r))
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, idx=None, **kw):
            return {"chosen_logps": self.pc[idx], "rejected_logps": self.pr[idx]}

    rc_t, rr_t = torch.tensor(ref_c), torch.tensor(ref_r)

    class FakeDPOTrainer(Trainer):                 # the TRL 0.18 call shape: prediction_step -> dpo_loss
        def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
            logits = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
            losses = -F.logsigmoid(0.1 * logits)
            return losses, 0.1 * (chosen_logps - ref_chosen_logps), 0.1 * (rejected_logps - ref_rejected_logps)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                out = model(**inputs)
                i = inputs["idx"]
                losses, _, _ = self.dpo_loss(out["chosen_logps"], out["rejected_logps"], rc_t[i], rr_t[i])
            return losses.mean().detach(), None, None

    data = [{"idx": i} for i in range(heavy.size)]

    def collate(feats):
        return {"idx": torch.tensor([f["idx"] for f in feats])}

    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(output_dir=tmp, report_to=[], per_device_eval_batch_size=7)
        try:
            args = TrainingArguments(use_cpu=True, **kw)
        except TypeError:
            args = TrainingArguments(no_cuda=True, **kw)
        dump = os.path.join(tmp, "dump")
        tr = make_gate_dpo_trainer(FakeDPOTrainer, dump_dir=dump)(model=PairModel(), args=args, eval_dataset=data,
                                                                  data_collator=collate)
        m = tr.evaluate()
        want = gate_vs_proxy(heavy)
        assert abs(m["eval_loss"] - want["score"]) < 1e-6, (m["eval_loss"], want["score"])   # fp32 per-pair
        assert abs(m["eval_loss_mean"] - heavy.mean()) < 1e-6 and abs(m["eval_pair_mean"] - heavy.mean()) < 1e-6
        files = glob.glob(os.path.join(dump, "dev-step*.json"))
        assert len(files) == 1
        js = json.load(open(files[0]))
        assert np.allclose(js["per_pair"], heavy, atol=1e-6)    # gathered in dataset order, 96 = 13x7 + 5
        tr2 = make_gate_dpo_trainer(FakeDPOTrainer, dump_dir=dump, select=False)(
            model=PairModel(), args=args, eval_dataset=data, data_collator=collate)
        m2 = tr2.evaluate()
        assert abs(m2["eval_loss"] - heavy.mean()) < 1e-6 and abs(m2["eval_gate_score"] - want["score"]) < 1e-6
    print(f"[ok] Trainer glue: eval_loss = gate score {want['score']:.5f} (mean {heavy.mean():.5f} kept), "
          f"per-pair dump in dataset order; select=False records only")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 DPO SELECT TESTS PASSED")
