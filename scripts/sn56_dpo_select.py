"""K5 (SN56_DPO_SELECT=gate, default mean): DPO checkpoint selection by the validator's paired gate.

Boss-round DPO tasks are decided by compare_paired_losses (sn56_paired_gate.py,
verbatim from G.O.D bfa699ad): the challenger needs win rate >= 0.55 over decided
pairs AND mean gap >= max(0.01 nats, 1% of the boss mean), BOTH at the one-sided
99% bootstrap lower bound (10k resamples, seed 20260808, dead zone 0.01), plus a
scalar no worse than the boss. The per-pair DPO loss is -logsigmoid(0.1 * delta)
with delta a SUM over completion tokens (validator/evaluation/evaluators/dpo.py
:200-272), unbounded above: a few catastrophic pairs set the bound. 26 Oct
198f2175: the challenger was 2.2% better on the mean (gap 0.0153 nats) and lost
on the bound (-0.0011).

The boss is unknown at selection time; the proxy is the base model. With the
reference = the base (as the validator scores it), the base's own per-pair loss is
exactly ln 2 on every pair. Each dev evaluation's per-pair losses (TRL's
dpo_loss: the same sigmoid formula, beta 0.1, summed log-probs) are compared to
that constant vector with the verbatim gate, and the selection score is

    score = mean(proxy) - mean_gap_lower_bound
          = ln 2 - LB99(mean(ln 2 - loss_i))  = the one-sided 99% bootstrap UPPER
            bound of the mean dev loss

lower is better, so it stands in for eval_loss: best checkpoint, greedy soup and
overfit early stop select on it (the mean stays in eval_loss_mean). A checkpoint
that buys its mean with a heavy tail scores worse than one that is slightly worse
on the mean but tight. SN56_DPO_DUMP_DIR=<dir> also writes every evaluation's
per-pair vector (rank 0) for offline A/B against the mean rule; with
SN56_DPO_SELECT=mean and a dump dir the vectors and gate stats are recorded but
eval_loss stays the mean (the control arm).
"""

import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

from sn56_paired_gate import challenger_takes_paired_task, compare_paired_losses

BASE_PROXY_LOSS = math.log(2.0)        # -logsigmoid(0): policy == reference


def gate_vs_proxy(pair_losses, proxy_losses=None) -> dict:
    """The verbatim gate with the candidate as challenger and the proxy as boss,
    plus the selection score (module doc). proxy None = base proxy (ln 2)."""
    x = np.asarray(pair_losses, dtype=np.float64).reshape(-1)
    proxy = (np.full_like(x, BASE_PROXY_LOSS) if proxy_losses is None
             else np.asarray(proxy_losses, dtype=np.float64).reshape(-1))
    comp = compare_paired_losses(proxy, x)
    d = comp.model_dump()
    ok = np.isfinite(x) & np.isfinite(proxy)
    xf = x[ok]
    d["mean_loss"] = float(xf.mean()) if xf.size else float("nan")
    d["p95_loss"] = float(np.percentile(xf, 95)) if xf.size else float("nan")
    d["max_loss"] = float(xf.max()) if xf.size else float("nan")
    d["proxy_mean"] = float(proxy[ok].mean()) if xf.size else float("nan")
    d["score"] = d["proxy_mean"] - d["mean_gap_lower_bound"] if xf.size else float("inf")
    return d


def rank_candidates(candidates: dict, rule: str = "gate", proxy_losses=None) -> list:
    """[(name, key, stats)] best first. rule: gate (score) | mean (mean_loss) | opp (opp_score;
    proxy_losses = the ID-aligned opponent vector, required)."""
    rows = []
    for name, vec in candidates.items():
        if rule == "opp":
            st = opp_gate_stats(vec, proxy_losses)
            rows.append((name, st["score"], st))
            continue
        st = gate_vs_proxy(vec, proxy_losses)
        rows.append((name, st["score"] if rule == "gate" else st["mean_loss"], st))
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


# ── SN56_DPO_SELECT=opp: selection against a cached OPPONENT vector ──
# The proxy above (ln 2 = the base) only bounds the candidate's own mean loss. The boss round
# compares against the boss per pair, so select against a lawful stand-in for it: a task-local
# reproduction of the frozen boss recipe, trained on this task's training rows only, scored on a
# split nobody trained, searched, souped or calibrated on (sn56_dpo_knobs SN56_DPO_GATE_SPLIT).
# Both bootstrap bounds of the verbatim gate count, plus the scalar guard; the ranking key is
#
#   slack_wr  = (win_rate_lb  - required_win_rate) / WR_SCALE
#   slack_gap = (mean_gap_lb - required_mean_gap) / GAP_SCALE
#   score     = opp_mean - GAP_SCALE * min(slack_wr, slack_gap)          (lower is better)
#
# so a candidate the gate would crown has score <= opp_mean and every failure > opp_mean, and
# among failures the one whose WORSE bound is closest (in these units) wins. The score lives on
# the loss scale (opp_mean ~ 0.6), keeping the inherited 5% overfit rule and the soup's
# "strictly lower" acceptance meaningful. WR_SCALE / GAP_SCALE are selection units, not fitted.
WR_SCALE = 0.05
GAP_SCALE = 0.01


def pair_id(prompt, chosen, rejected) -> str:
    """Content ID of a DPO row (the pair as the trainer reads it), for ID-aligned vectors."""
    raw = json.dumps([str(prompt), str(chosen), str(rejected)], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def dataset_ids(ds) -> list:
    """pair_id of every row, in dataset order (= the sequential eval order)."""
    return [pair_id(p, c, r) for p, c, r in zip(ds["prompt"], ds["chosen"], ds["rejected"])]


def opp_gate_stats(pair_losses, opp_losses) -> dict:
    """Verbatim gate + scalar guard with the candidate as challenger and the opponent as boss."""
    x = np.asarray(pair_losses, dtype=np.float64).reshape(-1)
    o = np.asarray(opp_losses, dtype=np.float64).reshape(-1)
    comp = compare_paired_losses(o, x)
    d = comp.model_dump()
    ok = np.isfinite(x) & np.isfinite(o)
    if not ok.any():
        d.update({"mean_loss": float("nan"), "opp_mean": float("nan"), "takes_task": False,
                  "slack_wr": float("-inf"), "slack_gap": float("-inf"), "score": float("inf")})
        return d
    d["mean_loss"] = float(x[ok].mean())
    d["opp_mean"] = float(o[ok].mean())
    d["p95_loss"] = float(np.percentile(x[ok], 95))
    takes, why = challenger_takes_paired_task(comp, d["opp_mean"], d["mean_loss"])
    d["takes_task"], d["verdict_reason"] = bool(takes), why
    d["slack_wr"] = (d["win_rate_lower_bound"] - d["required_win_rate"]) / WR_SCALE
    d["slack_gap"] = (d["mean_gap_lower_bound"] - d["required_mean_gap_nats"]) / GAP_SCALE
    worst = min(d["slack_wr"], d["slack_gap"])
    score = d["opp_mean"] - GAP_SCALE * worst
    if not takes:            # draw / scalar guard with both bounds met: still never ahead of a pass
        score = max(score, d["opp_mean"] + 1e-6)
    d["score"] = float(score)
    return d


def write_opp_vector(path, ids, losses, meta=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"ids": list(ids), "losses": [float(v) for v in losses], "meta": meta or {}}, f)
    os.replace(tmp, path)


def load_opp_vector(path, ids) -> np.ndarray:
    """The cached opponent losses aligned to ids (ValueError on any missing ID or conflicting duplicate)."""
    with open(path) as f:
        d = json.load(f)
    if len(d["ids"]) != len(d["losses"]):
        raise ValueError("opponent vector: ids/losses length mismatch")
    by_id = {}
    for i, v in zip(d["ids"], d["losses"]):
        v = float(v)
        if i in by_id and not (by_id[i] == v or (math.isnan(by_id[i]) and math.isnan(v))):
            raise ValueError(f"opponent vector: conflicting losses for duplicate id {i}")
        by_id[i] = v
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"opponent vector: {len(missing)}/{len(ids)} ids missing (first {missing[0]})")
    return np.asarray([by_id[i] for i in ids], dtype=np.float64)


def make_gate_dpo_trainer(base_cls, dump_dir: str = "", select: bool = True):
    """Subclass of a TRL DPOTrainer whose eval_loss is the gate score (module doc);
    select=False only records (control arm). Opponent mode: set trainer.sn56_opp to the
    ID-aligned opponent vector of the eval set -> eval_loss = opp_gate_stats score
    (select=True) and the opponent stats are logged / dumped; None = the ln 2 proxy."""
    _dump, _select = dump_dir or "", bool(select)

    class GateDPOTrainer(base_cls):
        _sn56_capture = False
        _sn56_pairs = None
        _sn56_evals = 0
        sn56_opp = None
        sn56_select = _select      # instance override: False = record only (e.g. opponent missing)

        def dpo_loss(self, *args, **kwargs):
            out = super().dpo_loss(*args, **kwargs)
            if self._sn56_capture:
                self._sn56_pairs = out[0].detach()
            return out

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            self._sn56_capture, self._sn56_pairs = True, None
            try:
                loss, _, _ = super().prediction_step(model, inputs, True, ignore_keys=ignore_keys)
            finally:
                self._sn56_capture = False
            pairs, self._sn56_pairs = self._sn56_pairs, None
            if pairs is None:
                return loss, None, None
            return loss, pairs.float().reshape(-1, 1), None

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                            metric_key_prefix="eval"):
            out = super().evaluation_loop(dataloader, description, prediction_loss_only=False,
                                          ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            key = f"{metric_key_prefix}_loss"
            try:
                vec = np.asarray(out.predictions, dtype=np.float64).reshape(-1)
                st = gate_vs_proxy(vec)
            except Exception as e:  # never lose an eval over the selector
                print(f"[sn56][dposel] vetor inválido ({e}); eval_loss médio mantido", file=sys.stderr, flush=True)
                return out
            if not math.isfinite(st["score"]):
                return out
            opp = None
            if self.sn56_opp is not None:
                try:
                    if len(self.sn56_opp) != vec.size:
                        raise ValueError(f"opponent {len(self.sn56_opp)} pairs, eval {vec.size}")
                    opp = opp_gate_stats(vec, self.sn56_opp)
                except Exception as e:  # never lose an eval over the selector
                    print(f"[sn56][dposel] oponente inválido ({e}); prova base", file=sys.stderr, flush=True)
                    opp = None
            out.metrics[f"{metric_key_prefix}_loss_mean"] = out.metrics.get(key)
            if self.sn56_select:
                out.metrics[key] = opp["score"] if opp is not None and math.isfinite(opp["score"]) else st["score"]
            if opp is not None:
                out.metrics[f"{metric_key_prefix}_opp_score"] = opp["score"]
                out.metrics[f"{metric_key_prefix}_opp_wr_lb"] = opp["win_rate_lower_bound"]
                out.metrics[f"{metric_key_prefix}_opp_gap_lb"] = opp["mean_gap_lower_bound"]
                out.metrics[f"{metric_key_prefix}_opp_takes"] = float(opp["takes_task"])
                print(f"[sn56][dposel] passo={self.state.global_step} vs oponente: wr={opp['win_rate']:.3f} "
                      f"lb={opp['win_rate_lower_bound']:.3f} gap={opp['mean_gap_nats']:+.5f} "
                      f"lb={opp['mean_gap_lower_bound']:+.5f} folga=({opp['slack_wr']:+.2f},{opp['slack_gap']:+.2f}) "
                      f"passa={opp['takes_task']} score={opp['score']:.5f}", file=sys.stderr, flush=True)
            out.metrics[f"{metric_key_prefix}_gate_score"] = st["score"]
            out.metrics[f"{metric_key_prefix}_gate_gap_lb"] = st["mean_gap_lower_bound"]
            out.metrics[f"{metric_key_prefix}_gate_wr_lb"] = st["win_rate_lower_bound"]
            out.metrics[f"{metric_key_prefix}_pair_mean"] = st["mean_loss"]
            out.metrics[f"{metric_key_prefix}_pair_p95"] = st["p95_loss"]
            print(f"[sn56][dposel] passo={self.state.global_step} n={st['n_examples']} média={st['mean_loss']:.5f} "
                  f"gap_lb={st['mean_gap_lower_bound']:.5f} wr_lb={st['win_rate_lower_bound']:.3f} "
                  f"p95={st['p95_loss']:.4f} score={st['score']:.5f} (média TRL {out.metrics[key + '_mean']}; "
                  f"seleção={('oponente' if opp is not None else 'gate') if self.sn56_select else 'média'})",
                  file=sys.stderr, flush=True)
            self._sn56_evals += 1
            if _dump and int(os.environ.get("LOCAL_RANK", "0") or 0) == 0:
                try:
                    os.makedirs(_dump, exist_ok=True)
                    keep = {k: v for k, v in st.items() if isinstance(v, (int, float, bool, str))}
                    rec = {"step": self.state.global_step, "eval_index": self._sn56_evals, "stats": keep,
                           "per_pair": [float(v) for v in vec]}
                    if opp is not None:
                        rec["opp"] = {k: v for k, v in opp.items() if isinstance(v, (int, float, bool, str))}
                    # e<k>: evaluation counter -- the soup's candidate evals reuse the final global_step
                    with open(os.path.join(_dump, f"dev-step{self.state.global_step:06d}-e{self._sn56_evals:03d}.json"),
                              "w") as f:
                        json.dump(rec, f)
                except Exception as e:
                    print(f"[sn56][dposel] dump falhou ({e})", file=sys.stderr, flush=True)
            return out

    GateDPOTrainer.__name__ = f"Gate{base_cls.__name__}"
    return GateDPOTrainer
