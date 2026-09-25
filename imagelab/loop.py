"""Budget-aware training loop with metric-replica checkpoint selection, single-member or cross-fit.

Runs inside ComfyUI's sampling lifecycle (see HijackSampler): the guider has loaded the model and processed
the conditioning, and calling it returns x0 for a noisy latent at a sigma. Every step draws one
(image, sigma-band) case from a cycle over all pairs, one fresh Gaussian noise, and alternates captioned and
empty prompts so the two halves of the score get equal gradient budget. Candidates (raw weights, EMA,
and an exact average of the last evaluated states) are scored on a holdout replica; the best is saved
atomically, so an artefact exists from the first minute on.

Cross-fit (members > 1): the wall is split into K slices; member k trains from a fresh initialisation
with holdout fold k (rotating windows over the hash order), keeps its own best state, and the deltas of
the members that beat their base are averaged exactly. Every image is trained on by K-1 members and
scores the selection of one, which is what a 16-image task cannot afford with a single 3-image holdout.

Adaptive soup (soup_rule="plateau", off by default): a fixed K-way split under-trains families whose holdout
optimum sits late in the wall (krea2 ~54 %, z-image up to 100 % in the champion's own run logs). Member 0 is
therefore trained on fold 0 with the single run's schedule and probes the family: after each evaluation round
it asks whether its holdout curve has plateaued early (best step <= plateau_frac of the steps so far, no gain
beyond plateau_tol for plateau_patience rounds). If so, and at least one more member of member 0's optimal
length fits in the time left, member 0 stops at its best and the remaining members (up to K-1, so K=2 is the
middle option) train and are averaged as in the cross-fit. If the curve keeps improving until no further
member would fit, the soup is abandoned and member 0 simply continues to the deadline (optionally with its
holdout fold added to the training set) and ships by the best/tail rule. The decision, its inputs and the
reason are written to summary.json under "soup_rule".
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch

from . import objective as obj
from .adapter import AdapterBank, merge_snapshots, save_adapter
from .boot import processed_conds
from .data import Item
from .replica import HoldoutReplica


@dataclass
class LoopConfig:
    deadline_ts: float
    lr: float = 2e-4
    lr_final_frac: float = 0.1
    warmup_steps: int = 20
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    up_lr_mult: float = 1.0
    ema: float = 0.99
    eval_every: int = 250
    delta_average: int = 2
    bands: int = obj.BANDS
    band_min: int = 0
    band_max: int = obj.BANDS
    band_weights: list[float] | None = None      # per band multiplier, mean 1; None = evaluator-identical
    band_repeat: dict[int, int] | None = None    # band -> copies in the training case cycle (sampling only, loss unweighted); None = each band once
    prompt_mode: str = "balanced"                # balanced | random
    empty_prompt_frac: float = 0.5
    select: str = "mean"                         # mean | p75 | worst
    ship: str = "best"                           # best (replica-selected) | tail (average of the last tail_frac of evaluated states, if not worse than base)
    tail_frac: float = 0.3
    members: int = 1                             # >1 = cross-fit soup over rotating holdout folds
    member_steps: int = 0                        # 0 = time-sliced; else stop each member at this many steps
    holdout_noises: int = 1
    holdout_noise_batch: int = 2
    soup_rule: str = "none"                      # none | plateau (adaptive soup, see module docstring)
    plateau_frac: float = 0.7                    # plateau: best step <= this fraction of member 0's steps so far
    plateau_patience: int = 2                    # ... and this many latest evaluation rounds without a real gain
    plateau_tol: float = 5e-4                    # relative gain below this (0.05 %) is not a gain
    member_mult: float = 1.0                     # soup members after the probe train best_step * this steps (evaluated at their end)
    continue_data: str = "fold"                  # single branch: fold (keep member 0's holdout) | all (add it, ship tail)
    save_reserve_s: float = 90.0
    seed: int = 0
    log: bool = True


class HijackSampler:
    """A ComfyUI sampler whose only job is to hand the loaded guider to the trainer."""

    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        self.trainer.run(model_wrap, extra_args)
        return latent_image


def rotating_folds(items: list[Item], members: int, fold_size: int) -> list[list[int]]:
    """Fold k = fold_size consecutive positions in the hash order starting at k * stride, wrapping."""
    order = sorted(range(len(items)), key=lambda index: items[index].sha256)
    n = len(order)
    fold_size = max(1, min(fold_size, n - 1))
    stride = max(1, math.ceil(n / members))
    return [[order[(k * stride + j) % n] for j in range(fold_size)] for k in range(members)]


class Trainer:
    def __init__(self, cfg: LoopConfig, model, bank: AdapterBank, items: list[Item], conds: dict,
                 replica: HoldoutReplica | None, out_dir: str, family: str):
        self.cfg, self.model, self.bank, self.items, self.conds = cfg, model, bank, items, conds
        self.replica, self.out_dir, self.family = replica, out_dir, family
        self.device = model.load_device
        self.rng = random.Random(cfg.seed)
        self.noise_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7919)
        self.summary = {"family": family, "steps": 0, "evals": [], "members": [], "saved": None}
        self.step_time: float | None = None
        self.eval_time: float | None = None
        self.base: dict | None = None
        self.best: tuple[float, str, dict] | None = None
        self._cond_cache: dict = {}
        self._guider = None
        self._extra = None
        self.train_items: list[Item] = [item for item in items if not item.holdout]
        self.export_transform = None    # optional: plain exported tensors -> (tensors, report), e.g. the fp8-aware export

    # ------------------------------------------------------------------ logging / persistence
    def log(self, *parts) -> None:
        if self.cfg.log:
            print(time.strftime("%H:%M:%S"), *parts, flush=True)

    def write_summary(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "summary.json"), "w") as handle:
            json.dump(self.summary, handle, indent=1)

    def save(self, snap: dict, score: float, tag: str) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        if self.export_transform is None:
            save_adapter(os.path.join(self.out_dir, "last.safetensors"), self.bank.export(snap),
                         {"imagelab": tag, "score": f"{score:.6f}"})
        else:
            plain = self.bank.export(snap)
            save_adapter(os.path.join(self.out_dir, "last.plain.safetensors"), plain, {"imagelab": tag, "score": f"{score:.6f}"})
            try:
                tensors, report = self.export_transform(plain)
            except Exception as error:  # the plain export is always a valid artefact; never lose a save to the transform
                self.log(f"export transform failed ({type(error).__name__}: {error}); saving the plain export")
                tensors, report = plain, {"error": f"{type(error).__name__}: {error}", "applied": 0}
            save_adapter(os.path.join(self.out_dir, "last.safetensors"), tensors,
                         {"imagelab": tag, "score": f"{score:.6f}", "export": "fp8aware" if report.get("applied") else "plain",
                          "export_layers": str(report.get("applied"))})
            with open(os.path.join(self.out_dir, "export-report.json"), "w") as handle:
                json.dump(report, handle, indent=1)
            self.summary["export"] = {key: report.get(key) for key in ("modes", "applied", "extra_rank_max",
                                                                      "move_abs_max_median", "move_abs_max_max", "error")}
            self.summary["export"]["skipped"] = len(report.get("skipped", {}))
        self.summary["saved"] = {"tag": tag, "score": score, "base": self.base["score"] if self.base else None}
        self.write_summary()
        self.log(f"saved {tag} score={score:.6f}")

    # ------------------------------------------------------------------ model calls
    def conditioning(self, item: Item, prompt: str):
        key = (item.sha256, prompt)
        if key not in self._cond_cache:
            self._cond_cache[key] = processed_conds(self._guider, self.conds[prompt], item.scaled.to(self.device), self.device)
        return self._cond_cache[key]

    def forward(self, item: Item, prompt: str, noisy: torch.Tensor, sigma: float) -> torch.Tensor:
        self._guider.conds = self.conditioning(item, prompt)
        timestep = torch.full((noisy.shape[0],), sigma, device=self.device, dtype=torch.float32)
        return self._guider(noisy, timestep, **self._extra)

    def case_loss(self, item: Item, prompt: str, sigma: float, noise: torch.Tensor) -> torch.Tensor:
        x = item.scaled.to(self.device)
        noisy, target = obj.noisy_and_target(x, noise, sigma)
        denoised = self.forward(item, prompt, noisy, sigma)
        return obj.velocity_error(noisy, denoised, target, sigma).mean()

    # ------------------------------------------------------------------ selection
    def metric(self, result: dict) -> float:
        return result[{"mean": "score", "p75": "p75", "worst": "worst"}[self.cfg.select]]

    def evaluate(self, tag: str, snap: dict | None, step: int, replica: HoldoutReplica | None = None, base: dict | None = None) -> dict:
        replica = replica or self.replica
        base = base or self.base
        self.bank.override(snap)
        try:
            result = replica.score(self.forward, self.device)
        finally:
            self.bank.override(None)
        self.eval_time = result["seconds"]
        rel = (result["score"] / base["score"] - 1) * 100 if base else 0.0
        self.log(f"[eval step {step}] {tag}: {result['score']:.6f} ({rel:+.2f}% vs base) text {result['text']:.5f} "
                 f"no_text {result['no_text']:.5f} worst {result['worst']:.5f} {result['seconds']:.0f}s")
        self.summary["evals"].append({"step": step, "tag": tag, "score": result["score"], "text": result["text"],
                                      "no_text": result["no_text"], "worst": result["worst"], "per_band": result["per_band"]})
        return result

    def consider(self, tag: str, snap: dict, result: dict) -> None:
        value = self.metric(result)
        if self.best is None or value < self.best[0]:
            self.best = (value, tag, snap)
            self.save(snap, result["score"], tag)

    # ------------------------------------------------------------------ schedule helpers
    def remaining(self) -> float:
        return self.cfg.deadline_ts - time.time()

    def band_weight(self, band: int) -> float:
        return 1.0 if not self.cfg.band_weights else float(self.cfg.band_weights[band])

    def training_bands(self) -> list[int]:
        """The bands of one case cycle; band_repeat lists a band that many times (the champion's edge_band_repeat)."""
        repeat = self.cfg.band_repeat or {}
        return [band for band in range(self.cfg.band_min, self.cfg.band_max) for _ in range(repeat.get(band, 1))]

    def next_cases(self, train_items: list[Item]) -> list[tuple[int, int]]:
        cases = [(index, band) for index in range(len(train_items)) for band in self.training_bands()]
        if not cases:
            raise RuntimeError("empty training case set")
        self.rng.shuffle(cases)
        return cases

    def learning_rate(self, step: int, progress: float) -> float:
        warm = min(1.0, (step + 1) / max(1, self.cfg.warmup_steps))
        cosine = self.cfg.lr_final_frac + (1 - self.cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        return self.cfg.lr * warm * cosine

    # ------------------------------------------------------------------ one member
    def train_member(self, tag: str, train_items: list[Item], replica: HoldoutReplica, base: dict,
                     budget_end: float, fixed_steps: int = 0, save_progress: bool = False, on_round=None,
                     eval_every: int | None = None):
        """Train from the bank's current (fresh) state until budget_end; return the member's best
        (metric, tag, snapshot, result) judged on `replica`, and the evaluated-state history.
        on_round(step, round_best_metric, best) is called after every periodic evaluation round; True stops."""
        cfg = self.cfg
        groups = [{"params": self.bank.up_parameters, "lr_mult": cfg.up_lr_mult},
                  {"params": self.bank.down_parameters, "lr_mult": 1.0}]
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8)
        ema = [p.detach().clone() for p in self.bank.parameters] if cfg.ema > 0 else None
        best = (self.metric(base), f"{tag}base", None, base)
        history: list[tuple[int, dict]] = []
        cases: list[tuple[int, int]] = []
        step, last_eval = 0, 0
        t0 = time.time()
        eval_count = 3.3 if cfg.delta_average > 1 else 2.2
        round_values: list[float] = []

        def judge(candidate_tag: str, snap: dict | None, at: int):
            nonlocal best
            result = self.evaluate(f"{tag}{candidate_tag}@{at}", snap, at, replica, base)
            if on_round is not None:
                round_values.append(self.metric(result))
            state = snap if snap is not None else self.bank.snapshot()
            if self.metric(result) < best[0]:
                best = (self.metric(result), f"{tag}{candidate_tag}@{at}", state, result)
                if save_progress:
                    self.consider(best[1], state, result)
            return state

        while True:
            now = time.time()
            est_step = self.step_time or 3.0
            est_eval = self.eval_time or 60.0
            if fixed_steps and step >= fixed_steps:
                break
            if now + est_step + est_eval * eval_count + 5 > budget_end:
                break
            progress = min(1.0, step / fixed_steps) if fixed_steps else min(1.0, (now - t0) / max(1.0, budget_end - t0))
            lr = self.learning_rate(step, progress)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["lr_mult"]
            if not cases:
                cases = self.next_cases(train_items)
            index, band = cases.pop()
            item = train_items[index]
            sigma = obj.sigma_of(band, cfg.bands)
            noise = torch.randn(item.scaled.shape, generator=self.noise_gen, dtype=torch.float32).to(self.device)
            if cfg.prompt_mode == "balanced":
                prompt = item.caption if step % 2 == 0 else ""
            else:
                prompt = "" if self.rng.random() < cfg.empty_prompt_frac else item.caption

            started = time.time()
            with torch.enable_grad():
                loss = self.case_loss(item, prompt, sigma, noise)
                (loss * self.band_weight(band)).backward()
            torch.nn.utils.clip_grad_norm_(self.bank.parameters, cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                with torch.no_grad():
                    for shadow, param in zip(ema, self.bank.parameters):
                        shadow.mul_(cfg.ema).add_(param.detach(), alpha=1 - cfg.ema)
            elapsed = time.time() - started
            self.step_time = elapsed if self.step_time is None else 0.9 * self.step_time + 0.1 * elapsed
            step += 1
            self.summary["steps"] += 1
            if step % 25 == 0:
                self.log(f"{tag}step {step} loss {loss.item():.4f} band {band} lr {lr:.2e} step_t {self.step_time:.2f}s left {budget_end - time.time():.0f}s")
            if step - last_eval >= (eval_every or cfg.eval_every):
                last_eval = step
                judge("raw", None, step)
                if ema is not None:
                    shadow = {name: (up.detach().clone(), down.detach().clone())
                              for name, (up, down) in self.bank.snapshot_from_flat(ema).items()}
                    judge("ema", shadow, step)
                history.append((step, self.bank.snapshot()))
                if on_round is not None:
                    round_best = min(round_values)
                    round_values.clear()
                    if on_round(step, round_best, best):
                        break

        if step > last_eval and time.time() + (self.eval_time or 60) * 2.1 < budget_end:
            judge("raw", None, step)
            if ema is not None:
                shadow = {name: (up.detach().clone(), down.detach().clone())
                          for name, (up, down) in self.bank.snapshot_from_flat(ema).items()}
                judge("ema", shadow, step)
            history.append((step, self.bank.snapshot()))
        if cfg.delta_average > 1 and len(history) >= cfg.delta_average and time.time() + (self.eval_time or 60) * 1.1 < budget_end:
            merged = merge_snapshots([snap for _, snap in history[-cfg.delta_average:]])
            judge(f"avg{cfg.delta_average}", merged, step)
        self.log(f"{tag}done: steps={step} best={best[1]} {best[3]['score']:.6f} ({(best[3]['score'] / base['score'] - 1) * 100:+.2f}% vs base)")
        return best, history, step

    # ------------------------------------------------------------------ main
    def run(self, guider, extra) -> None:
        cfg = self.cfg
        self._guider, self._extra = guider, extra
        for item in self.items:
            item.scaled = item.scaled.to(self.device)
        t_start = time.time()
        if cfg.soup_rule != "none":
            self.run_adaptive()
        elif cfg.members > 1:
            self.run_crossfit()
        else:
            self.run_single()
        self.summary["final"] = {"best": self.best[1] if self.best else None, "score": self.best[0] if self.best else None,
                                 "base": self.base["score"] if self.base else None, "steps": self.summary["steps"],
                                 "wall_s": time.time() - t_start, "step_time": self.step_time, "eval_time": self.eval_time}
        self.write_summary()
        self.log("final", json.dumps(self.summary["final"]))

    def run_single(self) -> None:
        cfg = self.cfg
        self.base = self.evaluate("base", self.bank.identity_snapshot(), 0)
        self.summary["base"] = {k: self.base[k] for k in ("score", "text", "no_text", "worst", "per_band")}
        self.log(f"base {self.base['score']:.6f} holdout={len(self.replica.items)} train={len(self.train_items)} "
                 f"lora_params={self.bank.count() / 1e6:.1f}M eval={self.base['seconds']:.0f}s")
        self.consider("identity", self.bank.identity_snapshot(), self.base)
        budget_end = cfg.deadline_ts - cfg.save_reserve_s
        best, history, step = self.train_member("", self.train_items, self.replica, self.base, budget_end, cfg.member_steps, save_progress=True)
        if best[2] is not None:
            self.consider(best[1], best[2], best[3])
        self._ship_tail(history, step)

    def _ship_tail(self, history: list, step: int, force: bool = False, replica: HoldoutReplica | None = None) -> None:
        cfg = self.cfg
        if (cfg.ship == "tail" or force) and history and self.remaining() > cfg.save_reserve_s + (self.eval_time or 60) * 1.1:
            cutoff = (1.0 - cfg.tail_frac) * step
            tail = [snap for at, snap in history if at >= cutoff] or [history[-1][1]]
            merged = merge_snapshots(tail)
            tag = f"tail{len(tail)}@{step}"
            result = self.evaluate(tag, merged, step, replica)
            if result["score"] <= self.base["score"]:
                self.best = (self.metric(result), tag, merged)
                self.save(merged, result["score"], tag)
            else:
                self.log(f"tail rule rejected: {result['score']:.6f} worse than base {self.base['score']:.6f}; keeping {self.best[1]}")

    def run_crossfit(self) -> None:
        cfg = self.cfg
        fold_size = len(self.replica.items) if self.replica is not None and self.replica.items else 3
        folds = rotating_folds(self.items, cfg.members, fold_size)
        self.log(f"cross-fit: {cfg.members} members, fold size {fold_size}, folds {folds}")
        members: list[tuple[dict, dict, dict, str]] = []     # (snapshot, result, base, tag)
        first_replica, first_base = None, None
        for k, fold in enumerate(folds):
            tag, train_items, replica, base = self._fold_member(k, fold)
            if k == 0:
                first_replica, first_base = replica, base
            remaining = cfg.deadline_ts - cfg.save_reserve_s - time.time() - (self.eval_time or 60) * 2.5
            slice_end = time.time() + remaining / (cfg.members - k)
            best, _, steps = self.train_member(tag, train_items, replica, base, slice_end, cfg.member_steps)
            self._record_member(k, fold, tag, best, steps, base, members)
            if self.remaining() < cfg.save_reserve_s + (self.eval_time or 60) * 3:
                break
        self._finish_soup(members, first_replica, first_base)

    def _fold_member(self, k: int, fold: list[int]):
        """Hold out `fold`, re-initialise the bank for member k and score its base on the fold."""
        cfg = self.cfg
        tag = f"m{k}:"
        fold_set = set(fold)
        for index, item in enumerate(self.items):
            item.holdout = index in fold_set
        holdout = [self.items[index] for index in fold]
        train_items = [item for index, item in enumerate(self.items) if index not in fold_set]
        replica = HoldoutReplica(holdout, noises=cfg.holdout_noises, noise_batch=cfg.holdout_noise_batch)
        self.bank.reinit(cfg.seed + 1000 * k)
        base = self.evaluate(f"{tag}base", self.bank.identity_snapshot(), 0, replica, {"score": 1.0})
        if k == 0:
            self.base = base
            self.summary["base"] = {key: base[key] for key in ("score", "text", "no_text", "worst", "per_band")}
            self.consider("identity", self.bank.identity_snapshot(), base)
        return tag, train_items, replica, base

    def _record_member(self, k: int, fold: list[int], tag: str, best, steps: int, base: dict, members: list) -> None:
        gain = (best[3]["score"] / base["score"] - 1) * 100
        self.summary["members"].append({"member": k, "fold": fold, "steps": steps, "best": best[1],
                                        "score": best[3]["score"], "base": base["score"], "gain_pct": gain})
        if best[2] is not None:
            members.append((best[2], best[3], base, best[1]))
            if k == 0:
                self.consider(best[1], best[2], best[3])
        else:
            self.log(f"{tag}no improvement over its base; excluded from the soup")

    def _finish_soup(self, members: list, first_replica: HoldoutReplica, first_base: dict) -> None:
        cfg = self.cfg
        if not members:
            self.log("cross-fit: no member beat its base; shipping identity")
            return
        merged = merge_snapshots([snap for snap, _, _, _ in members])
        tag = f"soup{len(members)}"
        if self.remaining() > cfg.save_reserve_s + (self.eval_time or 60) * 1.1:
            check = self.evaluate(tag, merged, self.summary["steps"], first_replica, first_base)
            if check["score"] <= first_base["score"]:
                self.best = (self.metric(check), tag, merged)
                self.save(merged, check["score"], tag)
                return
            self.log(f"soup rejected on fold 0: {check['score']:.6f} vs base {first_base['score']:.6f}")
        # fall back to the single member with the largest relative gain on its own fold
        snap, result, base, member_tag = min(members, key=lambda entry: entry[1]["score"] / entry[2]["score"])
        self.best = (self.metric(result), member_tag, snap)
        self.save(snap, result["score"], member_tag)

    # ------------------------------------------------------------------ adaptive soup (soup_rule="plateau")
    def run_adaptive(self) -> None:
        """Member 0 probes on the single run's schedule; the plateau rule decides soup (K' <= K members) or single."""
        cfg = self.cfg
        members_max = max(2, cfg.members)
        fold_size = len(self.replica.items) if self.replica is not None and self.replica.items else 3
        folds = rotating_folds(self.items, members_max, fold_size)
        tag, train_items, replica, base = self._fold_member(0, folds[0])
        probe = SoupProbe(cfg.plateau_frac, cfg.plateau_patience, cfg.plateau_tol, members_max, self.metric(base))
        rule = {"rule": cfg.soup_rule, "members_max": members_max, "folds": folds,
                "params": {"plateau_frac": cfg.plateau_frac, "plateau_patience": cfg.plateau_patience,
                           "plateau_tol": cfg.plateau_tol, "member_mult": cfg.member_mult,
                           "continue_data": cfg.continue_data, "eval_every": cfg.eval_every},
                "curve": probe.curve, "decision": None}
        self.summary["soup_rule"] = rule
        self.log(f"soup rule {cfg.soup_rule}: member 0 probes fold {folds[0]} on the single-run schedule "
                 f"(max {members_max} members)")

        def on_round(step: int, value: float, best) -> bool:
            probe.observe(step, value)
            if rule["decision"] is not None:
                return False
            member_steps = max(cfg.eval_every, int(round(probe.best_step * cfg.member_mult)))
            evals = 1 + (2 if cfg.ema > 0 else 1)           # base + one round at the member's end (its length is known)
            eval_s = self.eval_time or 60.0
            need = member_steps * (self.step_time or 3.0) + evals * eval_s + 5.0
            remaining = cfg.deadline_ts - cfg.save_reserve_s - time.time() - eval_s * 2.5
            fit = max(0, min(members_max - 1, int(remaining // need)))
            inputs = {"at_step": step, "best_step": probe.best_step, "best_value": probe.best_value,
                      "base_value": probe.base_value, "stale_rounds": probe.stale,
                      "best_frac": probe.best_step / step, "member_steps": member_steps, "member_need_s": need,
                      "remaining_s": remaining, "fit": fit, "step_time": self.step_time, "eval_time": eval_s}
            if probe.plateaued(step) and fit >= 1:
                rule["decision"] = {"choice": "soup", "members": fit + 1, **inputs,
                                    "why": f"holdout best at step {probe.best_step} = {probe.best_step / step:.2f} of {step} "
                                           f"(<= {cfg.plateau_frac}) with {probe.stale} round(s) without a "
                                           f"{cfg.plateau_tol:.2%} gain; {fit} more member(s) of {member_steps} steps fit"}
                self.log(f"soup rule: SOUP x{fit + 1} ({rule['decision']['why']})")
                self.write_summary()
                return True
            if fit < 1:
                rule["decision"] = {"choice": "single", "members": 1, **inputs,
                                    "why": f"holdout still improving (best at step {probe.best_step} of {step}) when no "
                                           f"further member of {member_steps} steps fits in {remaining:.0f}s; "
                                           f"member 0 continues to the deadline"}
                if cfg.continue_data == "all":
                    added = [self.items[index] for index in folds[0]]
                    train_items.extend(added)
                    for item in added:
                        item.holdout = False
                    rule["decision"]["why"] += f"; its {len(added)} holdout images join training (tail ship)"
                self.log(f"soup rule: SINGLE ({rule['decision']['why']})")
                self.write_summary()
            return False

        budget_end = cfg.deadline_ts - cfg.save_reserve_s
        if cfg.continue_data == "all":          # keep one evaluation for the tail's guard
            budget_end -= (self.eval_time or 60.0) * 1.2
        best, history, step = self.train_member(tag, train_items, replica, base, budget_end, cfg.member_steps,
                                                save_progress=True, on_round=on_round)
        members: list[tuple[dict, dict, dict, str]] = []
        self._record_member(0, folds[0], tag, best, step, base, members)
        decision = rule["decision"]
        if decision is None:
            rule["decision"] = decision = {"choice": "single", "members": 1, "at_step": step,
                                           "why": "the deadline came before any decision round"}
        if decision["choice"] == "single":
            self._ship_tail(history, step, force=cfg.continue_data == "all", replica=replica)
            return
        for k in range(1, decision["members"]):
            if self.remaining() < cfg.save_reserve_s + (self.eval_time or 60) * 3:
                break
            member_tag, member_items, member_replica, member_base = self._fold_member(k, folds[k])
            remaining = cfg.deadline_ts - cfg.save_reserve_s - time.time() - (self.eval_time or 60) * 2.5
            slice_end = time.time() + remaining / (decision["members"] - k)
            member_best, _, member_steps = self.train_member(member_tag, member_items, member_replica, member_base,
                                                             slice_end, decision["member_steps"],
                                                             eval_every=decision["member_steps"])
            self._record_member(k, folds[k], member_tag, member_best, member_steps, member_base, members)
        self._finish_soup(members, replica, base)


@dataclass
class SoupProbe:
    """The plateau rule's bookkeeping over member 0's evaluation rounds (no model calls; unit-testable)."""
    frac: float
    patience: int
    tol: float
    members_max: int
    base_value: float
    best_step: int = 0
    best_value: float = math.inf
    stale: int = 0
    curve: list = None

    def __post_init__(self):
        self.curve = [] if self.curve is None else self.curve

    def observe(self, step: int, value: float) -> None:
        """A round counts as a gain only if it beats the best so far (and the base) by more than tol, relative."""
        self.curve.append([step, value])
        if value < min(self.best_value, self.base_value) * (1.0 - self.tol):
            self.best_step, self.best_value, self.stale = step, value, 0
        else:
            self.stale += 1

    def plateaued(self, step: int) -> bool:
        return 0 < self.best_step <= self.frac * step and self.stale >= self.patience
