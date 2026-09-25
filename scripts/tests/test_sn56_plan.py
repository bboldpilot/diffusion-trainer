"""Planning math for SN56_EVAL_SKIP_FIX (F1) and SN56_TIME_PLAN (F2).

Run: python scripts/tests/test_sn56_plan.py   (no torch needed)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sn56_plan as P  # noqa: E402


def cosine_factor(step, total, warmup, min_lr_rate=0.25):
    """transformers cosine_with_min_lr (num_cycles=0.5) at `step`."""
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr_rate + (1 - min_lr_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _clear_env():
    for k in list(P.FLAGS) + [P.MASTER]:
        os.environ.pop(k, None)


def test_flags_defaults_old_values_and_master():
    _clear_env()
    assert P.flag_on("SN56_EVAL_SKIP_FIX") and P.flag_on("SN56_TIME_PLAN")
    # F3 (measured 21-25 Sep): the search is on by default; LR_MULT scales its pick
    assert P.flag_on("SN56_LR_SEARCH") and not P.flag_on("SN56_NEFTUNE")
    assert not P.flag_on("SN56_PER_EXAMPLE_LOSS")
    assert P.v5_pack_setting() == "auto" and P.max_epochs() == 4.0 and P.flag_float("SN56_LR_MULT") == 1.0
    # one env var per flag restores the inherited path / the A/B arm
    os.environ["SN56_LR_SEARCH"] = "0"
    os.environ["SN56_V5_PACK"] = "0"
    assert not P.flag_on("SN56_LR_SEARCH") and P.v5_pack_setting() == "0"
    os.environ["SN56_LR_MULT"] = "2"
    assert abs(P.lr_for_batch(1.25e-4, 64, 64, P.flag_float("SN56_LR_MULT")) - 2.5e-4) < 1e-12
    # garbage falls back to the default, not to "on"
    os.environ["SN56_NEFTUNE"] = "maybe"
    assert not P.flag_on("SN56_NEFTUNE")
    os.environ["SN56_MAX_EPOCHS"] = "-3"
    assert P.max_epochs() == 4.0
    # master switch: every flag at its old value, whatever the individual vars say
    _clear_env()
    os.environ["SN56_TEXT_FIX"] = "0"
    os.environ["SN56_TIME_PLAN"] = "1"
    assert not P.flag_on("SN56_EVAL_SKIP_FIX") and not P.flag_on("SN56_TIME_PLAN")
    assert P.flag_on("SN56_LR_SEARCH") and P.flag_on("SN56_NEFTUNE")
    assert P.v5_pack_setting() == "0" and not P.flag_on("SN56_PER_EXAMPLE_LOSS")
    assert "TEXT_FIX=0" in P.flags_summary()
    _clear_env()
    print("[ok] flags: defaults = new behaviour, per-flag and master rollback")


# The measured failure (2h LFM task): the inherited plan was E=1.3 with 1174
# planned steps; the run hit the final-save trigger at step 659 with no eval,
# no checkpoint selection, no soup, loss.txt = no_eval, LR still ~0.55x peak.
SPE = 903          # 1174 / 1.3
STOP_AT = 659      # step reached at the final-save trigger
T_STEP = 8.65      # s/step: (stop - now - startup) / 659
NOW = 0.0
END = 6300.0       # end_time, 1h45m after planning
RESERVE_MIN = 3 + 6  # save_before_remaining_time + dev-pass reserve
STOP = END - 60 * RESERVE_MIN


def test_measured_failure_case_legacy_reproduces():
    achievable = int((END - NOW) * 0.85 / T_STEP)
    e_old = P.legacy_epoch_plan(achievable, SPE, n_train_items=40000)
    total_old = int(SPE * e_old)
    assert e_old == 1.3, e_old                      # the 1.3 floor binds
    assert 1173 <= total_old <= 1175, total_old      # ~1174 planned
    # the callback's threshold uses the E=3 total divided by E=1.3
    thr_old = P.legacy_eval_skip_threshold(SPE * 3, e_old)
    assert thr_old == int(0.75 * SPE * 3 / 1.3) == 1562
    assert thr_old > total_old > STOP_AT, "no periodic eval can ever pass the skip"
    # cosine cut at 56% of the plan: LR still ~0.55x peak at the cutoff
    f_old = cosine_factor(STOP_AT, total_old, warmup=35)
    assert 0.5 < f_old < 0.6, f_old
    print(f"[ok] legacy: E=1.3, {total_old} planned, skip<{thr_old}, stop@{STOP_AT}, lr x{f_old:.2f} at stop")


def test_measured_failure_case_f1_threshold():
    # F1 alone on the legacy plan: 0.75 epoch of the real steps/epoch, capped at
    # half the plan -> evals are allowed from step 586 (old rule: 1562).
    thr = P.eval_skip_threshold(SPE, 1173, 1.3)
    assert thr == min(int(0.75 * SPE), int(0.5 * 1173)) == 586
    assert thr < STOP_AT
    print(f"[ok] F1 on the legacy plan: skip<{thr} (was 1562)")


def test_measured_failure_case_f1_f2_plan():
    eval_cost = 20.0
    p = P.plan_time_budget(now_ts=NOW, stop_ts=STOP, t_per_step=T_STEP, steps_per_epoch=SPE,
                           max_epochs=4, warmup_cap=200, eval_cost_s=eval_cost,
                           periodic_save_steps=500, startup_s=60.0)
    planned = p["planned_steps"]
    # fits before the real stop (not the end_time), net of the eval overhead
    assert 0.9 * STOP_AT <= planned <= STOP_AT, p
    assert p["train_s"] + p["eval_overhead_s"] <= p["available_s"] + T_STEP, p
    assert p["epochs"] < 1.0 and not p["capped"]           # no 1.3 floor
    assert p["eval_skip_threshold"] == 0                     # sub-epoch: nothing skipped
    assert p["first_eval_step"] is not None and p["first_eval_step"] <= planned // 4, p
    assert p["n_evals"] >= 6, p
    # the cosine reaches its floor at the cutoff instead of ~0.55x peak
    assert abs(cosine_factor(planned, planned, p["warmup"]) - 0.25) < 1e-9
    # even if the realised stop lands 5% later than planned, LR is at the floor
    assert cosine_factor(min(planned, int(1.05 * planned)), planned, p["warmup"]) < 0.26
    print(f"[ok] F1+F2: {planned} steps (E={p['epochs']:.2f}), eval every {p['eval_steps']}, "
          f"first eval @{p['first_eval_step']}, ~{p['n_evals']} evals")


def test_small_data_runs_multiple_epochs_until_budget_then_cap():
    # 1.5h task, tiny dataset: data is exhausted after 40 steps; the budget
    # (~3300 steps) would allow 80+ epochs -> planned to the SN56_MAX_EPOCHS cap.
    p = P.plan_time_budget(now_ts=0, stop_ts=5400 - 300, t_per_step=1.5, steps_per_epoch=40,
                           max_epochs=4, warmup_cap=90, eval_cost_s=10.0, periodic_save_steps=500)
    assert p["capped"] and p["planned_steps"] == 160 and p["epochs"] == 4.0, p
    assert p["eval_skip_threshold"] == 30 and p["first_eval_step"] >= 30, p
    p8 = P.plan_time_budget(now_ts=0, stop_ts=5400 - 300, t_per_step=1.5, steps_per_epoch=40,
                            max_epochs=8, warmup_cap=90, eval_cost_s=10.0, periodic_save_steps=500)
    assert p8["capped"] and p8["planned_steps"] == 320, p8
    # a mid-size dataset: the budget covers ~2.6 epochs -> E follows the budget,
    # no floor, no cap, inherited 1/8-epoch eval cadence
    p2 = P.plan_time_budget(now_ts=0, stop_ts=5400, t_per_step=2.0, steps_per_epoch=1000,
                            max_epochs=4, warmup_cap=90, eval_cost_s=15.0, periodic_save_steps=500)
    assert not p2["capped"] and 2.3 < p2["epochs"] < 2.7, p2
    assert p2["eval_steps"] == 125 and p2["eval_skip_threshold"] == 750, p2
    assert p2["train_s"] + p2["eval_overhead_s"] <= p2["available_s"] + 2.0, p2
    print(f"[ok] small data: capped at 4 ep (160) / 8 ep (320); mid data E={p2['epochs']:.2f}")


def test_eval_skip_threshold_rules():
    assert P.eval_skip_threshold(1000, 800, 0.8) == 0         # sub-epoch: never skip
    assert P.eval_skip_threshold(1000, 1000, 1.0) == 0
    assert P.eval_skip_threshold(100, 300, 3.0) == 75          # inherited 0.75 epoch
    assert P.eval_skip_threshold(1000, 1200, 1.2) == 600       # capped at half the plan
    # the legacy rule agrees only when E never changed from its initial value
    assert P.legacy_eval_skip_threshold(300, 3.0) == P.eval_skip_threshold(100, 300, 3.0)
    assert P.legacy_eval_skip_threshold(300, 1.5) == 150 > P.eval_skip_threshold(100, 150, 1.5)
    print("[ok] F1 threshold rules")


def test_eval_points_and_interval():
    assert P.eval_interval(903, 640) == 80                     # >= ~8 evals on a sub-epoch plan
    assert P.eval_interval(903, 5000) == 112                   # inherited 1/8 epoch
    assert P.eval_interval(40, 160) == 20                      # floor
    pts = P.eval_points(640, 80, 0, 903, 500)
    assert pts == [80, 160, 240, 320, 400, 480, 500, 560], pts
    assert P.eval_points(160, 20, 30, 40, 500) == [40, 60, 80, 100, 120, 140]
    print("[ok] eval grid incl. epoch/periodic triggers")


def test_eval_overhead_dominated_plan_does_not_collapse():
    # Regression (CPU smoke run B): 0.2 s steps, 6 s per eval+save. A naive
    # fixed point cycles 148 <-> 0 steps here; the plan must be the largest
    # step count whose training + implied evals fit the window.
    kw = dict(now_ts=0, stop_ts=79, t_per_step=0.214, steps_per_epoch=37, max_epochs=4,
              warmup_cap=10, eval_cost_s=6.0, periodic_save_steps=500, startup_s=30.0)
    p = P.plan_time_budget(**kw)
    assert 60 <= p["planned_steps"] < 148, p
    assert p["train_s"] + p["eval_overhead_s"] <= p["available_s"], p
    more = P.plan_time_budget(**dict(kw, stop_ts=79 + 30))
    assert more["planned_steps"] >= p["planned_steps"], (more, p)
    print(f"[ok] eval-dominated window: {p['planned_steps']} steps + {p['n_evals']} evals fit "
          f"{p['available_s']:.0f}s")


def test_plan_degenerate_inputs():
    p = P.plan_time_budget(now_ts=100, stop_ts=50, t_per_step=5.0, steps_per_epoch=10)
    assert p["planned_steps"] == 10 and p["warmup"] == 10      # no time: tiny floor, never 0
    p = P.plan_time_budget(now_ts=0, stop_ts=3600, t_per_step=0.0, steps_per_epoch=0, max_epochs=0)
    assert p["planned_steps"] >= 1 and p["capped"]
    print("[ok] degenerate inputs")


def test_devpass_reserve_matches_train_instruct_formula():
    secs, extra = P.devpass_reserve(n_dev_rows=500, world_size=1, t_per_step=2.0,
                                    grad_accum=1, per_device_bs=16)
    assert abs(secs - (2.0 * 500 * (2.0 / 16) + 120)) < 1e-9 and abs(extra - secs / 60) < 1e-9
    assert 2.0 <= P.devpass_reserve(10, 1, 0.1, 1, 64)[1] < 2.01   # +120 s save => >= 2 min
    assert P.devpass_reserve(10 ** 6, 1, 10.0, 1, 1)[1] == 15.0
    print("[ok] dev-pass reserve")


def test_lr_for_batch_and_token_correction():
    assert P.lr_for_batch(1e-4, 64, 64) == 1e-4
    assert abs(P.lr_for_batch(1e-4, 64, 16) - 5e-5) < 1e-15   # sqrt(1/4)
    assert abs(P.lr_for_batch(1e-4, 64, 64, mult=0.5) - 5e-5) < 1e-15
    assert P.lr_for_batch(1e-4, 0, 16) == 1e-4
    assert P.tokens_correction(None, 100) == 1.0
    assert P.tokens_correction(1000, 400) == 0.4
    assert P.tokens_correction(1000, 5000) == 1.0 and P.tokens_correction(1000, 10) == 0.25
    print("[ok] F3 LR re-centring, dyn-padding timing correction")


def test_time_helpers_roundtrip():
    ts = P.parse_utc("2026-09-28 11:00:00")
    assert P.fmt_utc(ts) == "2026-09-28 11:00:00"
    assert P.fmt_utc(ts - 60 * 9) == "2026-09-28 10:51:00"
    print("[ok] UTC helpers")


# ── The tail (25 Sep failures) ──────────────────────────────────────────────
# P-A: 3B (6.17 GB bf16), 1.5h, 8.4 s/step at bs 34 x ga 2, 186 packed dev
# blocks, 8 soup snapshots. text_trainer's end_time is the kill minus 180 s.
# The old trigger sat at T-5.8min (3 min + ~166 s dev-pass), which never
# covered the soup; the run then died on a full disk and its retries
# re-planned 20 and 10 steps into the kill.
KILL_A = 5400.0
END_A = KILL_A - 180.0
NOW_A = 4 * 60.0              # planning happens ~4 min after the start
PARAMS_3B = 3_085_000_000


def _tail_a(write_bps, **kw):
    return P.tail_plan(now_ts=NOW_A, end_ts=END_A, t_per_step=8.4, n_params=PARAMS_3B,
                       n_dev_rows=186, world_size=1, grad_accum=2, per_device_bs=34,
                       write_bps=write_bps, soup_pool=8, **kw)


def test_3b_1p5h_tail_leaves_submission_before_the_kill():
    # the pod's measured disk (6.17 GB in ~7 s) and the brief's 60-90 s save
    for write_bps, label in ((0.85e9, "0.85GB/s"), (6.17e9 / 75, "75s/save")):
        tail = _tail_a(write_bps)
        assert tail["n_soup"] == 8 and tail["devpass"] and not tail["shrunk"], tail
        parts = (tail["step_s"] + tail["eval_s"] + tail["save_s"] + tail["copy_s"] + tail["persist_s"]
                 + tail["soup_s"] + tail["devpass_s"] + tail["safety_s"])
        assert abs(tail["reserve_s"] - parts) < 1e-9
        assert tail["reserve_s"] > 5.8 * 60                       # more than the old T-5.8min
        assert tail["copy_s"] == 2 * tail["save_s"]               # read + write
        assert 10 <= tail["eval_s"] <= 30                         # measured 11-27 s per eval
        # the reserve scales with the disk: a 75 s save costs ~5 extra saves
        # the planned tail, walked at the planned costs: last write >= 3 min
        # + safety before the kill
        sim = P.simulate_tail(tail, END_A)
        assert sim["margin_s"] >= tail["safety_s"] - 1e-6, sim
        assert KILL_A - sim["last_write_ts"] >= 180 + tail["safety_s"] - 1e-6
        # a disk 1.5x slower and steps 1.3x slower than measured: the in-tail
        # deadlines cut the soup / dev-pass, the last write still precedes end_time
        slow = P.simulate_tail(tail, END_A, step_scale=1.3, save_scale=1.5)
        assert slow["last_write_ts"] < END_A, slow
        # the step plan ends at the trigger, net of the periodic eval+save cost
        eval_cost = tail["eval_s"] + tail["save_s"] + 0.5 * tail["copy_s"]
        p = P.plan_time_budget(now_ts=NOW_A, stop_ts=tail["trigger_ts"], t_per_step=8.4, steps_per_epoch=115,
                               max_epochs=4, warmup_cap=90, eval_cost_s=eval_cost, periodic_save_steps=500)
        assert p["train_s"] + p["eval_overhead_s"] <= p["available_s"] + 8.4
        # the pod's disk: most of the 1.5h is trained; a 75 s/save disk makes
        # every 20-step eval point cost ~170 s and the plan pays for it in steps
        assert p["planned_steps"] >= (400 if write_bps > 0.5e9 else 200), p
        print(f"[ok] 3B/1.5h ({label}): reserve {tail['reserve_s']:.0f}s "
              f"(save {tail['save_s']:.0f}s, soup {tail['soup_s']:.0f}s, dev-pass {tail['devpass_s']:.0f}s), "
              f"{p['planned_steps']} steps, last write {KILL_A - sim['last_write_ts']:.0f}s before the kill")
    # the old rule for the same run was the logged T-5.8min (3 min + the
    # dev-pass estimate); the priced tail is longer at the pod's disk rate
    old_reserve = 60 * (3 + P.devpass_reserve(186, 1, 8.4, 2, 34)[1])
    assert abs(old_reserve - 5.8 * 60) < 6 and _tail_a(0.85e9)["reserve_s"] > old_reserve


def test_2h_task_soup_gets_its_budget():
    # P-D (25 Sep, 2h): 1.336 s/step at bs 3 x ga 4, 348 dev blocks, 5.39 GB
    # weights, 8 snapshots. Old trigger T-6.3min: the soup ran out of time
    # after 5 candidates, the dev-pass was skipped (139 s left < 197 s).
    end, now = 7200.0 - 180.0, 6 * 60.0
    tail = P.tail_plan(now_ts=now, end_ts=end, t_per_step=1.336, n_params=2_695_000_000, n_dev_rows=348,
                       grad_accum=4, per_device_bs=3, write_bps=0.73e9, soup_pool=8)
    assert tail["n_soup"] == 8 and tail["devpass"] and not tail["shrunk"], tail
    assert tail["soup_s"] >= 8 * tail["eval_s"]                   # seed + 7 candidates, each an eval
    assert tail["reserve_s"] > 6.3 * 60
    # the old rule: T-6.3min, soup = half of what was left after the final
    # eval + save -> fewer than the 8 candidates (5 were evaluated on the pod)
    old_soup = 0.5 * (6.3 * 60 - tail["step_s"] - tail["eval_s"] - tail["save_s"] - tail["copy_s"])
    assert tail["eval_s"] * 4 < old_soup < tail["soup_s"], (old_soup, tail["soup_s"])
    # the soup deadline leaves the dev-pass its full budget and the margin
    assert tail["soup_deadline_ts"] + tail["devpass_s"] + tail["safety_s"] <= end + 1e-6
    sim = P.simulate_tail(tail, end)
    assert sim["margin_s"] >= tail["safety_s"] - 1e-6
    # at the planned costs every candidate fits before the soup deadline
    t = tail["trigger_ts"] + tail["step_s"] + tail["eval_s"] + tail["save_s"] + tail["copy_s"] + tail["eval_s"]
    for _ in range(7):
        t += tail["merge_s"] + tail["eval_s"]
    assert t <= tail["soup_deadline_ts"] - tail["persist_s"] + 1e-6
    print(f"[ok] 2h: soup budget {tail['soup_s']:.0f}s for 8 snapshots, dev-pass {tail['devpass_s']:.0f}s, "
          f"reserve {tail['reserve_s'] / 60:.1f}min (was 6.3)")


def test_tail_shrinks_soup_first_then_devpass_never_the_floor():
    # a 3B run with only 20 min left: the tail may take 25% -> soup trimmed
    tail = P.tail_plan(now_ts=0, end_ts=1200, t_per_step=8.4, n_params=PARAMS_3B, n_dev_rows=186,
                       grad_accum=2, per_device_bs=34, write_bps=0.85e9, soup_pool=8)
    assert 1 < tail["n_soup"] < 8 and tail["devpass"] and tail["shrunk"], tail
    assert tail["reserve_s"] <= tail["cap_s"] + 1e-9
    # a CPU-smoke-sized window: no soup, no dev-pass, but the floor stays
    tiny = P.tail_plan(now_ts=60, end_ts=300, t_per_step=0.2, n_params=1_000_000, n_dev_rows=40,
                       grad_accum=1, per_device_bs=2, write_bps=0.5e9, soup_pool=8)
    assert tiny["n_soup"] == 1 and not tiny["devpass"] and tiny["shrunk"], tiny
    floor = tiny["step_s"] + tiny["eval_s"] + tiny["save_s"] + tiny["copy_s"] + tiny["persist_s"] + tiny["safety_s"]
    assert abs(tiny["reserve_s"] - floor) < 1e-9 and tiny["soup_s"] == 0 and tiny["devpass_s"] == 0
    assert tiny["reserve_s"] > tiny["cap_s"]                       # the floor is never cut
    # no dev set: nothing to eval, soup or dev-pass
    nodev = P.tail_plan(now_ts=0, end_ts=3600, t_per_step=1.0, n_params=5e8, n_dev_rows=0)
    assert nodev["n_soup"] == 1 and not nodev["devpass"] and nodev["eval_s"] == 0
    print(f"[ok] shrink: 20min/3B -> soup {tail['n_soup']}, dev-pass kept; tiny window -> floor only")


def test_late_attempt_guard_on_the_measured_retries():
    # P-A's retries: end_time 11:16:54; attempt 2 started 11:05:23 (691 s
    # left), attempt 3 11:11:19 (335 s), 4 11:14:00, 5 11:16:33, 6 11:19:06.
    plan = dict(t_per_step=8.4, reserve_s=_tail_a(0.85e9)["reserve_s"])
    for left in (691.0, 335.0, 174.0, 21.0, -132.0):
        assert P.late_attempt(left, True, **plan), left           # keep the step-230 submission
    assert not P.late_attempt(40 * 60.0, True, **plan)             # 40 min left: a real retry
    assert not P.late_attempt(21.0, False, **plan)                 # nothing to keep: retry as before
    assert P.late_attempt(600.0, True) and not P.late_attempt(1000.0, True)   # no measurements: 15 min
    print("[ok] late-attempt guard: every 25 Sep retry keeps the submission")


def test_submission_and_disk_helpers(tmp_root=None):
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="sn56_plan_test_")
    try:
        sub = os.path.join(root, "sub")
        assert not P.submission_is_valid(sub) and not P.submission_is_valid("")
        os.makedirs(sub)
        open(os.path.join(sub, "config.json"), "w").close()
        assert not P.submission_is_valid(sub)                      # weights missing
        open(os.path.join(sub, "model.safetensors"), "w").close()
        assert P.submission_is_valid(sub)
        bps = P.measure_write_bps(root, size_bytes=8 * 1024 * 1024)
        assert P.WRITE_BPS_MIN <= bps <= P.WRITE_BPS_MAX and not any(n.startswith(".sn56_write_probe") for n in os.listdir(root))
        assert P.measure_write_bps(os.path.join(root, "nope", "x"), size_bytes=1024) >= P.WRITE_BPS_MIN
        assert P.save_time_s(PARAMS_3B, 0.85e9) > P.save_time_s(PARAMS_3B, 3e9) > P.SAVE_FIXED_S
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"[ok] submission check + write probe ({bps / 1e6:.0f} MB/s here)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} SN56 PLAN TESTS PASSED")
