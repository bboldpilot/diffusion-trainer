"""SN56_LOGIT_CAL_MODE (sn56_logit_cal.py). nopass: fold a into the weights it was measured on, no dev-pass.
gate: dev-pass on half of dev, the other half picks soup*a_pre or pass*a_post.

Run: python scripts/tests/test_sn56_logit_cal_mode.py   (torch, transformers 4.51.3, safetensors)
"""
import copy
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import sn56_logit_cal as C  # noqa: E402

GRID = (0.4, 0.6, 0.8, 1.0, 1.25)
ENV = "SN56_LOGIT_CAL_MODE"


def _set_mode(v):
    os.environ.pop(ENV, None)
    if v is not None:
        os.environ[ENV] = v


def _tiny_bloom(scale=40.0):
    from transformers import BloomConfig, BloomForCausalLM
    torch.manual_seed(2)
    m = BloomForCausalLM(BloomConfig(vocab_size=50, hidden_size=32, n_layer=2, n_head=4)).eval()
    with torch.no_grad():
        m.transformer.ln_f.weight.mul_(scale)     # untrained logits are ~0; give them a real temperature
        m.transformer.ln_f.bias.mul_(scale)
    return m


@torch.no_grad()
def _batches(model, a_true, n=8, seed=3):
    """Labels sampled from softmax(a_true * logits): the model is over-confident, its optimum is a_true."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(0, 50, (4, 40), generator=gen)
        z = model(input_ids=x).logits.double()
        y = torch.multinomial(torch.softmax(a_true * z[:, :-1], -1).reshape(-1, 50), 1, generator=gen).view(4, 39)
        lab = torch.full_like(x, -100)
        lab[:, 1:] = y
        lab[:, :2] = -100                          # prompt
        out.append({"input_ids": x, "attention_mask": torch.ones_like(x), "labels": lab})
    return out


class FakeTrainer:
    def __init__(self, model, batches):
        self.model, self.batches = model, batches
        self.model_wrapped = model
        self.accelerator = None
        self.optimizer = "kept until the dev-pass runs"

    def get_eval_dataloader(self):
        return list(self.batches)

    def _prepare_inputs(self, x):
        return x


@torch.no_grad()
def _loss(model, batches):
    s = torch.zeros(1, 2, dtype=torch.float64)
    for b in batches:
        fwd = {k: v for k, v in b.items() if k != "labels"}
        s += C.scaled_example_stats(model(**fwd).logits, b["labels"], (1.0,), None, b["attention_mask"])
    return float(s[0, 0] / s[0, 1])


def test_mode_from_env():
    for v, want in ((None, "dev"), ("", "dev"), ("garbage", "dev"), ("dev", "dev"), ("nopass", "nopass"),
                    (" NOPASS ", "nopass"), ("gate", "gate")):
        _set_mode(v)
        assert C.mode_from_env() == want, (v, want)
    _set_mode(None)
    print("[ok] mode: unset/unknown -> dev (old behaviour), nopass and gate parsed")


def test_dev_mode_keeps_loader_and_logs_curve():
    _set_mode(None)
    m = _tiny_bloom()
    t = FakeTrainer(m, _batches(m, 0.6))
    logs = []
    res = C.fit_on_trainer(t, grid=GRID, min_gain=0.0, log=logs.append)
    assert res["mode"] == "dev" and res["a"] != 1.0
    assert len(t.get_eval_dataloader()) == 8, "dev mode must leave the dev-pass alone"
    curve = [x for x in logs if "curva" in x]
    assert len(curve) == 1 and all(f"{a:g}=" in curve[0] for a in GRID), curve
    print(f"[ok] dev mode: loader intact, full curve in the log ({curve[0].split('curva ')[1]})")


def test_nopass_blocks_real_dev_pass_without_touching_weights():
    from dev_pass import run_dev_pass
    _set_mode("nopass")
    m = _tiny_bloom()
    t = FakeTrainer(m, _batches(m, 0.6))
    res = C.fit_on_trainer(t, grid=GRID, min_gain=0.0, log=lambda *_: None)
    assert res["mode"] == "nopass"
    before = copy.deepcopy(m.state_dict())
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "marker"), "w").close()
        try:
            run_dev_pass(t, submission_dir=d, min_lr=1e-3, max_grad_norm=1.0, train_per_device=1, train_grad_accum=1,
                         local_rank=-1, log=lambda *_: None)
            raise AssertionError("the dev-pass ran")
        except C.DevPassBlocked:
            pass
        assert os.listdir(d) == ["marker"], "submission dir must stay untouched"
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k
    _set_mode(None)
    print("[ok] nopass: the real run_dev_pass stops before any weight or file is touched")


def test_nopass_flow_folds_the_measured_weights_and_transfers():
    from transformers import BloomForCausalLM
    from dev_pass import run_dev_pass
    _set_mode("nopass")
    m = _tiny_bloom()
    fit, held = _batches(m, 0.6, seed=3), _batches(m, 0.6, seed=4)
    t = FakeTrainer(m, fit)
    with tempfile.TemporaryDirectory() as sub:
        m.save_pretrained(sub, safe_serialization=True)   # the soup already persisted in the submission
        # train_instruct's sequence: fit < dev-pass (guarded) < fold
        res = C.fit_on_trainer(t, grid=GRID, min_gain=0.001, log=lambda *_: None)
        try:
            run_dev_pass(t, submission_dir=sub, min_lr=1e-3, max_grad_norm=1.0, train_per_device=1,
                         train_grad_accum=1, local_rank=-1, log=lambda *_: None)
        except Exception as e:
            assert isinstance(e, C.DevPassBlocked), e
        assert res["a"] in (0.4, 0.6, 0.8), res
        assert C.apply_to_submission(sub, res["names"], res["a"], log=lambda *_: None) == 2
        shipped = BloomForCausalLM.from_pretrained(sub).eval()
    ids = held[0]["input_ids"]
    with torch.no_grad():
        err = (shipped(input_ids=ids).logits - res["a"] * m(input_ids=ids).logits).abs().max()
    assert float(err) < 1e-2, float(err)
    l0, l1 = _loss(m, held), _loss(shipped, held)
    assert l1 < l0 - 1e-3, (l0, l1)
    _set_mode(None)
    print(f"[ok] nopass flow: shipped logits = {res['a']:g} x measured; held-out fold {l0:.4f} -> {l1:.4f}")


def test_nopass_a1_still_blocks_and_writes_nothing():
    _set_mode("nopass")
    m = _tiny_bloom()
    t = FakeTrainer(m, _batches(m, 1.0))
    res = C.fit_on_trainer(t, grid=(0.9, 1.0, 1.1), min_gain=10.0, log=lambda *_: None)
    assert res["a"] == 1.0 and not res["names"]
    try:
        t.get_eval_dataloader()
        raise AssertionError("loader not blocked")
    except C.DevPassBlocked:
        pass
    _set_mode(None)
    print("[ok] nopass with a=1: dev-pass still off, nothing to fold (the arm measures the soup itself)")


def test_fit_failure_leaves_dev_pass_on():
    _set_mode("nopass")

    class Broken(FakeTrainer):
        def _prepare_inputs(self, x):
            raise ValueError("boom")

    m = _tiny_bloom()
    t = Broken(m, _batches(m, 1.0, n=1))
    try:
        C.fit_on_trainer(t, grid=GRID, min_gain=0.0, log=lambda *_: None)
        raise AssertionError("expected the fit to fail")
    except ValueError:
        pass
    assert len(t.get_eval_dataloader()) == 1, "a failed fit must not switch the dev-pass off"
    _set_mode(None)
    print("[ok] failed fit: ship path untouched (dev-pass stays on, no fold)")


def test_call_order_in_train_instruct():
    """The block relies on these: fit < dev-pass < fold, nothing else reads the dev loader between fit and fold,
    and the dev-pass exception is swallowed by the keep-best guard."""
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    i_fit, i_dp, i_fold = (src.index("fit_on_trainer(trainer"), src.index("run_dev_pass("),
                           src.index("apply_to_submission(train_request"))
    assert i_fit < i_dp < i_fold
    between = src[i_fit:i_fold]
    for frag in ("get_eval_dataloader", ".evaluate(", "evaluation_loop"):
        assert frag not in between, frag
    assert "mantendo melhor checkpoint" in between and "except Exception as _e" in between
    dp = open(os.path.join(HERE, "dev_pass.py")).read()
    body = dp[dp.index("def run_dev_pass"):]
    assert body.index("get_eval_dataloader()") < body.index("training_step(ddp_model")
    assert body.index("get_eval_dataloader()") < body.index("_save_weights_only(unwrapped")
    print("[ok] call order: fit < guarded dev-pass (loader first) < fold; nothing else uses the loader in between")



# ── gate ─────────────────────────────────────────────────────────────────────

GATE_SCALE = 3.0        # soft enough that CE(a*z) has a clear interior optimum


class ListCollator:
    def __call__(self, rows):
        return {k: torch.tensor([r[k] for r in rows]) for k in rows[0]}


class GateTrainer(FakeTrainer):
    """eval_dataset-driven loader, one row per batch (per_device_eval_batch_size=1 as in the trainer)."""
    def __init__(self, model, rows):
        super().__init__(model, None)
        self.eval_dataset = rows

    def get_eval_dataloader(self):
        c = ListCollator()
        return [c([self.eval_dataset[i]]) for i in range(len(self.eval_dataset))]


@torch.no_grad()
def _rows(model, a_true, n=240, seed=5, T=24):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(0, 50, (1, T), generator=gen)
        z = model(input_ids=x).logits.double()
        y = torch.multinomial(torch.softmax(a_true * z[0, :-1], -1), 1, generator=gen).squeeze(1)
        lab = [-100, -100] + y[1:].tolist()
        out.append({"input_ids": x[0].tolist(), "attention_mask": [1] * T, "labels": lab})
    return out


def _perturb(m, sd, seed=9):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.add_(torch.randn(p.shape, generator=gen) * sd)


def _logits(m, ids):
    with torch.no_grad():
        return m(input_ids=ids).logits


IDS = torch.tensor([[3, 8, 11, 4, 9, 21, 30, 2, 17, 5]])


def test_gate_real_trainer_harmful_pass_restores_soup_with_its_a():
    from transformers import BloomForCausalLM, Trainer, TrainingArguments
    from dev_pass import run_dev_pass
    _set_mode("gate")
    m = _tiny_bloom(GATE_SCALE)
    rows = _rows(m, 0.5)
    soup_logits = _logits(m, IDS)
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "sub")
        m.save_pretrained(sub, safe_serialization=True)
        args = TrainingArguments(output_dir=os.path.join(d, "o"), per_device_eval_batch_size=1, report_to=[],
                                 use_cpu=True, remove_unused_columns=False)
        t = Trainer(model=m, args=args, eval_dataset=rows, data_collator=ListCollator())
        logs = []
        res = C.fit_on_trainer(t, grid=GRID, min_gain=0.001, log=logs.append)
        assert res["a"] != res["a"] and res["mode"] == "gate", res
        assert len(t.get_eval_dataloader()) == 120, "the dev-pass must see one half of dev"
        a_pre = res["pre"]["a"]
        assert a_pre in (0.4, 0.6, 0.8), a_pre
        run_dev_pass(t, submission_dir=sub, min_lr=5e-2, max_grad_norm=1.0, train_per_device=1, train_grad_accum=8,
                     local_rank=-1, log=logs.append)                   # hot on purpose: wrecks the model
        assert C.apply_to_submission(sub, res["names"], res["a"], log=logs.append) == 2
        shipped = BloomForCausalLM.from_pretrained(sub).eval()
    err = (_logits(shipped, IDS) - a_pre * soup_logits).abs().max()
    assert float(err) < 1e-2, float(err)
    line = [x for x in logs if "gate: juri" in x]
    assert line and line[0].endswith("-> soup"), line
    assert not C._GATE, "state must not outlive the decision"
    _set_mode(None)
    print(f"[ok] gate (real Trainer + run_dev_pass): harmful pass -> soup restored x{a_pre:g}; {line[0].split('juri ')[1]}")


def test_gate_keeps_a_good_pass():
    from transformers import BloomForCausalLM
    _set_mode("gate")
    teacher = _tiny_bloom(GATE_SCALE)
    rows = _rows(teacher, 1.0, seed=6)
    teacher_logits = _logits(teacher, IDS)
    soup = copy.deepcopy(teacher)
    _perturb(soup, 0.06)
    t = GateTrainer(soup, rows)
    with tempfile.TemporaryDirectory() as sub:
        soup.save_pretrained(sub, safe_serialization=True)
        logs = []
        res = C.fit_on_trainer(t, grid=(0.8, 1.0, 1.25), min_gain=0.001, log=logs.append)
        soup.load_state_dict(teacher.state_dict())                     # the "pass" recovers the teacher
        soup.save_pretrained(sub, safe_serialization=True)
        C.apply_to_submission(sub, res["names"], res["a"], log=logs.append)
        shipped = BloomForCausalLM.from_pretrained(sub).eval()
    line = [x for x in logs if "gate: juri" in x][0]
    assert line.endswith("-> pass"), line
    a = float(line.split("| pass a=")[1].split()[0])
    assert float((_logits(shipped, IDS) - a * teacher_logits).abs().max()) < 1e-2
    _set_mode(None)
    print(f"[ok] gate: good pass kept x{a:g}; {line.split('juri ')[1]}")


def test_gate_margin_leans_to_the_pass():
    from transformers import BloomForCausalLM  # noqa: F401
    _set_mode("gate")
    m = _tiny_bloom(GATE_SCALE)
    rows = _rows(m, 1.0, seed=7)
    t = GateTrainer(m, rows)
    with tempfile.TemporaryDirectory() as sub:
        m.save_pretrained(sub, safe_serialization=True)
        logs = []
        res = C.fit_on_trainer(t, grid=(1.0,), min_gain=0.0, log=logs.append)
        pre = res["pre"]["loss_a"]
        _perturb(m, 0.002)                                             # barely worse than the soup
        m.save_pretrained(sub, safe_serialization=True)
        os.environ["SN56_LOGIT_CAL_GATE_MARGIN"] = "5"
        C.apply_to_submission(sub, res["names"], res["a"], log=logs.append)
        os.environ.pop("SN56_LOGIT_CAL_GATE_MARGIN")
    line = [x for x in logs if "gate: juri" in x][0]
    assert line.endswith("-> pass") and pre > 0, line
    _set_mode(None)
    print("[ok] gate: inside the margin the pass stays (status quo)")


def test_gate_pass_skipped_or_crashed_folds_soup_a():
    from transformers import BloomForCausalLM
    for crashed in (False, True):
        _set_mode("gate")
        m = _tiny_bloom(GATE_SCALE)
        rows = _rows(m, 0.5)
        soup_logits = _logits(m, IDS)
        t = GateTrainer(m, rows)
        with tempfile.TemporaryDirectory() as sub:
            m.save_pretrained(sub, safe_serialization=True)
            logs = []
            res = C.fit_on_trainer(t, grid=GRID, min_gain=0.001, log=logs.append)
            if crashed:
                _perturb(m, 0.05)                                      # in-memory moved, submission still the soup
            assert C.apply_to_submission(sub, res["names"], res["a"], log=logs.append) == 2
            shipped = BloomForCausalLM.from_pretrained(sub).eval()
        assert any("nao trocou" in x for x in logs), logs
        assert float((_logits(shipped, IDS) - res["pre"]["a"] * soup_logits).abs().max()) < 1e-2
    _set_mode(None)
    print("[ok] gate: pass skipped or crashed mid-way -> soup on disk x a_pre, in-memory drift ignored")


def test_gate_restore_failure_leaves_the_pass_unfolded():
    import dev_pass
    from transformers import BloomForCausalLM
    _set_mode("gate")
    m = _tiny_bloom(GATE_SCALE)
    rows = _rows(m, 0.5)
    t = GateTrainer(m, rows)
    real = dev_pass._save_weights_only
    with tempfile.TemporaryDirectory() as sub:
        m.save_pretrained(sub, safe_serialization=True)
        logs = []
        res = C.fit_on_trainer(t, grid=GRID, min_gain=0.001, log=logs.append)
        _perturb(m, 0.3)                                               # a clearly worse pass
        m.save_pretrained(sub, safe_serialization=True)
        passed = _logits(m, IDS)
        dev_pass._save_weights_only = lambda *a, **k: None             # a save that failed and kept the old files
        try:
            assert C.apply_to_submission(sub, res["names"], res["a"], log=logs.append) == 0
        finally:
            dev_pass._save_weights_only = real
        shipped = BloomForCausalLM.from_pretrained(sub).eval()
    assert float((_logits(shipped, IDS) - passed).abs().max()) < 1e-3, "no stale fold on the wrong weights"
    assert any("restauracao da soup falhou" in x for x in logs), logs
    _set_mode(None)
    print("[ok] gate: failed restore -> pass weights, no fold (never a stale a on the wrong weights)")


def test_gate_small_dev_and_fit_failure_stay_on_ship_path():
    _set_mode("gate")
    m = _tiny_bloom(GATE_SCALE)
    small = _rows(m, 1.0, n=2 * C.GATE_MIN_ROWS - 1)
    t = GateTrainer(m, small)
    res = C.fit_on_trainer(t, grid=GRID, min_gain=0.0, log=lambda *_: None)
    assert res["a"] == 1.0 and not res["names"] and t.eval_dataset is small and not C._GATE

    class Broken(GateTrainer):
        def _prepare_inputs(self, x):
            raise ValueError("boom")

    full = _rows(m, 1.0, n=2 * C.GATE_MIN_ROWS)
    b = Broken(m, full)
    try:
        C.fit_on_trainer(b, grid=GRID, min_gain=0.0, log=lambda *_: None)
        raise AssertionError("expected the fit to fail")
    except ValueError:
        pass
    assert b.eval_dataset is full and not C._GATE, "a failed fit must give the dev-pass all of dev back"
    _set_mode(None)
    print("[ok] gate: small dev or failed fit -> whole dev back to the dev-pass, nothing pending")


if __name__ == "__main__":
    test_mode_from_env()
    test_dev_mode_keeps_loader_and_logs_curve()
    test_nopass_blocks_real_dev_pass_without_touching_weights()
    test_nopass_flow_folds_the_measured_weights_and_transfers()
    test_nopass_a1_still_blocks_and_writes_nothing()
    test_fit_failure_leaves_dev_pass_on()
    test_call_order_in_train_instruct()
    test_gate_real_trainer_harmful_pass_restores_soup_with_its_a()
    test_gate_keeps_a_good_pass()
    test_gate_margin_leans_to_the_pass()
    test_gate_pass_skipped_or_crashed_folds_soup_a()
    test_gate_restore_failure_leaves_the_pass_unfolded()
    test_gate_small_dev_and_fit_failure_stay_on_ship_path()
    print("ALL 13 SN56 LOGIT_CAL_MODE TESTS PASSED")
