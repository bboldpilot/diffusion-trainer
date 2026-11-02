"""31 Oct flags: SN56_DEV_SCOREABLE, SN56_SOUP_CGROUP, SN56_CHAT_EVAL (+ SN56_CHAT_SEARCH_8BIT), SN56_R1_COOLDOWN,
SN56_R1_AVG2, SN56_FP32_NORMS (SN56_F6_GLOBAL has its own DDP test, test_sn56_f6_ddp.py).

Run: python scripts/tests/test_sn56_round4.py   (torch, transformers 4.51.3, safetensors)
"""
import datetime
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import sn56_plan as P  # noqa: E402
import sn56_retry as R  # noqa: E402
from sn56_devmetric import scoreable_rows, vali_sequence_len  # noqa: E402


def _clear_env():
    for k in list(P.FLAGS) + [P.MASTER, "SN56_LR_FIXED", "SN56_MIN_LR_RATE", "SN56_EVAL_FROM", "SN56_AVG_WITH"]:
        os.environ.pop(k, None)


def test_defaults():
    _clear_env()
    assert P.flag_on("SN56_LR_RETRY") and not P.flag_on("SN56_LOGIT_CAL")
    assert P.flag_on("SN56_DEV_SCOREABLE") and P.flag_on("SN56_SOUP_CGROUP")
    assert P.flag_on("SN56_R1_COOLDOWN") and P.flag_on("SN56_R1_AVG2")      # COOLAVG default since 1 Nov
    assert P.flag_on("SN56_CHAT_EVAL") and P.flag_on("SN56_CHAT_SEARCH_8BIT")  # boss-round Chat plan since 1 Nov
    for k in ("SN56_F6_GLOBAL", "SN56_FP32_NORMS", "SN56_TRAIN_SCOREABLE"):
        assert not P.flag_on(k), k
    os.environ["SN56_TEXT_FIX"] = "0"
    for k in ("SN56_LR_RETRY", "SN56_DEV_SCOREABLE", "SN56_SOUP_CGROUP", "SN56_R1_COOLDOWN", "SN56_R1_AVG2",
              "SN56_CHAT_EVAL", "SN56_CHAT_SEARCH_8BIT"):
        assert not P.flag_on(k), k
    _clear_env()
    print("[ok] defaults: LR_RETRY on, LOGIT_CAL off, DEV_SCOREABLE + SOUP_CGROUP + COOLAVG + CHAT_EVAL on, "
          "other arms off; master rollback")


def test_scoreable_dev():
    assert vali_sequence_len(32768) == 4096 and vali_sequence_len(8192) == 4096
    assert vali_sequence_len(4096) == 2048 and vali_sequence_len(4097) == 2049 and vali_sequence_len(None) == 4096
    rows = [{"input_ids": [0] * n} for n in (10, 4096, 4097, 9000)]
    kept, n = scoreable_rows(rows, 4096)
    assert [len(r["input_ids"]) for r in kept] == [10, 4096] and n == 2
    kept, n = scoreable_rows(rows[2:], 4096)        # none would survive -> validator retries longer
    assert len(kept) == 2 and n == 0
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    assert src.index("scoreable_rows(dev_ds.eval_dataset") < src.index("dev_ds = PackedDataset(")
    print("[ok] dev scoreable: validator seq rule (4096 / ceil(mpe/2)), rows > seq dropped before packing")


def test_cgroup_bound():
    with tempfile.TemporaryDirectory() as d:
        assert P.cgroup_free_bytes(d) is None
        open(os.path.join(d, "memory.max"), "w").write("max\n")
        open(os.path.join(d, "memory.current"), "w").write("100\n")
        assert P.cgroup_free_bytes(d) is None
        open(os.path.join(d, "memory.max"), "w").write(str(64 << 30))
        open(os.path.join(d, "memory.current"), "w").write(str(40 << 30))
        assert P.cgroup_free_bytes(d) == 24 << 30
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "memory"))
        open(os.path.join(d, "memory/memory.limit_in_bytes"), "w").write(str(1 << 62))
        open(os.path.join(d, "memory/memory.usage_in_bytes"), "w").write("5")
        assert P.cgroup_free_bytes(d) is None       # v1 unlimited sentinel
        open(os.path.join(d, "memory/memory.limit_in_bytes"), "w").write("1000")
        assert P.cgroup_free_bytes(d) == 995
    from checkpoint_avg_callback import AdaptiveTrainingCallback
    orig = P.cgroup_free_bytes
    try:
        P.cgroup_free_bytes = lambda root="/sys/fs/cgroup": 123
        assert AdaptiveTrainingCallback._available_ram_bytes() == 123
        os.environ["SN56_SOUP_CGROUP"] = "0"
        assert AdaptiveTrainingCallback._available_ram_bytes() != 123
    finally:
        P.cgroup_free_bytes = orig
        _clear_env()
    print("[ok] soup RAM bound: cgroup v2/v1 headroom, unlimited -> host, min(host, cgroup), flag rollback")


def test_chat_eval_scope():
    _clear_env()
    os.environ["SN56_CHAT_EVAL"] = "0"
    assert not P.sft_features("ChatTask", False, None, False)["sn56_on"]
    _clear_env()
    f = P.sft_features("ChatTask", False, None, False)
    assert f["sn56_on"] and f["per_example"] and f["scale_repair"]
    assert not P.sft_features("ChatTask", True, None, False)["sn56_on"]
    assert not P.sft_features("ChatTask", False, "ds.json", False)["sn56_on"]
    assert P.sft_features("InstructTextTask", False, None, False)["sn56_on"]
    _clear_env()
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    assert ('if (_task_type == "ChatTask" and sn56_plan.flag_on("SN56_CHAT_SEARCH_8BIT")\n'
            '            and _optim_name in ("paged_adamw_8bit", "adamw_8bit", "adamw_bnb_8bit")):\n'
            '        _probe_opt = bnb.optim.AdamW8bit') in src
    assert src.count("optimizer_cls=_probe_opt,") == 2                  # step timer + LR search
    assert src.index("_probe_opt = None") < src.index("measure_step_time(") < src.index("run_lr_search(")
    print("[ok] chat eval: ChatTask full FT joins the InstructText plan by default (SN56_CHAT_EVAL=0 opts out; "
          "LoRA/DS never); its probes use AdamW8bit")


def _rec(**kw):
    end = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=kw.pop("left", 2000))
    r = {"task_type": "InstructTextTask", "sn56_on": True, "use_kl": False, "dev_metric": "vali", "dev": 1.9117,
         "lr": 2.43e-5, "lr_fixed": None, "lr_argmin": 1.45e-5, "lr_search_s": 700.0, "t_start": 0.0,
         "t_train_end": 1565.0, "end_time": end.strftime("%Y-%m-%d %H:%M:%S"), "best_step": 100, "spe": 100,
         "stopped_early": True, "pre_devpass_dir": None}
    r.update(kw)
    return r


def test_cooldown_and_env():
    _clear_env()
    os.environ["SN56_R1_COOLDOWN"] = "0"; os.environ["SN56_R1_AVG2"] = "0"
    go, lr2, _ = R.decide(_rec(lr_argmin=2.3e-5))              # argmin not lower: no retry without cooldown
    assert not go
    os.environ["SN56_R1_COOLDOWN"] = "1"
    go, lr2, why = R.decide(_rec(lr_argmin=2.3e-5))
    assert go and lr2 == 2.43e-5 and "cooldown" in why, why
    go, lr2, _ = R.decide(_rec(lr_argmin=2.3e-5, stopped_early=False))
    assert not go
    go, lr2, _ = R.decide(_rec())
    assert go and lr2 == 1.45e-5
    env = R.second_run_env(_rec(), lr2)
    assert env == {"SN56_LR_SEARCH": "0", "SN56_LR_FIXED": repr(1.45e-5), "SN56_MAX_EPOCHS": "1.250",
                   "SN56_MIN_LR_RATE": "0.1", "SN56_EVAL_FROM": "0.25"}, env
    assert R.second_run_env(_rec(best_step=10, spe=100), lr2)["SN56_MAX_EPOCHS"] == "0.500"
    assert R.second_run_env(_rec(best_step=900, spe=100), lr2)["SN56_MAX_EPOCHS"] == "4.000"
    assert "SN56_MAX_EPOCHS" not in R.second_run_env(_rec(best_step=None), lr2)
    with tempfile.TemporaryDirectory() as d:
        os.environ["SN56_R1_AVG2"] = "1"
        assert R.second_run_env(_rec(pre_devpass_dir=d), lr2)["SN56_AVG_WITH"] == d
        assert "SN56_AVG_WITH" not in R.second_run_env(_rec(pre_devpass_dir=d + "-gone"), lr2)
    _clear_env()
    os.environ["SN56_R1_COOLDOWN"] = "0"; os.environ["SN56_R1_AVG2"] = "0"
    assert R.second_run_env(_rec(), 1e-5) == {"SN56_LR_SEARCH": "0", "SN56_LR_FIXED": repr(1e-5)}
    os.environ["SN56_MIN_LR_RATE"] = "0.1"; os.environ["SN56_EVAL_FROM"] = "0.25"
    assert P.env_pos_float("SN56_MIN_LR_RATE") == 0.1 and P.env_pos_float("SN56_EVAL_FROM") == 0.25
    os.environ["SN56_EVAL_FROM"] = "-1"; assert P.env_pos_float("SN56_EVAL_FROM") is None
    _clear_env()
    with tempfile.TemporaryDirectory() as d:
        assert P.best_step_from_loss_txt(d) is None
        open(os.path.join(d, "loss.txt"), "w").write("100,1.9214445354734984")
        assert P.best_step_from_loss_txt(d) == 100
        open(os.path.join(d, "loss.txt"), "w").write("93,no_eval")
        assert P.best_step_from_loss_txt(d) == 93
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    for frag in ('env_pos_float("SN56_MIN_LR_RATE")', 'env_pos_float("SN56_EVAL_FROM")', "try_blend(trainer",
                 '"best_step": _best_step', '"pre_devpass_dir": _pre_dir', "shutil.copytree(train_request"):
        assert frag in src, frag
    assert src.index("try_blend(trainer") < src.index("sn56_plan.write_run_record(") < src.index("run_dev_pass(")
    print("[ok] cooldown: same-LR retry only when cooled + early; env horizon 1.25x best epoch [0.5, 4], "
          "min_lr 0.1, evals from 0.25 ep; AVG2 plumbing; hooks in order blend < record < dev-pass")


def test_blend():
    from transformers import LlamaConfig, LlamaForCausalLM
    from safetensors.torch import save_file
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2)
    m, other = LlamaForCausalLM(cfg), LlamaForCausalLM(cfg)
    sd_o = {n: p.detach().clone() for n, p in other.named_parameters()}
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    bk = R.blend_into(m, sd_o, 0.5)
    for n, p in m.named_parameters():
        assert torch.allclose(p, 0.5 * (before[n] + sd_o[n]), atol=1e-6), n
    R.restore(m, bk)
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())

    class Out:
        def __init__(self, v):
            self.metrics = {"eval_loss": v}

    class FakeTrainer:
        def __init__(self, model, dev):
            self.model, self.dev = model, dev

        def get_eval_dataloader(self):
            return []

        def evaluation_loop(self, dl, desc, metric_key_prefix="eval"):
            return Out(self.dev)

    class Avg:
        best_loss = 1.90

    with tempfile.TemporaryDirectory() as d:
        od = os.path.join(d, "pre"); os.makedirs(od)
        save_file({k: v.contiguous() for k, v in sd_o.items()}, os.path.join(od, "model.safetensors"))
        sub = os.path.join(d, "sub"); os.makedirs(sub)
        m.save_pretrained(sub, safe_serialization=True)
        open(os.path.join(sub, "loss.txt"), "w").write("100,1.9")
        a = Avg()
        assert not R.try_blend(FakeTrainer(m, 1.95), od, a, sub, log=lambda *_: None)
        assert a.best_loss == 1.90 and all(torch.equal(p, before[n]) for n, p in m.named_parameters())
        assert R.try_blend(FakeTrainer(m, 1.80), od, a, sub, log=lambda *_: None)
        assert a.best_loss == 1.80
        from transformers import LlamaForCausalLM as L2
        m2 = L2.from_pretrained(sub)
        for n, p in m2.named_parameters():
            assert torch.allclose(p, 0.5 * (before[n] + sd_o[n]), atol=1e-6), n
        assert open(os.path.join(sub, "loss.txt")).read() == "100,1.9"
    print("[ok] avg2: 50:50 blend exact, restore exact; worse blend reverted, better blend kept + persisted "
          "(config/loss.txt preserved)")


def test_fp32_masters():
    from transformers import LlamaConfig, LlamaForCausalLM
    from sn56_fp32_master import Fp32MasterCallback, make_fp32_norm_trainer, one_d_params

    class Args:
        learning_rate = 1e-4

    class Base:
        def __init__(self, model):
            self.model, self.args, self.cbs = model, Args(), []

        def create_optimizer(self):
            decay = [p for n, p in self.model.named_parameters() if p.dim() > 1]
            nodecay = [p for n, p in self.model.named_parameters() if p.dim() == 1]
            return torch.optim.AdamW([{"params": decay, "weight_decay": 0.0, "lr": 1e-4},
                                      {"params": nodecay, "weight_decay": 0.0, "lr": 1e-4}], lr=1e-4)

        def add_callback(self, cb):
            self.cbs.append(cb)

    def run(fp32):
        torch.manual_seed(0)
        m = LlamaForCausalLM(LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=2)).to(torch.bfloat16)
        tr = (make_fp32_norm_trainer(Base) if fp32 else Base)(m)
        opt = tr.create_optimizer()
        cb = tr.cbs[0] if fp32 else None
        w0 = m.model.norm.weight.detach().float().clone()
        x = torch.randint(0, 40, (2, 8))
        for _ in range(60):
            out = m(input_ids=x, labels=x)
            out.loss.backward()
            if cb:
                cb.on_pre_optimizer_step(None, None, None)
            opt.step()
            if cb:
                cb.on_optimizer_step(None, None, None)
            opt.zero_grad(set_to_none=True); m.zero_grad(set_to_none=True)
        moved = float((m.model.norm.weight.float() - w0).abs().max())
        return moved, tr, opt

    moved_bf16, _, _ = run(False)
    moved_fp32, tr, opt = run(True)
    assert moved_bf16 == 0.0, moved_bf16                 # Adam steps of ~1e-4 round away at |w| ~ 1 (ulp 2^-7)
    assert moved_fp32 > 0.0, moved_fp32
    pairs = tr._sn56_fp32_pairs
    assert pairs and all(p.dtype == torch.bfloat16 and mm.dtype == torch.float32 for p, mm in pairs)
    for p, mm in pairs:
        assert torch.equal(p, mm.to(torch.bfloat16))       # model = round(master) after every step
    ids_model = {id(p) for p, _ in pairs}
    assert not any(id(p) in ids_model for g in opt.param_groups for p in g["params"])
    assert opt.param_groups[-1]["weight_decay"] == 0.0 and opt.param_groups[-1]["lr"] == 1e-4
    assert len(one_d_params(tr.model)) == len(pairs)
    print(f"[ok] fp32 norms: bf16 norm weight never moves at lr 1e-4 (0.0), with fp32 masters it does "
          f"({moved_fp32:.4f}); 1-D params out of the model groups, master group lr/wd, model == round(master)")


def test_hooks():
    src = open(os.path.join(HERE, "train_instruct.py")).read()
    assert 'global_norm=sn56_plan.flag_on("SN56_F6_GLOBAL")' in src
    assert 'if _feat["fp32_norms"]:' in src and "make_fp32_norm_trainer(_trainer_cls)" in src
    assert 'flag_on("SN56_TRAIN_SCOREABLE")' in src and not P.flag_on("SN56_TRAIN_SCOREABLE")
    assert src.index('flag_on("SN56_TRAIN_SCOREABLE")') < src.index("train_ds = _DatasetCls(")
    print("[ok] hooks: F6_GLOBAL into the loss trainer, FP32_NORMS trainer wrap")


if __name__ == "__main__":
    test_defaults()
    test_scoreable_dev()
    test_cgroup_bound()
    test_chat_eval_scope()
    test_cooldown_and_env()
    test_blend()
    test_fp32_masters()
    test_hooks()
    print("ALL 8 SN56 ROUND-4 TESTS PASSED")
