"""SN56_DPO_SELECT=opp plumbing on the REAL TRL DPOTrainer (trl 0.18.0 as in the trainer image), CPU.

The train_dpo.py wiring, step by step, on a tiny Llama:
  1. eval_dataset={"dev", "gate"} is prepared by TRL; after the swap the default eval loader is the gate split
     and get_eval_dataloader(dev_prepared) (the LR search's loader) is dev.
  2. The opponent = our own saved model dir: opponent_vector() scores it with THIS trainer (gate split, base
     reference), writes {ids, losses}, restores the policy bit-exact; re-reading aligns by ID; the vector equals
     scoring a trainer whose policy IS the opponent.
  3. sn56_opp set -> evaluate()'s eval_loss = opp_gate_stats score (both bounds + scalar guard), eval_loss_mean
     kept, opponent stats in the dump; sn56_select=False -> eval_loss = the mean.
  4. SN56_DPO_INIT: reference pinned to a base copy BEFORE load_weights_into -> TRL keeps ref = base, policy =
     the init dir (tied head handled); an untied missing tensor raises.
  5. calibrate(score_fn=opp score) picks the argmin of the opponent score, not of the mean.

Run: python scripts/tests/test_sn56_dpo_opp_trl.py   (needs trl, datasets, tokenizers; skips otherwise)
"""
import copy
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_opp_plumbing_real_trl():
    try:
        import numpy as np
        import torch
        import trl
        from datasets import Dataset
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from trl import DPOConfig, DPOTrainer
    except Exception as e:  # pragma: no cover
        print(f"[skip] real TRL test ({e})")
        return
    import sn56_dpo_knobs as K
    import sn56_dpo_opp as O
    import sn56_dpo_scale as S
    from sn56_dpo_select import dataset_ids, load_opp_vector, make_gate_dpo_trainer, opp_gate_stats

    words = [f"w{i}" for i in range(40)]
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>")
    rng = np.random.default_rng(1)
    lo, hi = words[:20], words[20:]
    def sent(ws, n): return " ".join(rng.choice(ws, n))
    def row(): return {"prompt": sent(words, int(rng.integers(3, 8))), "chosen": " " + sent(lo, int(rng.integers(2, 8))),
                       "rejected": " " + sent(hi, int(rng.integers(2, 8)))}
    full = Dataset.from_list([row() for _ in range(80)])
    dev = Dataset.from_list([row() for _ in range(15)])
    ids = dataset_ids(full)
    gi, ti = K.gate_split(ids, 25, min_train=20)
    gate, train = full.select(gi), full.select(ti)
    gate_ids = [ids[i] for i in gi]
    assert dataset_ids(gate) == gate_ids and len(gate) >= 25
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=True)
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        def args(sub, **kw):
            return DPOConfig(output_dir=os.path.join(tmp, sub), report_to=[], per_device_eval_batch_size=4,
                             per_device_train_batch_size=8, max_length=64, beta=0.1, use_cpu=True, logging_steps=1,
                             save_strategy="no", eval_strategy="no", **kw)
        # the opponent: a short real DPO run on the training rows only, saved like a submission
        opp_model = copy.deepcopy(base)
        DPOTrainer(model=opp_model, ref_model=copy.deepcopy(base), args=args("opp", learning_rate=5e-3, max_steps=10),
                   train_dataset=train, processing_class=tok).train()
        opp_dir = os.path.join(tmp, "opp-sub"); opp_model.save_pretrained(opp_dir, safe_serialization=True)

        # ── 1: candidate trainer, eval dict + swap (train_dpo.py opponent mode) ──
        cand = copy.deepcopy(base)
        with torch.no_grad():
            for p in cand.parameters():
                p.add_(0.02 * torch.randn_like(p))
        dump = os.path.join(tmp, "dump")
        cls = make_gate_dpo_trainer(DPOTrainer, dump_dir=dump, select=True)
        tr = cls(model=cand, ref_model=copy.deepcopy(base), args=args("cand"), train_dataset=train,
                 eval_dataset={"dev": dev, "gate": gate}, processing_class=tok)
        dev_eval = tr.eval_dataset["dev"]; tr.eval_dataset = tr.eval_dataset["gate"]
        assert len(S.dev_pair_losses(tr)) == len(gate)
        assert len(S.dev_pair_losses(tr, loader=tr.get_eval_dataloader(dev_eval))) == len(dev)
        print(f"[ok] eval dict prepared by TRL {trl.__version__}: default loader = gate ({len(gate)}), "
              f"get_eval_dataloader(dev) = dev ({len(dev)})")

        # ── 2: opponent vector from the saved dir ──
        before = [p.detach().clone() for p in tr.model.parameters()]
        vec_path = os.path.join(tmp, "opp-vec.json")
        v = O.opponent_vector(tr, gate_ids, vec_path=vec_path, model_dir=opp_dir)
        assert v is not None and len(v) == len(gate) and os.path.exists(vec_path)
        assert all(torch.equal(a, b) for a, b in zip(before, tr.model.parameters())), "policy not restored"
        assert np.allclose(load_opp_vector(vec_path, list(reversed(gate_ids))), v[::-1])
        tr_o = DPOTrainer(model=copy.deepcopy(opp_model), ref_model=copy.deepcopy(base), args=args("o2"),
                          train_dataset=train, eval_dataset=gate, processing_class=tok)
        direct = S.dev_pair_losses(tr_o)
        assert np.allclose(v, direct, atol=1e-6), (v[:4], direct[:4])
        v2 = O.opponent_vector(tr, gate_ids, vec_path=vec_path, model_dir="")     # cached read, no model needed
        assert np.allclose(v, v2)
        print(f"[ok] opponent vector: scored in the candidate trainer == the opponent's own trainer "
              f"(mean {v.mean():.5f}), cached by ID, policy bit-identical")

        # ── 3: selection metric ──
        tr.sn56_opp = v
        m = tr.evaluate()
        files = sorted(glob.glob(os.path.join(dump, "*.json")))
        rec = json.load(open(files[-1]))
        want = opp_gate_stats(rec["per_pair"], v)
        assert abs(m["eval_loss"] - want["score"]) < 1e-9 and "opp" in rec
        assert abs(m["eval_loss_mean"] - np.mean(rec["per_pair"])) < 1e-5
        tr.sn56_select = False
        m2 = tr.evaluate()
        assert abs(m2["eval_loss"] - m2["eval_loss_mean"]) < 1e-9
        tr.sn56_select = True
        print(f"[ok] eval_loss = opponent score {want['score']:.5f} (wr_lb {want['win_rate_lower_bound']:.3f}, "
              f"gap_lb {want['mean_gap_lower_bound']:+.4f}, passa={want['takes_task']}); sn56_select=False -> mean")

        # ── 4: SN56_DPO_INIT ──
        pol = copy.deepcopy(base)
        ref = copy.deepcopy(pol); ref.requires_grad_(False)
        info = O.load_weights_into(pol, opp_dir)
        assert info["loaded"] > 0
        sd_opp = dict(opp_model.named_parameters())
        assert all(torch.equal(p, sd_opp[n]) for n, p in pol.named_parameters())
        tr_i = DPOTrainer(model=pol, ref_model=ref, args=args("init"), train_dataset=train, eval_dataset=gate,
                          processing_class=tok)
        assert all(torch.equal(p, b) for p, b in zip(tr_i.ref_model.parameters(), base.parameters()))
        assert np.allclose(S.dev_pair_losses(tr_i), v, atol=1e-6)
        bad = os.path.join(tmp, "bad"); os.makedirs(bad)
        from safetensors.torch import save_file
        sd = {k: t.clone().contiguous() for k, t in opp_model.state_dict().items()
              if "layers.0.mlp" not in k and k != "lm_head.weight"}     # tied head stored once, as save_pretrained does
        save_file(sd, os.path.join(bad, "model.safetensors"))
        try:
            O.load_weights_into(copy.deepcopy(base), bad); raise AssertionError("missing tensors accepted")
        except ValueError:
            pass
        print(f"[ok] init: policy = opponent weights ({info['loaded']} tensors, tied {info['tied_skipped']}), "
              f"reference = base, missing tensors raise")

        # ── 5: scale picked by the opponent score ──
        out = S.calibrate(tr, [0.5, 1.0, 1.5, 2.0], min_gain=0.0, score_fn=lambda x: opp_gate_stats(x, v)["score"])
        curve = {float(k): val for k, val in out["losses"].items()}
        assert out["alpha"] == min(curve, key=curve.get)
        print(f"[ok] calibrate(score_fn=opponent): a={out['alpha']:g} (curve {({k: round(c, 4) for k, c in curve.items()})})")


if __name__ == "__main__":
    test_opp_plumbing_real_trl()
    print("\nALL 1 SN56 DPO OPP (TRL) TESTS PASSED")
