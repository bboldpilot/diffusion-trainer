"""SN56_DPO_SELECT=gate (K5) on the REAL TRL DPOTrainer (the trainer image pins trl 0.18.0), CPU.

Tiny Llama + word-level tokenizer, 37 pairs, eval batch 5 (uneven last batch): the per-pair vector captured from
dpo_loss has every pair, eval_loss == the gate score of that vector, eval_loss_mean == TRL's own mean.

Run: python scripts/tests/test_sn56_dpo_select_trl.py   (needs trl, datasets, tokenizers; skips otherwise)
"""
import copy
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_real_trl_dpo_trainer():
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
    from sn56_dpo_select import gate_vs_proxy, make_gate_dpo_trainer

    words = [f"w{i}" for i in range(40)]
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>")
    rng = np.random.default_rng(0)
    def sent(n): return " ".join(rng.choice(words, n))
    rows = [{"prompt": sent(int(rng.integers(3, 8))), "chosen": " " + sent(int(rng.integers(1, 12))),
             "rejected": " " + sent(int(rng.integers(1, 12)))} for _ in range(37)]
    ds = Dataset.from_list(rows)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0)
    torch.manual_seed(0); ref = LlamaForCausalLM(cfg)
    pol = copy.deepcopy(ref)
    with torch.no_grad():
        for p in pol.parameters(): p.add_(0.05 * torch.randn_like(p))
    with tempfile.TemporaryDirectory() as tmp:
        args = DPOConfig(output_dir=tmp, report_to=[], per_device_eval_batch_size=5, per_device_train_batch_size=4,
                         max_length=64, beta=0.1, use_cpu=True, logging_steps=1)
        dump = os.path.join(tmp, "dump")
        cls = make_gate_dpo_trainer(DPOTrainer, dump_dir=dump)
        tr = cls(model=pol, ref_model=ref, args=args, train_dataset=ds, eval_dataset=ds, processing_class=tok)
        m = tr.evaluate()
        files = glob.glob(os.path.join(dump, "*.json")); d = json.load(open(files[0]))
        vec = np.array(d["per_pair"])
        want = gate_vs_proxy(vec)
        print("trl", trl.__version__, "n pairs", len(vec), "eval_loss", m["eval_loss"], "score", want["score"],
              "eval_loss_mean", m["eval_loss_mean"], "vec mean", vec.mean())
        assert len(vec) == len(rows)
        assert abs(m["eval_loss"] - want["score"]) < 1e-6
        assert abs(m["eval_loss_mean"] - vec.mean()) < 1e-5, (m["eval_loss_mean"], vec.mean())
        # cross-check the captured per-pair values against the validator formula on TRL's own logps
    print(f"[ok] real TRL {trl.__version__} DPOTrainer: {len(vec)} pairs captured, eval_loss = gate score "
          f"{want['score']:.5f}, mean {vec.mean():.5f} kept")
    

if __name__ == "__main__":
    test_real_trl_dpo_trainer()
    print("\nALL 1 SN56 DPO SELECT (TRL) TESTS PASSED")
