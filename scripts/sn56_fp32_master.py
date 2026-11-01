"""SN56_FP32_NORMS (default off): fp32 master copies for the 1-D parameters (norm weights, biases).

The model trains in bf16 and the optimizer writes its step straight into the bf16 weights. For a norm weight
near 1.0 the bf16 spacing is 2^-7 (~0.0078), so an Adam step of ~lr (1e-5 .. 1e-4) rounds back to the same
value every time: those parameters do not train at all (Vikhr-7B boss probe, 28 Oct: 64-65% of weights
bit-identical to the base, every sampled large coordinate unchanged). Here the 1-D parameters are taken out of
the trainer's optimizer groups and replaced by fp32 masters in a group of their own (same lr schedule, no
weight decay): before each optimizer step the (clipped, DDP-averaged) bf16 gradients are copied onto the
masters, after it the masters are rounded once into the model. The forward pass, the saved model and every
other parameter are unchanged; matrices stay bf16 (a full fp32 master would be a different experiment).
"""
import torch
from transformers import TrainerCallback


def one_d_params(model):
    """[(name, param)] of trainable 1-D parameters (norm weights, biases)."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    return [(n, p) for n, p in m.named_parameters() if p.requires_grad and p.dim() == 1]


class Fp32MasterCallback(TrainerCallback):
    """Copies grads model -> masters before the step and weights masters -> model after it."""

    def __init__(self, pairs):
        self.pairs = pairs          # [(model param, fp32 master)]

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        for p, m in self.pairs:
            m.grad = None if p.grad is None else p.grad.detach().to(torch.float32)
            p.grad = None

    def on_optimizer_step(self, args, state, control, **kwargs):
        with torch.no_grad():
            for p, m in self.pairs:
                p.copy_(m.to(p.dtype))
                m.grad = None


def make_fp32_norm_trainer(base_cls):
    class Fp32NormTrainer(base_cls):
        _sn56_fp32_pairs = None

        def create_optimizer(self):
            opt = super().create_optimizer()
            if self._sn56_fp32_pairs is not None:
                return opt
            named = one_d_params(self.model)
            ids = {id(p) for _, p in named}
            lr = None
            for grp in opt.param_groups:
                if lr is None and grp.get("params"):
                    lr = grp.get("lr")
                grp["params"] = [p for p in grp["params"] if id(p) not in ids]
            pairs = [(p, p.detach().to(torch.float32).clone()) for _, p in named]
            if pairs:
                opt.add_param_group({"params": [m for _, m in pairs], "lr": lr if lr is not None else self.args.learning_rate,
                                     "weight_decay": 0.0})
                self.add_callback(Fp32MasterCallback(pairs))
                n_el = sum(p.numel() for p, _ in pairs)
                print(f"[sn56][fp32] {len(pairs)} parametros 1-D ({n_el / 1e6:.2f}M) com mestre fp32", flush=True)
            self._sn56_fp32_pairs = pairs
            return opt

    Fp32NormTrainer.__name__ = f"Fp32Norm{base_cls.__name__}"
    return Fp32NormTrainer
