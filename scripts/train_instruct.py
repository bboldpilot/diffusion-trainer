import quiet_mode  # noqa: F401,E402 — competition log gate; must precede heavy imports
from typing import Dict, Optional
import requests
import json
import random
from utility import log_info, MyDataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb
from checkpoint_avg_callback import AdaptiveTrainingCallback
from lr_search import run_lr_search, measure_step_time
from kl_trainer import KLRegularizedTrainer, load_base_model
import sn56_plan

# from packing.packed_dataset import PackedDataset
from transformers import (
    Trainer,
    TrainingArguments,
)

import os
import datetime
import shutil
import time
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
from state_manager import get_state, set_state
from v5_lane import dtype_kw, trainer_tokenizer_kw

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    request_path: Optional[str] = field(default=None)
    packing: Optional[bool] = field(default=False)
    max_packed_size: Optional[int] = field(default=-1)
    use_liger: Optional[bool] = field(default=False)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)
    use_attn_implementation: Optional[str] = field(default="")

@dataclass
class LoraArguments:
    lora_r: int = 128
    lora_alpha: int = 512
    lora_dropout: float = 0.1
    lora_target_modules: str = "all"  # all for all linear; "q_proj v_proj"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    
    
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            log_info(f"ajustável: {name}, pesos={num_params}")
            if "lm_head" in name or "embed_tokens" in name:
                embedding_lm_head_param_count += num_params
            else:
                lora_param_count += num_params
    trainable_params = embedding_lm_head_param_count + lora_param_count
    log_info(
        f"total={all_param:,d} || ajustáveis={trainable_params:,d} || fração={100 * trainable_params / all_param:.1f}%"
    )
    log_info(
        f"cabeça_emb={embedding_lm_head_param_count} ({embedding_lm_head_param_count * 100 / all_param:.1f}%)"
    )
    log_info(
        f"lora={lora_param_count} ({lora_param_count * 100 / all_param:.1f}%)"
    )
    

def load_lora_model(training_args: TrainingArguments, model_path: str, lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        **dtype_kw(torch.bfloat16),
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )
    # do not resize tokem embeddings in LOra --> will encounter size mismatch error in evaluation 
    # model.resize_token_embeddings(token_nums)
    # convert to lora
    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        modules = lora_args.lora_target_modules.split(" ")
        target_modules = [mod.strip() for mod in modules if len(mod.strip()) > 0]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        # modules_to_save=["lm_head", "embed_tokens"],  # because we retrain the embedding
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    # Activate computing load balancing loss iin MixtralForCausalLM
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    print_trainable_parameters(model)
    return model


def load_model(training_args: TrainingArguments, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        log_info("--- turbo ativo ---")
        model_class = AutoLigerKernelForCausalLM
    
    attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager"
    if training_args.use_attn_implementation:
        attn_implementation = training_args.use_attn_implementation
        log_info(f"atenção via {attn_implementation}")
    log_info(f"atenção: {attn_implementation}")
    
    model = model_class.from_pretrained(
        model_path,
        # trust_remote_code=True, remove this because we already filter the model architecture, it will not be used with liger-kernel 
        **dtype_kw(torch.bfloat16),
        attn_implementation=attn_implementation,
    )
    # model.resize_token_embeddings(token_nums)
    return model


def get_max_length_config():
    config_path = "test_axolotl.yml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def main():
    """Format of training requests"""
    argument_parser = transformers.HfArgumentParser((TrainingArguments, LoraArguments))
    _t_main0 = time.time()
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()
    # SN56_SEED (rehearsal replicates only; the validator passes no env, so the
    # inherited TrainingArguments seed 42 is what ships): data order + init seed.
    _seed_env = os.environ.get("SN56_SEED", "").strip()
    if _seed_env.lstrip("-").isdigit():
        training_args.seed = int(_seed_env)
    # SN56_MIN_LR_RATE (second run of SN56_R1_COOLDOWN, set by sn56_retry): cosine floor.
    _min_lr_rate_env = sn56_plan.env_pos_float("SN56_MIN_LR_RATE")
    if _min_lr_rate_env is not None and _min_lr_rate_env < 1:
        training_args.lr_scheduler_kwargs = {"min_lr_rate": _min_lr_rate_env}
    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    # log_info(f"Training request: {train_request}", "start")
    task_id = train_request["task_id"]

    # ── KL-regularised task contract (G.O.D feature/instruct-kl-training) ──
    # KL tasks set USE_KL=1 / KL_COEF=<float> on the container. The validator
    # scores eval_loss + kl_coef * KL(finetuned || base) over completion tokens,
    # so we add the identical term to the training loss (see kl_trainer.py).
    # Absent/unset => no KL, identical behaviour to before.
    use_kl = os.environ.get("USE_KL") == "1"
    _kl_coef_env = os.environ.get("KL_COEF")
    kl_coef = 0.0
    if use_kl and _kl_coef_env:
        try:
            kl_coef = float(_kl_coef_env)
        except (ValueError, TypeError):
            log_info(f"[sn56][kl] KL_COEF inválido ({_kl_coef_env!r}), desligando KL")
            kl_coef = 0.0
    use_kl = use_kl and kl_coef > 0
    if use_kl:
        log_info(f"[sn56][kl] USE_KL=1, kl_coef={kl_coef} — termo KL ativo")

    # ── SN56_* fixes (flags + math in sn56_plan.py) ──
    # F1-F4 + K4 gains: InstructText full fine-tune only; ChatTask (same
    # script), LoRA and DeepSpeed runs keep those inherited behaviours.
    # F6 (K1-b) + K4-repair: InstructText and ChatTask full fine-tunes
    # (SN56_K1_CHAT); K1-a dev metric: any InstructText / ChatTask run.
    _task_type = train_request.get("task_type", "InstructTextTask")
    _feat = sn56_plan.sft_features(_task_type, training_args.use_lora,
                                   getattr(training_args, "deepspeed", None), use_kl)
    _sn56_on = _feat["sn56_on"]
    _f_eval_fix = _sn56_on and sn56_plan.flag_on("SN56_EVAL_SKIP_FIX")
    _f_time_plan = _sn56_on and sn56_plan.flag_on("SN56_TIME_PLAN")
    _f_lr_search = (not _sn56_on) or sn56_plan.flag_on("SN56_LR_SEARCH")
    _f_neftune = (not _sn56_on) or sn56_plan.flag_on("SN56_NEFTUNE")
    _f_per_example = _feat["per_example"]
    _f6_alpha = _feat["alpha"]
    # K1-a: the validator's own dev metric (sn56_devmetric.py). Metric-only.
    _f_dev_vali = _feat["dev_vali"]
    # K4: fp32 per-tensor gains on augmentation-eligible weights (sn56_gain.py).
    _f_gain = _feat["gain"]
    # K4-repair: undo the validator's weight_scaling at load, blind (sn56_scale_detect.py). Not on KL
    # tasks: their KL reference is the augmented base.
    _f_scale_repair = _feat["scale_repair"]
    log_info(f"[sn56][k1] tarefa={_task_type} f6={_f_per_example} alpha={_f6_alpha:g} "
             f"dev={'vali' if _f_dev_vali else 'stock'} reparo={_f_scale_repair} ganho={_f_gain}")
    # F5: instruct_config decided the v5-lane mode (v5_pack.py); follow it.
    _v5_mode = train_request.get("packing_mode") if train_request.get("packing_mode") in ("v5_fa2", "v5_dyn") else None

    # ── Late-attempt guard (F2) ──
    # text_trainer re-runs this script after any crash. A retry that cannot
    # load the model, probe the step time, train a meaningful number of steps
    # and write before end_time must not re-plan the last minutes (25 Sep,
    # 3B/1.5h: six such attempts of 20 and 10 steps ran into the hard kill).
    # With a valid submission from an earlier attempt on disk it keeps it,
    # writes success.txt and exits. Rank 0 decides; the flag is broadcast so
    # DDP ranks exit together.
    if _sn56_on and train_request.get("end_time"):
        _late = False
        if is_main_process(LOCAL_RANK):
            _remaining0 = (sn56_plan.parse_utc(train_request["end_time"])
                           - datetime.datetime.now(datetime.timezone.utc).timestamp())
            _last_plan = get_state().get("sn56_last_plan") or {}
            _late = sn56_plan.late_attempt(
                _remaining0, sn56_plan.submission_is_valid(train_request.get("submission_dir", "")),
                _last_plan.get("t_per_step"), _last_plan.get("reserve_s"))
            if _late:
                log_info(
                    f"[sn56][guard] tentativa tardia: {_remaining0:.0f}s até end_time, submissão "
                    f"válida em {train_request.get('submission_dir')} — sem novo plano, submissão mantida"
                )
        if torch.distributed.is_initialized():
            _late_t = torch.tensor([1.0 if _late else 0.0],
                                   device=torch.device(f"cuda:{LOCAL_RANK}") if torch.cuda.is_available() else "cpu")
            torch.distributed.broadcast(_late_t, src=0)
            _late = bool(_late_t.item() > 0.5)
        if _late:
            if is_main_process(LOCAL_RANK):
                os.makedirs(training_args.output_dir, exist_ok=True)
                with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
                    f.write("Success")
            log_info("missão cumprida (guard)", "finish")
            return

    from model_utility import load_tokenizer
    tokenizer = load_tokenizer(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # wandb_init_success = init_wandb(train_request)
    # if not wandb_init_success:
    #     log_info("WANDB_API_KEY is not set, do not report to wandb")
    #     training_args.report_to = "none"    
    # else:
    #     log_info("WANDB_API_KEY is provided, we will report to wandb")
    #     training_args.report_to = "wandb"
        
    max_length = get_max_length_config()
    if "max_length" in train_request:
        max_length = train_request["max_length"]
    if _task_type == "ChatTask" and sn56_plan.flag_on("SN56_TRAIN_SCOREABLE"):
        # SN56_TRAIN_SCOREABLE: the validator scores only conversations <= its eval
        # sequence_len (4096); on the 26 Oct Chat data rows > 4096 are 11% of rows but
        # ~53% of the tokens trained at max_length 8192. Cap training at that length
        # (longer rows are truncated, not dropped) so the time goes to scored lengths.
        try:
            from sn56_devmetric import vali_sequence_len
            with open(os.path.join(train_request["model_path"], "config.json")) as _cf:
                _vseq_t = vali_sequence_len(json.load(_cf).get("max_position_embeddings"))
            if max_length > _vseq_t:
                log_info(f"[sn56][chatlen] max_length {max_length} -> {_vseq_t} (SN56_TRAIN_SCOREABLE)")
                max_length = _vseq_t
        except Exception as _e:
            log_info(f"[sn56][chatlen] falhou ({_e!r}), max_length {max_length}")

    # we already tokenize the data and save it to .pt (torch format, fast)
    _DatasetCls = MyDataset
    if _v5_mode is not None:
        # v5 modes pad per batch (V5Collator), never to max_length.
        from v5_pack import NoPadDataset
        _DatasetCls = NoPadDataset
    train_ds = _DatasetCls(
        tokenizer,
        f"datasets/train_tokenized_{task_id}.json",
        max_length
    )

    dev_ds = _DatasetCls(
        tokenizer,
        f"datasets/dev_tokenized_{task_id}.json",
        max_length
    )
    log_info(f"treino={len(train_ds)}, teste={len(dev_ds)}")
    if _f_dev_vali and sn56_plan.flag_on("SN56_DEV_SCOREABLE") and hasattr(dev_ds, "eval_dataset"):
        # K1-a parity: the validator drops eval rows longer than its sequence_len
        # (4096, or ceil(mpe/2) for short-context models) and scores the rest whole.
        try:
            from sn56_devmetric import scoreable_rows, vali_sequence_len
            with open(os.path.join(train_request["model_path"], "config.json")) as _cf:
                _mpe = json.load(_cf).get("max_position_embeddings")
            _vseq = vali_sequence_len(_mpe)
            dev_ds.eval_dataset, _n_drop = scoreable_rows(dev_ds.eval_dataset, _vseq)
            log_info(f"[sn56][devmetric] linhas do dev acima de {_vseq} (seq do validador, mpe={_mpe}): "
                     f"{_n_drop} removidas, {len(dev_ds.eval_dataset)} ficam")
        except Exception as _e:
            log_info(f"[sn56][devmetric] filtro de comprimento falhou ({_e!r}), dev intacto")
    if _f_dev_vali:
        _dev_long = sum(1 for dp in getattr(dev_ds, "eval_dataset", [])
                        if len(dp.get("input_ids", [])) > max_length)
        log_info(f"[sn56][devmetric] SN56_DEV_METRIC=vali: {len(dev_ds)} exemplos no dev, "
                 f"{_dev_long} > max_length {max_length} (truncados aqui; o validador pontua inteiros até 4096)")

    # ── Training data deduplication ──
    # Removes exact-duplicate tokenized samples. Especially impactful for
    # datasets with high near-duplicate rates (e.g. SciEntsBank at 59%).
    baseline_stats = train_request.get("baseline_stats")
    near_dup_rate = 0.0
    if baseline_stats:
        near_dup_rate = baseline_stats.get("dataset", {}).get("near_duplicate_rate", 0.0)
    from sn56_dev_testlike import training_dedup_enabled
    # The request records the split actually used, including on LR/OOM retries.
    _dev_testlike = (_task_type == "InstructTextTask" and not training_args.use_lora
                     and train_request.get("dev_testlike", False))
    if training_dedup_enabled(near_dup_rate, _dev_testlike):
        from data_filter import deduplicate_samples
        before = len(train_ds.eval_dataset)
        train_ds.eval_dataset = deduplicate_samples(train_ds.eval_dataset)
        log_info(f"[sn56][dedup] near_dup_rate={near_dup_rate:.2f}, deduped {before} -> {len(train_ds.eval_dataset)}")

    # ── (d) Prompt loss weight for prompt-dominated datasets ──
    # When prompt:completion > 5:1, unmask a small fraction of prompt tokens
    # so the model gets gradient signal about the input language/domain.
    # PLW scales inversely with ratio to avoid gradient dilution on extreme
    # ratios (e.g. 28:1 at flat 5% = 33% gradient dilution).
    if baseline_stats:
        _ds = baseline_stats.get("dataset", {})
        _prompt_tok = _ds.get("prompt_tokens", 0)
        _comp_tok = _ds.get("completion_tokens", 1)
        _pc_ratio = _prompt_tok / max(_comp_tok, 1)
        if _pc_ratio > 5 and not sn56_plan.flag_on("SN56_PLW"):
            log_info(f"[sn56][plw] prompt:comp={_pc_ratio:.1f}:1, off (SN56_PLW=0: completion-only loss)")
        elif _pc_ratio > 5:
            _plw = 0.05 / max(1.0, _pc_ratio / 5.0)
            from utility import apply_prompt_loss_weight
            train_ds.eval_dataset = apply_prompt_loss_weight(train_ds.eval_dataset, plw=_plw)
            log_info(f"[sn56][plw] prompt:comp={_pc_ratio:.1f}:1, PLW={_plw:.4f}")

    donot_pack = False
    original_train_size = len(train_ds)
    original_steps = original_train_size // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )  # number of steps in the original training
    # min_steps here is per epoch
    if original_steps < train_request["min_steps"]:
        donot_pack = True
        log_info(f"passos={original_steps} < mínimo={train_request['min_steps']}, sem compactação")

    min_data_size_num = (
        train_request["min_steps"]
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    
        
    log_info(f"limiar={min_data_size_num}, janela={max_length}")
    packing_mode = train_request.get("packing_mode", "fa")
    use_fa_packing = packing_mode == "fa"
    # Keep a handle on the RAW (unpacked) dataset: the post-search Gaussian
    # subsample mutates its eval_dataset and re-packs — PackedDataset doesn't
    # expose the underlying samples, so without this reference the subsample
    # can never fire on the packed (default) path.
    _raw_train_ds = train_ds
    _train_is_packed = False
    # F5 v5_fa2: padding-free packing (position_ids carry the example
    # boundaries); the 4.x seq-id-mask monkeypatch is NOT applied in the lane.
    _pack_free = _v5_mode == "v5_fa2"
    _pf_kw = {"padding_free": True} if _pack_free else {}
    if training_args.packing and not donot_pack:
        from monkeypatch import monkey_patch_packing_for_model, PackedDataset
        if _pack_free:
            log_info("[sn56][v5] costura sem preenchimento (position_ids), FA2 varlen")
        elif use_fa_packing:
            log_info("costurando sequências com atenção rápida")
            monkey_patch_packing_for_model(train_request["model_path"])
        else:
            log_info("costura ingênua, posições reiniciadas por sequência")

        t1 = datetime.datetime.now()
        train_ds = PackedDataset(
            train_ds,
            tokenizer,
            max_input_length=max_length,
            max_packed_size=training_args.max_packed_size,
            min_item_num=min_data_size_num,
            use_fa=use_fa_packing,
            **_pf_kw,
        )
        _train_is_packed = True
        t2 = datetime.datetime.now()
        log_info(f"costura treino: {(t2 - t1).total_seconds()}s")
        # Only pack eval with FA (proper sequence isolation).
        # Naive packing allows cross-attention between sequences which
        # corrupts eval loss — use unpacked eval for reliable checkpoint selection.
        if use_fa_packing or _pack_free:
            t1 = datetime.datetime.now()
            dev_ds = PackedDataset(
                dev_ds,
                tokenizer,
                max_input_length=max_length,
                max_packed_size=training_args.max_packed_size,
                use_fa=True,
                **_pf_kw,
            )
            t2 = datetime.datetime.now()
            log_info(f"costura teste: {(t2 - t1).total_seconds()}s")
        else:
            log_info("teste sem costura — isolamento necessário")
        log_info(f"treino compactado: {train_ds.stat()}")
        if hasattr(dev_ds, 'stat'):
            log_info(f"teste compactado: {dev_ds.stat()}")
        else:
            log_info(f"teste solto: {len(dev_ds)} amostras")

    log_info(f"nós={training_args.world_size}")
    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    log_info(f"passos/época={total_steps_per_epoch}")
    # consider reducing the batch_size if it is quite big
    # num_steps = len(train_ds) * training_args.num_train_epochs / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)
    # num_steps > min_step ->
    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = int(max_batch_size_theory)
    if max_batch_size_theory == 0:
        max_batch_size_theory = 1

    original_batch_size = training_args.per_device_train_batch_size
    if training_args.per_device_train_batch_size > max_batch_size_theory:
        # if batch_size is quite big set it to this value to make sure that we have at least min_steps
        if train_request.get("adjust_batch_size", True):
            log_info(
                f"lote grande demais ({training_args.per_device_train_batch_size}), cortando para {max_batch_size_theory}"
            )
            training_args.per_device_train_batch_size = max_batch_size_theory
            # need to update total_steps_per_epoch
            total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
            log_info(f"passos/época recalculados={total_steps_per_epoch}")

    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        # some model need to resize the token embeddings or encounter the size mismatch error; only for full-weight models
        resize_if_needed(train_request["model_name"], model, len(tokenizer))
        if _f_scale_repair:
            # before the LR search, so every later decision sees the repaired base
            from sn56_scale_detect import repair_model
            repair_model(model, train_request["model_path"], log=log_info)
    
    try:
        model.config.use_cache = False
    except:
        pass
    
    # some model need to set the generation config or encounter the invalid generation config error
    set_generation_config(train_request["model_name"], model)

    # ── F5: v5-lane batches (see v5_pack.py) ──
    _v5_collator = None
    if _v5_mode is not None:
        from v5_pack import V5Collator, patch_short_conv_seq_idx, SEQ_IDX_MODEL_TYPES, model_type_of
        _v5_mt = model_type_of(train_request["model_path"])
        _v5_seq_idx = _train_is_packed and _v5_mt in SEQ_IDX_MODEL_TYPES
        _v5_patched = patch_short_conv_seq_idx() if _v5_seq_idx else []
        _v5_collator = V5Collator(tokenizer.pad_token_id, pass_seq_idx=_v5_seq_idx)
        if not _train_is_packed:
            # Unpacked (v5_dyn, or too little data to pack): group similar
            # lengths so per-batch padding stays small.
            if hasattr(training_args, "train_sampling_strategy"):
                training_args.train_sampling_strategy = "group_by_length"
            else:
                training_args.group_by_length = True
        log_info(
            f"[sn56][v5] attn={getattr(model.config, '_attn_implementation', '?')} "
            f"packing={'padding-free' if _train_is_packed else 'off (dynamic padding, length-grouped)'} "
            f"bs={training_args.per_device_train_batch_size} ga={training_args.gradient_accumulation_steps} "
            f"max_length={max_length} mode={_v5_mode} model_type={_v5_mt} "
            f"seq_idx={_v5_seq_idx} conv_patch={_v5_patched}"
        )

    # Check if this is the main process and create the output directory
    if is_main_process(LOCAL_RANK):  # Only create directory on main process
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"destino criado: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    log_info(f"salvo periódico a cada {periodic_save_steps} passos")
    training_args.save_only_model = True  # only save the model, not the optimizer
    
    max_steps = train_request.get("max_steps", -1)
    log_info(f"teto de passos={max_steps}")
    
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    if "train" not in state:
        state["train"] = {}
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)
        
    total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
    
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs
    log_info(f"passos/época={total_steps_per_epoch}, total={total_steps_all_epochs}")

    warmup_from_ratio = max(10, int(0.03 * total_steps_all_epochs))
    training_args.warmup_steps = min(training_args.warmup_steps, warmup_from_ratio)
    log_info(f"Alongando por {training_args.warmup_steps} passos antes da corrida")

    # Adaptive eval frequency: only on final run (not LR search probes)
    _cm = train_request.get("checking_mode")
    is_final_run = _cm in ("none", None)
    log_info(f"checking_mode={_cm!r}, is_final_run={is_final_run}")
    if is_final_run:
        hours = train_request.get("hours_to_complete", 2)
        max_evals = max(3, int(hours * 4))
        adaptive_eval_steps = max(30, total_steps_all_epochs // min(10, max_evals))
        training_args.eval_strategy = "steps"
        training_args.eval_steps = adaptive_eval_steps
        training_args.save_strategy = "steps"
        training_args.save_steps = adaptive_eval_steps
        log_info(f"Olhando no espelho a cada {adaptive_eval_steps} passos")

    success_file = os.path.join(training_args.output_dir, "success.txt")
    # remove the success file if it exists
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)
    
    checking_step = train_request["checking_step"]
    if checking_step >= total_steps_per_epoch:
        checking_step = total_steps_per_epoch - 2
    
    # ── Always-on NEFTune ──
    # Start with alpha=1 (barely noticeable) so the hook is active.
    # Rollback can escalate to 5 → 10 → 15 when overfitting is detected.
    # Higher initial alpha for noisy datasets where regularization is needed from step 1.
    # SN56_NEFTUNE=0 (F4, default): no embedding noise — it is a regulariser for
    # generation quality, while the score is held-out CE of the plain model.
    if not _f_neftune:
        training_args.neftune_noise_alpha = None
        log_info("[sn56][neftune] desligado (SN56_NEFTUNE=0)")
    else:
        _neft_alpha = 1
        if baseline_stats:
            grad_noise = baseline_stats.get("training", {}).get("gradient_noise_scale", 0.0)
            if grad_noise > 1.0:
                _neft_alpha = 5
                log_info(f"[sn56][neftune] grad_noise={grad_noise:.2f} > 1.0, alpha=5")
        training_args.neftune_noise_alpha = _neft_alpha
        log_info(f"[sn56][neftune] alpha={_neft_alpha}")

    # Averaging mode: in-RAM window averaging for non-sharded models that fit
    # host RAM; disk-based averaging of consolidated checkpoints for sharded
    # (FSDP/DeepSpeed) or too-big models, where in-RAM snapshots are only shards.
    _shard_ds = getattr(training_args, "deepspeed", None) is not None
    _sharded = _shard_ds or len(training_args.fsdp or []) > 0
    _trainable_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2
    # RAM peak holds ~6x trainable bytes (best + 3-window + avg + stash).
    _avg_mode = "disk" if (_sharded or 6 * _trainable_bytes > 100e9) else "ram"
    log_info(
        f"[sn56][caldo] modo={_avg_mode} (treináveis={_trainable_bytes / 1e9:.0f}GB, "
        f"sharded={_sharded})"
    )
    ckpt_avg = (
        AdaptiveTrainingCallback(
            window=3, averaging_mode=_avg_mode, output_dir=training_args.output_dir
        )
        if is_final_run else None
    )
    if _sn56_on:
        # Bound the checkpoints on disk. Every eval point writes a weights-only
        # checkpoint; nothing pruned them, so a 3B run (6.2 GB each, every 20
        # steps) filled the disk at its 17th and died in safetensors (25 Sep).
        # The best checkpoint is copied into the submission dir the moment it
        # is written, so only the current one is needed afterwards; disk-mode
        # averaging reads the last `disk_members` from disk and keeps those.
        training_args.save_total_limit = (
            (ckpt_avg.disk_members if ckpt_avg is not None else 4) + 1 if _avg_mode == "disk" else 2
        )
        log_info(f"[sn56][disco] save_total_limit={training_args.save_total_limit}")
    if ckpt_avg is not None:
        ckpt_avg._submission_dir = train_request["submission_dir"]
        # end_time bounds the greedy soup's eval time at train end (it may use
        # at most half the remaining wall-clock; the dev-pass uses the rest).
        ckpt_avg.end_time = train_request["end_time"]

    # KL tasks use the KL-regularised subclass (only compute_loss differs, so all
    # callbacks/averaging/dev-pass/wall-clock behave identically); the base-model
    # reference is wired in after the LR search to keep it out of OOM-sensitive
    # search memory. Non-KL tasks use the stock Trainer unchanged.
    _trainer_cls = KLRegularizedTrainer if use_kl else Trainer
    if _f_per_example:
        # F6 (SN56_PER_EXAMPLE_LOSS=1): mean over examples of per-token CE,
        # the validator's test_loss weighting (sn56_loss.py); K1-b alpha != 1
        # weights example i by n_i^-alpha instead.
        from sn56_loss import make_per_example_trainer
        _trainer_cls = make_per_example_trainer(_trainer_cls, alpha=_f6_alpha,
                                                global_norm=sn56_plan.flag_on("SN56_F6_GLOBAL"))
        log_info(f"[sn56][perda] média por exemplo (SN56_PER_EXAMPLE_LOSS=1, alpha={_f6_alpha:g})")
    if _feat["fp32_norms"]:
        # SN56_FP32_NORMS: fp32 masters for norm weights / biases (sn56_fp32_master.py)
        from sn56_fp32_master import make_fp32_norm_trainer
        _trainer_cls = make_fp32_norm_trainer(_trainer_cls)
        log_info("[sn56][fp32] mestres fp32 para parametros 1-D (SN56_FP32_NORMS=1)")
    _gain_cb = None
    if _f_gain:
        # K4 (SN56_GAIN=1): gains trained in their own optimizer group, folded
        # into the weights before every eval/save and at train end (sn56_gain.py).
        # SN56_GAIN_INIT=<json {param name: g0}> sets initial gains (default 1.0).
        from sn56_gain import GainCallback, make_gain_trainer
        _gain_init = {}
        _gain_init_path = os.environ.get("SN56_GAIN_INIT", "").strip()
        if _gain_init_path:
            try:
                with open(_gain_init_path) as _f:
                    _gain_init = {str(k): float(v) for k, v in json.load(_f).items()}
            except Exception as _e:
                log_info(f"[sn56][ganho] SN56_GAIN_INIT ilegível ({_e}), ganhos em 1.0")
                _gain_init = {}
        _trainer_cls = make_gain_trainer(_trainer_cls, gain_lr=sn56_plan.flag_float("SN56_GAIN_LR"),
                                         init=_gain_init)
        _gain_cb = GainCallback()
        log_info(f"[sn56][ganho] SN56_GAIN=1 lr={sn56_plan.flag_float('SN56_GAIN_LR'):g} "
                 f"init={len(_gain_init)} valores")
    if _f_dev_vali:
        # K1-a (SN56_DEV_METRIC=vali): eval_loss = the validator's metric, so the
        # best checkpoint, soup, early stop and loss.txt select on it.
        from sn56_devmetric import make_vali_dev_trainer
        _trainer_cls = make_vali_dev_trainer(_trainer_cls, kl_coef=kl_coef if use_kl else 0.0)
        log_info(f"[sn56][devmetric] eval_loss = média por exemplo"
                 f"{f' + {kl_coef:g} x KL' if use_kl else ''} (SN56_DEV_METRIC=vali)")
    # Kept as a variable: the dev-pass budgeting below moves its end-time
    # trigger earlier once t_per_step is known (the trigger now also STOPS
    # training — see CustomEvalSaveCallback — which is what gives the dev-pass
    # its window to run in).
    when_to_eval_handler = WhenToEvalHandler(
        train_request["end_time"],
        train_request["save_before_remaining_time"],
        periodic_save_steps=periodic_save_steps,
        steps_per_epoch=total_steps_per_epoch,
        max_steps=max_steps,
    )
    _eval_cb = CustomEvalSaveCallback(
        when_to_eval_handler,
        train_request["submission_dir"],
        training_args.output_dir,
        train_request["model_name"],
        max_steps,
        checking_step=checking_step,
        total_steps_all_epochs=total_steps_all_epochs,
        end_time=train_request["end_time"],
        checking_mode=train_request.get("checking_mode", "none")
    )
    _eval_cb.sn56_atomic_copy = _sn56_on
    _trainer_kwargs = dict(
        model=model,
        **trainer_tokenizer_kw(tokenizer),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=[
            # K4 fold + unhook must precede the checkpoint callbacks' train end
            *([] if _gain_cb is None else [_gain_cb]),
            *([] if ckpt_avg is None else [ckpt_avg]),
            _eval_cb,
        ],
    )
    if _v5_collator is not None:
        _trainer_kwargs["data_collator"] = _v5_collator
    if use_kl:
        _trainer_kwargs["kl_coef"] = kl_coef
        _trainer_kwargs["use_lora_base"] = bool(training_args.use_lora)
    trainer = _trainer_cls(**_trainer_kwargs)
    if _gain_cb is not None:
        _gain_cb.trainer = trainer
    if _f_per_example and getattr(trainer, "_f6_mode", None) == "logits":
        # F6 fallback for a head the hidden-state path cannot drive: the loss
        # materialises the full (B, T, V) logits, so cap the per-device batch
        # to what fits (bf16 logits + the fp32 CE chunk) and keep the
        # effective batch with more accumulation.
        from sn56_loss import cap_batch_for_logits
        _V = int(getattr(model.config, "vocab_size", 0) or 0)
        _new_bs, _new_ga = cap_batch_for_logits(
            training_args.per_device_train_batch_size, training_args.gradient_accumulation_steps,
            max_length, _V)
        if (_new_bs, _new_ga) != (training_args.per_device_train_batch_size,
                                   training_args.gradient_accumulation_steps):
            log_info(f"[sn56][perda] F6 sem caminho oculto: lote {training_args.per_device_train_batch_size}x"
                     f"{training_args.gradient_accumulation_steps} -> {_new_bs}x{_new_ga} (logits {_V} x {max_length})")
            training_args.per_device_train_batch_size = _new_bs
            training_args.gradient_accumulation_steps = _new_ga
            trainer._train_batch_size = training_args.train_batch_size

    if ckpt_avg is not None:
        ckpt_avg.trainer = trainer
    try:
        trainer.tokenizer = tokenizer
    except AttributeError:
        # transformers 5 exposes the tokenizer as processing_class only
        pass

    # ── Estimate dataset coverage (rough; refined with real timing below) ──
    # Coverage = how much of one epoch we can train in the budget. Drives
    # Gaussian difficulty subsampling, applied AFTER the search so it can use the
    # measured per-step time. The params heuristic here is only a fallback for
    # when the search is skipped (cached LR / DeepSpeed).
    _param_nums = sum(p.numel() for p in model.parameters())
    _est_step_time = 4.0 + _param_nums / 1e9 * 3.0  # rough fallback heuristic
    _hours = train_request.get("hours_to_complete", 2)
    _eff_bs = max(1, training_args.per_device_train_batch_size
                  * training_args.gradient_accumulation_steps
                  * training_args.world_size)
    _steps_per_epoch = len(train_ds) // _eff_bs
    _est_total_steps = int(_hours * 3600 / _est_step_time)
    _est_coverage = _est_total_steps / max(1, _steps_per_epoch)
    log_info(f"[sn56][cobertura] steps/epoch={_steps_per_epoch}, est_total={_est_total_steps}, coverage~{_est_coverage:.1%} (heurística)")

    # ── LR search ──
    # Always run the search (its warmup measures real per-step time and
    # plan_budget decides skip/validate/tiny/full). The coverage heuristic no
    # longer gates the search — a good LR matters regardless of how much of the
    # dataset we cover. Coverage still drives Gaussian subsampling above.
    _use_deepspeed = getattr(training_args, "deepspeed", None) is not None
    t_per_step = None

    # Skip LR search if a previous attempt already found the best LR.
    _state_cache = get_state()
    _cached_lr = _state_cache.get("best_lr_found")
    _cached_bs = _state_cache.get("best_lr_batch_size")
    _lr_note = "busca" if _f_lr_search else ""
    _lr_search_s = None
    if not _f_lr_search:
        # ── F3 (SN56_LR_SEARCH=0): no search. The LR is the finder estimate
        # already in the command (lr_finder via instruct_config, x reg_ratio)
        # x SN56_LR_MULT, re-centred with the sqrt rule if the batch changed.
        # A few timed steps still measure t_per_step (and walk the same OOM
        # halving ladder) for the time plan; their weights are discarded.
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{LOCAL_RANK}")
            model = model.to(device)
            log_info(f"[sn56][cronometro] Modelo movido pra {device}")

        def _eff_bs_now():
            return (training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                    * training_args.world_size)

        # Reference = the first attempt's command LR and effective batch; OOM
        # retries (text_trainer halves the batch in the command) re-centre on it.
        _lr_ref = _state_cache.get("sn56_lr_ref")
        if not _lr_ref:
            _lr_ref = {"lr": float(training_args.learning_rate), "eff_bs": int(_eff_bs_now())}
            _state_cache["sn56_lr_ref"] = _lr_ref
            if is_main_process(LOCAL_RANK):
                set_state(_state_cache)

        def _halve_batch_timing_loader():
            cur = training_args.per_device_train_batch_size
            new_bs = max(1, cur // 2)
            training_args.per_device_train_batch_size = new_bs
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][cronometro] Cozinha quente, lote {cur} -> {new_bs}")
            return trainer.get_train_dataloader()

        original_bs = training_args.per_device_train_batch_size
        t_per_step, _timing = measure_step_time(
            model,
            trainer.get_train_dataloader(),
            lr=float(training_args.learning_rate),
            grad_accum_steps=training_args.gradient_accumulation_steps,
            max_grad_norm=training_args.max_grad_norm,
            dataloader_factory=_halve_batch_timing_loader,
        )
        # Length-grouped (unpacked v5) batches are served longest-first, so the
        # timed steps over-state the epoch-average step cost: scale by tokens.
        if (
            t_per_step is not None
            and _v5_mode is not None
            and not _train_is_packed
            and _timing.get("tokens_per_step")
        ):
            _lens = [min(len(dp["input_ids"]), max_length) for dp in _raw_train_ds.eval_dataset] or [1]
            _expected = (training_args.gradient_accumulation_steps
                         * training_args.per_device_train_batch_size
                         * (sum(_lens) / len(_lens)) * 1.05)
            _corr = sn56_plan.tokens_correction(_timing["tokens_per_step"], _expected)
            log_info(f"[sn56][cronometro] correção por tokens x{_corr:.2f} "
                     f"({_timing['tokens_per_step']:.0f} medidos vs ~{_expected:.0f}/passo)")
            t_per_step *= _corr

        if torch.distributed.is_initialized():
            _bs_t = torch.tensor(
                [training_args.per_device_train_batch_size],
                device=next(model.parameters()).device,
            )
            torch.distributed.all_reduce(_bs_t, op=torch.distributed.ReduceOp.MIN)
            _min_bs = int(_bs_t.item())
            if _min_bs < training_args.per_device_train_batch_size:
                log_info(
                    f"[sn56][cronometro] OOM em outro rank: lote "
                    f"{training_args.per_device_train_batch_size} -> {_min_bs} (sincronizado)"
                )
                training_args.per_device_train_batch_size = _min_bs
        if training_args.per_device_train_batch_size != original_bs:
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][cronometro] Lote reduzido: {original_bs} -> "
                     f"{training_args.per_device_train_batch_size} (carries to training)")
        if torch.distributed.is_initialized():
            _sync = torch.tensor([t_per_step or 0.0], device=next(model.parameters()).device)
            torch.distributed.broadcast(_sync, src=0)
            t_per_step = _sync[0].item() if _sync[0].item() > 0 else None
            for p in model.parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)

        _lr_mult = sn56_plan.flag_float("SN56_LR_MULT")
        best_lr = sn56_plan.lr_for_batch(_lr_ref["lr"], _lr_ref["eff_bs"], _eff_bs_now(), _lr_mult)
        _lr_note = f"finder x{_lr_mult:g} (lote efetivo {_lr_ref['eff_bs']}->{_eff_bs_now()})"
        _lr_fixed = sn56_plan.lr_fixed()
        if _lr_fixed is not None:
            # SN56_LR_FIXED: the dev-selected second run (sn56_retry.py) trains at
            # the first run's search argmin; the timed steps above still plan it.
            best_lr = _lr_fixed
            _lr_note = "fixo (SN56_LR_FIXED, 2a corrida)"
        trainer.args.learning_rate = best_lr
        log_info(f"[sn56][cronometro] sem busca de LR: lr={best_lr:.2e} ({_lr_note})")
    elif _cached_lr is not None:
        import math
        _current_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if _cached_bs is not None and _cached_bs != _current_bs:
            _scale = math.sqrt(_current_bs / _cached_bs)
            _cached_lr = _cached_lr * _scale
            log_info(f"[sn56][farejando] LR cached ajustado pra bs {_cached_bs}->{_current_bs}: {_cached_lr:.2e}")
        training_args.learning_rate = _cached_lr
        _lr_note = "cache (busca da tentativa anterior, mult já aplicado)"
        log_info(f"[sn56][farejando] Usando LR cached de tentativa anterior: {_cached_lr:.2e}")
    elif _use_deepspeed:
        log_info(f"[sn56][farejando] Pulando (DeepSpeed ativo)")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{LOCAL_RANK}")
            model = model.to(device)
            log_info(f"[sn56][farejando] Modelo movido pra {device}")

        search_loader = trainer.get_train_dataloader()

        def _halve_batch_dataloader():
            cur = training_args.per_device_train_batch_size
            new_bs = max(1, cur // 2)
            training_args.per_device_train_batch_size = new_bs
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Cozinha quente, lote {cur} -> {new_bs}")
            return trainer.get_train_dataloader()

        original_bs = training_args.per_device_train_batch_size
        _t_search0 = time.perf_counter()
        best_lr, t_per_step = run_lr_search(
            model=model,
            train_dataloader=search_loader,
            initial_lr=training_args.learning_rate,
            hours_to_complete=train_request["hours_to_complete"],
            grad_accum_steps=training_args.gradient_accumulation_steps,
            max_grad_norm=training_args.max_grad_norm,
            dataloader_factory=_halve_batch_dataloader,
            steps_per_epoch=_steps_per_epoch,
        )
        _lr_search_s = time.perf_counter() - _t_search0

        # Sync batch size across ranks: if ANY rank halved its batch on OOM
        # during the search, every rank must adopt the smallest surviving batch.
        # Otherwise divergent per-rank steps-per-epoch deadlock DDP at an epoch
        # boundary. all_reduce(MIN) is a collective — all ranks must call it.
        if torch.distributed.is_initialized():
            _bs_t = torch.tensor(
                [training_args.per_device_train_batch_size],
                device=next(model.parameters()).device,
            )
            torch.distributed.all_reduce(_bs_t, op=torch.distributed.ReduceOp.MIN)
            _min_bs = int(_bs_t.item())
            if _min_bs < training_args.per_device_train_batch_size:
                log_info(
                    f"[sn56][farejando] OOM em outro rank: lote "
                    f"{training_args.per_device_train_batch_size} -> {_min_bs} (sincronizado)"
                )
                training_args.per_device_train_batch_size = _min_bs

        if training_args.per_device_train_batch_size != original_bs:
            trainer._train_batch_size = training_args.train_batch_size
            log_info(f"[sn56][farejando] Lote reduzido durante busca: {original_bs} -> {training_args.per_device_train_batch_size} (carries to training)")

        if torch.distributed.is_initialized():
            # Sync best_lr AND t_per_step from rank 0: each rank measures its own
            # warmup, and downstream decisions (gauss subsample, epoch planning)
            # must be identical across ranks or the dataset/step counts diverge
            # and DDP deadlocks. 0.0 is the None sentinel.
            _sync = torch.tensor([best_lr, t_per_step or 0.0],
                                 device=next(model.parameters()).device)
            torch.distributed.broadcast(_sync, src=0)
            best_lr = _sync[0].item()
            t_per_step = _sync[1].item() if _sync[1].item() > 0 else None
            for p in model.parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)
            log_info(f"[sn56][farejando] Sincronizando lr={best_lr:.2e} + weights to all ranks")

        # F3: SN56_LR_MULT scales the searched LR too (InstructText full-FT);
        # the cached value below already carries it, so a retry never
        # multiplies twice.
        _lr_searched = best_lr
        _lr_mult = sn56_plan.flag_float("SN56_LR_MULT") if _sn56_on else 1.0
        if _lr_mult > 0 and _lr_mult != 1.0:
            best_lr = best_lr * _lr_mult
        trainer.args.learning_rate = best_lr
        _lr_note = f"busca {_lr_search_s:.0f}s: {_lr_searched:.3e} x{_lr_mult:g}"
        log_info(f"[sn56][farejando] Usando lr={best_lr:.2e} ({_lr_note})")
        # Cache LR + batch size so OOM retries don't repeat the search
        _s = get_state()
        _s["best_lr_found"] = best_lr
        _s["best_lr_batch_size"] = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if is_main_process(LOCAL_RANK):
            set_state(_s)

    # ── KL step-time overhead ──
    # The LR search measures t_per_step WITHOUT the base model (it isn't loaded
    # yet), but KL adds a base forward + completion-token KL each step. Inflate
    # the measured t_per_step so coverage/epoch planning don't over-plan; the
    # wall-clock stop is still the hard guarantee, this just keeps the cosine
    # schedule cooling properly. ~1.4x ≈ one extra forward (no backward).
    if use_kl and t_per_step is not None:
        _kl_overhead = 1.4
        t_per_step = t_per_step * _kl_overhead
        log_info(f"[sn56][kl] t_per_step inflado x{_kl_overhead} p/ overhead do termo KL: {t_per_step:.3f}s")

    # ── Gaussian difficulty subsampling for low-coverage runs ──
    # Uses the REAL measured per-step time (t_per_step, now synced across ranks)
    # instead of the params heuristic; falls back to the heuristic only when the
    # search was skipped (cached LR / DeepSpeed) and no measurement exists. Runs
    # before epoch planning so it sees the reduced dataset.
    #
    # Selection: keep the MEDIUM-DIFFICULTY core (Gaussian around the median
    # completion length) — both extremes (trivial / noise-length) are the least
    # informative. Sizing: coverage already dictates how MANY samples the budget
    # sees; the subsample only changes WHICH ones. Repeats buy ~nothing over
    # fresh data (Muennighoff 2023), so size the kept set to what the budget
    # covers ONCE (target = coverage * n) — breadth over repetition.
    #
    # Operates on the RAW dataset and re-packs: PackedDataset precomputes its
    # blocks, so mutating after packing is a no-op (this gate was dead on the
    # packed path before). Deterministic across ranks: t_per_step is broadcast,
    # gaussian_subsample is seeded, so every rank builds the same dataset.
    # Recompute from the CURRENT batch — it (and t_per_step) may have been
    # halved on OOM during the search; using the original batch would
    # overestimate coverage ~2x per halving and under-subsample.
    _cur_eff_bs = max(1, training_args.per_device_train_batch_size
                      * training_args.gradient_accumulation_steps
                      * training_args.world_size)
    if t_per_step is not None:
        _cur_spe = max(1, len(train_ds) // _cur_eff_bs)
        _coverage = _hours * 3600 * 0.85 / (_cur_spe * t_per_step)
        _cov_src = "medido"
    else:
        _coverage = _est_coverage
        _cov_src = "heurística"
    _n_raw = len(_raw_train_ds.eval_dataset)
    # MEASURED coverage only: the heuristic fallback (cached-LR retries /
    # DeepSpeed) over-estimates step time, which under-estimates coverage and
    # would over-cut see-once data on exactly the runs we know least about.
    if t_per_step is not None and _coverage < 0.5 and _n_raw > 2000:
        from utility import gaussian_subsample
        # Coverage is measured in (packed) steps but is a unitless fraction of
        # one epoch, so it translates to raw samples directly.
        #
        # INVARIANT — breadth pad: coverage is deliberately computed on the FULL
        # hours budget even though the LR search already spent up to 20% of it,
        # so the target runs ~20-25% generous. Training must outpace the
        # measured t_per_step by that margin before any sample is seen twice;
        # the expected case sees ~80% of the core exactly once. Do NOT "fix"
        # this to remaining-time — that would remove the hedge and turn
        # estimation error into discarded fresh data.
        _target = max(2000, int(_coverage * _n_raw))
        if _target < _n_raw:
            _raw_train_ds.eval_dataset = gaussian_subsample(
                _raw_train_ds.eval_dataset, _target
            )
            if _train_is_packed:
                from monkeypatch import PackedDataset
                _t1 = datetime.datetime.now()
                train_ds = PackedDataset(
                    _raw_train_ds,
                    tokenizer,
                    max_input_length=max_length,
                    max_packed_size=training_args.max_packed_size,
                    min_item_num=min_data_size_num,
                    use_fa=use_fa_packing,
                    **_pf_kw,
                )
                trainer.train_dataset = train_ds
                _t2 = datetime.datetime.now()
                log_info(f"[sn56][gauss] re-costura: {(_t2 - _t1).total_seconds():.1f}s")
            else:
                train_ds = _raw_train_ds
                trainer.train_dataset = train_ds
            log_info(
                f"[sn56][gauss] coverage={_coverage:.0%}<50% ({_cov_src}), "
                f"núcleo de {_target}/{_n_raw} amostras (~1 época no orçamento; "
                f"steps/epoch agora {len(train_ds) // _cur_eff_bs})"
            )

    # ── F2 (SN56_TIME_PLAN=1): optimizer steps from the time budget ──
    # Plan the steps that fit until the FINAL-SAVE trigger (end_time minus the
    # save + dev-pass reserve — where training really stops), net of the
    # eval/save overhead the plan implies, so the cosine reaches its floor at
    # the real cutoff. E = steps / steps-per-epoch with no floor, capped at
    # SN56_MAX_EPOCHS; the overfit early stop (AdaptiveTrainingCallback) stays.
    # Computed on rank 0 and broadcast, like the inherited planning below.
    _plan = None
    _tail = None          # the tail plan every rank applies (broadcast from rank 0)
    if t_per_step is not None and not _use_deepspeed and _f_time_plan:
        _plan_t = torch.zeros(14, dtype=torch.float64, device=next(model.parameters()).device)
        _cur_eff_bs_p = max(1, training_args.per_device_train_batch_size
                            * training_args.gradient_accumulation_steps
                            * training_args.world_size)
        _spe_p = max(1, len(train_ds) // _cur_eff_bs_p)
        _want_soup = is_final_run and _avg_mode == "ram" and ckpt_avg is not None
        _want_devpass = is_final_run and _avg_mode == "ram"
        if is_main_process(LOCAL_RANK):
            # ── The tail (sn56_plan.tail_plan): everything that runs after the
            # final-save trigger, priced from the model size, the measured step
            # time, a measured write rate of the checkpoint disk and the dev
            # size, with a safety margin. The trigger goes that far before
            # end_time; the soup and the dev-pass get their own deadlines.
            _end_ts = sn56_plan.parse_utc(train_request["end_time"])
            _write_bps = sn56_plan.measure_write_bps(training_args.output_dir)
            _tp = sn56_plan.tail_plan(
                now_ts=datetime.datetime.now(datetime.timezone.utc).timestamp(),
                end_ts=_end_ts,
                t_per_step=t_per_step,
                n_params=_param_nums,
                n_dev_rows=len(dev_ds) if is_final_run else 0,
                world_size=training_args.world_size,
                grad_accum=training_args.gradient_accumulation_steps,
                per_device_bs=training_args.per_device_train_batch_size,
                write_bps=_write_bps,
                soup_pool=(ckpt_avg.soup_max if ckpt_avg is not None else 0),
                want_soup=_want_soup,
                want_devpass=_want_devpass,
            )
            _stop_ts = _tp["trigger_ts"]
            # a periodic eval point: the eval, the checkpoint write and (often)
            # the copy of a new best into the submission dir
            _eval_cost = (_tp["eval_s"] + _tp["save_s"] + 0.5 * _tp["copy_s"]
                          if is_final_run and len(dev_ds) > 0 else 0.0)
            _p = sn56_plan.plan_time_budget(
                now_ts=datetime.datetime.now(datetime.timezone.utc).timestamp(),
                stop_ts=_stop_ts,
                t_per_step=t_per_step,
                steps_per_epoch=_spe_p,
                max_epochs=sn56_plan.max_epochs(),
                warmup_cap=training_args.warmup_steps,
                eval_cost_s=_eval_cost,
                periodic_save_steps=periodic_save_steps,
            )
            _plan_t[0] = float(_p["planned_steps"])
            _plan_t[1] = float(_p["epochs"])
            _plan_t[2] = float(_p["warmup"])
            _plan_t[3] = float(_p["eval_steps"])
            _plan_t[4] = 1.0 if _p["capped"] else 0.0
            _plan_t[5] = float(_tp["reserve_s"])
            _plan_t[6] = float(_tp["n_soup"])
            _plan_t[7] = 1.0 if _tp["devpass"] else 0.0
            _plan_t[8] = float(_tp["soup_deadline_ts"] - _tp["persist_s"])   # soup evals done by
            _plan_t[9] = float(_tp["devpass_s"])
            _plan_t[10] = float(_tp["save_s"])
            _plan_t[11] = float(_tp["safety_s"])
            _plan_t[12] = float(_tp["copy_s"])
            _plan_t[13] = float(_tp["persist_s"])
            log_info(
                f"[sn56][relogio] plano por tempo: {_p['planned_steps']} passos "
                f"(E={_p['epochs']:.2f}, teto {sn56_plan.max_epochs():g} épocas, capped={_p['capped']}), "
                f"t_per_step={t_per_step:.3f}s, parada final {sn56_plan.fmt_utc(_stop_ts)} "
                f"(T-{_tp['reserve_s'] / 60:.1f}min), evals~{_p['n_evals']} x {_eval_cost:.0f}s, "
                f"warmup={_p['warmup']}, eval_steps={_p['eval_steps']}"
            )
            log_info(
                f"[sn56][tail] reserva {_tp['reserve_s']:.0f}s = passo {_tp['step_s']:.1f}s + eval "
                f"{_tp['eval_s']:.0f}s + gravação {_tp['save_s']:.0f}s + cópia {_tp['copy_s']:.0f}s + "
                f"soup {_tp['n_soup']}x ({_tp['soup_s']:.0f}s) + persist {_tp['persist_s']:.0f}s + "
                f"dev-pass {'%.0fs' % _tp['devpass_s'] if _tp['devpass'] else 'não'} + margem "
                f"{_tp['safety_s']:.0f}s; disco {_write_bps / 1e9:.2f}GB/s, modelo "
                f"{2 * _param_nums / 1e9:.1f}GB, última gravação até {sn56_plan.fmt_utc(_tp['final_write_ts'])} "
                f"(end_time {sn56_plan.fmt_utc(_end_ts)}){' [cauda encolhida]' if _tp['shrunk'] else ''}"
            )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(_plan_t, src=0)
        if _plan_t[0] > 0:
            _plan = {
                "planned_steps": int(_plan_t[0].item()),
                "epochs": float(_plan_t[1].item()),
                "warmup": int(_plan_t[2].item()),
                "eval_steps": int(_plan_t[3].item()),
                "capped": bool(_plan_t[4].item() > 0.5),
                "steps_per_epoch": _spe_p,
            }
            _tail = {
                "reserve_s": float(_plan_t[5].item()),
                "n_soup": int(_plan_t[6].item()),
                "devpass": bool(_plan_t[7].item() > 0.5),
                "soup_eval_deadline_ts": float(_plan_t[8].item()),
                "devpass_s": float(_plan_t[9].item()),
                "save_s": float(_plan_t[10].item()),
                "safety_s": float(_plan_t[11].item()),
                "copy_s": float(_plan_t[12].item()),
                "persist_s": float(_plan_t[13].item()),
            }
            training_args.max_steps = _plan["planned_steps"]
            training_args.num_train_epochs = _plan["epochs"]
            training_args.warmup_steps = _plan["warmup"]
            if is_final_run:
                training_args.eval_steps = _plan["eval_steps"]
                training_args.save_steps = _plan["eval_steps"]
            # The final-save trigger: end_time minus the whole tail.
            when_to_eval_handler.save_before_remaining_time = _tail["reserve_s"] / 60.0
            if ckpt_avg is not None:
                ckpt_avg.soup_max = max(1, _tail["n_soup"])
                ckpt_avg.soup_deadline_ts = _tail["soup_eval_deadline_ts"]
            # First real checkpoint write is timed against save_s; a slower
            # disk moves the trigger earlier on every rank (customized_trainer).
            _eval_cb.sn56_tail = {
                "save_s": _tail["save_s"],
                "copy_factor": _tail["copy_s"] / max(1e-6, _tail["save_s"]),
                "n_saves_after": (1 if _tail["persist_s"] > 0 else 0) + (1 if _tail["devpass"] else 0),
                "handler": when_to_eval_handler,
            }
            if is_main_process(LOCAL_RANK):
                # for the late-attempt guard of a retry
                _s = get_state()
                _s["sn56_last_plan"] = {"t_per_step": float(t_per_step), "reserve_s": _tail["reserve_s"]}
                set_state(_s)

    # ── Time-aware epoch planning ──
    # Compute on rank 0, broadcast to all ranks to avoid NCCL deadlock.
    if t_per_step is not None and not _use_deepspeed and not _f_time_plan:
        _epoch_info = torch.tensor([0.0, 0.0, 0.0], device=next(model.parameters()).device)

        if is_main_process(LOCAL_RANK):
            _now = datetime.datetime.now(datetime.timezone.utc)
            _end_dt = datetime.datetime.strptime(
                train_request["end_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            remaining_secs = (_end_dt - _now).total_seconds()
            achievable_steps = int(remaining_secs * 0.85 / t_per_step)
            # Recompute from current batch size — may have been halved during LR search OOM
            _current_eff_bs = max(1,
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size)
            steps_per_epoch = len(train_ds) // _current_eff_bs

            if steps_per_epoch > 0:
                # Cap epochs: 4 for small datasets (<10K samples), 3 otherwise.
                # Beyond this, models memorize rather than generalize.
                _max_epochs = 4.0 if len(train_ds) < 10000 else 3.0
                # +25%: t_per_step is now the REAL post-search per-step time, so
                # achievable is accurate (only the 0.85 buffer is conservative). A
                # 1.25 stretch just spends that buffer — it plans slightly past the
                # deadline so the run uses the full budget, while keeping cosine
                # progress ~94% at the timer cut (LR ~min_lr, properly cooled). A
                # bigger stretch (e.g. 1.5) would leave the cosine at ~0.33x peak.
                # save_before_remaining_time is the real stop; floor 1.3 epochs so
                # even a tight job commits to a meaningful run.
                optimal_epochs = round(max(1.3, min(_max_epochs, 1.25 * achievable_steps / steps_per_epoch)), 2)
                new_total_steps = int(steps_per_epoch * optimal_epochs)
                warmup = min(training_args.warmup_steps, max(10, int(0.03 * new_total_steps)))
                eval_steps = float(training_args.eval_steps)
                if is_final_run:
                    # Eval every 1/8 epoch — frequent overfit detection and a
                    # denser snapshot trajectory for averaging. Evals before
                    # 0.75 epochs are skipped in the callback; the 10%-of-budget
                    # eval-timing governor widens this again if evals are slow.
                    eval_steps = float(max(20, steps_per_epoch // 8))
                _epoch_info[0] = optimal_epochs
                _epoch_info[1] = float(warmup)
                _epoch_info[2] = eval_steps

                log_info(
                    f"[sn56][relogio] Epochs {training_args.num_train_epochs} -> {optimal_epochs} "
                    f"(t_per_step={t_per_step:.3f}s, remaining={remaining_secs:.0f}s, "
                    f"achievable={achievable_steps}, steps/epoch={steps_per_epoch}, "
                    f"warmup={warmup}, eval_steps={int(eval_steps)})"
                )

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(_epoch_info, src=0)

        if _epoch_info[0] > 0:
            training_args.num_train_epochs = _epoch_info[0].item()
            training_args.warmup_steps = int(_epoch_info[1].item())
            if is_final_run:
                training_args.eval_steps = int(_epoch_info[2].item())
                training_args.save_steps = int(_epoch_info[2].item())

    # ── Dev-pass time budgeting ──
    # The end-time save now stops training, so the dev-pass actually runs — but
    # the stock T-3min trigger leaves it no room to finish. Move the trigger
    # earlier by the dev-pass's estimated cost: one epoch over dev at eval bs 1,
    # ~FLOP-equivalent per (packed) dev block to one train micro-batch row, so
    # per-block cost ≈ t_per_step / (grad_accum * per_device_bs). 2x safety for
    # the lost batch parallelism at bs 1, +120s for the weights-only save.
    # Deterministic across ranks (t_per_step was broadcast), so the trigger
    # fires in lockstep. Trades minutes of cosine-tail training (~min_lr, worth
    # little) for a full pass over held-out data.
    _dev_pass_secs = None
    _dev_pass_allowed = True
    _dev_safety_s = 0.0
    if _tail is not None:
        # F2 tail plan: the dev-pass budget and its deadline come from the plan
        # (it may have dropped the dev-pass when the tail would eat too much).
        _dev_pass_allowed = _tail["devpass"]
        _dev_pass_secs = _tail["devpass_s"] if _tail["devpass"] else None
        _dev_safety_s = _tail["safety_s"]
    elif is_final_run and _avg_mode == "ram" and len(dev_ds) > 0 and t_per_step is not None:
        _micro_cost = t_per_step / max(
            1,
            training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size,
        )
        _blocks_per_rank = len(dev_ds) / max(1, training_args.world_size)
        _dev_pass_secs = 2.0 * _blocks_per_rank * _micro_cost + 120
        _extra_min = min(15.0, max(2.0, _dev_pass_secs / 60))
        when_to_eval_handler.save_before_remaining_time = (
            train_request["save_before_remaining_time"] + _extra_min
        )
        log_info(
            f"[sn56][devfit] orçamento ~{_dev_pass_secs:.0f}s "
            f"({len(dev_ds)} blocos dev @ micro={_micro_cost:.3f}s/rank); "
            f"parada final em T-{when_to_eval_handler.save_before_remaining_time:.1f}min"
        )

    # ── Wire the KL reference model (after LR search / epoch planning) ──
    # LoRA: no separate model — compute_loss reads base logits via
    # model.disable_adapter() (exact + zero extra memory). Full-FT: load a frozen
    # copy of the original model on this rank's device, like the evaluator does.
    if use_kl and not training_args.use_lora:
        _kl_attn = "flash_attention_2" if not training_args.disable_fa else "eager"
        if training_args.use_attn_implementation:
            _kl_attn = training_args.use_attn_implementation
        _kl_device = next(model.parameters()).device
        trainer.base_model = load_base_model(train_request["model_path"], _kl_attn, _kl_device)
        log_info(f"[sn56][kl] modelo base congelado carregado em {_kl_device} (full-FT)")
    elif use_kl:
        log_info("[sn56][kl] usando disable_adapter do LoRA como base (sem cópia)")

    # ── F1 (SN56_EVAL_SKIP_FIX=1): eval-skip threshold from the CURRENT
    # steps/epoch and the planned total (the inherited rule divides a total
    # computed with E=3 by the re-planned E). ──
    _eff_bs_f = max(1, training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                    * training_args.world_size)
    _spe_f = max(1, len(train_ds) // _eff_bs_f)
    _total_f = (int(training_args.max_steps) if training_args.max_steps and training_args.max_steps > 0
                else int(_spe_f * training_args.num_train_epochs))
    if _f_eval_fix:
        _eval_cb.sn56_eval_skip_threshold = sn56_plan.eval_skip_threshold(
            _spe_f, _total_f, training_args.num_train_epochs)
        # SN56_EVAL_FROM (epochs; second run of SN56_R1_COOLDOWN): evals start no later than this
        _eval_from = sn56_plan.env_pos_float("SN56_EVAL_FROM")
        if _eval_from is not None:
            _eval_cb.sn56_eval_skip_threshold = min(_eval_cb.sn56_eval_skip_threshold,
                                                    int(_eval_from * _spe_f))
        _eval_cb.total_steps_all_epochs = _total_f
    if _sn56_on:
        _eval_cb.sn56_planned_steps = _total_f
        # One line with every flag and the resulting plan (visible with
        # SN56_VERBOSE=1, like every other log line here).
        _thr_f = (_eval_cb.sn56_eval_skip_threshold if _eval_cb.sn56_eval_skip_threshold is not None
                  else sn56_plan.legacy_eval_skip_threshold(total_steps_all_epochs, training_args.num_train_epochs))
        _es_f = int(training_args.eval_steps) if training_args.eval_steps else 0
        _pts_f = sn56_plan.eval_points(_total_f, _es_f, _thr_f, _spe_f, periodic_save_steps) if _es_f else []
        _save_ts_f = (sn56_plan.parse_utc(train_request["end_time"])
                      - 60.0 * float(when_to_eval_handler.save_before_remaining_time))
        log_info(
            f"[sn56][plan] flags={sn56_plan.flags_summary()} "
            f"lr={trainer.args.learning_rate:.3e} ({_lr_note or 'cache'}) "
            f"t_step={(f'{t_per_step:.3f}s' if t_per_step else 'n/a')} "
            f"mode={'time' if _plan else ('legacy' if t_per_step else 'none')} "
            f"steps={_total_f} E={training_args.num_train_epochs:.2f} steps/epoch={_spe_f} "
            f"cap={sn56_plan.max_epochs():g}ep capped={bool(_plan and _plan['capped'])} "
            f"warmup={training_args.warmup_steps} eval_every={_es_f} eval_skip<{_thr_f} "
            f"first_eval={_pts_f[0] if _pts_f else 'end'} evals~{len(_pts_f)} "
            f"final_save={sn56_plan.fmt_utc(_save_ts_f)} "
            f"(T-{float(when_to_eval_handler.save_before_remaining_time):.1f}min) "
            f"tail={'%.0fs' % _tail['reserve_s'] if _tail else 'legacy'} "
            f"soup={_tail['n_soup'] if _tail else '-'} devpass={_tail['devpass'] if _tail else '-'} "
            f"lr_search={'on' if _f_lr_search else 'off'} "
            f"lr_search_s={'%.0f' % _lr_search_s if _lr_search_s is not None else '-'} "
            f"lr_mult={sn56_plan.flag_float('SN56_LR_MULT'):g} "
            f"bs={training_args.per_device_train_batch_size} ga={training_args.gradient_accumulation_steps} "
            f"neftune={training_args.neftune_noise_alpha} per_example_loss={_f_per_example}"
            f"{'/' + getattr(trainer, '_f6_mode', '') if _f_per_example else ''} "
            f"alpha={_f6_alpha:g} dev_metric={'vali' if _f_dev_vali else 'stock'} gain={_f_gain} "
            f"scale_repair={_f_scale_repair} "
            f"v5={_v5_mode or '-'}"
        )
    else:
        log_info(f"[sn56][plan] correções F1-F4 inativas (tarefa={_task_type}, lora={training_args.use_lora}, "
                 f"deepspeed={_use_deepspeed}) — caminho herdado; per_example_loss={_f_per_example}"
                 f"{'/' + getattr(trainer, '_f6_mode', '') if _f_per_example else ''} alpha={_f6_alpha:g} "
                 f"dev_metric={'vali' if _f_dev_vali else 'stock'} scale_repair={_f_scale_repair}")

    log_info(f"fiscalização a cada {training_args.eval_steps} passos, {len(dev_ds)} na prova")
    try:
        trainer.train()
    except Exception as _train_err:
        # ── Crash guard (F2) ──
        # A crash inside the loop (25 Sep: ENOSPC in a periodic checkpoint
        # write) used to escape to text_trainer, which re-ran the whole script
        # and re-planned the minutes left. Once this loop has written a valid
        # submission there is nothing a re-plan can add: keep it, report
        # success, exit. OOM keeps its inherited path (batch halving in
        # text_trainer); a crash before any submission exists is retried as
        # before.
        _is_oom = isinstance(_train_err, getattr(torch, "OutOfMemoryError", ()) or ()) or \
            "out of memory" in str(_train_err).lower()
        if _sn56_on and not _is_oom and sn56_plan.submission_is_valid(train_request["submission_dir"]):
            log_info(
                f"[sn56][guard] treino interrompido no passo {trainer.state.global_step} "
                f"({type(_train_err).__name__}: {str(_train_err)[:160]}); submissão válida em "
                f"{train_request['submission_dir']} mantida, sem nova tentativa"
            )
            if is_main_process(LOCAL_RANK):
                with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
                    f.write("Success")
            log_info("missão cumprida (guard)", "finish")
            return
        raise

    # SN56_R1_AVG2, second run: try the 50:50 average with the first run's
    # pre-dev-pass weights (SN56_AVG_WITH, set by sn56_retry); kept only if the
    # validator-form dev metric improves (sn56_retry.try_blend).
    _avg_with = os.environ.get("SN56_AVG_WITH", "").strip()
    if (_avg_with and ckpt_avg is not None and training_args.world_size == 1
            and is_main_process(LOCAL_RANK)):
        try:
            from sn56_retry import try_blend
            try_blend(trainer, _avg_with, ckpt_avg, train_request["submission_dir"], log=log_info)
        except Exception as _e:
            log_info(f"[sn56][avg2] falhou ({_e!r}), mantendo a 2a corrida")

    # Run record for the dev-selected second run (sn56_retry.py, SN56_LR_RETRY):
    # the pre-dev-pass dev metric (soup / best checkpoint, the same number both
    # runs are compared on), the LR used, the search argmin and the wall-clock.
    if is_main_process(LOCAL_RANK):
        try:
            import lr_search as _lrs
            _best_dev = None
            if ckpt_avg is not None and getattr(ckpt_avg, "best_loss", None) is not None:
                _best_dev = float(ckpt_avg.best_loss)
            _best_step = sn56_plan.best_step_from_loss_txt(train_request["submission_dir"])
            _early = int(trainer.state.global_step) < 0.9 * max(1, int(training_args.max_steps or 0))
            _pre_dir = None
            if (sn56_plan.flag_on("SN56_R1_AVG2") and sn56_plan.flag_on("SN56_LR_RETRY") and _early
                    and sn56_plan.lr_fixed() is None and training_args.world_size == 1):
                # the first run's pre-dev-pass weights, for the second run's average
                _pre_dir = os.path.join(training_args.output_dir, "pre_devpass")
                shutil.rmtree(_pre_dir, ignore_errors=True)
                shutil.copytree(train_request["submission_dir"], _pre_dir)
            sn56_plan.write_run_record(training_args.output_dir, {
                "task_type": _task_type, "sn56_on": bool(_sn56_on), "use_kl": bool(use_kl),
                "dev_metric": "vali" if _f_dev_vali else "stock", "dev": _best_dev,
                "lr": float(trainer.args.learning_rate), "lr_fixed": sn56_plan.lr_fixed(),
                "lr_argmin": getattr(_lrs, "LAST_ARGMIN", None) if _lr_search_s is not None else None,
                "lr_search_s": _lr_search_s, "t_start": _t_main0, "t_train_end": time.time(),
                "global_step": int(trainer.state.global_step), "max_steps": int(training_args.max_steps or -1),
                "submission_dir": train_request["submission_dir"], "end_time": train_request["end_time"],
                "best_step": _best_step, "spe": int(_spe_f), "stopped_early": bool(_early),
                "pre_devpass_dir": _pre_dir, "world_size": int(training_args.world_size),
            })
        except Exception as _e:
            log_info(f"[sn56][lr2] registro da corrida falhou ({_e!r})")

    # SN56_LOGIT_CAL: fit one logit scale on dev with the validator's metric BEFORE the
    # dev-pass trains on dev; folded into the submission files after it (sn56_logit_cal.py).
    _logit_cal = None
    if _feat["logit_cal"] and training_args.world_size == 1 and is_main_process(LOCAL_RANK):
        try:
            from sn56_logit_cal import fit_on_trainer
            _logit_cal = fit_on_trainer(trainer, log=log_info)
        except Exception as _e:
            log_info(f"[sn56][logitcal] falhou ({_e!r}), sem calibração")
            _logit_cal = None

    # ── Final dev-data pass (data maximization on small datasets) ──
    # The dev split was held out only for selection; the scored test set is
    # separate. So reclaim it: one low-LR epoch over dev from the best checkpoint,
    # then weights-only save. Best-effort — any failure leaves the best
    # checkpoint untouched. Gated to RAM averaging (model reliably holds best
    # weights after train) and to datasets where dev is a meaningful fraction.
    try:
        _dev_n = len(dev_ds)
        # Window check: only start a dev-pass that can finish. The atomic swap
        # in dev_pass protects the submission either way; this just avoids
        # burning the final minutes on a doomed pass (e.g. when training
        # stopped via overfit early-stop near the deadline, or the budgeting
        # above never ran on a cached-LR/DeepSpeed path).
        _dev_window_ok = True
        if not _dev_pass_allowed:
            _dev_window_ok = False
            log_info("[sn56][devfit] sem orçamento no plano da cauda, pulando")
        elif _dev_pass_secs is not None:
            _end_dt = datetime.datetime.strptime(
                train_request["end_time"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            _remaining = (_end_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            # must finish AND leave the safety margin before end_time
            _dev_window_ok = _remaining > _dev_pass_secs + _dev_safety_s
            if not _dev_window_ok:
                log_info(
                    f"[sn56][devfit] sem janela ({_remaining:.0f}s restantes < "
                    f"~{_dev_pass_secs:.0f}s necessários + {_dev_safety_s:.0f}s de margem), pulando"
                )
        if is_final_run and _avg_mode == "ram" and _dev_n > 0 and _dev_window_ok:
            from dev_pass import run_dev_pass
            _min_lr_rate = 0.25
            _kw = training_args.lr_scheduler_kwargs or {}
            if isinstance(_kw, str):
                try:
                    _kw = json.loads(_kw)
                except (json.JSONDecodeError, ValueError):
                    _kw = {}
            try:
                _min_lr_rate = float(_kw.get("min_lr_rate", 0.25))
            except (AttributeError, TypeError, ValueError):
                pass
            _min_lr = trainer.args.learning_rate * _min_lr_rate
            log_info(
                f"[sn56][devfit] {_dev_n} amostras de dev, 1 época @ lr={_min_lr:.2e}"
            )
            run_dev_pass(
                trainer,
                submission_dir=train_request["submission_dir"],
                min_lr=_min_lr,
                max_grad_norm=training_args.max_grad_norm,
                train_per_device=training_args.per_device_train_batch_size,
                train_grad_accum=training_args.gradient_accumulation_steps,
                local_rank=LOCAL_RANK, log=log_info,
            )
        else:
            log_info(
                f"[sn56][devfit] pulando (final={is_final_run}, modo={_avg_mode}, "
                f"dev={_dev_n})"
            )
    except Exception as _e:
        log_info(f"[sn56][devfit] dev-pass falhou ({_e}), mantendo melhor checkpoint")

    if _logit_cal and _logit_cal.get("a", 1.0) != 1.0:
        try:
            from sn56_logit_cal import apply_to_submission
            apply_to_submission(train_request["submission_dir"], _logit_cal["names"], _logit_cal["a"], log=log_info)
        except Exception as _e:
            log_info(f"[sn56][logitcal] gravação falhou ({_e!r}); submissão sem calibração")

    if is_main_process(LOCAL_RANK):
        success_file = os.path.join(training_args.output_dir, "success.txt")
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("missão cumprida", "finish")

if __name__ == "__main__":
    main()
