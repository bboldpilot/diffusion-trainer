"""Fail-closed build patch for ai-toolkit pin 99be3d96 (FLUX loader only)."""
from pathlib import Path

OLD_0 = """            subfolder = 'transformer'
            transformer_path = model_path
            local_files_only = False
            # check if HF_DATASETS_OFFLINE or TRANSFORMERS_OFFLINE is set
            if os.path.exists(transformer_path):
                subfolder = None
                transformer_path = os.path.join(transformer_path, 'transformer')
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, 'text_encoder')
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                subfolder=subfolder,
                torch_dtype=dtype,
                # low_cpu_mem_usage=False,
                # device_map=None
            )
"""

NEW_0 = """            # [sn56_flux_single_file] Preserve every component supplied by the task.
            _flux_single = None
            if os.path.isfile(model_path):
                from flux_single_file import load_pipeline
                _flux_single = load_pipeline(model_path, dtype)
                transformer = _flux_single.transformer
            else:
                subfolder = 'transformer'
                transformer_path = model_path
                local_files_only = False
                # check if HF_DATASETS_OFFLINE or TRANSFORMERS_OFFLINE is set
                if os.path.exists(transformer_path):
                    subfolder = None
                    transformer_path = os.path.join(transformer_path, 'transformer')
                    # check if the path is a full checkpoint.
                    te_folder_path = os.path.join(model_path, 'text_encoder')
                    # if we have the te, this folder is a full checkpoint, use it as the base
                    if os.path.exists(te_folder_path):
                        base_model_path = model_path

                transformer = FluxTransformer2DModel.from_pretrained(
                    transformer_path,
                    subfolder=subfolder,
                    torch_dtype=dtype,
                    # low_cpu_mem_usage=False,
                    # device_map=None
                )
"""

OLD_1 = """            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
            self.print_and_status_update("Loading VAE")
            if self.model_config.vae_path is not None:
                vae = load_vae(self.model_config.vae_path, dtype)
            else:
                vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
            flush()
            
            self.print_and_status_update("Loading T5")
            tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_2", torch_dtype=dtype)
            text_encoder_2 = T5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder_2",
                                                            torch_dtype=dtype)

            text_encoder_2.to(self.device_torch, dtype=dtype)
            flush()

            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing T5")
                quantize(text_encoder_2, weights=get_qtype(self.model_config.qtype))
                freeze(text_encoder_2)
                flush()
                
            self.print_and_status_update("Loading CLIP")
            text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", torch_dtype=dtype)
            text_encoder.to(self.device_torch, dtype=dtype)

"""

NEW_1 = """            scheduler = (_flux_single.scheduler if _flux_single is not None else FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler"))
            self.print_and_status_update("Loading VAE")
            if self.model_config.vae_path is not None:
                vae = load_vae(self.model_config.vae_path, dtype)
            else:
                vae = (_flux_single.vae if _flux_single is not None else AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype))
            flush()
            
            self.print_and_status_update("Loading T5")
            tokenizer_2 = (_flux_single.tokenizer_2 if _flux_single is not None else T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_2", torch_dtype=dtype))
            text_encoder_2 = (_flux_single.text_encoder_2 if _flux_single is not None else T5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder_2",
                                                            torch_dtype=dtype))

            text_encoder_2.to(self.device_torch, dtype=dtype)
            flush()

            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing T5")
                quantize(text_encoder_2, weights=get_qtype(self.model_config.qtype))
                freeze(text_encoder_2)
                flush()
                
            self.print_and_status_update("Loading CLIP")
            text_encoder = (_flux_single.text_encoder if _flux_single is not None else CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype))
            tokenizer = (_flux_single.tokenizer if _flux_single is not None else CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", torch_dtype=dtype))
            text_encoder.to(self.device_torch, dtype=dtype)

"""

REPLACEMENTS = [(OLD_0, NEW_0), (OLD_1, NEW_1)]

def patch_source(source):
    if '[sn56_flux_single_file]' in source:
        return source
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            raise RuntimeError('FLUX loader anchor changed; refusing a partial patch')
        source = source.replace(old, new, 1)
    compile(source, '<patched flux loader>', 'exec')
    return source

if __name__ == '__main__':
    path = Path('/app/ai-toolkit/toolkit/stable_diffusion_model.py')
    path.write_text(patch_source(path.read_text()))
