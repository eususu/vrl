
import os

def unsloth_load():
    os.environ["XFORMERS_FLASH_ATTENTION"] = "1"
    from ..config import get_config

    project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()
    if unsloth_config is not None:
        from unsloth import FastLanguageModel, PatchDPOTrainer
        PatchDPOTrainer()
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = unsloth_config.base_model_name,
        max_seq_length = unsloth_config.max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=unsloth_config.r,
        target_modules=unsloth_config.target_modules,
    )

    return model, tokenizer
