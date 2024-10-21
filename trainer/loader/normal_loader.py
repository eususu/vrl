import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..config import get_config
project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()

def normal_load():
    model = AutoModelForCausalLM.from_pretrained(
        model_info.base_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="flash_attention_2",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_info.base_name,
        trust_remote_code=True)

    return model, tokenizer
