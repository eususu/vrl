import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


def lora_load(device_id):
    from trainer.config import get_config
    project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()
    if not isinstance(lora_config, LoraConfig):
        raise Exception("lora_config does not instance of LoraConfig")


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True if model_info.load_in_n_bit == 4 else False,
        load_in_8bit=True if model_info.load_in_n_bit == 8 else False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_info.base_name,
      torch_dtype=torch.bfloat16,
      quantization_config=quantization_config,
      device_map=device_id,
      attn_implementation=model_info.attn_implementation,
      )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    # LoRA 적용
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_info.base_name,
        trust_remote_code=True,
        )

    return model, tokenizer
