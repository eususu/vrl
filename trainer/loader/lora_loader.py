import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


def lora_load():
    from trainer.config import get_config
    project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()
    if not isinstance(lora_config, LoraConfig):
        raise Exception("lora_config does not instance of LoraConfig")


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True if model_info.load_in_n_bit == 4 else False,
        load_in_8bit=True if model_info.load_in_n_bit == 8 else False,
    )

    model = AutoModelForCausalLM.from_pretrained(model_info.base_name,
      torch_dtype=torch.bfloat16,
      quantization_config=quantization_config,
      device_map="auto",
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, tokenizer
