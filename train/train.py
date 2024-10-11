import os
from datasets import Dataset, concatenate_datasets
import wandb
from preprocess import preprocess

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from config import get_config

model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()
if unsloth_config is not None:
    from unsloth import FastLanguageModel, PatchDPOTrainer
    PatchDPOTrainer()
import torch

###############################################################################################

def unsloth_load():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = unsloth_config.model_name,
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

def lora_load():
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

def normal_load():
    model = AutoModelForCausalLM.from_pretrained(
        model_info.base_name,
        torch_dtype=torch.bfloat16,
        device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(
        model_info.base_name,
        trust_remote_code=True)

    return model, tokenizer

def dpo_train(ds:Dataset):

    os.environ["XFORMERS_FLASH_ATTENTION"] = "1"

    if unsloth_config is not None:
        model, tokenizer = unsloth_load()
    else:
        if model_info.load_in_n_bit == 16:
            print(f"### training ###")
            model, tokenizer = normal_load()
        else:
            print(f"### LORA {model_info.load_in_n_bit} bit training ###")
            model, tokenizer = lora_load()

    trainer = DPOTrainer(model,
               args=training_args,
               train_dataset=ds,
               tokenizer=tokenizer,
               )

    trainer.train()
    trainer.save_model()

def train():

    ds_list = []
    for ds_name in datasets:
        ds=preprocess(ds_name=ds_name)
        ds_list.append(ds)
    ds = concatenate_datasets(ds_list)
    print(ds)

    if wandb_config is not None:
        wandb.init(project=wandb_config.project, name=wandb_config.name)
    try:
        dpo_train(ds)
    finally:
        if wandb_config is not None:
            wandb.finish()
if __name__ == "__main__":
    train()
