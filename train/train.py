import os
from datasets import Dataset, concatenate_datasets
import wandb
from preprocess import preprocess

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from config import get_config
from loader.lora_loader import lora_load
from loader.unsloth_loader import unsloth_load

model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()
if unsloth_config is not None:
    from unsloth import FastLanguageModel, PatchDPOTrainer
    PatchDPOTrainer()
import torch

###############################################################################################
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
        ds=preprocess(ds_name=ds_name, target_template_name='gemma')
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
