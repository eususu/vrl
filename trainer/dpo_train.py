import os

from .loader.normal_loader import normal_load
from .config import get_config
from datasets import Dataset, concatenate_datasets
from .loader.lora_loader import lora_load
from .loader.unsloth_loader import unsloth_load

from trl import DPOTrainer, DPOConfig

model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()


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
