import os

from .config import get_config
from datasets import Dataset
from .loader.load import load

from trl import DPOTrainer, DPOConfig

project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()


def dpo_train(ds:Dataset):
  if not isinstance(training_args, DPOConfig):
    raise ValueError(f"input config is bad instance (need DPOConfig but {training_args})")

  model, tokenizer = load()

  trainer = DPOTrainer(model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    )

  trainer.train()
  trainer.save_model()
