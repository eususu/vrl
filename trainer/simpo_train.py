import os

from .extend.simpo.simpo_trainer import SimPOTrainer
from .extend.simpo.simpo_config import SimPOConfig

from .config import get_config
from datasets import Dataset
from .loader.load import load

project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()


def simpo_train(ds:Dataset):

  if not isinstance(training_args, SimPOConfig):
    raise ValueError(f"input config is bad instance (need SimPOConfig but {training_args})")

  model, tokenizer = load()

  trainer = SimPOTrainer(model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    )

  trainer.train()
  trainer.save_model()

