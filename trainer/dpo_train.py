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

  trainer.accelerator.print(f'{trainer.model}')
  if model_info.load_in_n_bit < 16:
    # peft lora
    trainer.model.print_trainable_parameters()

    if getattr(trainer.accelerator.state, 'fsdp_plugin', None):
      from peft.utils.other import fsdp_auto_wrap_policy
      fsdp_plugin = trainer.accelerator.state.fsdp_plugin
      fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

  trainer.train()

  if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
  trainer.save_model()
