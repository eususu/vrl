from ..config import get_config
model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()

from .normal_loader import normal_load
from .lora_loader import lora_load
from .unsloth_loader import unsloth_load


def load():
  if unsloth_config is not None:
      model, tokenizer = unsloth_load()
  else:
      if model_info.load_in_n_bit == 16:
          print(f"### training ###")
          model, tokenizer = normal_load()
      else:
          print(f"### LORA {model_info.load_in_n_bit} bit training ###")
          model, tokenizer = lora_load()

  return model, tokenizer