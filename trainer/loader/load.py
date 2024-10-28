import torch
from ..config import get_config
project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()

from .normal_loader import normal_load
from .lora_loader import lora_load
from .unsloth_loader import unsloth_load

from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

accelerator = Accelerator()
device_id = accelerator.local_process_index


def load():
  if unsloth_config is not None:
      model, tokenizer = unsloth_load()
  else:
      if model_info.load_in_n_bit == 16:
          print(f"### training ###")
          model, tokenizer = normal_load(device_id=device_id)
      else:
          print(f"### LORA {model_info.load_in_n_bit} bit training ###")
          model, tokenizer = lora_load(device_id=device_id)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device) # multi gpu 환경에서 종종 이게 필요함

  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'right'

  return model, tokenizer