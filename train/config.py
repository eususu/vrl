import os
from typing import List, Optional
from pydantic import BaseModel
from trl import DPOConfig
from peft import LoraConfig

class ModelConfig(BaseModel):
    base_name:str
    load_in_n_bit:int

class WandBConfig(BaseModel):
    project:str
    name:str

class UnslothConfig(BaseModel):
    model_name:str
    r:int
    target_modules:List[str]
    lora_alpha:int
    lora_dropout:float
    bias:str
    layers_to_transform:Optional[List[str]]
    layers_pattern:Optional[List[str]]
    use_gradient_checkpointing:bool
    random_state:int
    max_seq_length:int
    use_rslora:bool=False
    modules_to_save:str=None
    init_lora_weights:bool=True
    loftq_config:dict={}
    temporary_location:str="_unsloth_temporary_saved_buffers"

    pass

def get_config():
    model_info = ModelConfig(
        base_name= 'google/gemma-2-2b-it',
        load_in_n_bit= 4,
    )

    unsloth_config = UnslothConfig(
        model_name=model_info.base_name,
        r                   = 16,
        target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        lora_alpha          = 16,
        lora_dropout        = 0,
        bias                = "none",
        layers_to_transform = None,
        layers_pattern      = None,
        use_gradient_checkpointing = True,
        random_state        = 3407,
        max_seq_length      = 512,
    )

    wandb_config = WandBConfig(
        project="dpotest",
        name="vrl_user",
    )
    wandb_config = None

    datasets = [
    #'kuotient/orca-math-korean-dpo-pairs',
    'aiyets/argilla_dpo-mix-7k-ko',
    ]

    training_args = DPOConfig(
        './dpo_result',
        max_length = 128, #4096+512,
        max_prompt_length = 64, #4096,
        beta=0.1,
        warmup_ratio=0.1,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        bf16=True,
        logging_steps=1,
        seed=42,
        optim="paged_adamw_32bit",
        learning_rate=5e-7,
        lr_scheduler_type='cosine',
        dataset_num_proc=os.cpu_count() - 1, # dataset 크기가 커지면 꼭 필요함
        report_to=None if wandb_config is None else "wandb",
        run_name="vrl_user",
        push_to_hub=True,
        hub_model_id='aiyets/gemma-2-2b-it-dpo-dual_001',
        hub_strategy='checkpoint',
        )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return (
        model_info,
        unsloth_config,
        wandb_config,
        datasets,
        training_args,
        lora_config,
        )
