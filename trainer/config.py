import os
from typing import List, Optional
from pydantic import BaseModel
from trl import DPOConfig
from peft import LoraConfig
from .extend.simpo.simpo_config import SimPOConfig


class ProjectConfig(BaseModel):
    project_name:str
    output_path:str
    hub_model_id:Optional[str]
    noti_google_chat_url:Optional[str]=None

class ModelConfig(BaseModel):
    base_name:str
    attn_implementation:Optional[str]
    load_in_n_bit:int

class WandBConfig(BaseModel):
    project:str
    name:str

class UnslothConfig(BaseModel):
    base_model_name:str
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

import socket
hostname = socket.gethostname()
username = os.getenv('USER') or os.getenv('USERNAME')
run_name = f'{hostname}_{username}'

import importlib
try:
    user_config = importlib.import_module('trainer.user_config')
except Exception as e:
    print("## need user_config.py for you")
    print("## copy user_config.py from user_config.py_template")
    print("## and edit it")

def get_config():
    user_configs = user_config.get_config(run_name=run_name)
    #validate return values

    return user_configs
def example_get_config():
    project_info = ProjectConfig(
        project_name="example project name",
        output_path='train_output',
        noti_google_chat_url=None, # fill with your noti link
    )
    # 학습 대상 모델 기본 설정
    model_info = ModelConfig(
        base_name= 'google/gemma-2-2b-it',
        attn_implementation="eager", # gemma must be eager
        load_in_n_bit= 8,
    )

    # unsloth 설정
    unsloth_config = UnslothConfig(
        base_model_name=model_info.base_name,
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
    # unsloth_config = None

    # WANDB 설정
    wandb_config = WandBConfig(
        project="dpotest",
        name="vrl_user",
    )

    datasets = [
    'kuotient/orca-math-korean-dpo-pairs',
    'aiyets/argilla_dpo-mix-7k-ko',
    ]

    # TRL 설정
    training_args = DPOConfig(
        project_info.output_path,
        hub_model_id=None if project_info.hub_model_id is None else project_info.hub_model_id,
        save_total_limit=2,
        max_length = 2048,
        max_prompt_length = 2048,
        beta=0.1,
        warmup_ratio=0.1,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # deepspeed를 사용하면 여기서 옵션 빼던가, deepspeed거 쓰던거 해야함
        gradient_checkpointing=True,
        remove_unused_columns=False,
        bf16=True,
        logging_steps=1,
        seed=42,
        optim="paged_adamw_32bit",
        learning_rate=5e-7,
        lr_scheduler_type='cosine',
        dataset_num_proc=os.cpu_count() - 1, # dataset 크기가 커지면 꼭 필요함
        report_to="wandb",
        run_name="vrl_user",
        push_to_hub=True,
        hub_strategy='checkpoint',
        hub_private_repo=True, # 저장소 private
        )
    atraining_args = SimPOConfig(
        project_info.output_path,
        hub_model_id=None if project_info.hub_model_id is None else project_info.hub_model_id,
        save_total_limit=2,
        max_length = 2048,
        max_prompt_length = 2048,
        warmup_ratio=0.1,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        seed=42,
        optim="paged_adamw_32bit",
        learning_rate=8e-7,
        beta=10,
        gamma_beta_ratio=0.5,
        lr_scheduler_type='cosine',
        ### 잘 안바꾸는건 뒤에 몰아주기
        gradient_checkpointing=True,
        remove_unused_columns=False,
        bf16=True,
        logging_steps=1,
        dataset_num_proc=os.cpu_count() - 1, # dataset 크기가 커지면 꼭 필요함
        report_to="wandb",
        run_name="vrl_user",
        push_to_hub=True,
        hub_strategy='checkpoint',
        hub_private_repo=True, # 저장소 private
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # dict로 반환하고 싶지만, 직렬화가 허용된 객체가 아니라 이게 최선임
    return (
        project_info,
        model_info,
        unsloth_config,
        wandb_config,
        datasets,
        training_args,
        lora_config,
        )


if __name__ == "__main__":
    print(get_config())