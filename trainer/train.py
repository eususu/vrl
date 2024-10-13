import logging
import os
from datasets import Dataset, concatenate_datasets
import wandb


from .dpo_train import dpo_train
from .chat_template.detect import find_chat_template
from .preprocess import preprocess
from .config import get_config

model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()

logging.basicConfig(level=logging.INFO)

###############################################################################################

def train():
    # 1단계: 채팅 템플릿 확인
    chat_template = find_chat_template(model_info.base_name)
    logging.info(f"FOUND CHAT TEMPLATE ({chat_template})")

    # 2단계: 데이터셋을 채팅 템플릿과 튜닝목적에 맞게 변형
    ds_list = []
    for ds_name in datasets:
        ds=preprocess(ds_name=ds_name, target_template_name=chat_template)
        ds_list.append(ds)
    ds = concatenate_datasets(ds_list)
    print(ds)

    if wandb_config is not None:
        wandb.init(project=wandb_config.project, name=wandb_config.name)
    try:
        # 3단계: 학습 시작
        dpo_train(ds)
    finally:
        if wandb_config is not None:
            wandb.finish()
if __name__ == "__main__":
    train()
