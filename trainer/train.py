import logging
import sys
import requests
from datasets import Dataset, concatenate_datasets
import wandb

from trl import DPOConfig
from .extend.simpo.simpo_config import SimPOConfig

from .dpo_train import dpo_train
from .simpo_train import simpo_train
from .chat_template.detect import find_chat_template
from .preprocess import preprocess
from .config import get_config

project_info, model_info, unsloth_config, wandb_config, datasets, training_args, lora_config = get_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def noti():

    url = project_info.noti_google_chat_url
    if url is None:
        print("noti was disabled")

    message = {
        'text': f'{project_info.project_name} 학습 종료'
    }

    res = requests.post(url, json=message)

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
        if isinstance(training_args, DPOConfig):
            dpo_train(ds)
        elif isinstance(training_args, SimPOConfig):
            simpo_train(ds)
    finally:
        if wandb_config is not None:
            wandb.finish()

        noti()
if __name__ == "__main__":
    train()
