import argparse
import logging
import os
import sys
import re
from typing import List
from pydantic import BaseModel

from ._exceptions import AlreadyExistInstance
from ._types import Colors, Instance, Offer, RentOptions, RentState
from .vastapi import VastAPI, read_ssh_key

logging.basicConfig(level=logging.INFO)
import asyncssh

# asyncssh 로그 레벨을 WARNING으로 설정하여 불필요한 로그를 숨깁니다
asyncssh.logging.set_log_level(logging.WARNING)
logging.getLogger('paramiko').setLevel(logging.WARNING)

  
class VRL():
  api:VastAPI
  selected_offer:Offer

  def __init__(self):
    self.api = VastAPI()

  def __init_container(self, options:RentOptions, init_commands:List[str]=None):
    try:
      if self.api.is_running():
        instance = self.api.get_instance()
        if instance is not None:
          raise AlreadyExistInstance()

      self.api.search_offer(options.favor_gpu, options.num_gpus, min_down=options.min_down, min_up=options.min_up)
      self.api.create_instance(title=options.title, disk=options.disk)
    except AlreadyExistInstance as ae:
      pass

    from tqdm import tqdm
    import time

    start_time = time.time()
    with tqdm(desc=f"인스턴스(GPU={options.favor_gpu} x{options.num_gpus}) 준비 중", unit="초") as pbar:
        while True:
            instance = self.api.get_instance()
            pbar.set_postfix_str(f"상태: {instance.Status}")

            if instance.Status == 'running':
                break

            if time.time() - start_time > options.init_timeout:
                logging.warning(f"초기화 시간 초과: {options.init_timeout}초")
                break

            time.sleep(1)
            pbar.update(1)
    
    if instance.Status == 'running':
        logging.info(f'## {Colors.CYAN.to_str("인스턴스가 준비되었습니다")}')
    else:
        logging.error(f'## {Colors.RED.to_str("인스턴스 준비 실패")}')
        return

    ssh_key, pkey = read_ssh_key()
    self.api.init_ssh(ssh_key=ssh_key, pkey=pkey)
    self.api.launch_jobs(jobs=init_commands)


  def status(self):
    self.api.print_rent_state()

  def shell(self, cmd:str):
    return self.api.shell(cmd=cmd)

  def search(self, gpu_name:str, num_of_gpu:int, min_down:int):
    try:
      self.api.search_offer(gpu_name, num_of_gpu, min_down=min_down)
    except AlreadyExistInstance as ae:
      print(ae)

  def rent(self, options:RentOptions):
    token = os.getenv("HF_TOKEN")
    openai_apikey = os.getenv("OPENAI_API_KEY")
    wandb_apikey = os.getenv("WANDB_API_KEY")
    self.__init_container(options)

    commands = [ ]
    commands.append('echo export HF_HUB_ENABLE_HF_TRANSFER=1 >> ~/.profile')

    if token:
      logging.info('Passing HF_TOKEN to remote')
      commands.append(f'echo export HF_TOKEN={token} >> ~/.profile')
    if wandb_apikey:
      logging.info('Passing WANDB_API_KEY to remote')
      commands.append(f'echo export WANDB_API_KEY={wandb_apikey} >> ~/.profile')
    if openai_apikey:
      logging.info('Passing OPENAI_API_KEY to remote')
      commands.append(f'echo export OPENAI_API_KEY={openai_apikey} >> ~/.profile')

    #commands.append('pip install accelerate trl peft wandb')

    self.api.launch_jobs(jobs=commands)

    if options.auto_connect:
      self.ssh()

  def ssh(self):
    if not self.api.is_running():
      self.api.print_rent_state()
      exit(1)

    import subprocess

    cmds =[
      'ssh',
      '-oStrictHostKeyChecking=no',
      self.api.sshurl()
    ]
    logging.info(cmds)
    subprocess.call(cmds)

  def scp(self, remote:str, local:str):
    self.api.scp(remote, local)

  def logickor(self, model_name:str, lora_adapter:str=None, lora_revision:str=None):
    init_commands = [
      'git clone https://github.com/eususu/LogicKor.git'
    ]
    self.__init_container(init_commands=init_commands)


    extra_options = []
    if lora_adapter is not None:
      extra_options.append('-lm')
      extra_options.append(lora_adapter)

      if lora_revision is not None:
        extra_options.append('-lmr')
        extra_options.append(lora_revision)

    if len(extra_options) == 0:
      more_arg = ''
    else:
      more_arg = " ".join(extra_options)

    commands = [
      f'cd LogicKor && python3 cli.py -m {model_name} {more_arg} -f',
      #'SCP LogicKor/evaluated/*.jsonl .',
    ]
    self.api.launch_jobs(jobs=commands)

  def stop(self):
    self.api.destroy_instance()
