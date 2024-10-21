import logging
import os
import re
from typing import List
from pydantic import BaseModel

from _exceptions import AlreadyExistInstance
from _types import Instance, Offer, VRLOptions
from vastapi import VastAPI, read_ssh_key



logging.basicConfig(level=logging.INFO)


  
class VRL():
  options:VRLOptions
  api:VastAPI
  selected_offer:Offer

  def __init__(self, options:VRLOptions):
    self.api = VastAPI(options=options)
    self.options = options

    self.__assume_gpu_ram()

  def __assume_gpu_ram(self):
    BILLIONS = 1000000000
    IN_GB = 1024 * 1024 * 1024
    ram = self.options.lm_parameter * BILLIONS

    ram_in_32 = int(ram * (4 + 20) / IN_GB)
    ram_in_16 = int(ram * (2 + 20) / IN_GB)
    ram_in_8 = int(ram * (1 + 20) / IN_GB)
    logging.info(f'[32bit float] Assume GPU RAM is {ram_in_32} GB')
    logging.info(f'[16bit float] Assume GPU RAM is {ram_in_16} GB')
    logging.info(f'[8bit float] Assume GPU RAM is {ram_in_8} GB')

  def __init_container(self, init_commands:List[str]=None):
    try:
      if self.api.running_cid is not None:
        raise AlreadyExistInstance()

      self.api.search_offer(self.options.favor_gpu, self.options.num_gpus)
      self.api.create_instance()
    except AlreadyExistInstance as ae:
      pass

    cid = self.api.running_cid

    while(True):
      instance = self.api.get_instance(cid)
      print(instance)

      if instance.Status == 'running':
        break

      import time
      time.sleep(3)

    logging.info('## instance is ready')
    ssh_key, pkey = read_ssh_key()
    self.api.init_ssh(cid=cid, ssh_key=ssh_key, pkey=pkey)
    self.api.launch_jobs(jobs=init_commands)


  def shell(self, cmd:str):
    return self.api.shell(cmd=cmd)

  def search(self, gpu_name:str, num_of_gpu:int):
    try:
      self.api.search_offer(gpu_name, num_of_gpu)
    except AlreadyExistInstance as ae:
      print(ae)
      pass

  def rent(self):
    token = os.environ["HF_TOKEN"]
    wandb_apikey = os.environ["WANDB_API_KEY"]
    self.__init_container()
    commands = [
      f'echo export HF_TOKEN={token} >> ~/.profile',
      f'echo export WANDB_API_KEY={wandb_apikey} >> ~/.profile',
      'pip install accelerate trl peft xformers wandb deepspeed flash-attn',
      'nvidia-smi',
      #f'WANDB_PROJECT={self.options.title} WANDB_API_KEY={os.environ["WANDB_API_KEY"]} HF_TOKEN={os.environ["HF_TOKEN"]} accelerate launch --num_processes 1 -m trainer.train'
      ]
    self.api.launch_jobs(jobs=commands)

    #self.api.destroy_instance()
  def ssh(self):
    import subprocess


    cmds =[
      'ssh',
      '-oStrictHostKeyChecking=no',
      self.api.sshurl()
    ]

    print(cmds)

    subprocess.call(cmds)
  def scp(self, remote:str, local:str):
    self.api.scp(remote, local)
  def logickor(self, model_name:str):
    init_commands = [
      'git clone https://github.com/eususu/LogicKor.git'
    ]
    self.__init_container(init_commands=init_commands)

    commands = [
      'cd LogicKor && python3 cli.py --model aiyets/gemma-2-9b-it-dpo-1009_full -f',
      #'SCP LogicKor/evaluated/*.jsonl .',
    ]
    self.api.launch_jobs(jobs=commands)
  def stop(self):
    self.api.destroy_instance()


if __name__ == "__main__":
  options = VRLOptions(
    title="gemma_dpo_002_dualdata",
    lm_parameter=9,
    rl_optimization="DPO",
    favor_gpu='H100',
    num_gpus=8,
    disk=300,
  )
  vrl = VRL(options)

  import sys
  if len(sys.argv) > 1:
    if sys.argv[1] == 'rent':
      vrl.rent()
    if sys.argv[1] == 'shell':
      cmd = sys.argv[2] if len(sys.argv)>2 else ""
      ssh_cmd = vrl.shell(cmd)
      print(ssh_cmd)
    if sys.argv[1] == 'scp':
      remote = sys.argv[2]
      local = sys.argv[3]
      vrl.scp(remote, local)
    if sys.argv[1] == 'search':
      if len(sys.argv) < 3:
        raise "Arugment exception"
      vrl.search(sys.argv[2], sys.argv[3])
    if sys.argv[1] == 'ssh':
      vrl.ssh()
    if sys.argv[1] == 'stop':
      vrl.stop()
    if sys.argv[1] == 'logickor':
      if len(sys.argv)<2:
        raise Exception("need a huggingface modelpath for evaluation")
      vrl.logickor(sys.argv[1])
  else:
    raise Exception("unsupported command")
