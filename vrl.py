import argparse
import logging
import os
import sys
import re
from typing import List
from pydantic import BaseModel

from _exceptions import AlreadyExistInstance
from _types import Colors, Instance, Offer, RentOptions
from vastapi import VastAPI, read_ssh_key

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
      if self.api.running_cid is not None:
        raise AlreadyExistInstance()

      self.api.search_offer(options.favor_gpu, options.num_gpus, min_down=options.min_down)
      self.api.create_instance(title=options.title, disk=options.disk)
    except AlreadyExistInstance as ae:
      pass

    cid = self.api.running_cid

    from tqdm import tqdm
    import time

    start_time = time.time()
    with tqdm(desc=f"인스턴스(GPU={options.favor_gpu} x{options.num_gpus}) 준비 중", unit="초") as pbar:
        while True:
            instance = self.api.get_instance(cid)
            pbar.set_postfix_str(f"상태: {instance.Status}")

            if instance.Status == 'running':
                break

            if time.time() - start_time > options.init_timeout:
                logging.warning(f"초기화 시간 초과: {options.init_timeout}초")
                break

            time.sleep(1)
            pbar.update(1)
    
    if instance.Status == 'running':
        logging.info(f'## {Colors.to_str(Colors.CYAN, "인스턴스가 준비되었습니다")}')
    else:
        logging.error(f'## {Colors.to_str(Colors.RED, "인스턴스 준비 실패")}')
    ssh_key, pkey = read_ssh_key()
    self.api.init_ssh(cid=cid, ssh_key=ssh_key, pkey=pkey)
    self.api.launch_jobs(jobs=init_commands)


  def shell(self, cmd:str):
    return self.api.shell(cmd=cmd)

  def search(self, gpu_name:str, num_of_gpu:int, min_down:int):
    try:
      self.api.search_offer(gpu_name, num_of_gpu, min_down=min_down)
    except AlreadyExistInstance as ae:
      print(ae)

  def rent(self, options:RentOptions):
    token = os.environ["HF_TOKEN"]
    wandb_apikey = os.environ["WANDB_API_KEY"]
    self.__init_container(options)
    commands = [
      f'echo export HF_TOKEN={token} >> ~/.profile',
      f'echo export WANDB_API_KEY={wandb_apikey} >> ~/.profile',
      'pip install accelerate trl peft xformers wandb deepspeed flash-attn',
      ]
    self.api.launch_jobs(jobs=commands)

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


def rent(args:argparse.Namespace):
  import socket
  hostname=socket.gethostname()
  username = os.getenv('USER') or os.getenv('USERNAME')

  options = RentOptions(
    title=f"{hostname}_{username}",
    favor_gpu=args.gpu,
    num_gpus=args.num_gpu,
    disk=args.disk,
    min_down=args.min_down,
    init_timeout=args.init_timeout,
  )
  print(f'gogo rent:{args}')
  vrl.rent(options=options)
  vrl.shell('ls -al')

def stop(args:argparse.Namespace):
  vrl = VRL()
  vrl.stop()

def search(args:argparse.Namespace):
  vrl = VRL()
  vrl.search(args.gpu, args.num_gpu, min_down=args.min_down)

vrl:VRL = VRL()

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title='commands', dest='command')
rent_parser = subparsers.add_parser('rent', help='설정된 조건에 맞는 장비를 임대합니다.')
rent_parser.set_defaults(func=rent)
rent_parser.add_argument('-gpu', type=str, help='임대를 원하는 GPU의 이름을 입력합니다.(h100, a100, 4090)', required=True)
rent_parser.add_argument('-num_gpu', type=int, default=1, help='임대를 원하는 GPU의 개수를 입력합니다.')
rent_parser.add_argument('-disk', type=int, default=50, help='임대를 원하는 디스크 용량을 GB 단위로 입력합니다(기본 50GB).')
rent_parser.add_argument('-min_down', type=int, default=800, help='네트워크 다운로드 속도의 최하치를 Mbps 단위로 입력합니다(기본:800)')
rent_parser.add_argument('-init_timeout', type=int, default=120, help='지정된 시간동안 인스턴스를 생성하지 못하면 중단합니다(기본:120 초)')
#rent_parser.add_argument('-min_up', type=int, help='네트워크 업로드 속도의 최하치를 Mbps 입력합니다')

stop_parser = subparsers.add_parser('stop', help='임대된 장비를 반납합니다')
stop_parser.set_defaults(func=stop)

search_parser = subparsers.add_parser('search', help='설정된 조건에 맞는 장비를 조회합니다.')
search_parser.add_argument('-gpu', type=str, help='임대를 원하는 GPU의 이름을 입력합니다.(h100, a100, 4090)', required=True)
search_parser.add_argument('-num_gpu', type=int, default=1, help='임대를 원하는 GPU의 개수를 입력합니다.')
search_parser.add_argument('-min_down', type=int, default=800, help='네트워크 다운로드 속도의 최하치를 Mbps 단위로 입력합니다(기본:800)')
search_parser.set_defaults(func=search)

args = parser.parse_args()
if args.command is None:
  parser.print_help()
  exit(-1)

args.func(args)


if __name__ == "__main__":
  exit(0)

  options = RentOptions(
    title="gemma_dpo_002_dualdata",
    lm_parameter=9,
    rl_optimization="DPO",
    favor_gpu='H100',
    num_gpus=8,
    disk=300,
  )
  vrl = VRL(options)

  if len(sys.argv) > 1:
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
    if sys.argv[1] == 'logickor':
      if len(sys.argv)<2:
        raise Exception("need a huggingface modelpath for evaluation")
      vrl.logickor(sys.argv[1])
  else:
    raise Exception("unsupported command")
