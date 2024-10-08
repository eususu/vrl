import json
import logging
import os
import re
import ast
import socket
from typing import List, Tuple
from vastai import VastAI

from _exceptions import AlreadyExistInstance
from _types import Instance, Offer, VRLOptions

CIDFILE='RUNNING.CID'
api_key = None
home = os.path.expanduser("~")
with open(f'{home}/.vast_api_key', 'r') as file:
  api_key = file.read().replace('\n', '')

def read_cid()->int:
  try:
    with open(CIDFILE, 'r') as file:
      running_cid = file.read().strip()
      return int(running_cid)
  except Exception as e:
    raise Exception("no cid file")
def read_ssh_key()->int:
  try:
    with open(f'{home}/.ssh/id_ed25519.pub', 'r') as file:
      ssh_key = file.read().strip()
      return ssh_key
  except Exception as e:
    raise Exception("no cid file")

def retrieve_gpu_model(gpu_name:str)->str:
  list = None
  if gpu_name == 'A100':
    list = [
      'A100_SXM4',
      'A100_SXM',
      'A100X',
      'A100_PCIE',
    ]
  else:
    list = [gpu_name]

  return f'[{",".join(list)}]'

def parse_offers(offers:str)->List[Offer]:
  result = []
  rows = offers.split('\n')

  keys = list(Offer.model_fields.keys())
  for index, _row in enumerate(rows):
    if len(_row) == 0: # last line is empty.
      break

    row = re.sub(r'\s+', ' ', _row).strip()
    items = row.split(' ')

    if index == 0:
      continue

    obj = {}
    for iter in range(len(items)):
      obj[keys[iter]] = items[iter]
    obj = Offer(**obj)
    result.append(obj)

  return result



class VastAPI():
  api:VastAI
  options:VRLOptions

  def __init__(self, options:VRLOptions, api_key:str=api_key):
    self.options = options
    self.api = VastAI(api_key=api_key)

  @property
  def label(self):
    return f'VRL#{self.options.title}'

  def __parse_instances(self, lines:str)->List[Instance]:
    instances = []

    for index, line in enumerate(lines.split('\n')):

      row = re.sub(r'\s+', ' ', line).strip()
      if index == 0:
        continue
      if len(row) == 0:
        continue

      instance = Instance.from_str(row)
      instances.append(instance)

    logging.info(f'{len(instances)} are founded')
    return instances

  def get_instance(self, cid:str)->Instance:
    lines = self.api.show_instance(id=cid)
    return self.__parse_instances(lines)[0]

  def get_instances(self)->List[Instance]:
    lines = self.api.show_instances()
    return self.__parse_instances(lines)

  def __check_duplicated_label(self):
    try:
      with open(CIDFILE, 'r') as file:
        self.running_cid = file.read().strip()
    except FileNotFoundError:
      logging.debug(f'{CIDFILE} 파일이 없습니다.')
      self.running_cid = None
      return

    instances = self.get_instances()
    for instance in instances:
      if instance.Label == self.label and instance.ID == self.running_cid:
        logging.info(instance)
        raise AlreadyExistInstance(f"CID:{self.running_cid}, Label:{self.label} is already exists")

    logging.info('remove orphant CID file')
    os.remove(CIDFILE)

  def search_offer(self):
    self.__check_duplicated_label()

    _gpu_name = retrieve_gpu_model(self.options.favor_gpu)
    order='-inet_down'
    order='+dph'

    offer_conditions = [
      'reliability>0.99',
    ]

    # 요구하는 gpu ram에 따라 장치를 선택하게 해야함
    offer_conditions.append('inet_down > 800')
    offer_conditions.append('gpu_ram>=42')
    offer_conditions.append('num_gpus=1')
    offer_conditions.append(f'gpu_name in {_gpu_name}')
    offers_str = self.api.search_offers(query=' '.join(offer_conditions), order=order)
    offers = parse_offers(offers_str)

    for offer in offers:
      offer.print_summary()

    # select first item
    self.selected_offer = offers[0]

    logging.info("SELECTED OFFER:")
    self.selected_offer.print_summary()


  def create_instance(self)->str:
    result = self.api.create_instance(
      ID=self.selected_offer.ID,
      label=self.label,
      disk=self.options.disk,
      image="vllm/vllm-openai:latest",
      )

    logging.debug(f'create instance result={result}')

    j = result[result.find('{'):]
    res = ast.literal_eval(j)

    if not res['success']:
      logging.error(res)
      raise Exception(f"failed to create instance")

    with open('RUNNING.CID', 'w') as f:
      f.write(str(res['new_contract']))

    self.running_cid = res['new_contract']
    return self.running_cid

  def destroy_instance(self):
    res = self.api.destroy_instance(id=self.running_cid)
    logging.info(f'destory instance -> {res}')
    logging.info('remove CID file')
    os.remove(CIDFILE)

  def test(self, cid:int, ssh_key:str):
    res = self.api.attach_ssh(instance_id=cid, ssh_key=ssh_key)
    print(res)

    url = self.api.scp_url(id=cid)
    print(url)
    scheme_and_etc = url.split('://')
    user_and_host = scheme_and_etc[1].split('@')

    user = user_and_host[0]
    host_and_port = user_and_host[1].split(':')
    host = host_and_port[0]
    port = int(host_and_port[1])


    print(f'ssh -p {port} {host}')

    print(f'user is {user}')
    print(f'host is {host}')
    print(f'port is {port}')


    import paramiko
    from scp import SCPClient

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(host, port=port, username=user, password=ssh_key)


    local_file_path = './train'

    remote_base_path = '/root'
    print ("begin copy file")
    with SCPClient(ssh.get_transport()) as scp:
      scp.put(local_file_path, remote_path=f'{remote_base_path}/', recursive=True)
    print ("copy file is done")

    print ("exec is begin~~~~~~~~~~~~~~")
    environment = {
      'HF_TOKEN': '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'
    }

    commands = ['pip install trl peft', 'python3 train/train.py']
    contents = "\n".join(commands)
    stdin, stdout, stderr = ssh.exec_command('find train', environment=environment)
    print(stdout.read().decode())
    print(stderr.read().decode())

    stdin, stdout, stderr = ssh.exec_command('python3 train/train.py', environment=environment)
    #stdin, stdout, stderr = ssh.exec_command('/root/run.sh', environment=environment)
    print(stdout.read().decode())
    print(stderr.read().decode())

    print ("exec is done~~~~~~~~~~~~~~")

    pass

if __name__ == "__main__":
  options = VRLOptions(
    title="susu_dpo_001",
    lm_parameter=7,
    rl_optimization="DPO",
    favor_gpu='A100',
    disk=200,
  )
  api = VastAPI(options)

  cid = read_cid()
  ssh_key = read_ssh_key()
  api.test(cid=cid, ssh_key=ssh_key)
