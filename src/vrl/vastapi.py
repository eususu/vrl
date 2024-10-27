import logging
import os
import re
import ast
from typing import List
import paramiko
from pydantic import BaseModel
from vastai import VastAI

from vrl._ssh import read_ssh_key, ssh_exec_command_by_api
from vrl._types import Colors, Instance, Offer, RentState



api_key = None
home = os.path.expanduser("~")

try:
  with open(f'{home}/.vast_api_key', 'r') as file:
    api_key = file.read().replace('\n', '')
except Exception as e:
  print(e)
  print('# init vast ai first')
  print('  vastai set api-key [YOUR API_KEY]')
  exit(1)

def retrieve_gpu_model(gpu_name:str)->str:
  list = None
  min_ram = None
  if gpu_name.capitalize() == 'A100':
    list = [
      'A100_SXM4',
      'A100_SXM',
      'A100X',
      'A100_PCIE',
    ]
    min_ram = 80
  elif gpu_name.capitalize() == 'H100':
    list = [
      'H100_NVL',
      'H100_SXM',
    ]
    min_ram = None
  elif '4090' in gpu_name:
    list = [
      'RTX_4090',
    ]
  elif '6000' in gpu_name:
    list = [
      'RTX_6000Ada',
    ]
  else:
    list = [gpu_name.capitalize()]

  return f'[{",".join(list)}]', min_ram

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
  ssh_key:str
  ssh_pkey:str

  rentState:RentState

  def __init__(self, api_key:str=api_key):
    self.api = VastAI(api_key=api_key)
    self.ssh_key, self.ssh_pkey = read_ssh_key()

    try:
      self.__check_running_pid()
    except Exception as e:
      pass

  def is_running(self):
    if self.rentState and self.rentState.running_cid:
      return True
    return False

  def print_rent_state(self):
    if not self.is_running():
      print(Colors.GREY.to_str("현재 임대중인 장치가 없습니다"))
      return

    print("Running CID=" + Colors.CYAN.to_str(self.rentState.running_cid))
    print("installed ssh key=" + Colors.CYAN.to_str(self.rentState.install_ssh_key))


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

    return instances

  def get_instance(self)->Instance:
    cid = self.rentState.running_cid
    try:
      lines = self.api.show_instance(id=cid)
    except:
      return None
    logging.debug(lines)
    return self.__parse_instances(lines)[0]

  def get_instances(self)->List[Instance]:
    lines = self.api.show_instances()
    return self.__parse_instances(lines)

  def __check_running_pid(self):
    rentState = RentState.load()
    self.rentState = rentState

  def search_offer(self, gpu_name:str, num_of_gpu:int, min_down:int=None):
    _gpu_name, min_gpu_ram = retrieve_gpu_model(gpu_name)
    order='-inet_down'
    order='+dph'

    offer_conditions = [
      'reliability>0.96',
    ]

    # 요구하는 gpu ram에 따라 장치를 선택하게 해야함
    if min_down:
      offer_conditions.append(f'inet_down > {min_down}')
    if min_gpu_ram:
      offer_conditions.append(f'gpu_ram>={min_gpu_ram}')
    offer_conditions.append(f'num_gpus={num_of_gpu}')
    offer_conditions.append(f'gpu_name in {_gpu_name}')
    logging.debug(offer_conditions)
    offers_str = self.api.search_offers(query=' '.join(offer_conditions), order=order)
    offers = parse_offers(offers_str)
    if len(offers) == 0:
      raise Exception(f"there are no available GPU({_gpu_name}x{num_of_gpu}).")

    for offer in offers:
      offer.print_summary()

    # select first item
    self.selected_offer = offers[0]

    logging.info("SELECTED OFFER:")
    self.selected_offer.print_summary()


  def create_instance(self, title:str, disk:int):
    result = self.api.create_instance(
      ID=self.selected_offer.ID,
      label=title,
      disk=disk,
      image="vllm/vllm-openai:latest",
      env="HF_TOKEN=1234",
      )

    logging.debug(f'create instance result={result}')

    j = result[result.find('{'):]
    res = ast.literal_eval(j)

    if not res['success']:
      logging.error(res)
      raise Exception(f"failed to create instance")

    running_cid = res['new_contract']
    rentState = RentState(running_cid=running_cid)
    rentState.save()
    self.rentState = rentState
    return

  def _checkInstance(self):
    if self.rentState is None:
      raise Exception("아직 장치를 임대하지 않았습니다")

  def destroy_instance(self):
    self._checkInstance()
    res = self.api.destroy_instance(id=self.rentState.running_cid)
    logging.info(f'destory instance -> {res}')
    RentState.remove()

    sshjson = f'ssh_{self.rentState.running_cid}.json' # vastai generate it.

    os.remove(sshjson)

  def sshurl(self):
    cid = self.rentState.running_cid
    url = self.api.ssh_url(id=cid).strip()
    print(f'ssh url={url}')
    return url

  def scp(self, remote, local):
    self._checkInstance()

    cid = self.rentState.running_cid
    url = self.api.ssh_url(id=cid)
    print(f'ssh {url}')

    scheme_and_etc = url.split('://')
    user_and_host = scheme_and_etc[1].split('@')

    user = user_and_host[0]
    host_and_port = user_and_host[1].split(':')
    host = host_and_port[0]
    port = int(host_and_port[1])

    #print(f'user is {user}')
    #print(f'host is {host}')
    #print(f'port is {port}')

    import paramiko
    from scp import SCPClient

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_key, pkey = read_ssh_key()
    ssh.connect(host, port=port, username=user, pkey=pkey)

    print ("begin copy file")
    with SCPClient(ssh.get_transport()) as scp:
      scp.get(remote, local_path=local)
    print ("copy file is done")

  def shell(self, cmd:str):
    self._checkInstance()

    cid = self.rentState.running_cid
    url = self.api.ssh_url(id=cid)
    print(url)

    ssh_exec_command_by_api(url, [cmd])
    

  def init_ssh(self, ssh_key:str, pkey:str):
    cid = self.rentState.running_cid
    url = self.api.ssh_url(id=cid)
    print(Colors.GREY.to_str(url))

    if not self.rentState.install_ssh_key:
      print(Colors.CYAN.to_str('SSH 키를 서버에 등록합니다'))
      res = self.api.attach_ssh(instance_id=cid, ssh_key=ssh_key)
      print(res)

    scheme_and_etc = url.split('://')
    user_and_host = scheme_and_etc[1].split('@')

    user = user_and_host[0]
    host_and_port = user_and_host[1].split(':')
    host = host_and_port[0]
    port = int(host_and_port[1])

    #print(f'user is {user}')
    #print(f'host is {host}')
    #print(f'port is {port}')

    import paramiko
    from scp import SCPClient

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    while True:
      try:
        logging.info("waiting apply SSH key pair")
        ssh.connect(host, port=port, username=user, pkey=pkey)
        self.rentState.install_ssh_key = True
        self.rentState.save()
        logging.info("confirm applied SSH key pair")
        break
      except Exception as e:
        continue


    local_file_path = './trainer'

    remote_base_path = '/root'
    print ("begin copy file")
    with SCPClient(ssh.get_transport()) as scp:
      scp.put(local_file_path, remote_path=f'{remote_base_path}/', recursive=True)
    print ("copy file is done")


  def launch_jobs(self, jobs:List[str]):
    if not jobs:
      return
    self._checkInstance()

    url = self.api.ssh_url(id=self.rentState.running_cid)

    print ("exec is begin~~~~~~~~~~~~~~")
    ssh_exec_command_by_api(url=url, cmdlist=jobs)
    print ("exec is done~~~~~~~~~~~~~~")
