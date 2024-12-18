import logging
import os
from pydantic import BaseModel

class RentOptions(BaseModel):
  title:str # project title,
  favor_gpu:str
  num_gpus:int=1
  disk:int
  min_down:int
  min_up:int
  init_timeout:int
  auto_connect:bool


class Colors:
    default = '\033[0m'
    cyan = '\033[92m'
    yellow = '\033[93m'
    grey = '\033[90m'
    red = '\033[91m'

    value: str

    def __init__(self, value: str):
        self.value = value

    def to_str(self, msg: str):
        return f'{self.value}{msg}{self.default}'


Colors.DEFAULT = Colors(Colors.default)
Colors.CYAN = Colors(Colors.cyan)
Colors.YELLOW = Colors(Colors.yellow)
Colors.GREY = Colors(Colors.grey)
Colors.RED = Colors(Colors.red)

class Offer(BaseModel):
  ID:str
  CUDA:str
  N:str
  Model:str
  PCIE:str
  cpu_ghz:str
  vCPUs:str
  RAM:str
  Disk:str
  price:str
  DLP:str
  DLP_usd:str
  score:str
  NV_driver:str
  Net_up:str
  Net_down:str
  R:str
  Max_Days:str
  mach_id:str
  status:str
  ports:str
  country:str

  def print_summary(self):
    print(f'{Colors.CYAN.to_str(self.Model)} {Colors.YELLOW.to_str(self.price)} - {Colors.GREY.to_str(self)}')
    logging.info(f'{Colors.CYAN.to_str(self.Model)} {Colors.YELLOW.to_str(self.price)} - {Colors.GREY.to_str(self)}')
    

class Instance(BaseModel):
  ID:str
  Machine:str
  Status:str
  num:str
  Model:str
  Util:str
  vCPUs:str
  RAM:str
  Storage:str
  SSH_Addr:str
  SSH_Port:str
  price:str
  Image:str
  Net_up:str
  Net_down:str
  R:str
  Label:str
  age:str

  @classmethod
  def from_str(cls, line: str) -> 'Instance':
      fields = line.split(' ')
      return Instance(
          ID=fields[0],
          Machine=fields[1],
          Status=fields[2],
          num=fields[3],
          Model=fields[4],
          Util=fields[5],
          vCPUs=fields[6],
          RAM=fields[7],
          Storage=fields[8],
          SSH_Addr=fields[9],
          SSH_Port=fields[10],
          price=fields[11],
          Image=fields[12],
          Net_up=fields[13],
          Net_down=fields[14],
          R=fields[15],
          Label=fields[16],
          age=fields[17]
      )

import json
_CIDFILE='RUNNING.CID'
class RentState(BaseModel):
  running_cid:int
  install_ssh_key:bool=False
  complete_first_command:bool=False

  @classmethod
  def load(cls):
    try:
      with open(_CIDFILE, 'r') as file:
        data = json.load(file)
        return cls.model_validate(data)
    except FileNotFoundError:
      logging.debug(f'{_CIDFILE} 파일이 없습니다.')
      return None
  def save(self):
    try:
      with open(_CIDFILE, 'w') as file:
        file.write(self.model_dump_json())
    except Exception as e:
      print(e)

  @classmethod
  def remove(cls):
    logging.info('remove CID file')
    os.remove(_CIDFILE)
