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

  def shell(self, cmd:str):
    return self.api.shell(cmd=cmd)

  def search(self):
    try:
      self.api.search_offer()
    except AlreadyExistInstance as ae:
      pass

  def train(self):
    try:
      if self.api.running_cid is not None:
        raise AlreadyExistInstance()

      self.api.search_offer()
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
    self.api.launch_jobs()

    #self.api.destroy_instance()
  def search(self):
    self.api.search_offer()
  def sshurl(self):
    self.api.sshurl()
  def stop(self):
    self.api.destroy_instance()


if __name__ == "__main__":
  options = VRLOptions(
    title="susu_dpo_001",
    lm_parameter=7,
    num_gpus=2,
    rl_optimization="DPO",
    favor_gpu='A100',
    disk=200,
  )
  vrl = VRL(options)

  import sys
  if len(sys.argv) > 1:
    if sys.argv[1] == 'shell':
      cmd = sys.argv[2] if len(sys.argv)>2 else ""
      ssh_cmd = vrl.shell(cmd)
      print(ssh_cmd)
    if sys.argv[1] == 'search':
      vrl.search()
    if sys.argv[1] == 'sshurl':
      vrl.sshurl()
    if sys.argv[1] == 'stop':
      vrl.stop()
  else:
    vrl.train()