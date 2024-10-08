import logging
import os
import re
from typing import List
from pydantic import BaseModel

from _exceptions import AlreadyExistInstance
from _types import Instance, Offer, VRLOptions
from vastapi import VastAPI



logging.basicConfig(level=logging.INFO)


  
class VRL():
  options:VRLOptions
  api:VastAPI
  selected_offer:Offer

  def __init__(self, options:VRLOptions):
    self.api = VastAPI(options=options)
    self.options = options

  def train(self):
    try:
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

    logging.info('instance is ready')

    self.api.destroy_instance()



if __name__ == "__main__":
  options = VRLOptions(
    title="susu_dpo_001",
    lm_parameter=7,
    rl_optimization="DPO",
    favor_gpu='A100',
    disk=200,
  )
  vrl = VRL(options)
  vrl.train()