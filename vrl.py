import os
import re
from typing import List
from vastai import VastAI
from pydantic import BaseModel

from _types import Instance, Offer

api_key = None
home = os.path.expanduser("~")
with open(f'{home}/.vast_api_key', 'r') as file:
  api_key = file.read().replace('\n', '')


order='-inet_down'




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



class VRLOptions(BaseModel):
  title:str # project title,
  lm_parameter:int # 7B, 11B, ...
  favor_gpu:str
  rl_optimization:str # DPO, OROP, ...
  disk:int

  
class VRL():
  options:VRLOptions
  api:VastAI
  selected_offer:Offer

  def __init__(self, options:VRLOptions):
    self.api = VastAI(api_key=api_key)
    self.options = options



  def __find_instance(self):
    _gpu_name = retrieve_gpu_model(self.options.favor_gpu)

    offer_conditions = [
      'reliability>0.99',
    ]

    # 요구하는 gpu ram에 따라 장치를 선택하게 해야함
    offer_conditions.append('gpu_ram>=42')
    offer_conditions.append('num_gpus=1')
    offer_conditions.append(f'gpu_name in {_gpu_name}')
    offers_str = self.api.search_offers(query=' '.join(offer_conditions), order=order)
    offers = parse_offers(offers_str)

    for offer in offers:
      offer.print_summary()

    # select first item
    self.selected_offer = offers[0]

    print("SELECTED OFFER:")
    self.selected_offer.print_summary()

  def __create_instance(self):
    self.api.create_instance(
      ID=self.selected_offer.ID,
      label=f'VRL#{self.options.title}',
      disk=self.options.disk,
      image="vllm/vllm-openai:latest"
      )

  def train(self):
    self.__find_instance()
    #self.__create_instance()
    instances = self.api.show_instances()
    print(instances)



if __name__ == "__main__":

  lines="""ID        Machine  Status   Num  Model      Util. %  vCPUs    RAM  Storage  SSH Addr      SSH Port  $/hr    Image                    Net up  Net down  R     Label             age(hours)
13022723  22952    running   1x  A100_SXM4  0.0      32.0   515.6  200      ssh8.vast.ai  22722     1.6400  vllm/vllm-openai:latest  946.8   8766.9    99.5  VRL#susu_dpo_001  2036.33
  """

  for index, line in enumerate(lines.split('\n')):

    row = re.sub(r'\s+', ' ', line).strip()
    if index == 0:
      continue
    if len(row) == 0:
      continue


    instance = Instance.from_str(row)
    print(instance)
  exit(0)



  options = VRLOptions(
    title="susu_dpo_001",
    lm_parameter=7,
    rl_optimization="DPO",
    favor_gpu='A100',
    disk=200,
  )
  vrl = VRL(options)
  vrl.train()