import os
import re
from typing import List
from vastai import VastAI
from pydantic import BaseModel

api_key = None
home = os.path.expanduser("~")
with open(f'{home}/.vast_api_key', 'r') as file:
  api_key = file.read().replace('\n', '')


order='-inet_down'


class Offers(BaseModel):
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


def parse_offers(offers:str)->List[Offers]:
  result = []
  rows = offers.split('\n')

  keys = list(Offers.model_fields.keys())
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
    obj = Offers(**obj)
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

  return f'[{",".join(list)}]'


def vrl(gpu_name:str):
  api = VastAI(api_key=api_key)

  _gpu_name = retrieve_gpu_model(gpu_name)

  offer_conditions = [
    'reliability>0.99',
  ]

  # 요구하는 gpu ram에 따라 장치를 선택하게 해야함
  offer_conditions.append('gpu_ram<=80')
  offer_conditions.append('num_gpus=1')
  offer_conditions.append(f'gpu_name in {_gpu_name}')
  offers_str = api.search_offers(query=' '.join(offer_conditions), order=order)
  offers = parse_offers(offers_str)

  for offer in offers:
    print(f'\033[92m{offer.Model} \033[93m${offer.price}\033[0m - \033[90m{offer}')



if __name__ == "__main__":
  vrl('A100')