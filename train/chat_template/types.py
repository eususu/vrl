from typing import List
from pydantic import BaseModel
from datasets import Dataset


class Prompt(BaseModel):
    content:str
    role:str

class DPOFormat(BaseModel):
    chosen:str
    rejected: str

class DPOFormat_argilla(BaseModel):
    chosen:List[Prompt]
    rejected:List[Prompt]


class ChatTemplate():

    def detect(self, data:dict):
        pass
    def to_dataset(self, name:str, origin:Dataset)->Dataset:
        pass

class Conversation(BaseModel):
    messages:List[dict]


