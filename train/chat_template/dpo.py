import pandas as pd
from tqdm import tqdm
from .types import ChatTemplate, Conversation, DPOFormat, DPOFormat_argilla, Prompt
from .c2d import apply_template
from datasets import Dataset, load_dataset, load_from_disk

def _is_dpo_format(data:dict):
    try:
        _data:DPOFormat = DPOFormat.model_validate(data)
        print("## this format(DPOFormat)")
        print(_data)
        return DPOFormat
    except Exception as e:
        print("## not DPOFormat")


    try:
        _data:DPOFormat_argilla = DPOFormat_argilla.model_validate(data)
        print("## this format(DPOFormat_agrilla)")
        print(_data)
        return DPOFormat_argilla
    except Exception as e:
        print("## not DPOFormat_agrilla")

class ChatTemplate_DPO(ChatTemplate):
    target_type:None

    def to_dataset(self, name:str, origin:Dataset)->Dataset:

        datatype = _is_dpo_format(origin[0])
        if datatype == DPOFormat:
            return self._to_dataset_DPOFormat(name=name, origin=origin)
        if datatype == DPOFormat_argilla:
            return self._to_dataset_DPOFormat_agrilla(name=name, origin=origin)
        raise Exception("unsupported dop dataformat")

    def _to_dataset_DPOFormat(self, name:str, origin:Dataset)->Dataset:
        list = []

        import os
        ds_path = f"temp_dataset_{name.replace('/','_')}"
        if os.path.exists(ds_path):
            return load_from_disk(ds_path)

        dummy_message = {'role': 'user', 'content': 'this is a dummy message'}
        dummy = apply_template(Conversation(messages=[dummy_message]), target_template='gemma')

        for row in tqdm(origin):
            system = {'role': 'system', 'content': row['system']}
            prompt = [{'role': 'user', 'content': row['question']}]
            chosen = [dummy_message, {'role': 'assistant', 'content': row['chosen']}]
            rejected = [dummy_message, {'role': 'assistant', 'content': row['rejected']}]

            new_prompt = apply_template(Conversation(messages=prompt), target_template='gemma')
            new_chosen = apply_template(Conversation(messages=chosen), target_template='gemma').removeprefix(dummy)
            new_rejected = apply_template(Conversation(messages=rejected), target_template='gemma').removeprefix(dummy)

            list.append({'prompt': new_prompt, 'chosen': new_chosen, 'rejected': new_rejected})
            
        new_dataset = Dataset.from_list(list)
        new_dataset.save_to_disk(ds_path)
        return new_dataset

    def _to_dataset_DPOFormat_agrilla(self, name:str, origin:Dataset)->Dataset:
        list = []

        import os
        ds_path = f"temp_dataset_{name.replace('/','_')}"
        if os.path.exists(ds_path):
            return load_from_disk(ds_path)
            

        dummy_message = {'role': 'user', 'content': 'this is a dummy message'}
        dummy = apply_template(Conversation(messages=[dummy_message]), target_template=None)
        for row in tqdm(origin):

            prompt = row['chosen'][:-1]
            chosen = [dummy_message, row['chosen'][-1]]
            rejected = [dummy_message, row['rejected'][-1]]

            new_prompt = apply_template(Conversation(messages=prompt), target_template=None)
            new_chosen = apply_template(Conversation(messages=chosen), target_template=None).removeprefix(dummy)
            new_rejected = apply_template(Conversation(messages=rejected), target_template=None).removeprefix(dummy)

            list.append({'prompt': new_prompt, 'chosen': new_chosen, 'rejected': new_rejected})
            
        new_dataset = Dataset.from_list(list)
        new_dataset.save_to_disk(ds_path)
        return new_dataset


"""
## DPO dataset format example from https://huggingface.co/docs/trl/main/en/dpo_trainer

dpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ],
}
"""
