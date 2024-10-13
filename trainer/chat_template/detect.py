from .c2d import templates
from huggingface_hub import hf_hub_download

def find_chat_template(model_name:str)->str:
    """
    모델의 tokenizer_config.json을 읽어서 모델이 지원하는 chat_template의 이름을 찾아주는 기능
    """
    import json
    f = hf_hub_download(model_name, filename="tokenizer_config.json")

    with open(f, 'r') as fp:
        content = json.load(fp)
        chat_template = content['chat_template']
        return _detect_chat_template(chat_template=chat_template)

def _detect_chat_template(chat_template:str)->str:

  for (key, template) in templates.items():
    if template == chat_template:
      return key

  raise Exception(f"unsupported chat template: {chat_template}")
