from datasets import load_dataset, DatasetDict,Dataset
from .chat_template.types import ChatTemplate
from .chat_template.dpo import ChatTemplate_DPO

def preprocess(ds_name:str, target_template_name:str, split:str='train')->Dataset:
    """
    주어진 데이터셋에 대해서 전처리 과정을 수행합니다.
    학습의 목적에 맞는 chat template를 찾아내고, 주어진 데이터셋의 형식을 탐지하여 openai chat template로 변환합니다.
    최종으로 openai chat template로 변환된 데이터셋은 다시 주어진 target_template_name에 맞도록 변환됩니다.

    Args:
        ds_name:
            변환할 데이터셋 이름(huggingface name)
        target_template_name:
            최종 변환을 원하는 chat template의 이름 (chatml, gemma, llama, ...)
        split:
            변환 할 데이터셋의 구분자 이름 (train, test)
    """
    ct:ChatTemplate = ChatTemplate_DPO(target_template_name)
    _dataset:DatasetDict = load_dataset(ds_name)

    try:
        print("Convert dataset format for DPO training")
        for _split, _data in _dataset.items():
            if not split == _split:
                continue

            prep_ds = ct.to_dataset(name=ds_name, origin=_data)
            print(f"PROMPT: {prep_ds['prompt'][0]}")
            print(f"CHOSEN: {prep_ds['chosen'][0]}")
            print(f"REJECTED: {prep_ds['rejected'][0]}")
            return prep_ds
    except Exception as e:
        print("bad format", e)
        import traceback
        print(traceback.format_exc())
    raise Exception("not found matched split")


if __name__ == "__main__":
    chat_template = 'gemma'
    preprocess(ds_name='kuotient/orca-math-korean-dpo-pairs', target_template_name=chat_template)
    preprocess(ds_name='aiyets/argilla_dpo-mix-7k-ko', target_template_name=chat_template)
