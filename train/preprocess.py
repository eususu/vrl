from datasets import load_dataset, DatasetDict,Dataset
from chat_template.types import ChatTemplate
from chat_template.dpo import ChatTemplate_DPO

def preprocess(ds_name:str, target_template_name:str)->Dataset:
    ct:ChatTemplate = ChatTemplate_DPO(target_template_name)
    _dataset:DatasetDict = load_dataset(ds_name)

    try:
        print("Convert dataset format for DPO training")
        for split, _data in _dataset.items():
            prep_ds = ct.to_dataset(name=ds_name, origin=_data)
            print(f"PROMPT: {prep_ds['prompt'][0]}")
            print(f"CHOSEN: {prep_ds['chosen'][0]}")
            print(f"REJECTED: {prep_ds['rejected'][0]}")
            return prep_ds
            break
    except Exception as e:
        print("bad format", e)
        import traceback
        print(traceback.format_exc())




if __name__ == "__main__":
    #preprocess()
    preprocess(ds_name='kuotient/orca-math-korean-dpo-pairs')
    preprocess(ds_name='aiyets/argilla_dpo-mix-7k-ko')
