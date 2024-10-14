all:
	python3 vrl.py

search:
	python3 -m vrl search

bash:
	python3 -m vrl shell

ssh:
	python3 -m vrl ssh

shell:
	python3 -m vrl shell nvidia-smi

eval:
	python3 -m vrl logickor

scp:
	python3 -m vrl scp ./LogicKor/evaluated/default.jsonl .

stop:
	python3 -m vrl stop


tt:
	python3 -m trainer.train