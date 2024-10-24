all:
	python3 -m vrl rent -gpu 4090

search:
	python3 -m vrl search -gpu 6000 -num_gpu 8

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