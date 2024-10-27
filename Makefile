all: status

status:
	vrl status

rent:
	vrl rent -gpu 4090

search:
	vrl search -gpu 6000 -num_gpu 8

bash:
	vrl shell

ssh:
	vrl ssh

shell:
	vrl shell nvidia-smi

eval:
	vrl logickor

scp:
	vrl scp ./LogicKor/evaluated/default.jsonl .

stop:
	vrl stop

tt:
	trainer.train

build:
	python3 -m build
	pip install -e .

install:
	python3 setup.py install
