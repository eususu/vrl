all:
	python3 vrl.py


search:
	python3 -m vrl search

bash:
	python3 -m vrl shell

sshurl:
	python3 -m vrl sshurl

shell:
	python3 -m vrl shell nvidia-smi

stop:
	python3 -m vrl stop