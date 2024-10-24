import os
import logging
import subprocess
import asyncio, asyncssh, sys
from typing import List, Tuple

from _types import Colors

import paramiko

logging.basicConfig(level=logging.INFO)

home = os.path.expanduser("~")

def _extract_ssh_url(url:str)->Tuple[str, str, int]:
  scheme_and_etc = url.split('://')
  user_and_host = scheme_and_etc[1].split('@')

  if len(user_and_host) == 1:
    user = None
    host_and_port = user_and_host[0].split(':')
  else:
    user = user_and_host[0]
    host_and_port = user_and_host[1].split(':')

  if len(host_and_port) == 1:
    host = host_and_port[0]
    port = 22
  else:
    host = host_and_port[0]
    port = int(host_and_port[1])
  return user, host, port

class MySession(asyncssh.SSHClientSession):
  def data_received(self, data: str, datatype: asyncssh.DataType) -> None:
    print(data, end='')

  def connection_lost(self, exc: Exception | None) -> None:
    print('ssh session closed', exc)

def ssh_exec_command_by_api(url:str, cmdlist:List[str], environment:dict=None):
  user, host, port = _extract_ssh_url(url)
  logging.info(f'SSH command info ({user}@{host}:{port})')
  async def run():
    ssh_info = {
      'port': port,
      'known_hosts': None,
    }
    if user is not None:
      ssh_info['username'] = user

    async with asyncssh.connect(host, **ssh_info) as conn:
      for cmd in cmdlist:
        if len(cmd) == 0:
          async def read_output(proc):
            try:
              while True:
                line = await proc.stdout.readline()
                if not line:
                  break
                print(line, end='')
            except Exception as e:
              print(e)

          async def send_input(proc):
            try:
              while True:
                command = await asyncio.get_event_loop().run_in_executor(None, input, '$ ')
                proc.stdin.write(command + '\n')
            except Exception as e:
              print(e)

          async with conn.create_process() as proc:
            await asyncio.gather(
              read_output(proc),
              send_input(proc)
            )
          break

        chan, session = await conn.create_session(MySession, cmd, env=environment)
        await chan.wait_closed()

  asyncio.get_event_loop().run_until_complete(run())


def ssh_exec_command(url:str, cmd:str):
  cmdline = f'ssh {url} {cmd}'
  logging.info(cmdline)

  process = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
  while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
      break
    if output:
      print(f'remote => {Colors.GREY}{output.strip()}{Colors.DEFAULT}')
  rc = process.poll()
  if rc != 0:
    print(process.stderr.read().decode())

def read_ssh_key():
  ssh_key=None
  pkey=None
  pub:str=None
  pri:str=None
  key_names =[
    'id_ed25519',
    'id_rsa'
  ]
  selected_key_name:str=None
  for key_name in key_names:
    pri = f'{home}/.ssh/{key_name}'
    pub = f'{home}/.ssh/{key_name}.pub'

    if os.path.exists(pri) and os.path.exists(pub):
      selected_key_name = key_name
      break

  if not pri or not pub:
    raise Exception(f"need .ssh keypair in ({home}/.ssh)")
  try:
    with open(pub, 'r') as file:
      ssh_key = file.read().strip()

    if selected_key_name == 'id_ed25519':
      pkey = paramiko.Ed25519Key.from_private_key_file(pri)
    elif selected_key_name == 'id_rsa':
      pkey = paramiko.RSAKey.from_private_key_file(pri)
    else:
      raise Exception (f'unknown ssh key type:{selected_key_name}')
  except Exception as e:
    raise Exception("failed to access pub or pkey", e)

  return ssh_key, pkey