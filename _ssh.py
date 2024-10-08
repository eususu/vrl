import logging
import subprocess
import asyncio, asyncssh, sys
from typing import List, Tuple

from _types import Colors

from paramiko import SSHClient

logging.basicConfig(level=logging.INFO)


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

if __name__ == "__main__":
  ssh_exec_command_by_api(url='ssh://172.16.10.14', cmdlist=['uptime','ls -al'])