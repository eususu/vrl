import argparse
import logging
import os
import sys
import re
from typing import List
from pydantic import BaseModel


from vrl.vrl import VRL
from vrl._exceptions import AlreadyExistInstance
from vrl._types import Colors, Instance, Offer, RentOptions, RentState
from vrl.vastapi import VastAPI, read_ssh_key

logging.basicConfig(level=logging.INFO)
import asyncssh

# asyncssh 로그 레벨을 WARNING으로 설정하여 불필요한 로그를 숨깁니다
asyncssh.logging.set_log_level(logging.WARNING)
logging.getLogger('paramiko').setLevel(logging.WARNING)

##########################################################
vrl:VRL = VRL()
def status(args:argparse.Namespace):
  vrl.status()

def rent(args:argparse.Namespace):
  import socket
  hostname=socket.gethostname()
  username = os.getenv('USER') or os.getenv('USERNAME')


  options = RentOptions(
    title=f"{hostname}_{username}",
    favor_gpu=args.gpu,
    num_gpus=args.num_gpu,
    disk=args.disk,
    min_down=args.min_down,
    min_up=args.min_up,
    init_timeout=args.init_timeout,
  )
  print(f'gogo rent:{args}')
  vrl.rent(options=options)
  vrl.shell('ls -al')

def stop(args:argparse.Namespace):
  vrl.stop()

def search(args:argparse.Namespace):
  vrl.search(args.gpu, args.num_gpu, min_down=args.min_down)

def ssh(args:argparse.Namespace):
  vrl.ssh()

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title='commands', dest='command')
status_parser = subparsers.add_parser('status', help='현재 상태를 확인합니다')
status_parser.set_defaults(func=status)
rent_parser = subparsers.add_parser('rent', help='설정된 조건에 맞는 장비를 임대합니다.')
rent_parser.set_defaults(func=rent)
rent_parser.add_argument('-gpu', type=str, help='임대를 원하는 GPU의 이름을 입력합니다.(h100, a100, 4090)', required=True)
rent_parser.add_argument('-num_gpu', type=int, default=1, help='임대를 원하는 GPU의 개수를 입력합니다.')
rent_parser.add_argument('-disk', type=int, default=50, help='임대를 원하는 디스크 용량을 GB 단위로 입력합니다(기본 50GB).')
rent_parser.add_argument('-min_down', type=int, default=1000, help='네트워크 다운로드 속도의 최하치를 Mbps 단위로 입력합니다(기본:800)')
rent_parser.add_argument('-init_timeout', type=int, default=120, help='지정된 시간동안 인스턴스를 생성하지 못하면 중단합니다(기본:120 초)')
rent_parser.add_argument('-min_up', type=int, default=800, help='네트워크 업로드 속도의 최하치를 Mbps 입력합니다')

ssh_parser = subparsers.add_parser('ssh', help='ssh에 접속합니다')
ssh_parser.set_defaults(func=ssh)

stop_parser = subparsers.add_parser('stop', help='임대된 장비를 반납합니다')
stop_parser.set_defaults(func=stop)

search_parser = subparsers.add_parser('search', help='설정된 조건에 맞는 장비를 조회합니다.')
search_parser.add_argument('-gpu', type=str, help='임대를 원하는 GPU의 이름을 입력합니다.(h100, a100, 4090)', required=True)
search_parser.add_argument('-num_gpu', type=int, default=1, help='임대를 원하는 GPU의 개수를 입력합니다.')
search_parser.add_argument('-min_down', type=int, default=800, help='네트워크 다운로드 속도의 최하치를 Mbps 단위로 입력합니다(기본:800)')
search_parser.set_defaults(func=search)



def main():
  args = parser.parse_args()
  if args.command is None:
    parser.print_help()
    exit(-1)
  args.func(args)


if __name__ == "__main__":
  main()
