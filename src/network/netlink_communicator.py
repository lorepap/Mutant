import os
from pickle import TRUE
import socket
import struct
import subprocess
import time
from src.utils.config import Config

NETLINK_TEST = 25
END_COMM_FLAG = 0
INIT_COMM_FLAG = 1
SET_PROTO_FLAG = 2
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class NetlinkCommunicator():
    _socket_obj = None

    def __init__(self, config: Config):
        self.config = config
        self._init_proto()
        self.socket = self.create_socket()
        self.socket.setblocking(False)
        self.set_socket_buffer_size(recv_size=262144, send_size=262144) # 256KB
        self.set_protocol(self.config.protocols['cubic']) # set initial protocol
    
    @classmethod
    def create_socket(cls):
        print("Creating netlink socket to communicate with the kernel...")
        if cls._socket_obj is None:
            s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_TEST)
            s.bind((os.getpid(), 0))
            cls._socket_obj = s
        return cls._socket_obj
    
    def _init_proto(self) -> None:
        if self.is_kernel_initialized():
            print('Mutant protocol has been already set up\n')
            return
        cmd1 = os.path.join(PROJ_ROOT, "ins_proto.sh")
        cmd2 = os.path.join(PROJ_ROOT, "init_kernel.sh")
        subprocess.call(cmd1)
        subprocess.call(cmd2)
        print("Kernel module set up")

    def initialize_protocols(self):
        print("Initializing protocols...")
        for p, id in self.config.protocols.items():
            print(f"Initializing protocol: {p} ({id})")
            start = time.time()
            while time.time() - start < 0.5:
                msg = self.create_netlink_msg(
                    'SENDING ACTION', msg_flags=SET_PROTO_FLAG, msg_seq=int(id))
                self.send_msg(msg)

    def change_cca(self, protocol):
        msg = self.create_netlink_msg(
            'SENDING ACTION', msg_flags=2, msg_seq=protocol)
        self.send_msg(msg)

    def close_socket(self):
        self.socket.close()

    def create_netlink_msg(self, data, msg_type=0, msg_flags=0, msg_seq=0, msg_pid=os.getpid()):
        payload = f'{data}\0'
        header_size = 16
        payload_size = len(payload)
        msg_len = header_size + payload_size
        header = struct.pack("=LHHLL", msg_len, msg_type, msg_flags, msg_seq, msg_pid)
        msg = header + payload.encode()
        return msg

    def send_msg(self, msg):
        self.socket.send(msg)

    def set_socket_buffer_size(self, recv_size=10000, send_size=10000):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_size)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_size)

    def receive_msg(self):
        return self.socket.recv(8192)

    def read_netlink_msg(self, msg):
        value_len, value_type, value_flags, value_seq, value_pid = struct.unpack("=LHHLL", msg[:16])
        data = msg[16:value_len]
        return data
    
    def init_kernel_communication(self):
        print("Initiating communication...")
        msg = self.create_netlink_msg(
            'INIT_COMMUNICATION', msg_flags=INIT_COMM_FLAG)
        # Send init communication message to kernel: response is handled by kernel thread
        self.send_msg(msg)
        print("Communication initiated")

    def stop_kernel_communication(self) -> None:
        msg = self.create_netlink_msg(
            'END_COMMUNICATION', msg_flags=END_COMM_FLAG)
        self.send_msg(msg)
        self.close_socket()

    def set_protocol(self, protocol_id):
        msg = self.create_netlink_msg(
            'SENDING ACTION', msg_flags=SET_PROTO_FLAG, msg_seq=protocol_id)
        self.send_msg(msg)

    @staticmethod
    def is_kernel_initialized() -> bool:
        cmd = ['cat', '/proc/sys/net/ipv4/tcp_congestion_control']
        res = subprocess.check_output(cmd)
        protocol = res.strip().decode('utf-8')
        return protocol == 'mutant'
    