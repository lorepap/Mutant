# src/core/comm_manager.py

from src.network.iperf_client import IperfClient
from src.network.iperf_server import IperfServer
from src.network.netlink_communicator import NetlinkCommunicator
from src.utils.config import Config
import subprocess
import os

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class CommManager:
    def __init__(self, config: Config):
        self.config = config
        self.client = IperfClient(config)
        self.server = IperfServer(config)
        self.netlink_communicator = NetlinkCommunicator()

    def start_communication(self):
        self.netlink_communicator.init_kernel_communication()
        self.server.start()
        self.client.start()

    def stop_communication(self):
        self.client.stop()
        self.server.stop()
        self.netlink_communicator.stop_kernel_communication()