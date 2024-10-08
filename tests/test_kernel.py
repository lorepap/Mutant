import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.network.iperf_client import IperfClient
from src.network.iperf_server import IperfServer
from src.network.netlink_communicator import NetlinkCommunicator
from src.utils.config import Config
from src.core.kernel_request import KernelRequest
import time

def main():
    config = Config('config.yaml')
    client = IperfClient(config)
    server = IperfServer(config)
    netlink_communicator = NetlinkCommunicator(config)
    kernel_thread = KernelRequest(config.num_fields_kernel, netlink_communicator)
    n_steps = 10

    netlink_communicator.init_kernel_communication()
    server.start()
    client.start()

    kernel_thread.start()
    # for step in range(n_steps):
    #     step_data  = collect_step_data(kernel_thread, config)
    #     print(f"Step {step + 1}/{n_steps}")
    netlink_communicator.set_protocol(0)
    collect_step_data(kernel_thread, config)
    kernel_thread.stop()
    print("Raw data collection completed. Processing data...")

def collect_step_data(kernel_thread: KernelRequest, config: Config):
    step_data = []
    # step_start = time.time()
    # kernel_thread.enable()
    while True:
        raw_data = kernel_thread.read_data()
        kernel_thread.queue.task_done()
        print(raw_data)
    kernel_thread.disable()
    kernel_thread.flush()

main()