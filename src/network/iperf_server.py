# src/network/iperf_server.py

import subprocess
import threading
import os

class IperfServer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None

    def start(self):
        cmd = ['iperf3', '-s', '-p', str(self.config.server_port),
                '--logfile', 
                os.path.join(self.config.iperf_dir, 'server.log')]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, text=True)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()