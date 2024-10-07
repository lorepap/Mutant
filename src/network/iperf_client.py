# src/network/iperf_client.py

import subprocess
import threading
import os
import logging
from src.utils.config import Config
import tempfile

class IperfClient(threading.Thread):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.process = None
        self._tag = f'{self.config.trace_u}-{self.config.bw}-{self.config.rtt}-{self.config.q_size}-{self.config.bw_factor}x'
        self._up_log_file = f'uplink-{self._tag}.log'
        self._down_log_file = f'downlink-{self._tag}.log'
        self._client_log_file = f'iperf_client-{self._tag}.log'
        self._setup_logging()


    def _setup_logging(self):
        log_dir = os.path.join(self.config.iperf_dir, 'iperf_client')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, self._client_log_file)
        
        self.logger = logging.getLogger(f'IperfClient-{self._tag}')
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    def run(self):
        self.logger.info("Starting IperfClient")
        # self._set_ip_forwarding()
        cmd = self._build_command()
        self.logger.info(f"Executing command: {cmd}")
        self.logger.info(f"[IPERF CLIENT] Mahimahi network scenario:\n rtt(ms) = {self.config.rtt}\n bw(Mbps) = {self.config.bw}\n q_size (pkts) = {self.config.q_size}\n bw_factor = {self.config.bw_factor}\n")
        self.logger.info(f"[IPERF CLIENT] Mahimahi traces:\n D: {self.config.trace_d}\n U: {self.config.trace_u}\n")
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = self.process.communicate()

            if stdout:
                self.logger.info(stdout.strip())
            if stderr:
                self.logger.error(stderr.strip())

        except Exception as e:
            self.logger.exception(f"Error while running IperfClient: {e}")
        # self.process.wait()

    def _build_command(self):
        
        iperf_script_path = self._generate_iperf_script()

        mahimahi_cmd = [
            'mm-delay', str(int(self.config.rtt/2)),
            'mm-link',
            self.config.get_trace_path(self.config.trace_u),
            self.config.get_trace_path(self.config.trace_d),
            '--uplink-queue=droptail',
            f'--uplink-queue-args=packets={self.config.q_size}',
            '--downlink-queue=droptail',
            f'--downlink-queue-args=packets={self.config.q_size}'
        ]

        if self.config.log_mahimahi:
            mahimahi_cmd.extend([
                f'--uplink-log={os.path.join(self.config.mahimahi_dir, self._up_log_file)}',
                f'--downlink-log={os.path.join(self.config.mahimahi_dir, self._down_log_file)}'
            ])

        # Add the script to be executed within mahimahi
        # iperf_cmd = [
        #     'sh',
        #     f'{iperf_script_path}'
        #     '-c',
        #     self.config.server_ip,
        #     '-t',
        #     str(self.config.iperf_time),
        #     '-p',
        #     str(self.config.server_port),
        #     '-o',
        #     os.path.join(self.config.iperf_dir, f'{self._tag}.txt'
        #     )
        # ]

        iperf_cmd = ['sh', iperf_script_path]

        cmd = mahimahi_cmd + iperf_cmd
        cmd = ' '.join(cmd)
        print(cmd)

        return cmd

    # def _set_ip_forwarding(self):
    #     cmd = ['sudo', 'sysctl', '-w', 'net.ipv4.ip_forward=1']
    #     res = subprocess.call(cmd)
    #     if res != 0:
    #         raise Exception("Unable to set ipv4 forwarding")

    def _generate_iperf_script(self):
        script_content = f"""#!/bin/bash

        # Run iperf3 with the specified parameters
        iperf3 -c "{self.config.server_ip}" \\
            -p "{self.config.server_port}" \\
            -t "{self.config.iperf_time}" \\
            --logfile "{os.path.join(self.config.iperf_dir, f'{self._tag}.txt')}"
        """
                
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name

        # Make the script executable
        os.chmod(temp_file_path, 0o755)

        return temp_file_path

    def stop(self):
        if self.process:
            self.process.terminate()
    
    def __del__(self):
        # Clean up the temporary script file
        script_path = self._build_command()[-1]
        if os.path.exists(script_path):
            os.unlink(script_path)