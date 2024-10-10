import subprocess
import time
import json
import os
import sys
import argparse

OUT_DIR = 'out'
TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'traces/5G')

class IperfServer:
    def __init__(self, port=5201):
        self.port = port
        self.process = None

    def start(self):
        cmd = ['iperf3', '-s', '-p', str(self.port), '-1', '-J']
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Iperf3 server started on port {self.port}")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Iperf3 server stopped")

class MahimahiIperfClient:
    def __init__(self, server_ip='localhost', port=5201, duration=5, delay=10, protocol='cubic', trace='default'):
        self.server_ip = server_ip
        self.port = port
        self.duration = duration
        self.delay = delay  # delay in ms
        self.protocol = protocol
        self.trace = trace

    def run(self):
        # Construct the iperf3 command
        iperf_cmd = f"./iperf.sh"
        trace_path = os.path.join(TRACE_DIR, f'{self.trace}')
        print(f"Using trace: {trace_path}")

        # Construct file names
        uplink_log = f'uplink_{self.protocol}_{self.trace}.log'
        downlink_log = f'downlink_{self.protocol}_{self.trace}.log'
        
        # Wrap the iperf3 command with mahimahi mm-delay and mm-link
        cmd = [
            'mm-delay', str(self.delay),
            'mm-link',
            trace_path,
            trace_path,
            f'--uplink-log={uplink_log}',
        ]
        cmd = ' '.join(cmd) + f" {iperf_cmd}"

        # Run the command using shell=True to properly handle the mahimahi piping
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        
        if stderr:
            print(f"Error: {stderr.decode()}")
            return None

        try:
            result = json.loads(stdout)
            return result, uplink_log
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {stdout.decode()}")
            return None, uplink_log

def run_mm_thr(log_file, output_file):
    cmd = f"./mm-thr 100 {log_file} > {output_file}"
    subprocess.run(cmd, shell=True, check=True)

def run_test(args):
    server = IperfServer()
    client = MahimahiIperfClient(delay=50, protocol=args.protocol, trace=args.trace)
    try:
        server.start()
        time.sleep(2)  # Give the server some time to start up
        subprocess.run("sudo sysctl net.ipv4.ip_forward=1", shell=True)
        subprocess.run(f"sudo sysctl net.ipv4.tcp_congestion_control={args.protocol}", shell=True)
        print("Starting mahimahi iperf3 client test...")
        result, uplink_log = client.run()

        if result:
            end_info = result.get('end', {})
            sender_info = end_info.get('streams', [{}])[0].get('sender', {})
            receiver_info = end_info.get('streams', [{}])[0].get('receiver', {})

            print("\nTest Results:")
            print(f"Sent: {sender_info.get('bytes', 0) / 1e6:.2f} MB")
            print(f"Received: {receiver_info.get('bytes', 0) / 1e6:.2f} MB")
            print(f"Bitrate: {sender_info.get('bits_per_second', 0) / 1e6:.2f} Mbps")
            print(f"Retransmits: {sender_info.get('retransmits', 0)}")
            print(f"RTT: {end_info.get('streams', [{}])[0].get('sender', {}).get('mean_rtt', 0)} ms")
        else:
            print("Test failed to produce results.")

        # Run mm-thr on uplink and downlink logs
        uplink_output = os.path.join(OUT_DIR, f'mm-thr_uplink_{args.protocol}_{args.trace}.txt')
        # downlink_output = os.path.join(OUT_DIR, f'mm-thr_downlink_{args.protocol}_{args.trace}.txt'
        
        print("Running mm-thr on uplink log...")
        run_mm_thr(uplink_log, uplink_output)
        # print("Running mm-thr on downlink log...")
        # run_mm_thr(downlink_log, downlink_output)

        print(f"mm-thr results saved to {uplink_output}")

    finally:
        server.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mahimahi iperf3 test")
    parser.add_argument("--protocol", type=str, default='cubic', help="TCP congestion control algorithm to use")
    parser.add_argument("--trace", type=str, default='5GBeachStationary_new_format', help="Name of the trace file (without .csv extension)")
    run_test(args=parser.parse_args())