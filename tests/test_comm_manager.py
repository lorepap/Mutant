import subprocess
import time
import json
import os

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
    def __init__(self, server_ip='localhost', port=5201, duration=5, delay=50):
        self.server_ip = server_ip
        self.port = port
        self.duration = duration
        self.delay = delay  # delay in ms

    def run(self):
        # Construct the iperf3 command
        iperf_cmd = f"./iperf.sh"
        
        # Wrap the iperf3 command with mahimahi mm-delay
        cmd = f"mm-delay {self.delay} {iperf_cmd}"
        
        # Run the command using shell=True to properly handle the mahimahi piping
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        
        if stderr:
            print(f"Error: {stderr.decode()}")
            return None

        try:
            result = json.loads(stdout)
            return result
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {stdout.decode()}")
            return None

def run_test():
    server = IperfServer()
    client = MahimahiIperfClient(delay=50)  # 50ms delay

    try:
        server.start()
        time.sleep(2)  # Give the server some time to start up

        print("Starting mahimahi iperf3 client test...")
        result = client.run()

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

    finally:
        server.stop()

if __name__ == "__main__":
    run_test()