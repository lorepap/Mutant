# src/core/kernel_request.py

import queue
import threading
import select
from src.network.netlink_communicator import NetlinkCommunicator

class KernelRequest(threading.Thread):
    def __init__(self, num_fields: int, comm: NetlinkCommunicator):
        super().__init__()
        self.num_fields = num_fields
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self.comm = comm
        self._enabled = False

    def run(self):
        while not self._stop_event.is_set():
            try:
                if self.comm.socket.fileno() == -1:
                    # Invalid file descriptor, break out of the loop
                    print("[KERNEL THREAD] Invalid file descriptor. Exiting...")
                    break
                
                # Use select with a timeout to implement a timer
                readable, _, _ = select.select([self.comm.socket], [], [], 10) 

                if not readable:
                    # No data received within the timeout period
                    print("[KERNEL THREAD] Timeout occurred. Exiting...")
                    break
                
                data = self._read_from_kernel()
                if data:
                    # data = self.comm.read_netlink_msg(data) # extract the data from the message
                    data_decoded = data.decode('utf-8')
                    if data_decoded == "0":
                        # Received "0" as a notification of completed setup
                        print("[KERNEL THREAD] ACK Received: Communication setup completed.")
                    elif data_decoded == "-1":
                        print("[KERNEL THREAD] Communication terminated")
                        break
                    else: # data received
                        split_data = data_decoded.split(';')
                        entry = [int(field) if field.isdigit() or (field[1:].isdigit() and field[0] == '-') else field for field in split_data]
                        # if self._enabled:
                        self.queue.put(entry)
                else:
                    print("[KERNEL THREAD] Exit event set. Exiting...")
                    break
            
            except Exception as e:
                print("[KERNEL THREAD] Exception occurred:", e)

    def _read_from_kernel(self):
         # print("[KERNEL THREAD return self.socket.return self.socket.recv(8192)recv(8192)D] Waiting for message...")
        msg = self.comm.receive_msg()
        # print("[KERNEL THREAD] Received message:", msg)
        if msg:
            data = self.comm.read_netlink_msg(msg)
            return data
        return None

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


    def read_data(self):
        return self.queue.get()

    def stop(self):
        self._stop_event.set()

    def flush(self):
        while not self.queue.empty():
            self.queue.get()
    
    @property
    def enabled(self):
        return self._enabled