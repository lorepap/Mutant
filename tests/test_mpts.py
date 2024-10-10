import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from src.mab.encoding_network import EncodingNetwork
from src.core.kernel_request import KernelRequest
from src.core.comm_manager import CommManager
from src.utils.trace_manager import CellularTraceManager
from src.utils.paths import ProjectPaths

import json
import time
import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from argparse import ArgumentParser 

from src.utils.feature_extractor import FeatureExtractor
from src.utils.logger import Logger
from src.utils.change_detection import PageHinkley, ADWIN
from src.core.mpts import MPTS
from src.utils.config import Config

from collections import deque, Counter
from src.core.kernel_request import KernelRequest


def compute_reward(kappa, zeta, thr, loss_rate, rtt):
        # Reward is normalized if the normalize_rw is true, otherwise max_rw = 1
        return (pow(abs((thr - zeta * loss_rate)), kappa) / (rtt*10**-3) )  # thr in Mbps; rtt in s

def save_arms_to_file(arms_list, filename):
    with open(filename, 'w') as f:
        json.dump(arms_list, f)
    print(f"Arms selection saved to {filename}")

def compute_most_selected_arms(arms_list):
    flat_arms = [arm for step_arms in arms_list for arm in step_arms]
    counter = Counter(flat_arms)
    most_common = counter.most_common()
    return most_common

def save_most_selected_arms(most_common, filename):
    with open(filename, 'w') as f:
        json.dump(most_common, f)
    print(f"Most selected arms saved to {filename}")

            
if __name__ == "__main__":
        parser = ArgumentParser()
        # Accept a list of policies to be used in the environment - if it's not passed, use all of them
        parser.add_argument('--rtt', '-r', default=20, type=int)
        # parser.add_argument('--bw', '-b', default=12, type=int)
        parser.add_argument('--bdp_mult', '-q', default=1, type=int)
        parser.add_argument('--bw_factor', '-f', default=1, type=int)
        parser.add_argument('-T', '-t', default=100, type=int)
        parser.add_argument('-K', '-k', default=4, type=int)
        # parser.add_argument('--trace', '-n', nargs='+', default=list('beach-stationary'))
        args = parser.parse_args()

        trace_man = CellularTraceManager(paths=ProjectPaths())
        traces = trace_man.cellular_names
        for trace_name in traces:
            if os.path.exists(f'most_selected_arms_{trace_name}.json'):
                print(f"Skipping trace {trace_name} as it has already been processed.")
                continue
            config = Config('config.yaml')
            feature_settings = config.train_features
            policies = config.protocols
            config.trace_type = 'cellular'
            config.cellular_trace_name = trace_name
        
            # Reward
            zeta = config.zeta
            kappa = config.kappa

            # Change detection
            detector = ADWIN(delta=1e-8)
            
            # Communication setup (comm manager + kernel thread)
            logdir = 'test_mpts/log'
            comm_manager = CommManager(config) #iperf_dir, time
            k_thread = KernelRequest(len(config.kernel_info), comm_manager.netlink_communicator)

            # Loop
            step_cnt = 0
            
            map_proto = {i: policies[p] for i, p in enumerate(policies)}
            # MPTS
            mpts = MPTS(config, map_proto, thread=k_thread, net_channel=comm_manager.netlink_communicator)
            
            k_thread.start() # start the kernel thread
            comm_manager.start_communication()
            
            selected_arms = []
            n_steps = 10
            while step_cnt < n_steps:
                    comm_manager.netlink_communicator.set_protocol(0) # initialize with the first protocol
                    start_cmp_time = time.time()
                    arms = mpts.run()
                    # Expected outcome: the arms selected should be the same for the same trace
                    print(f"Step {step_cnt} Selected arms: {arms}")
                    selected_arms.append(arms)
                    print(f"Time taken: {time.time() - start_cmp_time}")
                    step_cnt+=1
            
            comm_manager.stop_communication()

            # Save selected arms to file
            save_arms_to_file(selected_arms, 'selected_arms.json')

            # Compute and save most selected arms
            most_common_arms = compute_most_selected_arms(selected_arms)
            save_most_selected_arms(most_common_arms, f'most_selected_arms_{trace_name}.json')

            print("Execution completed. Selected arms and most common arms have been saved.")