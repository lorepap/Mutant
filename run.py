# src/main.py

import argparse
import tensorflow as tf
from tf_agents.specs import tensor_spec
from src.core.environment import MabEnvironment
from src.core.runner import RLRunner
from src.core.comm_manager import CommManager
from src.core.kernel_request import KernelRequest
from src.utils.config import Config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run collection with specified parameters")
    parser.add_argument("--n_steps", type=int, help="Number of steps", default=5)
    parser.add_argument("--bw", type=int, help="Bandwidth in Mbps")
    parser.add_argument("--bw_factor", type=float, help="Bandwidth factor")
    parser.add_argument("--delay", type=int, help="Delay in ms")
    parser.add_argument("--rtt", type=int, help="RTT in ms")
    parser.add_argument("--bdp_mult", type=float, help="BDP multiplier")
    parser.add_argument("--num_steps", type=int, help="Number of steps")
    parser.add_argument("--num_fields_kernel", type=int, help="Number of fields in kernel")
    parser.add_argument("--zeta", type=float, help="Zeta value for reward calculation")
    parser.add_argument("--kappa", type=float, help="Kappa value for reward calculation")
    parser.add_argument("--step_wait", type=float, help="Wait time between steps")
    parser.add_argument("--pool_size", type=int, help="Pool size")
    return parser.parse_args()

def update_config_from_args(config, args):
    if args.bw is not None:
        config.bw = args.bw
    if args.bw_factor is not None:
        config.bw_factor = args.bw_factor
    if args.delay is not None:
        config.delay = args.delay
    if args.rtt is not None:
        config.rtt = args.rtt
    if args.bdp_mult is not None:
        config.bdp_mult = args.bdp_mult
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.num_fields_kernel is not None:
        config.num_fields_kernel = args.num_fields_kernel
    if args.zeta is not None or args.kappa is not None:
        reward = config.reward.copy()
        if args.zeta is not None:
            reward['zeta'] = args.zeta
        if args.kappa is not None:
            reward['kappa'] = args.kappa
        config.reward = reward
    if args.step_wait is not None:
        config.step_wait = args.step_wait
    if args.pool_size is not None:
        config.pool_size = args.pool_size

def setup_environment(config, kernel_thread):
    observation_spec = tensor_spec.TensorSpec(
        shape=(len(config.train_features),), dtype=tf.float32, name='observation')
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=len(config.pool)-1, name='action')
    
    environment = MabEnvironment(
        config,
        kernel_thread,
        observation_spec, 
        action_spec, 
        normalize_rw=True, 
        change_detection=True
    )
    return environment

def main():
    args = parse_arguments()
    config = Config('config.yaml')
    update_config_from_args(config, args)
    comm_manager = CommManager(config)
    kernel_request = KernelRequest(config.num_fields_kernel, comm_manager.netlink_communicator) # we can just setup a universe with all instantiated object
    pool = [0, 1, 2, 3] # debug
    config.pool = pool
    env = setup_environment(config, kernel_request)
    runner = RLRunner(config, env)

    try:
        comm_manager.start_communication()
        kernel_request.start()
        # Here we run MPTS first
        # comm_manager.netlink_communicator.initialize_protocols() # only for mpts
        # pool = self.mpts.run() # TODO check
        print("Running Mutant...")
        runner.train()
    finally:
        comm_manager.stop_communication()

if __name__ == "__main__":
    main()