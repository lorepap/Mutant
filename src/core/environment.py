import numpy as np
import os
from tf_agents.specs import array_spec
from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.specs import array_spec
from gym import spaces
import time
import yaml
import numpy as np
import traceback
from typing import Dict, List

from src.utils.feature_extractor import FeatureExtractor
from src.utils.logger import Logger
from src.core.kernel_request import KernelRequest
from src.network.netlink_communicator import NetlinkCommunicator
from src.utils.change_detection import ADWIN
from src.network.netlink_communicator import SET_PROTO_FLAG
from src.utils.data_preprocessor import DataPreprocessor

from collections import deque

class MabEnvironment(bandit_py_environment.BanditPyEnvironment):

    def __init__(self, config, kernel_thread: KernelRequest, observation_spec, action_spec, 
                 batch_size=1, normalize_rw: bool = False, change_detection: bool = False,
                ):
        super(MabEnvironment, self).__init__(observation_spec, action_spec)
        self.config = config
        self.kernel_thread = kernel_thread
        self._action_spec = action_spec
        self._batch_size = batch_size
        self.n_actions = int(self._action_spec.maximum+1)

        # Logger
        self.logger = Logger(self.config, log_dir=self.config.run_log_dir) # TODO check if we want to keep run_log_dir as the default
        self.logger.features = self.config.log_train_features # with epoch, step and reward (for logging and post-training analysis)
        # Load the configuration
        # self.proto_config = utils.parse_protocols_config()

        # Step counter
        self.num_steps = self.config.num_steps
        self._step_counter = 0
        self.crt_action = None

        # Reward
        # There's a lot of garbage here
        self._normalize_rw = normalize_rw
        self.curr_reward = 0
        self.epoch = 0
        self.step_wait = self.config.step_wait
        self.zeta = self.config.zeta
        self.kappa = self.config.kappa
        self._thr_history = deque(maxlen=1000)
        self._rtt_history = deque(maxlen=1000)

        if self.config.pool:
            self._map_actions = {action: int(p_id) for action, p_id in enumerate(self.config.pool)}
            self._inv_map_actions = {int(v): int(k) for k, v in self.map_actions.items()}
        else:
            raise ValueError("No pool of protocols selected")
        
        # Feature extractor
        # self.feature_settings = utils.parse_features_config()
        # self.feature_names = self.feature_settings['kernel_info'] # List of features
        # self.stat_features= self.feature_settings['train_stat_features']
        # self.non_stat_features = self.feature_settings['train_non_stat_features']
        # self.window_sizes = self.feature_settings['window_sizes']
        # self.training_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        # self.training_features = utils.get_training_features(all_features=self.non_stat_features, 
        #                         stat_features=self.stat_features, pool_size=len(self.proto_config))
        self.training_features = self.config.train_features
        # self.feat_extractor = FeatureExtractor(self.config.train_stat_features, self.config.window_sizes) # window_sizes=(10, 200, 1000)
        self.data_preprocessor = DataPreprocessor(config)

        # Netlink communicator
        self.netlink_communicator = NetlinkCommunicator(config)

        # For the logger (The runner should invoke set_logger() to enable logging)
        # self.logger = None
        # self.log_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        # self.initiated = False
        # self.curr_reward = 0
        # self._timestamp = utils.time_to_str()
        
        # Thread for kernel info
        # self.kernel_thread = KernelRequest(self.netlink_communicator, self.num_fields_kernel)
        # self._init_communication()

        # MPTS: returns top-K arms given all protocols in the pool at each run
        # We mantain the mapping of all the protocols for remapping after a change is detected (MPTS run)
        # self.map_all_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)}
        # self.inv_map_all_proto = {v: k for k, v in self.map_all_proto.items()} # protocol id -> action id
        # self._map_actions = None
        # self._inv_map_actions = None

        # Change detection: we keep a detector for each of the protocols in the pool
        self.detectors = None
        if change_detection:
            self.detectors = {int(i): ADWIN(delta=1e-9) for i in range(self.n_actions)}

    # def initialize(self):
    #     """
    #     Initialize the environment with the first set of protocols to run.
    #     MPTS algorithm is run first to select the first pool of K protocol.
    #     This has to be run before the first step.
    #     """
    #     self._initialize_protocols()
    #     time.sleep(0.5)
    #     first_set = self.mpts.run()
    #     print("[DEBUG] First set of protocols: ", [p for p in first_set])
    #     self.map_actions = {i: int(p) for i, p in enumerate(first_set)} # action id -> protocol id
    #     self.crt_action = None
    #     return self.map_actions

    def _observe(self, step_wait=None):
        s_tmp = np.array([])
        _log_tmp = []

        # Callbacks data
        self.features = []

        # Block to kernel thread to avoid samples from the next protocol
        self.kernel_thread.flush()
        self.kernel_thread.enable()

        if step_wait is None:
            step_wait = self.step_wait
        
        # Read and record data for step_wait seconds
        start = time.time()
        while float(time.time() - start) <= float(step_wait):
            # Read kernel features
            data = self._read_data()
            crt_proto_id = data[self.config.kernel_info.index('crt_proto_id')]
            processed_data = self._process_raw_data(data)
            if processed_data is None:
                continue
            _log_tmp.append(processed_data)
            self._thr_history.append(processed_data['thruput'])
            self._rtt_history.append(processed_data['rtt'])

            # Each protocol is equipped with a change detector (ADWIN) to detect changes in the network
            # When a protocol is run, at each step the corresponding window is updated with the average throughput value
            # Throughput history is cleared when a change in the network is detected -> new max reward is computed in apply_action()
            # We have to select all the actions to get the maximum throughput achievable (bandwidth estimation) and set the new max for the "new" network scenario
            if self.detectors:
                self.detectors[self._inv_map_actions[crt_proto_id]] \
                    .add_element(processed_data['thruput'])

            feat_vector = np.array([processed_data[feature] for feature in self.config.train_features])
            if s_tmp.size == 0:
                s_tmp = np.array(feat_vector).reshape(1, -1)
            else:
                s_tmp = np.vstack((s_tmp, np.array(feat_vector).reshape(1, -1)))

        self.kernel_thread.disable()

        # Observation as a mean of the samples
        # Check if the crt_proto_id is the same for all the samples (it must be, otherwise is an error))
        # print("[DEBUG] Current proto id", self.log_values[0][7])
        self.log_values = np.mean(s_tmp, axis=0).reshape(1, -1)
        # crt_proto_id_idx = self.non_stat_features.index('crt_proto_id')
        # self.log_values[0][crt_proto_id_idx] = int(collected_data['crt_proto_id'])

        self._observation = np.array(np.mean(s_tmp, axis=0), dtype=np.float32).reshape(1, -1)
        # self._observation = np.array(s_tmp)

        if self._observation.shape[1] != self._observation_spec.shape[0]:
            raise ValueError('The number of features in the observation should match the observation spec.')

        # We detect the change after the step_wait to collect all the samples on time
        if self.detectors:
            if self.detectors[self._inv_map_actions[processed_data['crt_proto_id']]].detected_change():
                print(f"Change detected at step {self._step_counter} | Thr: {processed_data['thruput']} | RTT: {processed_data['rtt']} | Loss: {processed_data['loss_rate']} | Proto: {processed_data['crt_proto_id']}")
                self.update_network_params() # TODO: potential sigfault here
                # if self._enable_mpts:
                    # accepted_actions = self.mpts.run()
                    # Remap new actions
                    # self.map_actions = {i: int(a) for i, a in enumerate(accepted_actions)}
                    # print("[DEBUG] New pool: ", [p for p in accepted_actions])
            
        self.kernel_thread.flush()

        return self._observation

    def _log_data(self, processed_data: List[Dict]):
        self.logger.log(processed_data)

    def _is_valid_sample(self, data: Dict) -> bool:
        return data['thruput'] <= 300 and data['rtt'] >= self.config.rtt
    
    def _process_raw_data(self, raw_data) -> List[Dict]:
        data = dict(zip(self.config.kernel_info, raw_data))
        data = self._convert(data)
        # Filter skewed samples
        if self._is_valid_sample(data):
            # Add statistic features
            return self.data_preprocessor.preprocess(data)
        return None

    def _change_cca(self, action):
        msg = self.netlink_communicator.create_netlink_msg(
            'SENDING ACTION', msg_flags=SET_PROTO_FLAG, msg_seq=int(self._map_actions[action[0]]))
        self.netlink_communicator.send_msg(msg)

    def _apply_action(self, action):
        # TODO Apply the action and get the reward. Here the reward is not the reward of the action selected, the the previous one
        self.crt_action = action
        # Change the CCA
        self._change_cca(action)
        self._observation = self._observe() # Update the observation to get the fresh reward

        # Compute the reward given the mean of the collected samples as the observation (shape: (1, num_features))
        # data = {name: value for name, value in zip(self.training_features, self._observation[0])}
        data = dict(zip(self.training_features, self._observation[0])) # TODO: check, potential bug (observation is a ndarray)
        
        # Compute the reward. Absolute value of the reward is kept for logging.
        # reward = self._compute_reward(data['thruput'], data['loss_rate'], data['rtt'])
        reward = self._compute_reward(data['thruput'], data['loss_rate'], data['rtt'])
        
        # Log the single observation with the absolute reward
        if self.logger:
            log_data = [self._step_counter] + [val for val in self.log_values[0]] + [reward]
            self._log_data(log_data)
            self._step_counter+=1
        
        # Reward normalization
        if self._normalize_rw:
            if len(self._thr_history) > 0:
                max_thr = max(self._thr_history)
                min_rtt = min(self._rtt_history)
            self._max_rw = self._compute_reward(max_thr, 0, min_rtt)
            reward = reward/self._max_rw

        reward = np.array(reward).reshape(self.batch_size)
        
        return reward
        
    def _compute_reward(self, thr, loss_rate, rtt):
        return (pow(abs((thr - self.zeta * loss_rate)), self.kappa) / (rtt*10**-3) )  # thr in Mbps; rtt in s
    
    def update_network_params(self):
        # In the same step we "refresh" the value of the max reward by running all the actions and get the throughput of the network
        # This approach will avoid that the reward is normalized with a value that is not the maximum achievable and the policy gets stuck on a suboptimal action
        self._thr_history.clear()
        self._rtt_history.clear()
        for a in range(self._action_spec.maximum+1):
            self._change_cca([a])
            self.kernel_thread.enable()
            time.sleep(0.1)
            self.kernel_thread.disable()
            while not self.kernel_thread.queue.empty():
                _d = self._read_data()
                _c_d = dict(zip(self.config.kernel_info, _d))
                thr = _c_d['thruput']*1e-6
                rtt = _c_d['rtt']*1e-3
                # print("[DEBUG] Update: action", self._map_proto[a], "Thr: ", thr, "RTT: ", rtt)
                if thr < 192 and rtt >= self.config.rtt: # TODO remove the bw check here
                    self._thr_history.append(thr)
                    self._rtt_history.append(rtt)

    def close(self):
        self.kernel_thread.stop()

    def _read_data(self):
        kernel_info = self.kernel_thread.queue.get()
        # self.kernel_thread.queue.task_done()
        return kernel_info

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    def batched(self) -> bool:
        return True

    @staticmethod
    def _convert(data: Dict) -> Dict:
        data['thruput'] *= 1e-6  # bps -> Mbps
        data['rtt'] *= 1e-3  # us -> ms
        data['loss_rate'] *= 0.01  # percentage -> ratio
        return data

    # Gargabe
    @property
    def step_counter(self):
        return self._step_counter
    
    @step_counter.setter
    def step_counter(self, value):
        self._step_counter = value

    @property
    def map_actions(self):
        return self._map_actions
    
    
    @map_actions.setter
    def map_actions(self, value):
        self._map_actions = value # actions -> protocols ID
        self._inv_map_actions = {int(v): int(k) for k, v in self._map_actions.items()} # protocols ID -> actions

    @property 
    def inv_map_actions(self):
        return self._inv_map_actions
    
    @property
    def timestamp(self):
        return self._timestamp
    
    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value 