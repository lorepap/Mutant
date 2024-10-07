# src/utils/config.py
from src.utils.paths import ProjectPaths
import yaml

class Config:
    def __init__(self, config_name: str):
        self.paths = ProjectPaths()
        config_path = self.paths.get_config_path(config_name)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.__dict__.update(config)

        # Initialize params
        self._bw = config.get('bw')
        self._bw_factor = config.get('bw_factor')
        self._rtt = config.get('rtt')
        self._bdp_mult = config.get('bdp_mult')
        self._num_steps = config.get('num_steps')
        self._num_fields_kernel = config.get('num_fields_kernel')
        self._reward = config.get('reward')
        self._step_wait = config.get('step_wait')
        self._pool_size = config.get('pool_size')

        self._update_derived_values()

        self.mahimahi_dir = self.paths.MAHIMAHI_LOG_DIR
        self.iperf_dir = self.paths.IPERF_LOG_DIR
        self.collection_dir = self.paths.COLLECTION_LOG_DIR

        # log booleans
        self.log_mahimahi = True
        self.log_iperf = True
        
        # connection params
        self.server_port = 5201
        self.server_ip = '10.172.13.12' # TODO: parametrize
        self.iperf_time = 86400
        self.train_features = self._get_train_features()

        

    @property
    def bw(self):
        return self._bw

    @bw.setter
    def bw(self, value):
        self._bw = value
        self._update_derived_values()

    @property
    def bw_factor(self):
        return self._bw_factor

    @bw_factor.setter
    def bw_factor(self, value):
        self._bw_factor = int(value)
        self._update_derived_values()

    @property
    def rtt(self):
        return self._rtt

    @rtt.setter
    def rtt(self, value):
        self._rtt = value
        self._update_derived_values()

    @property
    def bdp_mult(self):
        return self._bdp_mult

    @bdp_mult.setter
    def bdp_mult(self, value):
        self._bdp_mult = value
        self._update_derived_values()

    @property
    def num_steps(self):
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value):
        self._num_steps = value

    @property
    def num_fields_kernel(self):
        return self._num_fields_kernel

    @num_fields_kernel.setter
    def num_fields_kernel(self, value):
        self._num_fields_kernel = value

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value

    @property
    def step_wait(self):
        return self._step_wait

    @step_wait.setter
    def step_wait(self, value):
        self._step_wait = value

    @property
    def pool_size(self):
        return self._pool_size

    @pool_size.setter
    def pool_size(self, value):
        self._pool_size = value

    def _update_derived_values(self):
        # Calculate q_size
        bdp = self.bw * self.rtt  # Mbits
        mss = 1488  # bytes
        self.q_size = int(self.bdp_mult * bdp * 10**3 / (8*mss))  # packets

        # Set trace files
        if self.bw_factor == 1:
            self.trace_d = f'wired{int(self.bw)}' # TODO: add trace type (e.g., 'wired', 'cellular')
            self.trace_u = self.trace_d
        else:
            self.trace_d = f'wired{int(self.bw)}-{int(self.bw_factor)}x-d'
            self.trace_u = f'wired{int(self.bw)}-{int(self.bw_factor)}x-u'

    def get_trace_path(self, trace_name):
        return str(self.paths.get_trace_path(trace_name))

    def get_log_path(self, log_type, filename):
        return str(self.paths.get_log_path(log_type, filename))

    def _get_train_features(self):
        train_features = []
        train_features.extend(self.train_non_stat_features)
        train_features.extend([feat for feat in self.train_stat_features if feat not in self.train_non_stat_features])
        for stat_feature in self.train_stat_features:
            for w_size in self.window_sizes:
                train_features.extend([f"{stat_feature}_avg_{w_size}", f"{stat_feature}_min_{w_size}", f"{stat_feature}_max_{w_size}"])
        # One hot encoding features. N_actions = 2**n_features, so n_features= log2(n_actions)
        train_features.extend([f"arm_{i}" for i in range(len(self.protocols))])
        return train_features