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

        # Calculate q_size
        bdp = self.bw * self.rtt  # Mbits
        mss = 1488  # bytes
        self.q_size = int(self.bdp_mult * bdp * 10**3 / (8*mss))  # packets

        # Set trace files
        if self.bw_factor == 1:
            self.trace_d = f'wired{int(self.bw)}' # TODO: add trace type (e.g., 'wired', 'cellular')
            self.trace_u = self.trace_d
        else:
            self.trace_d = f'wired{int(self.bw)}-{self.bw_factor}x-d'
            self.trace_u = f'wired{int(self.bw)}-{self.bw_factor}x-u'
        
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

    def get_trace_path(self, trace_name):
        return str(self.paths.get_trace_path(trace_name))

    def get_log_path(self, log_type, filename):
        return str(self.paths.get_log_path(log_type, filename))

    def _get_train_features(self):
        train_features = []
        train_features.extend(self.train_non_stat_features)
        train_features.extend(self.train_stat_features)
        for stat_feature in self.train_stat_features:
            for w_size in self.window_sizes:
                train_features.extend([f"{stat_feature}_avg_{w_size}", f"{stat_feature}_min_{w_size}", f"{stat_feature}_max_{w_size}"])
        # One hot encoding features. N_actions = 2**n_features, so n_features= log2(n_actions)
        train_features.extend([f"arm_{i}" for i in range(len(self.protocols))])
        return train_features