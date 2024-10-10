from src.utils.paths import ProjectPaths
from src.utils.trace_manager import TraceManager
import yaml
from typing import Dict, Any
import os

class Config:
    def __init__(self, config_name: str):
        self._paths = ProjectPaths()
        self._trace_man = TraceManager(self._paths)
        self._config = self._load_config(config_name)
        self._derived_values: Dict[str, Any] = {}
        self._add_directory_paths()
        self._update_derived_values()

    def _load_config(self, config_name: str) -> Dict[str, Any]:
        config_path = self._paths.get_config_path(config_name)
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def __getattr__(self, name):
        if name == 'paths':
            return self._paths
        if name == 'trace_man':
            return self._trace_man
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ['_paths', '_trace_man', '_config']:
            super().__setattr__(name, value)
        elif hasattr(self, '_config'):
            # Add or update the attribute in _config
            self._config[name] = value
            self._update_derived_values()
        else:
            super().__setattr__(name, value)

    def _add_directory_paths(self):
        # Add all directory attributes from ProjectPaths to the config
        for attr_name in dir(self._paths):
            if attr_name.endswith('_DIR') and not attr_name.startswith('_'):
                dir_path = getattr(self._paths, attr_name)
                self._config[attr_name.lower()] = dir_path
                # Create the directory if it doesn't exist
                os.makedirs(dir_path, exist_ok=True)

    def _update_derived_values(self):
        if all(key in self._config for key in ['bw', 'rtt', 'bdp_mult']):
            self._config['q_size'] = self._calculate_q_size()
            if 'trace_type' in self._config and self._config['trace_type'] == 'wired':
                self._config['trace_d'], self._config['trace_u'] = self._get_traces()
        
        if all(key in self._config for key in ['trace_type', 'cellular_trace_name']) and self._config['trace_type'] == 'cellular':
            self._config['trace_d'], self._config['trace_u'] = self._get_traces()
        
        if all(key in self._config for key in ['train_non_stat_features', 'train_stat_features', 'window_sizes', 'protocols']):
            self._config['train_features'] = self._get_train_features()
            self._config['log_train_features'] = ['step'] + self._config['train_features'] + ['reward']

    def _calculate_q_size(self) -> int:
        bdp = self._config['bw'] * self._config['rtt']  # Mbits
        mss = 1488  # bytes
        return int(self._config['bdp_mult'] * bdp * 10**3 / (8*mss))  # packets

    def _get_traces(self):
        trace_type = self._config.get('trace_type')
        if trace_type == 'wired':
            trace_d = self._trace_man.get_trace_name(trace_type, bw=self._config['bw'], bw_factor=self._config['bw_factor'], direction='down')
            trace_u = self._trace_man.get_trace_name(trace_type, bw=self._config['bw'], bw_factor=self._config['bw_factor'], direction='up')
        elif trace_type == 'cellular':
            cellular_trace = self.trace_man.get_trace_name('cellular', name=self._config.get('cellular_trace_name'))
            trace_d = trace_u = cellular_trace
        else:
            if not trace_type:
                raise ValueError("Trace type not provided")
            else:
                raise ValueError(f"Unknown trace type '{trace_type}'")

        if not self._trace_man.get_trace_path(trace_d):
            raise ValueError(f"Downlink trace '{trace_d}' not found")
        if not self._trace_man.get_trace_path(trace_u):
            raise ValueError(f"Uplink trace '{trace_u}' not found")

        return trace_d, trace_u

    def _get_train_features(self):
        features = self._config.get('train_non_stat_features', []) + [
            feat for feat in self._config.get('train_stat_features', [])
            if feat not in self._config.get('train_non_stat_features', [])
        ]
        for stat_feature in self._config.get('train_stat_features', []):
            for w_size in self._config.get('window_sizes', []):
                features.extend([f"{stat_feature}_avg_{w_size}", f"{stat_feature}_min_{w_size}", f"{stat_feature}_max_{w_size}"])
        features.extend([f"arm_{i}" for i in range(len(self._config.get('protocols', {})))])
        return features

    def get_trace_path(self, trace_name):
        return self._trace_man.get_trace_path(trace_name)

    def get_log_path(self, log_type, filename):
        return str(self._paths.get_log_path(log_type, filename))

    def get_id(self, protocol):
        return self._config['protocols'][protocol]

    def get_protocol_name(self, id):
        return next(p for p, pid in self._config['protocols'].items() if pid == id)