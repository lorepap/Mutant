# src/utils/data_preprocessor.py

import numpy as np
from typing import Dict
from src.utils.config import Config
from src.utils.feature_extractor import FeatureExtractor

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = FeatureExtractor(config.train_stat_features, config.window_sizes)
        self.map_all_proto = {int(i): config.protocols[p] for i, p in enumerate(config.protocols)}
        self.inv_map_all_proto = {int(v): k for k, v in self.map_all_proto.items()}
        self.feature_vector = None

    def preprocess(self, data: Dict) -> np.ndarray:
        # Prepare feature vector
        data: dict = self._add_statistic_features(data)
        data: dict = self._filter_only_train_features(data)
        data: dict = self._add_one_hot(data)
        # self.feature_vector = np.array([val for feat, val in data.items() if feat in self.config.train_features])
        # preprocessed_data = np.hstack((self.feature_vector, one_hot_proto_id))
        # data = {feat: val for feat, val in data.items() if feat in self.config.train_features}
        # preprocessed_data = {**data, **{f'arm_{i}': one_hot_proto_id[i] for i in range(len(one_hot_proto_id))}}
        return data

    def _add_statistic_features(self, data: Dict):
        data_only_stats = {feature: data[feature] for feature in self.config.train_stat_features}
        self.feature_extractor.update(data_only_stats)        
        stats = self.feature_extractor.get_statistics()
        for window_size in self.config.window_sizes:
            for feature in self.config.train_stat_features:
                data[f'{feature}_avg_{window_size}'] = stats[window_size]['avg'][feature]
                data[f'{feature}_min_{window_size}'] = stats[window_size]['min'][feature]
                data[f'{feature}_max_{window_size}'] = stats[window_size]['max'][feature]
        return data

    def _filter_only_train_features(self, data: Dict):
        return {feature: data[feature] for feature in data.keys() if feature in self.config.train_features} 

    def _add_one_hot(self, data: Dict):
        one_hot_proto_id = self._one_hot_encode(self.inv_map_all_proto[data['crt_proto_id']], len(self.config.protocols))
        preprocessed_data = {**data, **{f'arm_{i}': one_hot_proto_id[i] for i in range(len(one_hot_proto_id))}}
        return preprocessed_data

    @staticmethod
    def _one_hot_encode(id: int, nchoices: int) -> np.ndarray:
        vector = np.zeros(nchoices, dtype=int)
        vector[id] = 1
        return vector

    @property
    def features(self):
        return self.config.all_features