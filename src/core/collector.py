"""
TODO: add data preprocessor for one-hot encoding
"""


from typing import Dict, List
import time
import numpy as np
from src.core.kernel_request import KernelRequest
from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.feature_extractor import FeatureExtractor
from src.utils.data_preprocessor import DataPreprocessor
from src.network.netlink_communicator import NetlinkCommunicator

class Collector:
    def __init__(self, n_steps: int, config: Config, netcomm: NetlinkCommunicator):
        self.n_steps = n_steps
        self.config = config
        self.netcomm = netcomm
        self.kernel_request = KernelRequest(config.num_fields_kernel, netcomm)
        self.data_preprocessor = DataPreprocessor(config)
        self.logger = Logger(config)
        self.features = self.config.kernel_info
        self.raw_data = []

    def run_collection(self):
        self.kernel_request.start()
        # list of protocols
        for protocol, id in self.config.protocols.items():
            print(f"Running {protocol} ({id}) for {self.n_steps} steps...")
            self.netcomm.change_cca(id)
            for step in range(self.n_steps):
                step_data = self._collect_step_data()
                self.raw_data.extend(step_data)
                # print(f"Step {step + 1}/{self.n_steps}")

        self.kernel_request.stop()
        print("Raw data collection completed. Processing data...")

        processed_data = self._process_raw_data()
        self._log_data(processed_data)
        
        print("Data processing and logging completed.")

        self.logger.flush()

    def _process_raw_data(self) -> List[Dict]:
        processed_data = []
        
        for raw_sample in self.raw_data:
            data = dict(zip(self.features, raw_sample))
            data = self._convert(data)
            
            # Filter skewed samples
            if self._is_valid_sample(data):
                # Add statistic features
                preprocessed_data = self.data_preprocessor.preprocess(data)
                processed_data.append(preprocessed_data)
        
        return processed_data

    def _log_data(self, processed_data: List[Dict]):
        for data in processed_data:
            self.logger.log(data)
        self.logger.flush()

    def _collect_step_data(self) -> List[List]:
        step_data = []
        step_start = time.time()
        self.kernel_request.enable()
        while time.time() - step_start <= self.config.step_wait:
            raw_data = self.kernel_request.read_data()
            self.kernel_request.queue.task_done()
            # print(raw_data)
            if raw_data:
                step_data.append(raw_data)
        # print(raw_data)
        self.kernel_request.disable()
        self.kernel_request.flush()

        return step_data

    def _is_valid_sample(self, data: Dict) -> bool:
        return data['thruput'] <= 300 and data['rtt'] >= self.config.rtt

    # def _collect_step_data(self) -> List[Dict]:
    #     step_data = []
    #     step_start = time.time()
    #     while time.time() - step_start <= self.config.step_wait:
    #         data = self.kernel_request.read_data()
    #         self.kernel_request.queue.task_done()
    #         data = {feature: data[i] for i, feature in enumerate(self.features)}
    #         if self._is_valid_data(data):
    #             step_data.append(self._preprocess_data(data))
    #     return step_data

    # def _is_valid_data(self, data: Dict) -> bool:
    #     return data['thruput'] <= 192 and data['rtt'] >= self.config.rtt


    @staticmethod
    def _convert(data: Dict) -> Dict:
        data['thruput'] *= 1e-6  # bps -> Mbps
        data['rtt'] *= 1e-3  # us -> ms
        data['loss_rate'] *= 0.01  # percentage -> ratio
        return data

    # def _process_and_log_data(self, step_data: List[Dict]):
    #     avg_data = self._calculate_average_data(step_data)
    #     features = self.feature_extractor.extract_features(avg_data)
    #     self.logger.log(features)

    # def _calculate_average_data(self, step_data: List[Dict]) -> Dict:
    #     return {
    #         feature: np.mean([d[feature] for d in step_data])
    #         for feature in self.config.feature_names
    #     }