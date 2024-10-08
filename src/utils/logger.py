# src/utils/logger.py

import os
import csv
from typing import List, Dict
import pandas as pd
from src.utils.config import Config

class Logger:
    def __init__(self, config: Config, log_dir: str = None):
        self.config = config
        self.log_dir = config.collection_dir if log_dir is None else log_dir
        self.data_buffer = []
        self.buffer_dim = 10 # Every buffer_dim samples, write to disk
        self._features = config.train_features # can be set by setter
        self.csv_file = self._create_csv_file()

    def _create_csv_file(self) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        # filename = f'collection_log_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        trace_name = "".join(self.config.get_trace_path(
            self.config.trace_d).split("/")[-1]) \
            .split(".csv")[0]

        filename = 'collection_' \
            + trace_name + '_' \
            + f'{self.config.rtt}_{self.config.q_size}.csv'
        
        csv_file = os.path.join(self.log_dir, filename)
        print(f"Logging data to {csv_file}")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._features)
            writer.writeheader()

        return csv_file

    def log(self, data: Dict[str, float]):
        """
        Log a single data point.
        """
        self.data_buffer.append(data)

        if len(self.data_buffer) >= self.buffer_dim:
            self._write_buffer_to_csv()

    def _write_buffer_to_csv(self):
        """
        Write the buffered data to the CSV file.
        """
        with open(self.csv_file, 'a', newline='') as f:
            if self.data_buffer and isinstance(self.data_buffer[0], dict):
                writer = csv.DictWriter(f, fieldnames=self._features)
                writer.writerows(self.data_buffer)
            else:
                writer = csv.writer(f)
                writer.writerows(self.data_buffer)
        
        self.data_buffer.clear()

    def flush(self):
        """
        Write any remaining buffered data to the CSV file.
        """
        if self.data_buffer:
            self._write_buffer_to_csv()

    def get_logged_data(self) -> pd.DataFrame:
        """
        Read the logged data into a pandas DataFrame.
        """
        return pd.read_csv(self.csv_file)

    def __del__(self):
        """
        Ensure all data is written when the Logger object is destroyed.
        """
        self.flush()

    @property
    def features(self) -> List[str]:
        return self._features
    
    @features.setter
    def features(self, features: List[str]):
        self._features = features
        self.csv_file = self._create_csv_file()