# src/utils/logger.py

import os
import csv
from typing import List, Dict
import pandas as pd
from src.utils.config import Config

class Logger:
    def __init__(self, config: Config):
        self.config = config
        self.log_dir = config.collection_dir
        self.data_buffer = []
        # self.buffer_size = 100  # Number of entries to keep in memory before writing to file
        self.features = config.train_features
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
            writer = csv.DictWriter(f, fieldnames=self.features)
            writer.writeheader()

        return csv_file

    def log(self, data: Dict[str, float]):
        """
        Log a single data point.
        """
        # Ensure all features are present in the data
        for feature in self.features:
            if feature not in data:
                data[feature] = None  # or some default value

        self.data_buffer.append(data)

        # if len(self.data_buffer) >= self.buffer_size:
        self._write_buffer_to_csv()

    def _write_buffer_to_csv(self):
        """
        Write the buffered data to the CSV file.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.features)
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
