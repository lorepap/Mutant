import os
from pathlib import Path

class ProjectPaths:
    def __init__(self):
        # Get the absolute path to the project root directory
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

        # Define paths relative to the project root
        self.SRC_DIR = self.PROJECT_ROOT / 'src'
        self.CONFIG_DIR = self.PROJECT_ROOT / 'config'
        self.LOG_DIR = self.PROJECT_ROOT / 'log'
        self.TRACES_DIR = self.PROJECT_ROOT / 'traces'

        # Specific subdirectories
        self.MAHIMAHI_LOG_DIR = self.LOG_DIR / 'mahimahi'
        self.IPERF_LOG_DIR = self.LOG_DIR / 'iperf'
        self.COLLECTION_LOG_DIR = self.LOG_DIR / 'collection'

        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.TRACES_DIR, exist_ok=True)
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        os.makedirs(self.MAHIMAHI_LOG_DIR, exist_ok=True)
        os.makedirs(self.IPERF_LOG_DIR, exist_ok=True)
        os.makedirs(self.COLLECTION_LOG_DIR, exist_ok=True)

    def get_config_path(self, config_name):
        return self.CONFIG_DIR / config_name

    def get_trace_path(self, trace_name):
        return self.TRACES_DIR / trace_name

    def get_log_path(self, log_type, filename):
        if log_type == 'mahimahi':
            return self.MAHIMAHI_LOG_DIR / filename
        elif log_type == 'iperf':
            return self.IPERF_LOG_DIR / filename
        elif log_type == 'collection':
            return self.COLLECTION_LOG_DIR / filename
        else:
            raise ValueError(f"Unknown log type: {log_type}")