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
        self.MAHIMAHI_DIR = self.LOG_DIR / 'mahimahi'
        self.IPERF_DIR = self.LOG_DIR / 'iperf'
        self.COLLECTION_DIR = self.LOG_DIR / 'collection'
        self.RL_DIR = self.LOG_DIR / 'rl'
        self.CHECKPOINT_DIR = self.RL_DIR / 'checkpoints'
        self.MODEL_DIR = self.RL_DIR / 'models'
        self.HISTORY_DIR = self.RL_DIR / 'history'
        self.RUN_DIR = self.RL_DIR / 'run_logs'


        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.TRACES_DIR, exist_ok=True)
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        os.makedirs(self.MAHIMAHI_DIR, exist_ok=True)
        os.makedirs(self.IPERF_DIR, exist_ok=True)
        os.makedirs(self.COLLECTION_DIR, exist_ok=True)
        os.makedirs(self.RL_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.HISTORY_DIR, exist_ok=True)
        os.makedirs(self.RUN_DIR, exist_ok=True)

    def get_config_path(self, config_name):
        return self.CONFIG_DIR / config_name

    def get_trace_path(self, trace_name):
        return self.TRACES_DIR / trace_name

    def get_log_path(self, log_type, filename):
        if log_type == 'mahimahi':
            return self.MAHIMAHI_DIR / filename
        elif log_type == 'iperf':
            return self.IPERF_DIR / filename
        elif log_type == 'collection':
            return self.COLLECTION_DIR / filename
        elif log_type == 'rl':
            return self.RL_DIR / filename
        elif log_type == 'checkpoint':
            return self.CHECKPOINT_DIR / filename
        elif log_type == 'model':
            return self.MODEL_DIR / filename
        elif log_type == 'history':
            return self.HISTORY_DIR / filename
        elif log_type == 'run_log':
            return self.RUN_DIR / filename
        else:
            raise ValueError(f"Unknown log type: {log_type}")