# src/utils/trace_manager.py

from src.utils.paths import ProjectPaths
from abc import ABC, abstractmethod

class BaseTraceManager(ABC):
    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    @abstractmethod
    def get_trace_name(self, **kwargs):
        pass

    def get_trace_path(self, trace_name):
        return str(self.paths.get_trace_path(trace_name))

class WiredTraceManager(BaseTraceManager):
    def get_trace_name(self, bw, bw_factor, direction):
        if bw_factor == 1:
            return f'wired{int(bw)}'
        else:
            dir_suffix = 'd' if direction == 'down' else 'u'
            return f'wired{int(bw)}-{int(bw_factor)}x-{dir_suffix}'

class CellularTraceManager(BaseTraceManager):
    def __init__(self, paths: ProjectPaths):
        super().__init__(paths)
        self.cellular_traces = [
            "5GBeachStationary.csv",
            "5GBeachStationary2.csv",
            "5GBeachStationary3.csv",
            "5GCityDrive.csv",
            "5GCorniche.csv",
            "5GCornicheWalking.csv",
            "5GParkStationary1.csv",
            "5GParkStationary2.csv",
        ]
    def get_trace_name(self, name):
        switcher = {
            "beach-stationary": "5GBeachStationary.csv",
            "beach-stationary2": "5GBeachStationary2.csv",
            "beach-stationary3": "5GBeachStationary3.csv",
            "city-drive": "5GCityDrive.csv",
            "corniche": "5GCorniche.csv",
            "corniche-walking": "5GCornicheWalking.csv",
            "park-stationary1": "5GParkStationary1.csv",
            "park-stationary2": "5GParkStationary2.csv",
        }
        return switcher.get(name, name)

class TraceManager:
    def __init__(self, paths: ProjectPaths):
        self.wired_manager = WiredTraceManager(paths)
        self.cellular_manager = CellularTraceManager(paths)

    def get_trace_name(self, trace_type, **kwargs):
        if trace_type == 'wired':
            return self.wired_manager.get_trace_name(**kwargs)
        elif trace_type == 'cellular':
            return self.cellular_manager.get_trace_name(**kwargs)
        else:
            return trace_type  # For traces like '5GBeachStationary.csv'

    def get_trace_path(self, trace_name):
        # Both managers use the same method from BaseTraceManager
        return self.wired_manager.get_trace_path(trace_name)