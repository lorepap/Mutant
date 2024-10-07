# src/main.py

from src.core.collector import Collector
from src.core.comm_manager import CommManager
from src.utils.config import Config

def main():
    config = Config('config.yaml')
    comm_manager = CommManager(config)
    n_steps = 50
    collector = Collector('cubic', n_steps, config, comm_manager.netlink_communicator)

    try:
        comm_manager.start_communication()
        print("Running collection...")
        collector.run_collection()
    finally:
        comm_manager.stop_communication()

if __name__ == "__main__":
    main()