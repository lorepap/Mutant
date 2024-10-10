import subprocess
import sys
import os
import logging
from datetime import datetime

# Assuming the CellularTraceManager class is in a module named 'trace_managers'
from src.utils.trace_manager import CellularTraceManager
from src.utils.paths import ProjectPaths  # Assuming you have this class defined

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_trace(trace_name):
    command = f"python run.py --trace_type cellular --trace_name {trace_name}"
    logging.info(f"Running command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"Command for trace {trace_name} completed successfully")
        logging.debug(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command for trace {trace_name} failed with error code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while running trace {trace_name}: {str(e)}")

def main():
    # Initialize ProjectPaths (you may need to adjust this based on your project structure)
    project_paths = ProjectPaths()
    
    # Initialize CellularTraceManager
    trace_manager = CellularTraceManager(project_paths)
    
    # Get the list of cellular trace names
    trace_names = trace_manager.cellular_names
    
    # Create a directory for this run's logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"cellular_trace_run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Run each trace
    for trace_name in trace_names:
        log_file = os.path.join(log_dir, f"{trace_name}_log.txt")
        file_handler = logging.FileHandler(log_file)
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Starting run for trace: {trace_name}")
        run_trace(trace_name)
        logging.info(f"Completed run for trace: {trace_name}")
        
        logging.getLogger().removeHandler(file_handler)

if __name__ == "__main__":
    main()
    logging.info("All trace runs completed.")