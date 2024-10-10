import subprocess
import sys
import os
import logging
from datetime import datetime
from src.utils.trace_manager import CellularTraceManager
from src.utils.paths import ProjectPaths

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_trace(trace_name):
    command = f"python tests/test_mpts.py --trace {trace_name}"
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
    # Initialize ProjectPaths and CellularTraceManager
    project_paths = ProjectPaths()
    trace_manager = CellularTraceManager(project_paths)
    
    # Get the list of cellular trace names
    trace_names = trace_manager.cellular_names
    
    # Create a directory for this run's logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"mpts_trace_run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Run each trace
    for trace_name in trace_names:
        # Check if the trace has already been processed
        if os.path.exists(f'most_selected_arms_{trace_name}.json'):
            logging.info(f"Skipping trace {trace_name} as it has already been processed.")
            continue
        
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