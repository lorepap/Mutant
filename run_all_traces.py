import subprocess
import sys
import os
import logging
from datetime import datetime

from src.utils.trace_manager import CellularTraceManager
from src.utils.paths import ProjectPaths

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_trace(trace_name, log_file):
    command = f"python run.py --trace_type cellular --trace_name {trace_name}"
    logging.info(f"Running command: {command}")
    
    try:
        with open(log_file, 'a') as f:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print to console
                f.write(line)  # Write to log file
                f.flush()  # Ensure it's written immediately
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                logging.error(f"Command for trace {trace_name} failed with return code {return_code}")
            else:
                logging.info(f"Command for trace {trace_name} completed successfully")
    
    except Exception as e:
        logging.error(f"An unexpected error occurred while running trace {trace_name}: {str(e)}")

def main():
    project_paths = ProjectPaths()
    trace_manager = CellularTraceManager(project_paths)
    trace_names = trace_manager.cellular_names
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"cellular_trace_run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    for trace_name in trace_names:
        log_file = os.path.join(log_dir, f"{trace_name}_log.txt")
        
        logging.info(f"Starting run for trace: {trace_name}")
        run_trace(trace_name, log_file)
        logging.info(f"Completed run for trace: {trace_name}")

if __name__ == "__main__":
    main()
    logging.info("All trace runs completed.")