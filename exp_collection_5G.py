# run_experiments.py

import json
import subprocess
import os
from src.utils.paths import ProjectPaths
from src.utils.trace_manager import TraceManager
import time

def load_experiments():
    # Each experiment is just --trace_type cellular and --cellular_trace_name [trace_name], for all trace names
    trace_manager = TraceManager(ProjectPaths())
    experiments = []
    for trace_name in trace_manager.cellular_manager.cellular_traces:
        experiments.append({
            "trace_type": "cellular",
            "cellular_trace_name": trace_name
        })
    return experiments

def run_experiment(experiment):
    cmd = ["python", "collect.py"]
    for key, value in experiment.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running experiment: {experiment}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False

def run_all_experiments():
    experiments = load_experiments()
    total = len(experiments)
    
    executed = 0
    skipped = 0
    failed = 0

    for i, experiment in enumerate(experiments, 1):
        print(f"\nProcessing experiment {i} of {total}")
        result = run_experiment(experiment)
        if result is True:
            executed += 1
        elif result is False:
            skipped += 1
        else:
            failed += 1
        time.sleep(2)

    print(f"\nExperiment execution completed.")
    print(f"Total experiments: {total}")
    print(f"Executed: {executed}")
    print(f"Skipped (no trace): {skipped}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    run_all_experiments()