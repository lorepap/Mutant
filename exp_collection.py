# run_experiments.py

import json
import subprocess
import os
from src.utils.paths import ProjectPaths
import time

def load_experiments(filename="src/utils/experiments.json"):
    with open(filename, "r") as f:
        return json.load(f)

def trace_exists(bw, bw_factor):
    paths = ProjectPaths()
    if bw_factor == 1:
        trace_name = f'wired{int(bw)}'
        return os.path.exists(paths.get_trace_path(trace_name))
    else:
        trace_d = f'wired{int(bw)}-{bw_factor}x-d'
        trace_u = f'wired{int(bw)}-{bw_factor}x-u'
        return os.path.exists(paths.get_trace_path(trace_d)) and os.path.exists(paths.get_trace_path(trace_u))

def run_experiment(experiment):
    if not trace_exists(experiment['bw'], experiment['bw_factor']):
        print(f"Skipping experiment: {experiment} - Trace file not found")
        return False

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