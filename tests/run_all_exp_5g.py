import subprocess
import argparse
import os
from itertools import product

# List of protocols and their corresponding numbers
PROTOCOLS = {
    'cubic': 0,
    'hybla': 1,
    'bbr': 2,
    'westwood': 3,
    'veno': 4,
    'vegas': 5,
    'yeah': 6,
    'bic': 7,
    'htcp': 8,
    'highspeed': 9,
    'illinois': 10
}

# List of traces
TRACES = [
    '5GBeachStationary.csv',
    '5GBeachStationary2.csv',
    '5GBeachStationary3.csv',
    '5GCityDrive.csv',
    '5GCorniche.csv',
    '5GCornicheWalking.csv',
    '5GParkStationary1.csv',
    '5GParkStationary2.csv'
]

def run_experiment(protocol, trace):
    """Run a single experiment with the given protocol and trace."""
    # trace_name = os.path.splitext(os.path.basename(trace))[0]
    cmd = f"python run_exp.py --protocol {protocol} --trace {trace}"
    print(f"Running experiment: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Experiment completed: {protocol} - {trace}")
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {protocol} - {trace}")
        print(f"Error message: {e}")

def run_all_experiments(selected_protocols=None, selected_traces=None):
    """Run experiments for all combinations of protocols and traces."""
    protocols = selected_protocols if selected_protocols else PROTOCOLS.keys()
    traces = selected_traces if selected_traces else TRACES

    for protocol, trace in product(protocols, traces):
        run_experiment(protocol, trace)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for multiple protocols and traces")
    parser.add_argument("--proto", nargs='+', choices=PROTOCOLS.keys(), help="Specific protocols to run")
    parser.add_argument("--trace", nargs='+', help="Specific traces to use")
    args = parser.parse_args()

    if args.proto or args.trace:
        run_all_experiments(args.proto, args.trace)
    else:
        run_all_experiments()

    print("All experiments completed.")