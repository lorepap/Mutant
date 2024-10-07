# generate_experiments.py

import itertools
import json
import os

def generate_experiments():
    bw_values = [6, 12, 24, 48, 96, 192, 300]
    rtt_values = list(range(10, 121, 20))
    bdp_mult_values = list(range(1, 21, 5))
    bw_factor_values = [1, 2, 4, 8]

    experiments = list(itertools.product(bw_values, rtt_values, bdp_mult_values, bw_factor_values))
    
    formatted_experiments = [
        {
            "bw": bw,
            "rtt": rtt,
            "bdp_mult": bdp_mult,
            "bw_factor": bw_factor
        }
        for bw, rtt, bdp_mult, bw_factor in experiments
    ]

    return formatted_experiments

def save_experiments(experiments, filename="experiments.json"):
    with open(filename, "w") as f:
        json.dump(experiments, f, indent=2)
    print(f"Generated {len(experiments)} experiments and saved to {filename}")

if __name__ == "__main__":
    experiments = generate_experiments()
    save_experiments(experiments)