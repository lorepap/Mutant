import os
import sys

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import Config
from train_encoder import AdaptiveRewardCalculator


def generate_synthetic_data(n_samples=1000, change_points=[250, 500, 750]):
    np.random.seed(42)
    
    throughput = np.zeros(n_samples)
    rtt = np.zeros(n_samples)
    loss_rate = np.zeros(n_samples)
    
    # Base values
    throughput[:] = 10 + np.random.normal(0, 1, n_samples)
    rtt[:] = 20 + np.random.normal(0, 2, n_samples)
    loss_rate[:] = 0.01 + np.random.normal(0, 0.001, n_samples)
    
    # Introduce changes
    throughput[change_points[0]:change_points[1]] += 5  # Increase in bandwidth
    throughput[change_points[1]:change_points[2]] -= 3  # Decrease in bandwidth
    rtt[change_points[2]:] += 10  # Increase in RTT
    
    # Ensure non-negative values
    throughput = np.maximum(throughput, 0)
    rtt = np.maximum(rtt, 1)
    
    df = pd.DataFrame({
        'thruput': throughput,
        'rtt': rtt,
        'loss_rate': loss_rate
    })
    
    return df

def test_reward_calculator():
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Initialize reward calculator
    config = Config('config.yaml')
    calculator = AdaptiveRewardCalculator(config)
    
    # Process data and calculate rewards
    processed_df = calculator.process_dataframe(df)
    
    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    
    axs[0].plot(df['thruput'])
    axs[0].set_title('Throughput')
    axs[0].set_ylabel('Mbps')
    
    axs[1].set_ylabel('Ratio')
    
    axs[2].plot(df['rtt'])
    axs[2].set_title('RTT')
    axs[2].set_ylabel('ms')
    
    axs[3].plot(processed_df['normalized_reward'])
    axs[3].set_title('Normalized Reward')
    axs[3].set_ylabel('Reward')
    axs[3].set_xlabel('Sample')
    
    for ax in axs:
        ax.axvline(x=250, color='r', linestyle='--')
        ax.axvline(x=500, color='r', linestyle='--')
        ax.axvline(x=750, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('reward_calculator_test.png')
    plt.close()

    # Analyze results
    change_points = [250, 500, 750]
    for point in change_points:
        before = processed_df['normalized_reward'][point-10:point].mean()
        after = processed_df['normalized_reward'][point:point+10].mean()
        print(f"Change at {point}: Before = {before:.4f}, After = {after:.4f}, Difference = {after-before:.4f}")

if __name__ == "__main__":
    test_reward_calculator()