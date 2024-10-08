import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from src.utils.config import Config
from src.utils.change_detection import ADWIN


class AdaptiveRewardCalculator:
    def __init__(self, config):
        self.config = config
        self.throughput_adwin = ADWIN(delta=1e-3)
        self.rtt_adwin = ADWIN(delta=1e-5)
        self.max_throughput = 0
        self.min_rtt = float('inf')

    def calculate_reward(self, row):
        throughput = row['thruput']
        rtt = row['rtt']
        loss_rate = row['loss_rate']

        # Update ADWIN and get change detection
        self.throughput_adwin.add_element(throughput)
        self.rtt_adwin.add_element(rtt)

        # Update baselines if change is detected or new extreme is observed
        if throughput >  self.max_throughput:
            self.max_throughput = throughput
        elif self.throughput_adwin.detected_change():
            print(f"Change detected in throughput: current value -> {throughput} | max value -> {self.max_throughput}")
            self.max_throughput = min(self.max_throughput, throughput) # Yes, min because it's the maximum thruput for that period
        if rtt < self.min_rtt:
            self.min_rtt = rtt
            print(f"New min RTT: current value -> {rtt} | min value -> {self.min_rtt}")

        # Calculate the actual reward
        reward = self.compute_reward(throughput, loss_rate, rtt)

        # Calculate the max possible reward with current conditions
        max_reward = self.compute_reward(self.max_throughput, loss_rate, self.min_rtt)

        # Normalize the reward
        normalized_reward = reward / max_reward if max_reward > 0 else 0

        return normalized_reward

    def compute_reward(self, thr, loss_rate, rtt):
        return pow((thr - self.config.zeta * loss_rate), self.config.kappa) / (rtt * 1e-3)

    def process_dataframe(self, df):
        rewards = []
        normalized_rewards = []
        max_reward = 0

        for _, row in df.iterrows():
            reward = self.calculate_reward(row)
            rewards.append(reward)

            # Update max_reward
            max_reward = max(max_reward, reward)

            # Normalize reward
            normalized_reward = reward / max_reward if max_reward > 0 else 0
            normalized_rewards.append(normalized_reward)

        # df['reward'] = rewards
        df['normalized_reward'] = normalized_rewards
        return df

class Trainer:
    def __init__(self):
        self.config = Config('config.yaml')
        self.obs_size = len(self.config.train_non_stat_features) + len(self.config.train_stat_features)*3*3 + len(self.config.protocols) # self.config.train_features
        self.dataset = None
        self.reward_calculator = AdaptiveRewardCalculator(self.config)

    def process_file(self, file_path):
        df = pd.read_csv(file_path)[:1000]
        return self.reward_calculator.process_dataframe(df)

    def build_dataset(self):
        datasets = []
        for f in tqdm(os.listdir(self.config.collection_dir)):
            if f.endswith('.csv'):
                print(f"Processing file {f}...")
                file_path = os.path.join(self.config.collection_dir, f)
                datasets.append(self.process_file(file_path))
        
        print("Finished processing the dataset")
        self.dataset = pd.concat(datasets, ignore_index=True)
        output_file = 'collection_dataset.csv'
        self.dataset.to_csv(output_file, index=False)
        print(f"Dataset shape: {self.dataset.shape}")
        print(f"Dataset columns: {self.dataset.columns}")
        print(f"Dataset head:\n{self.dataset.head()}")

    def run(self):
        encoding_dim = 16
        encoding_net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.obs_size, activation='relu'),
            tf.keras.layers.Reshape((1, self.obs_size)),
            tf.keras.layers.GRU(32, return_sequences=True),
            tf.keras.layers.Dense(encoding_dim, activation='relu')
        ])

        reward_predictor = tf.keras.layers.Dense(1)

        _observation = self.dataset.drop(['normalized_reward'], axis=1)
        norm_rw = self.dataset['normalized_reward']

        combined_model = tf.keras.models.Sequential([
            encoding_net,
            reward_predictor
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        combined_model.compile(optimizer=optimizer, loss='mse')
        combined_model.build(input_shape=(None, self.obs_size))
        combined_model.summary()

        # Add early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train the combined model (encoder and reward predictor) with early stopping
        history = combined_model.fit(
            _observation, 
            norm_rw, 
            epochs=10, 
            batch_size=32, 
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        # Extract and save the encoder model
        encoder_model = tf.keras.models.Sequential(encoding_net.layers)
        encoder_model.build(input_shape=(None, self.obs_size))
        encoder_model.summary()

        # Save the weights of the encoder model
        encoder_model.save_weights('encoder_weights.h5')
        
        print(f"Model trained for {len(history.history['loss'])} epochs")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.build_dataset()
    trainer.run()