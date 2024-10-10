import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
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
        self.obs_size = len(self.config.train_non_stat_features) + len(self.config.train_stat_features)*3*3 + len(self.config.protocols)
        self.reward_calculator = AdaptiveRewardCalculator(self.config)
        self.encoding_dim = 16
        self.encoder, self.model = self.build_model()
        self.scaler = MinMaxScaler()

    def build_model(self):
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.obs_size, activation='relu'),
            tf.keras.layers.Reshape((1, self.obs_size)),
            tf.keras.layers.GRU(32, return_sequences=False),
            tf.keras.layers.Dense(self.encoding_dim, activation='relu')
        ])

        reward_predictor = tf.keras.layers.Dense(1)

        combined_model = tf.keras.models.Sequential([
            encoder,
            reward_predictor
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        combined_model.compile(optimizer=optimizer, loss='mse')
        combined_model.build(input_shape=(None, self.obs_size))
        return encoder, combined_model

    def normalize_data(self, df: pd.DataFrame):
        # Separate features and target
        features = df.drop(['normalized_reward'], axis=1)
        target = df['normalized_reward']

        # Fit the scaler on the features and transform
        normalized_features = self.scaler.fit_transform(features)

        # Create a new dataframe with normalized features
        normalized_df = pd.DataFrame(normalized_features, columns=features.columns)

        # Add back the target column
        normalized_df['normalized_reward'] = target

        return normalized_df


    def process_file(self, file_path):
        df = pd.read_csv(file_path)[:1000]
        df = self.reward_calculator.process_dataframe(df)
        return self.normalize_data(df)

    def train_on_trace(self, file_path):
        print(f"Processing and training on file: {file_path}")
        df = self.process_file(file_path)
        
        _observation = df.drop(['normalized_reward'], axis=1)
        norm_rw = df['normalized_reward']

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            _observation, 
            norm_rw, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        print(f"Model trained for {len(history.history['loss'])} epochs on {file_path}")

    def save_weights(self, file_name='encoder_weights_5G.h5'):
        self.encoder.save_weights(file_name)
        print(f"Model weights saved to {file_name}")

    def load_weights(self, file_name='encoder_weights_5G.h5'):
        if os.path.exists(file_name):
            self.encoder.load_weights(file_name)
            print(f"Model weights loaded from {file_name}")
        else:
            print(f"No weights file found at {file_name}. Starting with fresh weights.")

def main():
    trainer = Trainer()
    
    for f in tqdm(os.listdir(trainer.config.collection_dir)):
        if f.endswith('.csv'):
            file_path = os.path.join(trainer.config.collection_dir, f)
            print(f"Training on trace: {file_path}")
            
            # Load weights from previous trace (if exists)
            trainer.load_weights()
            
            # Train on current trace
            trainer.train_on_trace(file_path)
            
            # Save weights after training on this trace
            trainer.save_weights()

if __name__ == '__main__':
    main()