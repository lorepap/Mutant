import numpy as np
import time

class OnlineMinMaxScaler:
    def __init__(self, feature_dim, feature_range=(0, 1), epsilon=1e-8):
        self.feature_dim = feature_dim
        self.feature_range = feature_range
        self.min = np.full(feature_dim, np.inf)
        self.max = np.full(feature_dim, -np.inf)
        self.epsilon = epsilon  # Small value to avoid division by zero

    def update(self, x):
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

    def scale(self, x):
        numerator = (x - self.min) * (self.feature_range[1] - self.feature_range[0])
        denominator = (self.max - self.min) + self.epsilon
        scaled = numerator / denominator + self.feature_range[0]
        return np.clip(scaled, self.feature_range[0], self.feature_range[1])