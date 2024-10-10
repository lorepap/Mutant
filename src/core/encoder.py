import tensorflow as tf
from tf_agents.networks import network

class EncoderNetwork(network.Network):
    def __init__(self, input_tensor_spec, name='EncoderNetwork'):
        super(EncoderNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        self.encoding_dim = 16
        
        # self._encoder = tf.keras.models.Sequential([
        #     tf.keras.layers.Dense(input_tensor_spec.shape[0], activation='relu'),
        #     tf.keras.layers.Reshape((1, input_tensor_spec.shape[0])),
        #     tf.keras.layers.GRU(256, return_sequences=True),
        #     tf.keras.layers.Dense(self.encoding_dim, activation='relu'),
        #     tf.keras.layers.Flatten()  # Flatten the output to (batch_size, encoding_dim)
        # ])

        self._encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_tensor_spec.shape[0], activation='relu'),
            tf.keras.layers.Reshape((1, input_tensor_spec.shape[0])),
            tf.keras.layers.GRU(32, return_sequences=False),
            tf.keras.layers.Dense(self.encoding_dim, activation='relu')
        ])

        
    def call(self, observations, step_type=None, network_state=()):
        encoding = self._encoder(observations)
        return encoding, network_state

class Encoder:
    def __init__(self, config):
        self.config = config
        self.input_tensor_spec = tf.TensorSpec(shape=(len(self.config.train_features)), 
                                               dtype=tf.float32)
        self.net = EncoderNetwork(self.input_tensor_spec)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr)
        self.net._encoder.compile(optimizer='adam', loss='mse')
        self.net._encoder.build(input_shape=(None, self.input_tensor_spec.shape[0]))
        self.load_weights()

    def load_weights(self, weights_path='encoder_weights_5G.h5'):
        try:
            self.net._encoder.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Continuing with initialized weights.")