import os
import time
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.utils.common import Checkpointer
from tf_agents.utils import common

from src.core.agent import NeuralUCBMabAgent
# from src.core.mpts import MPTS
from src.utils.config import Config
from src.core.encoder import Encoder
from src.core.environment import MabEnvironment
from tf_agents.policies import policy_saver, tf_policy


class RLRunner:
    def __init__(self, config: Config, environment: MabEnvironment):
        self.config = config
        self.encoder = Encoder(config)
        self.environment = environment
        # self.environment.set_initial_protocol() # It can be set from the environment!
        # time.sleep(1)
        self.load_model_path = os.path.join(self.config.model_dir, self.config.model_name)
        self.initialize()

    def initialize(self):
        self.setup_agent()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.environment)
        self.replay_buffer = self.setup_replay_buffer()
        self.driver = self.setup_driver(self.tf_env, self.replay_buffer)
        self.setup_checkpointing()
        self.setup_policy_saver()
        if self.load_model_path:
            self.load_model()

    def setup_agent(self):
        time_step_spec = ts.time_step_spec(self.environment.observation_spec())
        encoding_dim = 16
        self.agent = NeuralUCBMabAgent(
            time_step_spec=time_step_spec,
            action_spec=self.environment.action_spec(),
            alpha=1.0,
            gamma=0.9,
            encoding_network=self.encoder.net,
            encoding_network_num_train_steps=-1,
            encoding_dim=encoding_dim,
            optimizer=self.encoder.optimizer
        )
    
    def setup_policy_saver(self):
        self.policy_saver = policy_saver.PolicySaver(self.agent.policy)

    def train(self) -> Any:

        print("Start experiment...")
        # print(f"Training encoder for {self.agent._encoding_network_num_train_steps} steps")
        start = time.time()
        # total_steps = self.config.num_steps + self.agent._encoding_network_num_train_steps # Encoder is pre-trained
        for step in range(self.config.num_steps):
            print(f"Step {step}/{self.config.num_steps}")
            self.run_training_step(step)
        
        # self.train_checkpointer.save(self.global_step)
        self.training_time = time.time() - start
        print(f"Training finished in {self.training_time} seconds")
        self.save_model()
        
        # utils.update_log(os.path.join(self.config.run_log_dir, 'settings.json'), 
        #                  self.settings, 'success', self.training_time, int(self.global_step.numpy()))

    def run_training_step(self, step):
        self.driver.run()
        sel_actions = self.replay_buffer.gather_all().action.numpy()[0]
        rewards = self.replay_buffer.gather_all().reward.numpy()[0]
        for a, r in zip(sel_actions, rewards):
            print(f"[Step {step}] Action: {self.config.get_protocol_name(self.environment.map_actions[a])} | Reward: {r} | (DEBUG) Max rw: {self.environment._max_rw}\n")
        self.agent.train(self.replay_buffer.gather_all())
        self.replay_buffer.clear()

    def shut_down_env(self) -> None:
        self.environment.close()

    def setup_checkpointing(self):
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(self.config.checkpoint_dir, self.config.model_name),
            max_to_keep=5,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step
        )

    # def get_checkpoint_filename(self):
    #     if self.config.restore:
    #         ckpt_filename = utils.get_latest_ckpt_dir(self.settings)
    #         if not ckpt_filename or not os.path.isdir(ckpt_filename):
    #             print("No checkpoint found, using current timestamp")
    #             ckpt_filename = os.path.join(self.config.checkpoint_dir, utils.time_to_str())
    #         else:
    #             print(f"Restoring checkpoint from {ckpt_filename}")
    #             self.environment.timestamp = ckpt_filename.split('/')[-1]
    #     else:
    #         ckpt_filename = os.path.join(self.config.checkpoint_dir, utils.time_to_str())
    #     return ckpt_filename

    def setup_replay_buffer(self):
        return tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.policy.trajectory_spec,
            batch_size=1,
            max_length=self.config.steps_per_loop
        )

    def setup_driver(self, tf_env, replay_buffer):
        return dynamic_step_driver.DynamicStepDriver(
            env=tf_env,
            policy=self.agent.collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=self.config.steps_per_loop
        )

    def save_model(self):
        self.checkpointer.save(self.global_step)

    def load_model(self):
        load_dir = os.path.join(self.config.model_dir, self.config.model_name)
        if not os.path.exists(load_dir):
            print(f"No saved model found at {load_dir}. Starting with a fresh agent.")
            self.checkpointer.initialize_or_restore()
            return
        self.checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_global_step()
        print(f"Agent loaded ({self.config.model_name}) at step {self.global_step.numpy()}")
