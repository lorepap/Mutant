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

from src.core.agent import NeuralUCBMabAgent
# from src.core.mpts import MPTS
from src.utils.config import Config
from src.core.encoder import Encoder
from src.core.environment import MabEnvironment


class RLRunner:
    def __init__(self, config: Config, environment: MabEnvironment):
        self.config = config
        self.encoder = Encoder(config)
        self.encoder.load_weights()
        self.environment = environment
        # self.environment.set_initial_protocol() # It can be set from the environment!
        # time.sleep(1)
        self.initialize()

    def initialize(self):
        self.setup_agent()
        self.setup_mpts()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.environment)
        self.replay_buffer = self.setup_replay_buffer()
        self.driver = self.setup_driver(self.tf_env, self.replay_buffer)
        self.setup_checkpointing()

    def setup_agent(self):
        time_step_spec = ts.time_step_spec(self.environment.observation_spec())
        encoding_dim = 16
        self.agent = NeuralUCBMabAgent(
            time_step_spec=time_step_spec,
            action_spec=self.environment.action_spec(),
            alpha=0.1,
            gamma=0.9,
            encoding_network=self.encoder.net,
            encoding_network_num_train_steps=-1,
            encoding_dim=encoding_dim,
            optimizer=self.encoder.optimizer
        )

    def setup_mpts(self):
        # TODO: fix MPTS class
        # self.mpts = MPTS(
        #     arms=self.config.protocols,
        #     k=len(self.config.pool_size),
        #     T=int(self.config.mpts_T),
        #     thread=self.environment.kernel_thread,
        #     net_channel=self.cm.netlink_communicator,
        #     step_wait=self.config.mpts_step_wait
        # )
        pass

    def train(self) -> Any:

        print("Start experiment...")
        # print(f"Training encoder for {self.agent._encoding_network_num_train_steps} steps")
        start = time.time()
        # total_steps = self.config.num_steps + self.agent._encoding_network_num_train_steps # Encoder is pre-trained
         
        for step in range(self.global_step.numpy(), self.config.num_steps):
            self.run_training_step(step)
        
        # self.train_checkpointer.save(self.global_step)
        self.training_time = time.time() - start
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
    #     self.settings = {
    #         'timestamp': utils.time_to_str(),
    #         **self.config.get_network_params(),
    #         **self.config.__dict__,
    #         'action_mapping': {action: self.config.protocols[p_id] for action, p_id in enumerate(self.environment.map_actions)}
    #     }
    #     self.ckpt_filename = self.get_checkpoint_filename()
    #     self.settings['checkpoint_dir'] = self.ckpt_filename
    #     utils.log_settings(os.path.join(self.config.run_log_dir, 'settings.json'), self.settings, 'failed')

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.train_checkpointer = Checkpointer(
            ckpt_dir=self.config.checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step
        )
        self.train_checkpointer.initialize_or_restore()
        self.environment.step_counter = self.global_step.numpy()

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
