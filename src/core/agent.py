import abc
import numpy as np
import tensorflow as tf

import tf_agents
from tf_agents.agents import tf_agent
from tf_agents.bandits.agents.linear_thompson_sampling_agent import LinearThompsonSamplingAgent
from tf_agents.bandits.agents.neural_linucb_agent import NeuralLinUCBAgent
# from tf_agents.bandits.agents.lin_ucb_agent import LinUCBAgent
from tf_agents.bandits.policies import linear_thompson_sampling_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.typing import types


nest = tf.nest

# TODO: Not all parameters are needed

class LinTSMabAgent(LinearThompsonSamplingAgent):
    def __init__(self, 
                 time_step_spec: ts.TimeStep,
                 action_spec: tf_agents.typing.types.BoundedTensorSpec,
                 variable_collection = None,
                 alpha: float = 1.0,
                 gamma: float = 1.0,
                 use_eigendecomp: bool = False,
                 tikhonov_weight: float = 1.0,
                 add_bias: bool = False,
                 emit_policy_info = (),
                 observation_and_action_constraint_splitter = None,
                 accepts_per_arm_features: bool = False,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False,
                 enable_summaries: bool = True,
                 dtype: tf.DType = tf.float32,
                 name = None):
        super(LinTSMabAgent, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            alpha=alpha,
            gamma=gamma,
            use_eigendecomp=use_eigendecomp,
            tikhonov_weight=tikhonov_weight,
            add_bias=add_bias,
            emit_policy_info=emit_policy_info,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            accepts_per_arm_features=accepts_per_arm_features,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            enable_summaries=enable_summaries,
            dtype=dtype,
            name=name
        )

class NeuralUCBMabAgent(NeuralLinUCBAgent):
    def __init__(self,
            time_step_spec: types.TimeStep,
            action_spec: types.BoundedTensorSpec,
            encoding_network: types.Network,
            encoding_network_num_train_steps: int,
            encoding_dim: int,
            optimizer: types.Optimizer,
            alpha: float,
            gamma: float
        ):
        super(NeuralUCBMabAgent, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            encoding_network=encoding_network,
            encoding_network_num_train_steps=encoding_network_num_train_steps,
            encoding_dim=encoding_dim,
            optimizer=optimizer,
            alpha=alpha,
            gamma=gamma
        )
