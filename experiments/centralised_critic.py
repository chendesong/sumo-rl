"""
Centralised Critic Model for MAPPO in RLlib.

The actor (policy network) sees only the local observation.
The critic (value network) sees the global state = all agents' observations concatenated.

Global state is passed via the info dict under key "global_obs".
"""

import numpy as np
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn


class CentralisedCriticModel(TorchModelV2, nn.Module):
    """Actor uses local obs, Critic uses global obs (all agents' obs concatenated).

    Expects obs_space to be a Dict with:
        "local_obs": Box(local_obs_dim,)
        "global_obs": Box(global_obs_dim,)

    The policy (actor) only uses "local_obs".
    The value function (critic) only uses "global_obs".
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Extract dimensions from the Dict obs space
        local_obs_space = obs_space.original_space["local_obs"]
        global_obs_space = obs_space.original_space["global_obs"]

        local_dim = local_obs_space.shape[0]
        global_dim = global_obs_space.shape[0]

        # Actor network: local_obs → action logits
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        self.actor = nn.Sequential(
            nn.Linear(local_dim, hiddens[0]),
            nn.Tanh(),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.Tanh(),
            nn.Linear(hiddens[1], num_outputs),
        )

        # Critic network: global_obs → value
        self.critic = nn.Sequential(
            nn.Linear(global_dim, hiddens[0]),
            nn.Tanh(),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.Tanh(),
            nn.Linear(hiddens[1], 1),
        )

        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        local_obs = obs["local_obs"]
        global_obs = obs["global_obs"]

        # Actor forward: local obs → action logits
        logits = self.actor(local_obs)

        # Critic forward: global obs → value (stored for value_function())
        self._cur_value = self.critic(global_obs).squeeze(-1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "Must call forward() first"
        return self._cur_value