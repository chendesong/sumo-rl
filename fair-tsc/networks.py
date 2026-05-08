"""
Shared-parameter actor and critic networks for Fair-TSC.

Architecture (paper §III.A):
  - Single shared actor π_θ:  local_obs (D_l)  + one-hot agent ID (N) → logits (action_dim)
  - Single shared critic V_φ: global_obs (D_g) + one-hot agent ID (N) → scalar value

Both used twice in Fair-TSC:
  - π_θ, V_φ^MARL : main coordination policy / critic
  - π_UE, V^UE    : selfish baseline (same architecture, separate weights)

Categorical actions over Discrete(action_dim).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim: int, hidden: List[int], out_dim: int, activation=nn.Tanh) -> nn.Sequential:
    """Fully-connected stack: in_dim → h0 → h1 → ... → out_dim (linear last layer)."""
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    """Shared-parameter categorical policy.

    Input:  local_obs (D_l) ⊕ one-hot agent ID (N)   shape: [B, D_l + N]
    Output: Categorical over action_dim
    """

    def __init__(self, local_obs_dim: int, num_agents: int, action_dim: int, hidden: List[int]):
        super().__init__()
        self.local_obs_dim = local_obs_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.net = _mlp(local_obs_dim + num_agents, hidden, action_dim)

    def forward(self, local_obs: torch.Tensor, agent_idx: torch.Tensor) -> Categorical:
        """
        Args:
            local_obs: float tensor [B, D_l]
            agent_idx: long  tensor [B]   (values in [0, N))
        Returns:
            Categorical distribution over [0, action_dim)
        """
        onehot = torch.nn.functional.one_hot(agent_idx, num_classes=self.num_agents).float()
        x = torch.cat([local_obs, onehot], dim=-1)
        logits = self.net(x)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(
        self, local_obs: torch.Tensor, agent_idx: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action + return log-prob.

        Returns:
            action:  long  [B]
            logprob: float [B]
        """
        dist = self.forward(local_obs, agent_idx)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def evaluate(
        self, local_obs: torch.Tensor, agent_idx: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For PPO update: log-prob of given action + entropy of dist.

        Returns:
            logprob: float [B]
            entropy: float [B]
        """
        dist = self.forward(local_obs, agent_idx)
        return dist.log_prob(action), dist.entropy()


class SharedCritic(nn.Module):
    """Shared-parameter centralised state-value critic.

    Input:  global_obs (D_g) ⊕ one-hot agent ID (N)   shape: [B, D_g + N]
    Output: scalar value V(s, i)

    Used to instantiate both V^MARL (trained on D_MARL) and V^UE (trained on D_UE).
    Same architecture, different parameters.
    """

    def __init__(self, global_obs_dim: int, num_agents: int, hidden: List[int]):
        super().__init__()
        self.global_obs_dim = global_obs_dim
        self.num_agents = num_agents
        self.net = _mlp(global_obs_dim + num_agents, hidden, 1)

    def forward(self, global_obs: torch.Tensor, agent_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_obs: float tensor [B, D_g]
            agent_idx:  long  tensor [B]
        Returns:
            value: float [B]
        """
        onehot = torch.nn.functional.one_hot(agent_idx, num_classes=self.num_agents).float()
        x = torch.cat([global_obs, onehot], dim=-1)
        return self.net(x).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak target-network update (paper Eq. 33): φ⁻ ← τφ + (1−τ)φ⁻."""
    with torch.no_grad():
        for p_tgt, p_src in zip(target.parameters(), source.parameters()):
            p_tgt.mul_(1.0 - tau).add_(p_src.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module):
    """Copy source → target."""
    target.load_state_dict(source.state_dict())