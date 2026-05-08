"""
Rollout buffer for on-policy PPO training in Fair-TSC.

Stores one rollout's worth of per-agent transitions, then runs GAE to
produce advantages & returns. Used twice:

  - D_MARL : populated under π_θ  (main MARL training)
  - D_UE   : populated under π_UE (selfish baseline)

Each transition stored:
    local_obs   [D_l]    : actor input
    global_obs  [D_g]    : critic input  (concat of all agents' local_obs)
    agent_idx   ()       : 0..N-1, for one-hot in shared networks
    action      ()       : int
    logprob     ()       : float
    reward      ()       : float (paper Eq. 15)
    value       ()       : float (V(s,i) at collection time)
    done        ()       : bool  (terminal at this step)
    C_p         ()       : float (paper Eq. 6)  — only used by MARL buffer
    C_s         ()       : float (paper Eq. 10) — only used by MARL buffer

The buffer is FLAT: all (agent, timestep) pairs concatenated. GAE is
computed PER-AGENT in temporal order before flattening shuffles them.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class AgentTrajectory:
    """One agent's contiguous trajectory within a rollout."""
    local_obs:  List[np.ndarray] = field(default_factory=list)
    global_obs: List[np.ndarray] = field(default_factory=list)
    actions:    List[int]        = field(default_factory=list)
    logprobs:   List[float]      = field(default_factory=list)
    rewards:    List[float]      = field(default_factory=list)
    values:     List[float]      = field(default_factory=list)
    dones:      List[bool]       = field(default_factory=list)
    c_p:        List[float]      = field(default_factory=list)
    c_s:        List[float]      = field(default_factory=list)


class RolloutBuffer:
    """Per-agent buffer; supports GAE + flat-batch mini-batch sampling."""

    def __init__(self, agent_ids: List[str], num_agents: int):
        self.agent_ids = agent_ids
        self.num_agents = num_agents
        self.agent_idx = {aid: i for i, aid in enumerate(agent_ids)}
        self.trajs: dict = {a: AgentTrajectory() for a in agent_ids}

        # Filled after compute_gae()
        self.flat = None  # dict of torch tensors

    def reset(self):
        self.trajs = {a: AgentTrajectory() for a in self.agent_ids}
        self.flat = None

    # ─────────────────────────────────────────────────────────────────
    # Collection
    # ─────────────────────────────────────────────────────────────────

    def add(
        self,
        agent_id: str,
        local_obs: np.ndarray,
        global_obs: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        value: float,
        done: bool,
        c_p: float = 0.0,
        c_s: float = 0.0,
    ):
        t = self.trajs[agent_id]
        t.local_obs.append(local_obs.astype(np.float32))
        t.global_obs.append(global_obs.astype(np.float32))
        t.actions.append(int(action))
        t.logprobs.append(float(logprob))
        t.rewards.append(float(reward))
        t.values.append(float(value))
        t.dones.append(bool(done))
        t.c_p.append(float(c_p))
        t.c_s.append(float(c_s))

    # ─────────────────────────────────────────────────────────────────
    # GAE — per-agent, temporal order preserved
    # ─────────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        last_values: dict,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        """Compute advantages + returns per agent, then flatten into one batch.

        Args:
            last_values: {agent_id: float}  V(s_T)  — bootstrap value at end of rollout
            gamma:      discount factor
            gae_lambda: GAE λ
            device:     torch device for the resulting flat tensors
        """
        all_local, all_global, all_idx = [], [], []
        all_act, all_logp, all_val = [], [], []
        all_adv, all_ret, all_cp, all_cs = [], [], [], []

        for aid in self.agent_ids:
            t = self.trajs[aid]
            T = len(t.rewards)
            if T == 0:
                continue

            rewards = np.asarray(t.rewards, dtype=np.float32)
            values  = np.asarray(t.values,  dtype=np.float32)
            dones   = np.asarray(t.dones,   dtype=np.bool_)
            last_v  = float(last_values.get(aid, 0.0))

            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for tt in reversed(range(T)):
                next_v = last_v if tt == T - 1 else values[tt + 1]
                next_nonterminal = 0.0 if dones[tt] else 1.0
                delta = rewards[tt] + gamma * next_v * next_nonterminal - values[tt]
                gae = delta + gamma * gae_lambda * next_nonterminal * gae
                advantages[tt] = gae
            returns = advantages + values

            agent_idx = np.full(T, self.agent_idx[aid], dtype=np.int64)

            all_local .append(np.stack(t.local_obs))
            all_global.append(np.stack(t.global_obs))
            all_idx   .append(agent_idx)
            all_act   .append(np.asarray(t.actions,  dtype=np.int64))
            all_logp  .append(np.asarray(t.logprobs, dtype=np.float32))
            all_val   .append(values)
            all_adv   .append(advantages)
            all_ret   .append(returns)
            all_cp    .append(np.asarray(t.c_p, dtype=np.float32))
            all_cs    .append(np.asarray(t.c_s, dtype=np.float32))

        self.flat = {
            "local_obs":  torch.from_numpy(np.concatenate(all_local)).to(device),
            "global_obs": torch.from_numpy(np.concatenate(all_global)).to(device),
            "agent_idx":  torch.from_numpy(np.concatenate(all_idx)).to(device),
            "action":     torch.from_numpy(np.concatenate(all_act)).to(device),
            "logprob":    torch.from_numpy(np.concatenate(all_logp)).to(device),
            "value":      torch.from_numpy(np.concatenate(all_val)).to(device),
            "advantage":  torch.from_numpy(np.concatenate(all_adv)).to(device),
            "return":     torch.from_numpy(np.concatenate(all_ret)).to(device),
            "c_p":        torch.from_numpy(np.concatenate(all_cp)).to(device),
            "c_s":        torch.from_numpy(np.concatenate(all_cs)).to(device),
        }

    # ─────────────────────────────────────────────────────────────────
    # Mini-batch iterator for PPO update
    # ─────────────────────────────────────────────────────────────────

    def iter_minibatches(self, minibatch_size: int, shuffle: bool = True):
        """Yield dicts of tensor slices for PPO update."""
        assert self.flat is not None, "call compute_gae() before iter_minibatches()"
        N = self.flat["local_obs"].shape[0]
        idx = torch.randperm(N) if shuffle else torch.arange(N)
        for s in range(0, N, minibatch_size):
            sel = idx[s : s + minibatch_size]
            yield {k: v[sel] for k, v in self.flat.items()}

    def total_size(self) -> int:
        if self.flat is None:
            return sum(len(self.trajs[a].rewards) for a in self.agent_ids)
        return self.flat["local_obs"].shape[0]