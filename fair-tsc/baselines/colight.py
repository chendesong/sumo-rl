"""CoLight-style graph-attention cooperation baseline.

The official CoLight uses graph attention to communicate between traffic
signals. This local baseline keeps the same SUMO/PPO scaffold as the
other comparisons and implements the core mechanism: a shared actor
attends over neighboring intersection observations before choosing a
phase. No fairness term is used.
"""

import math
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from evaluate import MetricsCollector, compute_deltas_from_rollout, evaluate_run, load_shared_ue_critic
from networks import SharedCritic
from ppo_core import bootstrap_last_values, ppo_update
from rollout_buffer import RolloutBuffer
from sumo_env import FairTSCEnv

from baselines.coordination_utils import build_neighbor_map


COLIGHT_MAX_NEIGHBORS = int(os.environ.get("FAIR_TSC_COLIGHT_MAX_NEIGHBORS", "4"))
COLIGHT_HIDDEN = int(os.environ.get("FAIR_TSC_COLIGHT_HIDDEN", "128"))


def _mlp(in_dim, hidden, out_dim):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class CoLightActor(nn.Module):
    """Graph-attention actor over fixed-size padded neighborhoods."""

    def __init__(self, obs_dim: int, max_neighbors: int, num_agents: int, action_dim: int, hidden):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.max_neighbors = int(max_neighbors)
        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.local_obs_dim = self.obs_dim * (1 + self.max_neighbors) + self.max_neighbors

        h = int(hidden[0]) if hidden else COLIGHT_HIDDEN
        self.self_enc = nn.Linear(self.obs_dim, h)
        self.neighbor_enc = nn.Linear(self.obs_dim, h)
        self.query = nn.Linear(h, h, bias=False)
        self.key = nn.Linear(h, h, bias=False)
        self.value = nn.Linear(h, h, bias=False)
        self.out = _mlp(2 * h + self.num_agents, list(hidden[1:]) if len(hidden) > 1 else [h], action_dim)

    def _split(self, local_obs: torch.Tensor):
        own = local_obs[:, : self.obs_dim]
        start = self.obs_dim
        end = start + self.max_neighbors * self.obs_dim
        neighbors = local_obs[:, start:end].reshape(-1, self.max_neighbors, self.obs_dim)
        mask = local_obs[:, end:end + self.max_neighbors]
        return own, neighbors, mask

    def forward(self, local_obs: torch.Tensor, agent_idx: torch.Tensor) -> Categorical:
        own, neighbors, mask = self._split(local_obs)
        own_h = torch.tanh(self.self_enc(own))
        nei_h = torch.tanh(self.neighbor_enc(neighbors))

        q = self.query(own_h).unsqueeze(1)
        k = self.key(nei_h)
        v = self.value(nei_h)
        scores = (q * k).sum(dim=-1) / math.sqrt(max(k.shape[-1], 1))

        mask = (mask > 0.5).float()
        weights = torch.softmax(scores.masked_fill(mask <= 0.0, -1e9), dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        context = (weights.unsqueeze(-1) * v).sum(dim=1)

        onehot = torch.nn.functional.one_hot(agent_idx, num_classes=self.num_agents).float()
        logits = self.out(torch.cat([own_h, context, onehot], dim=-1))
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, local_obs: torch.Tensor, agent_idx: torch.Tensor, deterministic: bool = False):
        dist = self.forward(local_obs, agent_idx)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, local_obs: torch.Tensor, agent_idx: torch.Tensor, action: torch.Tensor):
        dist = self.forward(local_obs, agent_idx)
        return dist.log_prob(action), dist.entropy()


def _pack_colight_obs(obs: Dict[str, np.ndarray], agent_ids, neighbor_map, obs_dim: int, max_neighbors: int):
    out = {}
    zeros = np.zeros(obs_dim, dtype=np.float32)
    for agent in agent_ids:
        own = obs[agent].astype(np.float32)
        neighbors = [n for n in neighbor_map.get(agent, []) if n in obs][:max_neighbors]
        parts = [own]
        mask = []
        for i in range(max_neighbors):
            if i < len(neighbors):
                parts.append(obs[neighbors[i]].astype(np.float32))
                mask.append(1.0)
            else:
                parts.append(zeros)
                mask.append(0.0)
        parts.append(np.asarray(mask, dtype=np.float32))
        out[agent] = np.concatenate(parts, axis=0).astype(np.float32)
    return out


def collect_episode_colight(
    env,
    actor,
    critic,
    buffer,
    device,
    seed=None,
    coll: Optional[MetricsCollector] = None,
    rollout: Optional[list] = None,
    neighbor_map=None,
):
    obs = env.reset(seed=seed)
    if neighbor_map is None:
        neighbor_map = build_neighbor_map(env.agent_ids, max_neighbors=actor.max_neighbors)
    done = False
    ep_R = {a: 0.0 for a in env.agent_ids}

    while not done:
        global_obs = env.get_global_obs(obs)
        packed = _pack_colight_obs(obs, env.agent_ids, neighbor_map, env.local_obs_dim, actor.max_neighbors)
        local_b = torch.from_numpy(np.stack([packed[a] for a in env.agent_ids])).to(device)
        global_b = torch.from_numpy(np.tile(global_obs, (env.num_agents, 1))).to(device)
        idx_b = torch.arange(env.num_agents, device=device)

        with torch.no_grad():
            action, logprob = actor.act(local_b, idx_b)
            value = critic(global_b, idx_b)

        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, raw_R, _cp, _cs, done, info = env.step(action_dict)
        raw_vec = np.array([raw_R[a] for a in env.agent_ids], dtype=np.float32)

        for i, agent in enumerate(env.agent_ids):
            buffer.add(
                agent_id=agent,
                local_obs=packed[agent],
                global_obs=global_obs,
                action=int(action[i].item()),
                logprob=float(logprob[i].item()),
                reward=float(raw_R[agent]),
                value=float(value[i].item()),
                done=done,
            )
            ep_R[agent] += float(raw_R[agent])

        if coll is not None:
            coll.add(info, mean_reward=float(raw_vec.mean()) if raw_vec.size else 0.0)
        if rollout is not None:
            rollout.append({"global_obs": global_obs, "rewards_array": raw_vec.copy()})
        if next_obs:
            obs = next_obs

    if C.REWARD_NORMALIZE:
        buffer.normalize_rewards(center=C.REWARD_NORM_CENTER, clip=C.REWARD_NORM_CLIP, eps=C.REWARD_NORM_EPS)
    last_v = bootstrap_last_values(critic, env.get_global_obs(obs), env.agent_ids, env.num_agents, device)
    buffer.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    return ep_R


def train_colight(
    num_episodes: int = 150,
    seed: Optional[int] = None,
    v_ue=None,
    save_actor: bool = True,
    additional_sumo_cmd: Optional[str] = None,
) -> Dict:
    if seed is None:
        seed = C.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    try:
        neighbor_map = build_neighbor_map(env.agent_ids, max_neighbors=COLIGHT_MAX_NEIGHBORS)
        actor = CoLightActor(
            obs_dim=env.local_obs_dim,
            max_neighbors=COLIGHT_MAX_NEIGHBORS,
            num_agents=env.num_agents,
            action_dim=env.action_dim,
            hidden=C.ACTOR_HIDDEN,
        ).to(device)
        critic = SharedCritic(env.global_obs_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)
        o_a = torch.optim.Adam(actor.parameters(), lr=C.ACTOR_LR)
        o_c = torch.optim.Adam(critic.parameters(), lr=C.CRITIC_LR)

        for ep in range(num_episodes):
            buf = RolloutBuffer(env.agent_ids, env.num_agents)
            ep_R = collect_episode_colight(
                env, actor, critic, buf, device, seed=seed + ep, neighbor_map=neighbor_map
            )
            if v_ue is None and ep == 0:
                v_ue = load_shared_ue_critic(env=env, device=device)
            st = ppo_update(
                actor=actor, critic=critic, actor_optim=o_a, critic_optim=o_c, buffer=buf,
                ppo_epochs=C.PPO_EPOCHS, minibatch_size=C.MINIBATCH_SIZE,
                clip_eps=C.CLIP_EPS, entropy_coeff=C.ENTROPY_COEFF,
                vf_coeff=C.VF_COEFF, grad_clip=C.GRAD_CLIP,
            )
            rR = np.array(list(ep_R.values()), dtype=np.float32)
            print(f"[CoLight] ep={ep+1:3d}/{num_episodes} Rbar(raw)={rR.mean():+.1f} "
                  f"ploss={st['policy_loss']:+.4f} H={st['entropy']:.3f}")

        if save_actor:
            out_dir = os.path.join(C.BASE_DIR, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "local_obs_dim": actor.local_obs_dim,
                    "global_obs_dim": env.global_obs_dim,
                    "num_agents": env.num_agents,
                    "max_neighbors": COLIGHT_MAX_NEIGHBORS,
                    "num_episodes": num_episodes,
                    "seed": seed,
                    "baseline": "colight_graph_attention",
                },
                os.path.join(out_dir, "colight_baseline.pt"),
            )

        coll = MetricsCollector()
        buf = RolloutBuffer(env.agent_ids, env.num_agents)
        rollout = []
        collect_episode_colight(
            env, actor, critic, buf, device,
            seed=seed + num_episodes, coll=coll, rollout=rollout, neighbor_map=neighbor_map,
        )
        env_metrics = coll.finalize(env)
        if v_ue is None:
            v_ue = load_shared_ue_critic(env=env, device=device)
        deltas_TN = compute_deltas_from_rollout(rollout, v_ue=v_ue, num_agents=env.num_agents, gamma=C.GAMMA)
        result = evaluate_run(deltas_TN, env_metrics, delta_valid=True)
        print(f"[CoLight eval] {result}")
        return result
    finally:
        env.close()


def main(v_ue=None, additional_sumo_cmd: Optional[str] = None, **_unused):
    return train_colight(num_episodes=50, v_ue=v_ue, additional_sumo_cmd=additional_sumo_cmd)


if __name__ == "__main__":
    main()
