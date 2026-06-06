"""SocialLight-style distributed cooperation baseline.

SocialLight learns cooperation with a locally-centralized critic over a
small neighborhood and uses counterfactual reasoning to estimate each
agent's marginal contribution. This implementation keeps our SUMO/PPO
scaffold but mirrors that core mechanism:

  actor:  decentralized, uses the local intersection observation.
  critic: Q_i(neighborhood observations, neighborhood actions).
  actor advantage: Q_i(z_i, a_i, a_Ni) - E_{a'_i~pi_i} Q_i(z_i, a'_i, a_Ni).

No fairness reward or fairness constraint is used.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from comparison_artifacts import write_green_split_episode, write_train_episode
from evaluate import MetricsCollector, compute_deltas_from_rollout, evaluate_run, load_shared_ue_critic
from networks import SharedActor
from sumo_env import FairTSCEnv

from baselines.coordination_utils import build_neighbor_map, graph_distances, spatially_discounted_rewards


SOCIALLIGHT_MAX_NEIGHBORS = int(os.environ.get("FAIR_TSC_SOCIALLIGHT_MAX_NEIGHBORS", "4"))
SOCIALLIGHT_COMM_GAMMA = float(os.environ.get("FAIR_TSC_SOCIALLIGHT_COMM_GAMMA", "0.9"))
SOCIALLIGHT_CRITIC_EPOCHS = int(os.environ.get("FAIR_TSC_SOCIALLIGHT_CRITIC_EPOCHS", "8"))


def _mlp(in_dim, hidden, out_dim):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SocialLightCritic(nn.Module):
    """Neighborhood Q critic used for counterfactual marginal contribution."""

    def __init__(self, packed_obs_dim: int, action_pack_dim: int, num_agents: int, hidden):
        super().__init__()
        self.packed_obs_dim = int(packed_obs_dim)
        self.action_pack_dim = int(action_pack_dim)
        self.num_agents = int(num_agents)
        self.net = _mlp(packed_obs_dim + action_pack_dim + num_agents, hidden, 1)

    def forward(self, packed_obs: torch.Tensor, action_pack: torch.Tensor, agent_idx: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(agent_idx, num_classes=self.num_agents).float()
        x = torch.cat([packed_obs, action_pack, onehot], dim=-1)
        return self.net(x).squeeze(-1)


def _pack_obs(obs: Dict[str, np.ndarray], agent_ids, neighbor_map, obs_dim: int, max_neighbors: int):
    out = {}
    zeros = np.zeros(obs_dim, dtype=np.float32)
    for agent in agent_ids:
        neighbors = [n for n in neighbor_map.get(agent, []) if n in obs][:max_neighbors]
        parts = [obs[agent].astype(np.float32)]
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


def _pack_actions(action_dict: Dict[str, int], agent_ids, neighbor_map, action_dim: int, max_neighbors: int):
    out = {}
    zeros = np.zeros(action_dim, dtype=np.float32)
    for agent in agent_ids:
        neighbors = [n for n in neighbor_map.get(agent, []) if n in action_dict][:max_neighbors]
        parts = []
        own = np.zeros(action_dim, dtype=np.float32)
        own[int(action_dict[agent])] = 1.0
        parts.append(own)
        for i in range(max_neighbors):
            if i < len(neighbors):
                vec = np.zeros(action_dim, dtype=np.float32)
                vec[int(action_dict[neighbors[i]])] = 1.0
                parts.append(vec)
            else:
                parts.append(zeros)
        out[agent] = np.concatenate(parts, axis=0).astype(np.float32)
    return out


@dataclass
class SocialRollout:
    local_obs: List[np.ndarray]
    packed_obs: List[np.ndarray]
    action_pack: List[np.ndarray]
    actions: List[int]
    logprobs: List[float]
    probs: List[np.ndarray]
    agent_idx: List[int]
    train_rewards_tn: List[np.ndarray]
    raw_rollout: List[dict]


def _discounted_returns(rewards_tn: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards_tn, dtype=np.float32)
    running = np.zeros(rewards_tn.shape[1], dtype=np.float32)
    for t in range(rewards_tn.shape[0] - 1, -1, -1):
        running = rewards_tn[t] + gamma * running
        returns[t] = running
    return returns


def collect_episode_sociallight(
    env,
    actor,
    device,
    seed=None,
    neighbor_map=None,
    distances=None,
    coll: Optional[MetricsCollector] = None,
    rollout: Optional[list] = None,
) -> tuple[Dict[str, float], SocialRollout]:
    obs = env.reset(seed=seed)
    if neighbor_map is None:
        neighbor_map = build_neighbor_map(env.agent_ids, max_neighbors=SOCIALLIGHT_MAX_NEIGHBORS)
    if distances is None:
        distances = graph_distances(env.agent_ids, neighbor_map)

    data = SocialRollout([], [], [], [], [], [], [], [], [])
    ep_R = {a: 0.0 for a in env.agent_ids}
    done = False

    while not done:
        global_obs = env.get_global_obs(obs)
        packed_obs = _pack_obs(obs, env.agent_ids, neighbor_map, env.local_obs_dim, SOCIALLIGHT_MAX_NEIGHBORS)
        local_b = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
        idx_b = torch.arange(env.num_agents, device=device)
        with torch.no_grad():
            dist = actor(local_b, idx_b)
            action = dist.sample()
            logprob = dist.log_prob(action)
            probs = dist.probs.detach().cpu().numpy()

        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        action_pack = _pack_actions(action_dict, env.agent_ids, neighbor_map, env.action_dim, SOCIALLIGHT_MAX_NEIGHBORS)
        next_obs, raw_R, _cp, _cs, done, info = env.step(action_dict)
        train_R = spatially_discounted_rewards(raw_R, env.agent_ids, distances, gamma=SOCIALLIGHT_COMM_GAMMA)

        raw_vec = np.array([raw_R[a] for a in env.agent_ids], dtype=np.float32)
        train_vec = np.array([train_R[a] for a in env.agent_ids], dtype=np.float32)
        data.train_rewards_tn.append(train_vec)
        data.raw_rollout.append({"global_obs": global_obs, "rewards_array": raw_vec.copy()})

        for i, agent in enumerate(env.agent_ids):
            data.local_obs.append(obs[agent].astype(np.float32))
            data.packed_obs.append(packed_obs[agent])
            data.action_pack.append(action_pack[agent])
            data.actions.append(int(action[i].item()))
            data.logprobs.append(float(logprob[i].item()))
            data.probs.append(probs[i].astype(np.float32))
            data.agent_idx.append(i)
            ep_R[agent] += float(raw_R[agent])

        if coll is not None:
            coll.add(info, mean_reward=float(raw_vec.mean()) if raw_vec.size else 0.0)
        if rollout is not None:
            rollout.append({"global_obs": global_obs, "rewards_array": raw_vec.copy()})
        if next_obs:
            obs = next_obs

    return ep_R, data


def _flatten_rollout(data: SocialRollout, device):
    return {
        "local_obs": torch.from_numpy(np.stack(data.local_obs)).to(device),
        "packed_obs": torch.from_numpy(np.stack(data.packed_obs)).to(device),
        "action_pack": torch.from_numpy(np.stack(data.action_pack)).to(device),
        "actions": torch.tensor(data.actions, dtype=torch.long, device=device),
        "logprobs": torch.tensor(data.logprobs, dtype=torch.float32, device=device),
        "probs": torch.from_numpy(np.stack(data.probs)).to(device),
        "agent_idx": torch.tensor(data.agent_idx, dtype=torch.long, device=device),
    }


def _compute_targets(data: SocialRollout, num_agents: int, device):
    rewards = np.stack(data.train_rewards_tn).astype(np.float32)
    returns = _discounted_returns(rewards, gamma=C.GAMMA).reshape(-1)
    if C.REWARD_NORMALIZE:
        std = float(np.std(returns))
        returns = returns / max(std, C.REWARD_NORM_EPS)
        if C.REWARD_NORM_CLIP and C.REWARD_NORM_CLIP > 0:
            returns = np.clip(returns, -C.REWARD_NORM_CLIP, C.REWARD_NORM_CLIP)
    return torch.from_numpy(returns.astype(np.float32)).to(device)


def _train_social_critic(critic, optim, flat, targets, minibatch_size: int):
    n = targets.shape[0]
    stats = []
    for _ in range(SOCIALLIGHT_CRITIC_EPOCHS):
        perm = torch.randperm(n, device=targets.device)
        for s in range(0, n, minibatch_size):
            idx = perm[s:s + minibatch_size]
            pred = critic(flat["packed_obs"][idx], flat["action_pack"][idx], flat["agent_idx"][idx])
            loss = F.mse_loss(pred, targets[idx])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), C.GRAD_CLIP)
            optim.step()
            stats.append(float(loss.item()))
    return float(np.mean(stats)) if stats else 0.0


@torch.no_grad()
def _counterfactual_advantages(actor, critic, flat, action_dim: int):
    q_taken = critic(flat["packed_obs"], flat["action_pack"], flat["agent_idx"])
    q_all = []
    for a in range(action_dim):
        cf_pack = flat["action_pack"].clone()
        cf_pack[:, :action_dim] = 0.0
        cf_pack[:, a] = 1.0
        q_all.append(critic(flat["packed_obs"], cf_pack, flat["agent_idx"]))
    q_all = torch.stack(q_all, dim=1)
    baseline = (flat["probs"] * q_all).sum(dim=1)
    adv = q_taken - baseline
    return adv


def _actor_update(actor, optim, flat, advantages, minibatch_size: int):
    n = advantages.shape[0]
    stats = {"policy_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for _ in range(C.PPO_EPOCHS):
        perm = torch.randperm(n, device=advantages.device)
        for s in range(0, n, minibatch_size):
            idx = perm[s:s + minibatch_size]
            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            new_logp, entropy = actor.evaluate(flat["local_obs"][idx], flat["agent_idx"][idx], flat["actions"][idx])
            old_logp = flat["logprobs"][idx]
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - C.CLIP_EPS, 1.0 + C.CLIP_EPS) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss - C.ENTROPY_COEFF * entropy.mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), C.GRAD_CLIP)
            optim.step()

            with torch.no_grad():
                stats["policy_loss"].append(float(policy_loss.item()))
                stats["entropy"].append(float(entropy.mean().item()))
                stats["approx_kl"].append(float((old_logp - new_logp).mean().item()))
                stats["clip_frac"].append(float(((ratio - 1.0).abs() > C.CLIP_EPS).float().mean().item()))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def train_sociallight(
    num_episodes: int = 150,
    seed: Optional[int] = None,
    v_ue=None,
    save_critic: bool = True,
    additional_sumo_cmd: Optional[str] = None,
    artifact_dir: Optional[str] = None,
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
        neighbor_map = build_neighbor_map(env.agent_ids, max_neighbors=SOCIALLIGHT_MAX_NEIGHBORS)
        distances = graph_distances(env.agent_ids, neighbor_map)
        packed_obs_dim = env.local_obs_dim * (1 + SOCIALLIGHT_MAX_NEIGHBORS) + SOCIALLIGHT_MAX_NEIGHBORS
        action_pack_dim = env.action_dim * (1 + SOCIALLIGHT_MAX_NEIGHBORS)

        actor = SharedActor(env.local_obs_dim, env.num_agents, env.action_dim, C.ACTOR_HIDDEN).to(device)
        critic = SocialLightCritic(packed_obs_dim, action_pack_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=C.ACTOR_LR)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=C.CRITIC_LR)

        for ep in range(num_episodes):
            ep_R, data = collect_episode_sociallight(
                env, actor, device, seed=seed + ep, neighbor_map=neighbor_map, distances=distances
            )
            write_train_episode(artifact_dir, "sociallight", env, ep + 1, ep_R, seed=seed)
            write_green_split_episode(
                artifact_dir, "sociallight", env, ep + 1, stage="train", seed=seed
            )
            if v_ue is None and ep == 0:
                v_ue = load_shared_ue_critic(env=env, device=device)
            flat = _flatten_rollout(data, device)
            targets = _compute_targets(data, env.num_agents, device)
            critic_loss = _train_social_critic(critic, critic_optim, flat, targets, C.MINIBATCH_SIZE)
            advantages = _counterfactual_advantages(actor, critic, flat, env.action_dim)
            st = _actor_update(actor, actor_optim, flat, advantages, C.MINIBATCH_SIZE)
            rR = np.array(list(ep_R.values()), dtype=np.float32)
            print(f"[SocialLight] ep={ep+1:3d}/{num_episodes} Rbar(raw)={rR.mean():+.1f} "
                  f"closs={critic_loss:.4f} ploss={st['policy_loss']:+.4f} H={st['entropy']:.3f}")

        if save_critic:
            out_dir = os.path.join(C.BASE_DIR, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "packed_obs_dim": packed_obs_dim,
                    "action_pack_dim": action_pack_dim,
                    "num_agents": env.num_agents,
                    "num_episodes": num_episodes,
                    "seed": seed,
                    "comm_gamma": SOCIALLIGHT_COMM_GAMMA,
                    "baseline": "sociallight_counterfactual_neighborhood",
                },
                os.path.join(out_dir, "sociallight_baseline.pt"),
            )

        coll = MetricsCollector()
        rollout = []
        collect_episode_sociallight(
            env, actor, device, seed=seed + num_episodes,
            neighbor_map=neighbor_map, distances=distances, coll=coll, rollout=rollout,
        )
        write_green_split_episode(
            artifact_dir, "sociallight", env, num_episodes + 1, stage="eval", seed=seed
        )
        env_metrics = coll.finalize(env)
        if v_ue is None:
            v_ue = load_shared_ue_critic(env=env, device=device)
        deltas_TN = compute_deltas_from_rollout(rollout, v_ue=v_ue, num_agents=env.num_agents, gamma=C.GAMMA)
        result = evaluate_run(deltas_TN, env_metrics, delta_valid=True)
        print(f"[SocialLight eval] {result}")
        return result
    finally:
        env.close()


def main(v_ue=None, additional_sumo_cmd: Optional[str] = None, artifact_dir: Optional[str] = None,
         num_episodes: Optional[int] = None, seed: Optional[int] = None, **_unused):
    return train_sociallight(
        num_episodes=50 if num_episodes is None else int(num_episodes),
        seed=seed,
        v_ue=v_ue,
        additional_sumo_cmd=additional_sumo_cmd,
        artifact_dir=artifact_dir,
    )


if __name__ == "__main__":
    main()
