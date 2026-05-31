"""MA2C-style coordination baseline.

This is a lightweight SUMO/PPO implementation of the core idea in
Chu et al.'s MA2C baseline: each agent observes neighborhood traffic
information and neighbor policy fingerprints, while training on a
spatially discounted cooperative reward. No fairness term is used.
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from evaluate import MetricsCollector, compute_deltas_from_rollout, evaluate_run, load_shared_ue_critic
from networks import SharedActor, SharedCritic
from ppo_core import bootstrap_last_values, ppo_update
from rollout_buffer import RolloutBuffer
from sumo_env import FairTSCEnv

from baselines.coordination_utils import (
    build_neighbor_map,
    graph_distances,
    mean_or_zeros,
    spatially_discounted_rewards,
)


MA2C_COMM_GAMMA = float(os.environ.get("FAIR_TSC_MA2C_COMM_GAMMA", "0.9"))


def _augment_obs(
    obs: Dict[str, np.ndarray],
    agent_ids,
    neighbor_map,
    prev_policy,
    action_dim: int,
) -> Dict[str, np.ndarray]:
    out = {}
    for agent in agent_ids:
        own = obs[agent].astype(np.float32)
        neighbors = [n for n in neighbor_map.get(agent, []) if n in obs]
        neighbor_obs = mean_or_zeros((obs[n].astype(np.float32) for n in neighbors), own)
        if neighbors:
            fp = np.mean(np.stack([prev_policy.get(n, np.zeros(action_dim, dtype=np.float32)) for n in neighbors]), axis=0)
        else:
            fp = np.zeros(action_dim, dtype=np.float32)
        out[agent] = np.concatenate([own, neighbor_obs, fp.astype(np.float32)], axis=0).astype(np.float32)
    return out


def collect_episode_ma2c(
    env,
    actor,
    critic,
    buffer,
    device,
    seed=None,
    coll: Optional[MetricsCollector] = None,
    rollout: Optional[list] = None,
    neighbor_map=None,
    distances=None,
):
    obs = env.reset(seed=seed)
    if neighbor_map is None:
        neighbor_map = build_neighbor_map(env.agent_ids)
    if distances is None:
        distances = graph_distances(env.agent_ids, neighbor_map)

    prev_policy = {
        a: np.full(env.action_dim, 1.0 / max(env.action_dim, 1), dtype=np.float32)
        for a in env.agent_ids
    }
    done = False
    ep_R = {a: 0.0 for a in env.agent_ids}
    n_steps = 0

    while not done:
        global_obs = env.get_global_obs(obs)
        aug_obs = _augment_obs(obs, env.agent_ids, neighbor_map, prev_policy, env.action_dim)
        local_b = torch.from_numpy(np.stack([aug_obs[a] for a in env.agent_ids])).to(device)
        global_b = torch.from_numpy(np.tile(global_obs, (env.num_agents, 1))).to(device)
        idx_b = torch.arange(env.num_agents, device=device)

        with torch.no_grad():
            dist = actor(local_b, idx_b)
            action = dist.sample()
            logprob = dist.log_prob(action)
            probs = dist.probs.detach().cpu().numpy()
            value = critic(global_b, idx_b)

        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, raw_R, _cp, _cs, done, info = env.step(action_dict)
        train_R = spatially_discounted_rewards(raw_R, env.agent_ids, distances, gamma=MA2C_COMM_GAMMA)
        raw_vec = np.array([raw_R[a] for a in env.agent_ids], dtype=np.float32)

        for i, agent in enumerate(env.agent_ids):
            buffer.add(
                agent_id=agent,
                local_obs=aug_obs[agent],
                global_obs=global_obs,
                action=int(action[i].item()),
                logprob=float(logprob[i].item()),
                reward=float(train_R[agent]),
                value=float(value[i].item()),
                done=done,
            )
            ep_R[agent] += float(raw_R[agent])
            prev_policy[agent] = probs[i].astype(np.float32)

        if coll is not None:
            coll.add(info, mean_reward=float(raw_vec.mean()) if raw_vec.size else 0.0)
        if rollout is not None:
            rollout.append({"global_obs": global_obs, "rewards_array": raw_vec.copy()})
        if next_obs:
            obs = next_obs
        n_steps += 1

    if C.REWARD_NORMALIZE:
        buffer.normalize_rewards(center=C.REWARD_NORM_CENTER, clip=C.REWARD_NORM_CLIP, eps=C.REWARD_NORM_EPS)
    last_v = bootstrap_last_values(critic, env.get_global_obs(obs), env.agent_ids, env.num_agents, device)
    buffer.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    return ep_R, n_steps


def train_ma2c(
    num_episodes: int = 150,
    seed: Optional[int] = None,
    v_ue=None,
    save_critic: bool = True,
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
        neighbor_map = build_neighbor_map(env.agent_ids)
        distances = graph_distances(env.agent_ids, neighbor_map)
        aug_dim = env.local_obs_dim * 2 + env.action_dim
        actor = SharedActor(aug_dim, env.num_agents, env.action_dim, C.ACTOR_HIDDEN).to(device)
        critic = SharedCritic(env.global_obs_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)
        o_a = torch.optim.Adam(actor.parameters(), lr=C.ACTOR_LR)
        o_c = torch.optim.Adam(critic.parameters(), lr=C.CRITIC_LR)

        for ep in range(num_episodes):
            buf = RolloutBuffer(env.agent_ids, env.num_agents)
            ep_R, _n = collect_episode_ma2c(
                env, actor, critic, buf, device, seed=seed + ep,
                neighbor_map=neighbor_map, distances=distances,
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
            print(f"[MA2C] ep={ep+1:3d}/{num_episodes} Rbar(raw)={rR.mean():+.1f} "
                  f"ploss={st['policy_loss']:+.4f} H={st['entropy']:.3f}")

        if save_critic:
            out_dir = os.path.join(C.BASE_DIR, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "aug_dim": aug_dim,
                    "global_obs_dim": env.global_obs_dim,
                    "num_agents": env.num_agents,
                    "hidden": C.CRITIC_HIDDEN,
                    "num_episodes": num_episodes,
                    "seed": seed,
                    "comm_gamma": MA2C_COMM_GAMMA,
                    "baseline": "ma2c_neighbor_fingerprint",
                },
                os.path.join(out_dir, "ma2c_baseline.pt"),
            )

        coll = MetricsCollector()
        buf = RolloutBuffer(env.agent_ids, env.num_agents)
        rollout = []
        collect_episode_ma2c(
            env, actor, critic, buf, device,
            seed=seed + num_episodes, coll=coll, rollout=rollout,
            neighbor_map=neighbor_map, distances=distances,
        )
        env_metrics = coll.finalize(env)
        if v_ue is None:
            v_ue = load_shared_ue_critic(env=env, device=device)
        deltas_TN = compute_deltas_from_rollout(rollout, v_ue=v_ue, num_agents=env.num_agents, gamma=C.GAMMA)
        result = evaluate_run(deltas_TN, env_metrics, delta_valid=True)
        print(f"[MA2C eval] {result}")
        return result
    finally:
        env.close()


def main(v_ue=None, additional_sumo_cmd: Optional[str] = None, **_unused):
    return train_ma2c(num_episodes=50, v_ue=v_ue, additional_sumo_cmd=additional_sumo_cmd)


if __name__ == "__main__":
    main()
