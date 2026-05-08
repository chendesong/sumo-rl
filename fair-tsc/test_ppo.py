"""Smoke test: collect 1 episode, run 1 PPO update, print training stats."""

import numpy as np
import torch

from sumo_env import FairTSCEnv
from networks import SharedActor, SharedCritic
from rollout_buffer import RolloutBuffer
from ppo_core import ppo_update, bootstrap_last_values
import config as C


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cpu")

    # 1) Env
    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE, out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
    )
    print(f"agents={env.agent_ids}  N={env.num_agents}  D_l={env.local_obs_dim}  D_g={env.global_obs_dim}  A={env.action_dim}")

    # 2) Networks + optimisers
    actor  = SharedActor(env.local_obs_dim, env.num_agents, env.action_dim, C.ACTOR_HIDDEN).to(device)
    critic = SharedCritic(env.global_obs_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)
    actor_optim  = torch.optim.Adam(actor.parameters(),  lr=C.ACTOR_LR)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=C.CRITIC_LR)

    # 3) Collect 1 episode
    buf = RolloutBuffer(env.agent_ids, env.num_agents)
    obs = env.reset(seed=C.SEED)
    done = False
    step = 0

    while not done:
        global_obs = env.get_global_obs(obs)
        local_batch  = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
        global_batch = torch.from_numpy(np.tile(global_obs, (env.num_agents, 1))).to(device)
        idx_batch    = torch.arange(env.num_agents, device=device)

        with torch.no_grad():
            action, logprob = actor.act(local_batch, idx_batch)
            value = critic(global_batch, idx_batch)

        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, R, Cp, Cs, done, _ = env.step(action_dict)

        for i, a in enumerate(env.agent_ids):
            buf.add(
                agent_id=a,
                local_obs=obs[a], global_obs=global_obs,
                action=int(action[i].item()), logprob=float(logprob[i].item()),
                reward=R[a], value=float(value[i].item()), done=done,
                c_p=Cp[a], c_s=Cs[a],
            )
        obs = next_obs
        step += 1

    print(f"\nepisode collected: {step} steps × {env.num_agents} agents = {buf.total_size()} transitions")

    # 4) Bootstrap + GAE
    final_global_obs = env.get_global_obs(obs)
    last_v = bootstrap_last_values(critic, final_global_obs, env.agent_ids, env.num_agents, device)
    buf.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)

    print(f"\n  return     : mean={buf.flat['return'].mean().item():+.2f}  std={buf.flat['return'].std().item():.2f}")
    print(f"  advantage  : mean={buf.flat['advantage'].mean().item():+.2f}  std={buf.flat['advantage'].std().item():.2f}")

    # 5) PPO update
    print(f"\nrunning PPO update: {C.PPO_EPOCHS} epochs × ~{buf.total_size() // C.MINIBATCH_SIZE} minibatches")
    stats = ppo_update(
        actor=actor, critic=critic,
        actor_optim=actor_optim, critic_optim=critic_optim,
        buffer=buf,
        ppo_epochs=C.PPO_EPOCHS,
        minibatch_size=C.MINIBATCH_SIZE,
        clip_eps=C.CLIP_EPS,
        entropy_coeff=C.ENTROPY_COEFF,
        vf_coeff=C.VF_COEFF,
        grad_clip=C.GRAD_CLIP,
    )

    print(f"\nPPO update stats:")
    for k, v in stats.items():
        print(f"  {k:15s} = {v:+.4f}")

    env.close()


if __name__ == "__main__":
    main()