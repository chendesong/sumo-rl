"""Smoke test: random-policy rollout + GAE, verify shapes flow end-to-end."""

import numpy as np
import torch

from sumo_env import FairTSCEnv
from networks import SharedActor, SharedCritic
from rollout_buffer import RolloutBuffer
import config as C


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cpu")

    # 1) Build env
    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
    )
    print(f"agents={env.agent_ids}  N={env.num_agents}  D_l={env.local_obs_dim}  D_g={env.global_obs_dim}  A={env.action_dim}")

    # 2) Build networks (use random init, just verify shapes + forward pass)
    actor = SharedActor(
        local_obs_dim=env.local_obs_dim,
        num_agents=env.num_agents,
        action_dim=env.action_dim,
        hidden=C.ACTOR_HIDDEN,
    ).to(device)
    critic = SharedCritic(
        global_obs_dim=env.global_obs_dim,
        num_agents=env.num_agents,
        hidden=C.CRITIC_HIDDEN,
    ).to(device)

    n_actor_params  = sum(p.numel() for p in actor.parameters())
    n_critic_params = sum(p.numel() for p in critic.parameters())
    print(f"actor params  : {n_actor_params:,}")
    print(f"critic params : {n_critic_params:,}")

    # 3) Roll one episode using the actor (random init = essentially random policy)
    buf = RolloutBuffer(env.agent_ids, env.num_agents)
    obs = env.reset(seed=C.SEED)
    done = False
    step = 0

    while not done:
        global_obs = env.get_global_obs(obs)  # [D_g]

        # Batch all 4 agents in one forward pass
        local_batch  = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)  # [N, D_l]
        global_batch = torch.from_numpy(np.tile(global_obs, (env.num_agents, 1))).to(device)   # [N, D_g]
        idx_batch    = torch.arange(env.num_agents, device=device)                             # [N]

        with torch.no_grad():
            action, logprob = actor.act(local_batch, idx_batch)
            value = critic(global_batch, idx_batch)

        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}

        next_obs, R, Cp, Cs, done, _ = env.step(action_dict)

        for i, a in enumerate(env.agent_ids):
            buf.add(
                agent_id=a,
                local_obs=obs[a],
                global_obs=global_obs,
                action=int(action[i].item()),
                logprob=float(logprob[i].item()),
                reward=R[a],
                value=float(value[i].item()),
                done=done,
                c_p=Cp[a],
                c_s=Cs[a],
            )

        obs = next_obs
        step += 1

    print(f"episode finished: {step} steps, buffer size = {buf.total_size()}")

    # 4) Bootstrap last_values + run GAE
    final_global_obs = env.get_global_obs(obs)
    final_global_batch = torch.from_numpy(np.tile(final_global_obs, (env.num_agents, 1))).to(device)
    idx_batch = torch.arange(env.num_agents, device=device)
    with torch.no_grad():
        last_v = critic(final_global_batch, idx_batch).cpu().numpy()
    last_values = {a: float(last_v[i]) for i, a in enumerate(env.agent_ids)}

    buf.compute_gae(last_values, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)

    # 5) Verify flat tensor shapes
    print("\nflat tensors after GAE:")
    for k, v in buf.flat.items():
        print(f"  {k:12s} shape={tuple(v.shape)}  dtype={v.dtype}")

    # 6) Iterate one mini-batch to sanity-check
    mb = next(buf.iter_minibatches(minibatch_size=C.MINIBATCH_SIZE))
    print(f"\nfirst minibatch keys+shapes:")
    for k, v in mb.items():
        print(f"  {k:12s} shape={tuple(v.shape)}")

    # 7) Sanity: advantage & return statistics
    adv = buf.flat["advantage"].cpu().numpy()
    ret = buf.flat["return"].cpu().numpy()
    print(f"\nadvantage  : mean={adv.mean():+.3f}  std={adv.std():.3f}  min={adv.min():+.3f}  max={adv.max():+.3f}")
    print(f"return     : mean={ret.mean():+.3f}  std={ret.std():.3f}  min={ret.min():+.3f}  max={ret.max():+.3f}")
    print(f"C_p        : mean={buf.flat['c_p'].mean().item():.4f}  max={buf.flat['c_p'].max().item():.4f}")
    print(f"C_s        : mean={buf.flat['c_s'].mean().item():.4f}  max={buf.flat['c_s'].max().item():.4f}")

    env.close()


if __name__ == "__main__":
    main()