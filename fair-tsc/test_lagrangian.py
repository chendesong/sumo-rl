"""Smoke test: Theil + sacrifice gap + dual updates + fair advantage."""

import numpy as np
import torch

from sumo_env import FairTSCEnv
from networks import SharedActor, SharedCritic
from rollout_buffer import RolloutBuffer
from ppo_core import bootstrap_last_values
from lagrangian import (
    theil_t_index, theil_t_batch,
    compute_sacrifice_gaps, reshape_deltas_to_step_agent,
    LagrangianMultipliers,
)
import config as C


def test_theil_unit():
    """Unit test: Theil = 0 when all equal, > 0 when unequal."""
    print("─── Theil unit tests ───")
    eq    = np.array([1.0, 1.0, 1.0, 1.0])
    uneq  = np.array([0.0, 0.0, 0.0, 4.0])
    mid   = np.array([0.5, 1.0, 1.0, 1.5])
    print(f"  equal      [1,1,1,1]    → T = {theil_t_index(eq):.6f}   (expected ~0)")
    print(f"  one big    [0,0,0,4]    → T = {theil_t_index(uneq):.6f}   (expected large)")
    print(f"  mild       [0.5..1.5]   → T = {theil_t_index(mid):.6f}   (small but >0)")
    assert theil_t_index(eq) < 1e-3, "T should be ~0 for equal gaps"
    assert theil_t_index(uneq) > theil_t_index(mid), "more inequality → larger T"
    print("  ✓ unit tests passed\n")


def main():
    test_theil_unit()

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cpu")

    # 1) Env + networks
    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE, out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
    )

    actor       = SharedActor (env.local_obs_dim,  env.num_agents, env.action_dim, C.ACTOR_HIDDEN ).to(device)
    marl_critic = SharedCritic(env.global_obs_dim, env.num_agents,                  C.CRITIC_HIDDEN).to(device)
    ue_critic   = SharedCritic(env.global_obs_dim, env.num_agents,                  C.CRITIC_HIDDEN).to(device)

    # 2) Collect 1 episode
    buf = RolloutBuffer(env.agent_ids, env.num_agents)
    obs = env.reset(seed=C.SEED)
    done = False
    while not done:
        global_obs = env.get_global_obs(obs)
        local_batch  = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
        global_batch = torch.from_numpy(np.tile(global_obs, (env.num_agents, 1))).to(device)
        idx_batch    = torch.arange(env.num_agents, device=device)
        with torch.no_grad():
            action, logprob = actor.act(local_batch, idx_batch)
            value = marl_critic(global_batch, idx_batch)
        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, R, Cp, Cs, done, _ = env.step(action_dict)
        for i, a in enumerate(env.agent_ids):
            buf.add(
                agent_id=a, local_obs=obs[a], global_obs=global_obs,
                action=int(action[i].item()), logprob=float(logprob[i].item()),
                reward=R[a], value=float(value[i].item()), done=done,
                c_p=Cp[a], c_s=Cs[a],
            )
        obs = next_obs

    # GAE
    last_v = bootstrap_last_values(marl_critic, env.get_global_obs(obs),
                                   env.agent_ids, env.num_agents, device)
    buf.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    print(f"buffer ready: B = {buf.total_size()}")

    # 3) Sacrifice gaps
    print("\n─── Sacrifice gap (paper Eq. 17) ───")
    deltas = compute_sacrifice_gaps(buf, ue_critic, marl_critic, device)
    print(f"  deltas shape    : {tuple(deltas.shape)}")
    print(f"  deltas mean     : {deltas.mean().item():.4f}")
    print(f"  deltas max      : {deltas.max().item():.4f}")
    print(f"  deltas min      : {deltas.min().item():.4f}   (should be ≥ 0 due to clamp)")
    print(f"  fraction > 0    : {(deltas > 0).float().mean().item():.3f}")

    # Reshape to [T, N] for Theil
    deltas_TN = reshape_deltas_to_step_agent(
        deltas, buf.flat["agent_idx"], env.num_agents
    )
    print(f"  deltas reshaped : [T={deltas_TN.shape[0]}, N={deltas_TN.shape[1]}]")

    # 4) Theil over batch
    print("\n─── Theil index ───")
    T_avg = theil_t_batch(deltas_TN, eps=C.THEIL_EPS)
    print(f"  T̄_B = {T_avg:.6f}   (paper threshold T_max = {C.T_MAX})")

    # 5) Dual variables
    print("\n─── Lagrangian multipliers ───")
    lagr = LagrangianMultipliers(
        agent_ids=env.agent_ids, d_p=C.D_P, d_s=C.D_S, t_max=C.T_MAX,
        eta_lambda=C.ETA_LAMBDA, eta_mu=C.ETA_MU,
    )
    print(f"  initial: λ^p={lagr.lambda_p}, λ^s={lagr.lambda_s}, µ={lagr.mu}")

    # Compute batch-mean C^p, C^s per agent
    agent_idx_np = buf.flat["agent_idx"].cpu().numpy()
    c_p_np = buf.flat["c_p"].cpu().numpy()
    c_s_np = buf.flat["c_s"].cpu().numpy()
    mean_c_p, mean_c_s = {}, {}
    for n, a in enumerate(env.agent_ids):
        mask = agent_idx_np == n
        mean_c_p[a] = float(c_p_np[mask].mean())
        mean_c_s[a] = float(c_s_np[mask].mean())
    print(f"  mean C_p per agent: {mean_c_p}")
    print(f"  mean C_s per agent: {mean_c_s}")

    lagr.update_intra(mean_c_p, mean_c_s)
    lagr.update_inter(T_avg)
    print(f"\n  after 1 update: λ^p={lagr.lambda_p}")
    print(f"                  λ^s={lagr.lambda_s}")
    print(f"                  µ  ={lagr.mu:.6f}")

    # 6) Apply fair advantage
    print("\n─── Fair advantage (paper Eq. 28) ───")
    agent_idx_to_id = {i: a for i, a in enumerate(env.agent_ids)}
    adv_before = buf.flat["advantage"].clone()
    lagr.apply_fair_advantage(buf, deltas, agent_idx_to_id)
    adv_after = buf.flat["advantage"]
    diff = (adv_after - adv_before).abs().mean().item()
    print(f"  |Δ adv| mean = {diff:.6f}   (should be small if λ,µ all ≈ 0)")
    print(f"  adv before  : mean={adv_before.mean().item():+.2f} std={adv_before.std().item():.2f}")
    print(f"  adv after   : mean={adv_after.mean().item():+.2f} std={adv_after.std().item():.2f}")

    print("\n  ✓ end-to-end pipeline OK\n")
    env.close()


if __name__ == "__main__":
    main()