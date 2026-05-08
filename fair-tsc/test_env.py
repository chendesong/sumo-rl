"""Smoke test: random policy for 1 episode, print per-step (R, C_p, C_s)."""

import numpy as np

from sumo_env import FairTSCEnv
import config as C


def main():
    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
    )
    print(f"agents     : {env.agent_ids}")
    print(f"num_agents : {env.num_agents}")
    print(f"local dim  : {env.local_obs_dim}")
    print(f"global dim : {env.global_obs_dim}")
    print(f"action dim : {env.action_dim}")

    rng = np.random.default_rng(42)
    obs = env.reset(seed=42)

    total_R = {a: 0.0 for a in env.agent_ids}
    total_Cp = {a: 0.0 for a in env.agent_ids}
    total_Cs = {a: 0.0 for a in env.agent_ids}

    step = 0
    done = False
    while not done:
        actions = {a: int(rng.integers(0, env.action_dim)) for a in env.agent_ids}
        obs, R, Cp, Cs, done, info = env.step(actions)

        for a in env.agent_ids:
            total_R[a]  += R[a]
            total_Cp[a] += Cp[a]
            total_Cs[a] += Cs[a]
        step += 1

        if step % 100 == 0:
            sample = env.agent_ids[0]
            print(f"step {step:4d}  agent={sample}  R={R[sample]:+.2f}  C_p={Cp[sample]:.3f}  C_s={Cs[sample]:.2f}")

    print(f"\nepisode finished after {step} steps\n")
    for a in env.agent_ids:
        print(f"  {a}:  ΣR = {total_R[a]:>10.1f}   ΣC_p = {total_Cp[a]:>8.3f}   ΣC_s = {total_Cs[a]:>8.1f}")

    env.close()


if __name__ == "__main__":
    main()