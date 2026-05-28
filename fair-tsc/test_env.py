"""Smoke test: random policy for one episode, print reward and phase intervals."""

import numpy as np

import config as C
from sumo_env import FairTSCEnv


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
    total_reward = {a: 0.0 for a in env.agent_ids}

    step = 0
    done = False
    while not done:
        actions = {a: int(rng.integers(0, env.action_dim)) for a in env.agent_ids}
        obs, reward, _cp, _cs, done, _info = env.step(actions)
        for a in env.agent_ids:
            total_reward[a] += reward[a]
        step += 1
        if step % 100 == 0:
            sample = env.agent_ids[0]
            print(f"step {step:4d}  agent={sample}  R={reward[sample]:+.2f}")

    summary = env.get_phase_service_summary()
    print(f"\nepisode finished after {step} steps")
    print(f"theil_intra        : {summary['theil_intra']:.6f}")
    print(f"max_phase_interval : {summary['max_phase_interval']:.1f}")
    for a in env.agent_ids:
        print(f"  {a}: sum_R={total_reward[a]:>10.1f}  T_intra={summary.get(f'theil_intra_{a}', 0.0):.6f}")
    env.close()


if __name__ == "__main__":
    main()
