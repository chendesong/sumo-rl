"""Fixed-time baseline: each intersection cycles green phases at a fixed period.

No learning, no value function — but under the unified δ formula

    δ_i(t) = max( V^UE(s_t, i) − G_t(i), 0 )

every method (including Fixed-time) produces a real number: V^UE is the
shared frozen Fair-TSC critic and G_t(i) is the realized discounted
return from the raw env rewards on this controller's eval rollout.
delta_valid=True everywhere.
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

# Ensure parent fair-tsc dir is on path so `import config`, `import sumo_env`
# resolve when this file is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from comparison_artifacts import write_green_split_episode
from sumo_env import FairTSCEnv
from evaluate import (
    MetricsCollector,
    compute_deltas_from_rollout,
    evaluate_run,
    load_shared_ue_critic,
)


# Fixed cycle: spend FIXED_PHASE_DURATION simulated seconds in each green
# phase before advancing.  At the configured decision rate, this is
# FIXED_PHASE_DURATION // Δt decision steps per phase.
FIXED_PHASE_DURATION = 30   # seconds per green phase per intersection


def run_fixed_time_episode(
    env: FairTSCEnv,
    seed: int = 0,
    v_ue=None,
    artifact_dir: Optional[str] = None,
    episode: int = 1,
    stage: str = "eval",
    method_name: str = "fixed_time",
) -> Dict:
    """Roll out one episode with fixed-time control on every intersection.

    Args:
        env:   a FairTSCEnv (will be reset).
        seed:  env seed.
        v_ue:  pre-loaded shared V^UE `SharedCritic`. None → lazy-load
               from default ckpt after the env is reset.

    Returns the dict produced by `evaluate_run` (theil_ema, efficiency, ...).
    `delta_valid` is True (G-based δ applies to every method uniformly).
    """
    obs = env.reset(seed=seed)
    phase_step = FIXED_PHASE_DURATION // C.DELTA_TIME   # decision-steps per phase
    if phase_step < 1:
        phase_step = 1

    if v_ue is None:
        v_ue = load_shared_ue_critic(env=env)

    coll = MetricsCollector()
    rollout = []   # per-step {global_obs, rewards_array} for δ computation
    step_counter = 0
    done = False
    current_phase = {a: 0 for a in env.agent_ids}

    while not done:
        # All agents advance phase in lock-step.  Number of green phases
        # may differ per agent, so each modulates by its own action_dim.
        if step_counter > 0 and step_counter % phase_step == 0:
            for a in env.agent_ids:
                current_phase[a] = (current_phase[a] + 1) % env.action_dim

        g_t = env.get_global_obs(obs)
        action_dict = {a: int(current_phase[a]) for a in env.agent_ids}
        next_obs, R, Cp, Cs, done, info = env.step(action_dict)

        r_arr = np.array([R[a] for a in env.agent_ids], dtype=np.float32)
        rollout.append({"global_obs": g_t, "rewards_array": r_arr})

        mean_r = float(r_arr.mean()) if r_arr.size else 0.0
        coll.add(info, mean_reward=mean_r)

        obs = next_obs
        step_counter += 1

    write_green_split_episode(
        artifact_dir, method=method_name, env=env, episode=episode, stage=stage, seed=seed
    )
    env_metrics = coll.finalize(env)

    if len(rollout) == 0:
        deltas_TN = np.zeros((1, env.num_agents), dtype=np.float32)
    else:
        deltas_TN = compute_deltas_from_rollout(
            rollout, v_ue=v_ue, num_agents=env.num_agents, gamma=C.GAMMA,
        )
    return evaluate_run(deltas_TN, env_metrics, delta_valid=True)


def main(
    v_ue=None,
    additional_sumo_cmd: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    episode: int = 1,
    stage: str = "eval",
    seed: Optional[int] = None,
    **_unused,
):
    """Entry point. `v_ue` may be a pre-loaded shared SharedCritic.
    Accepts and ignores legacy kwargs (e.g. v_ue_fn=) so the caller in
    run_comparison can hand the same kwargs to every baseline without
    branching.
    """
    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    try:
        result = run_fixed_time_episode(
            env,
            seed=C.SEED if seed is None else int(seed),
            v_ue=v_ue,
            artifact_dir=artifact_dir,
            episode=episode,
            stage=stage,
            method_name="fixed_time",
        )
        print(f"[fixed_time] {result}")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    main()
