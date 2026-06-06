"""Max-Pressure baseline.

No learning, no value function — but under the unified δ formula

    δ_i(t) = max( V^UE(s_t, i) − G_t(i), 0 )

every method (including Max-Pressure) produces a real number: V^UE is
the shared frozen Fair-TSC critic and G_t(i) is the realized discounted
return from the raw env rewards on this controller's eval rollout.
delta_valid=True everywhere.

Pressure proxy (per intersection, per phase):

    pressure_k = Σ_{l in incoming lanes activated by phase k} queue(l)
                 - Σ_{l in outgoing lanes for those moves}     density(l)

We do NOT have direct per-phase lane masks at the FairTSCEnv level, so
we use a simplified-but-standard approximation built on sumo_rl's
`get_lanes_queue`, `get_out_lanes_density`, and `green_phases[k].state`.
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

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


def _choose_mp_action_for_ts(ts) -> int:
    """Return the green-phase index that maximises pressure for this signal.

    ts: sumo_rl.environment.traffic_signal.TrafficSignal instance.

    Pressure(phase k) = (Σ queue on incoming lanes whose link is 'G' under
    phase k) - (Σ outgoing-lane density for those same links).

    If lane membership per phase cannot be resolved (older sumo_rl
    versions), fall back to "pick the phase whose state string has the
    most G characters * mean queue" — strictly worse but never crashes.
    """
    try:
        num_phases = ts.num_green_phases
        green_phases = ts.green_phases
        in_lanes = ts.lanes
        controlled = ts.sumo.trafficlight.getControlledLinks(ts.id)
        in_q  = ts.get_lanes_queue()
        out_d = ts.get_out_lanes_density()

        in_q_by_lane  = {ln: in_q[i]  for i, ln in enumerate(in_lanes)}  if len(in_q) == len(in_lanes) else {}
        out_d_by_lane = {ln: out_d[i] for i, ln in enumerate(ts.out_lanes)} if len(out_d) == len(ts.out_lanes) else {}

        best_k, best_p = 0, -1e18
        for k in range(num_phases):
            state = green_phases[k].state
            press = 0.0
            for s_idx, ch in enumerate(state):
                if ch not in ("G", "g"):
                    continue
                if s_idx >= len(controlled):
                    continue
                for link in controlled[s_idx]:
                    if not link:
                        continue
                    in_l, out_l = link[0], link[1]
                    press += in_q_by_lane.get(in_l, 0.0)
                    press -= out_d_by_lane.get(out_l, 0.0)
            if press > best_p:
                best_p = press
                best_k = k
        return int(best_k)
    except Exception:
        try:
            mean_q = float(np.mean(ts.get_lanes_queue())) if ts.lanes else 0.0
            scores = []
            for ph in ts.green_phases:
                g_count = sum(1 for c in ph.state if c in ("G", "g"))
                scores.append(g_count * mean_q)
            return int(np.argmax(scores)) if scores else 0
        except Exception:
            return 0


def _mp_actions_for_all(env: FairTSCEnv) -> Dict[str, int]:
    sumo_env = env._walk_to_sumo_env()
    actions = {}
    for a in env.agent_ids:
        ts = sumo_env.traffic_signals.get(a)
        if ts is None:
            actions[a] = 0
        else:
            actions[a] = _choose_mp_action_for_ts(ts)
    return actions


def run_max_pressure_episode(
    env: FairTSCEnv,
    seed: int = 0,
    v_ue=None,
    artifact_dir: Optional[str] = None,
    episode: int = 1,
    stage: str = "eval",
    method_name: str = "max_pressure",
) -> Dict:
    obs = env.reset(seed=seed)
    if v_ue is None:
        v_ue = load_shared_ue_critic(env=env)

    coll = MetricsCollector()
    rollout = []
    done = False

    while not done:
        g_t = env.get_global_obs(obs)
        action_dict = _mp_actions_for_all(env)
        next_obs, R, Cp, Cs, done, info = env.step(action_dict)

        r_arr = np.array([R[a] for a in env.agent_ids], dtype=np.float32)
        rollout.append({"global_obs": g_t, "rewards_array": r_arr})

        mean_r = float(r_arr.mean()) if r_arr.size else 0.0
        coll.add(info, mean_reward=mean_r)
        obs = next_obs

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
    Ignores legacy kwargs (e.g. v_ue_fn=)."""
    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    try:
        result = run_max_pressure_episode(
            env,
            seed=C.SEED if seed is None else int(seed),
            v_ue=v_ue,
            artifact_dir=artifact_dir,
            episode=episode,
            stage=stage,
            method_name="max_pressure",
        )
        print(f"[max_pressure] {result}")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    main()
