"""Run only fixed-time and max-pressure baselines on the current config.

This lightweight driver is for quick efficiency checks while learning runs
continue elsewhere.  It does not require V^UE and does not train any policy.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Callable, Dict

import numpy as np

import config as C
from evaluate import MetricsCollector, attach_tripinfo_metrics, make_tripinfo_sumo_cmd
from sumo_env import FairTSCEnv
from baselines.fixed_time import FIXED_PHASE_DURATION
from baselines.max_pressure import _mp_actions_for_all


def _components(env: FairTSCEnv, env_metrics: Dict) -> Dict[str, float]:
    vehicle_queue_series = env_metrics.get("agents_total_stopped_series", [])
    ped_queue_series = env_metrics.get("agents_total_ped_queued_series", [])
    denom = max(float(env.num_agents) * float(C.REWARD_SCALE), 1e-9)
    vehicle_term = -float(np.sum(vehicle_queue_series)) / denom
    ped_term = -float(C.OMEGA_P) * float(np.sum(ped_queue_series)) / denom
    return {
        "reward_vehicle_component": vehicle_term,
        "reward_ped_component": ped_term,
        "reward_env_component_sum": vehicle_term + ped_term,
        "vehicle_queue_mean": float(np.mean(vehicle_queue_series)) if vehicle_queue_series else 0.0,
        "ped_queue_mean": float(np.mean(ped_queue_series)) if ped_queue_series else 0.0,
    }


def _summarize(method: str, env: FairTSCEnv, coll: MetricsCollector, tripinfo_path: str) -> Dict[str, float]:
    env_metrics = coll.finalize(env)
    row = {
        "method": method,
        "demand": C.DEMAND_LEVEL,
        "route_file": os.path.abspath(C.ROUTE_FILE),
        "seed": C.SEED,
        "num_seconds": C.NUM_SECONDS,
        "time_to_teleport": C.TIME_TO_TELEPORT,
        "reward_efficiency": float(np.mean(env_metrics.get("reward_series", [0.0]))),
        "queue_efficiency": -float(np.mean(env_metrics.get("system_total_waiting_time_series", [0.0]))),
        "system_wait_mean": float(np.mean(env_metrics.get("system_total_waiting_time_series", [0.0]))),
        "ped_wait_mean": float(np.mean(env_metrics.get("agents_total_ped_waiting_time_series", [0.0]))),
        "ped_expected_violations": float(np.mean(env_metrics.get("agents_total_expected_violations_series", [0.0]))),
        "teleported_total": float(env_metrics.get("teleported_total", 0.0) or 0.0),
        "departed_total": float(env_metrics.get("departed_total", 0.0) or 0.0),
        "arrived_total": float(env_metrics.get("arrived_total", 0.0) or 0.0),
        "completion_rate_departed": float(env_metrics.get("completion_rate_departed", 0.0) or 0.0),
        "completion_rate_demand": float(env_metrics.get("completion_rate_demand", 0.0) or 0.0),
        "unfinished_vehicle_demand": float(env_metrics.get("unfinished_vehicle_demand", 0.0) or 0.0),
        "active_vehicle_count_end": float(env_metrics.get("active_vehicle_count_end", 0.0) or 0.0),
        "pending_vehicle_count_end": float(env_metrics.get("pending_vehicle_count_end", 0.0) or 0.0),
        "min_expected_number_end": float(env_metrics.get("min_expected_number_end", 0.0) or 0.0),
        "theil_intra": float(env_metrics.get("theil_intra", 0.0) or 0.0),
        "max_phase_interval": float(env_metrics.get("max_phase_interval", 0.0) or 0.0),
    }
    row.update(_components(env, env_metrics))
    attach_tripinfo_metrics(row, tripinfo_path, horizon_s=C.NUM_SECONDS)
    return row


def run_controller(method: str, action_fn: Callable[[FairTSCEnv, int], Dict[str, int]], out_dir: str) -> Dict:
    tripinfo_path = os.path.join(out_dir, f"{method}.tripinfo.xml")
    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
        additional_sumo_cmd=make_tripinfo_sumo_cmd(tripinfo_path),
    )
    try:
        obs = env.reset(seed=C.SEED)
        coll = MetricsCollector()
        done = False
        step = 0
        while not done:
            action = action_fn(env, step)
            next_obs, rewards, _cp, _cs, done, info = env.step(action)
            mean_reward = float(np.mean([rewards[a] for a in env.agent_ids])) if rewards else 0.0
            coll.add(info, mean_reward=mean_reward)
            obs = next_obs if next_obs else obs
            step += 1
        return _summarize(method, env, coll, tripinfo_path)
    finally:
        env.close()


def fixed_time_action(env: FairTSCEnv, step: int) -> Dict[str, int]:
    phase_step = max(1, FIXED_PHASE_DURATION // C.DELTA_TIME)
    phase = (step // phase_step) % max(env.action_dim, 1)
    return {agent: int(phase) for agent in env.agent_ids}


def max_pressure_action(env: FairTSCEnv, _step: int) -> Dict[str, int]:
    return _mp_actions_for_all(env)


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(C.BASE_DIR, "outputs", f"rule_based_{C.DEMAND_LEVEL}_s{C.SEED}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for method, fn in [
        ("fixed_time", fixed_time_action),
        ("max_pressure", max_pressure_action),
    ]:
        print(f"===== {method} =====")
        row = run_controller(method, fn, out_dir)
        rows.append(row)
        print(row)

    csv_path = os.path.join(out_dir, "rule_based_baselines.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_rule_based_baselines] wrote {csv_path}")


if __name__ == "__main__":
    main()
