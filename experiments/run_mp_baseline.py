"""
Max-Pressure Baseline for 2x2 SUMO Grid

Each intersection has:
  - lanes[0:8]: 8 real vehicle incoming lanes (_1, _2 suffixes)
  - lanes[8:12]: 4 walking area lanes (skip for vehicle pressure)
  - green_phases[0..3]: 4 green phases
  - phase state: 20 chars, positions 16-19 = crossings c0-c3

Pressure per phase:
  veh = Σ halting(green vehicle lanes) - downstream_avg
  ped = Σ persons_on_crossing(green crossings) × ω_MP_p
"""

import os
import sys
import csv

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import sumo_rl
from sumo_rl.environment.observations import PedestrianObservationFunction

BASE_DIR = "C:/Users/ucemdc3/PycharmProjects/sumo-rl"
NET_FILE = os.path.join(BASE_DIR, "nets/2x2grid/01.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, "nets/2x2grid/02.rou.xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/mp_baseline")

NUM_SECONDS = 3600
DELTA_TIME = 5
MIN_GREEN = 5
NUM_EPISODES = 5
OMEGA_MP_PED = 0.2
MP_DECISION_INTERVAL = 3  # 15s


def get_sumo_env(par_env):
    obj = par_env
    for _ in range(10):
        if hasattr(obj, 'traffic_signals'):
            return obj
        if hasattr(obj, 'aec_env'):
            obj = obj.aec_env
        elif hasattr(obj, 'env'):
            obj = obj.env
        else:
            break
    raise RuntimeError("Could not find SumoEnvironment")


def compute_pressure(ts_obj):
    """Compute pressure for each green phase at one intersection."""
    sumo = ts_obj.sumo
    pressures = {}

    # Downstream avg: only real vehicle out_lanes (not : prefixed)
    real_out = [l for l in ts_obj.out_lanes if not l.startswith(':')]
    if real_out:
        downstream_avg = sum(sumo.lane.getLastStepVehicleNumber(l) for l in real_out) / len(real_out)
    else:
        downstream_avg = 0

    # Ped queue per crossing (fast edge API)
    ped_q = {}
    for c_id in ts_obj.crossing_ids:
        try:
            ped_q[c_id] = len(sumo.edge.getLastStepPersonIDs(c_id))
        except Exception:
            ped_q[c_id] = 0

    for idx, phase in enumerate(ts_obj.green_phases):
        state = phase.state

        # Vehicle: only lanes[0:8], skip walking areas
        veh_p = 0.0
        for i in range(min(8, len(state), len(ts_obj.lanes))):
            if state[i] in ('G', 'g'):
                veh_p += sumo.lane.getLastStepHaltingNumber(ts_obj.lanes[i]) - downstream_avg

        # Pedestrian: state[16:19] = crossings
        ped_p = 0.0
        for p in range(4):
            pos = 16 + p
            if pos < len(state) and state[pos] in ('G', 'g'):
                if p < len(ts_obj.crossing_ids):
                    ped_p += ped_q.get(ts_obj.crossing_ids[p], 0)

        pressures[idx] = veh_p + OMEGA_MP_PED * ped_p
    return pressures


def select_mp_action(ts_obj):
    pressures = compute_pressure(ts_obj)
    if not pressures:
        return 0
    best = max(pressures, key=pressures.get)
    current = ts_obj.green_phase
    if current in pressures and pressures.get(current, 0) >= pressures[best] * 0.7:
        return current
    return best


def run_mp_episode(episode_num):
    par_env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        min_green=MIN_GREEN,
        reward_fn="diff-waiting-time-with-pedestrian",
        observation_class=PedestrianObservationFunction,
        sumo_warnings=False,
    )

    obs, info = par_env.reset()
    agents = list(par_env.agents)
    sumo_env = get_sumo_env(par_env)

    records = []
    step_count = 0
    cached_actions = {a: 0 for a in agents}

    while par_env.agents:
        if step_count % MP_DECISION_INTERVAL == 0:
            for a in par_env.agents:
                cached_actions[a] = select_mp_action(sumo_env.traffic_signals[a])

        actions = {a: cached_actions[a] for a in par_env.agents}
        obs, rewards, terminations, truncations, infos = par_env.step(actions)
        step_count += 1

        record = {"step": step_count * DELTA_TIME, "episode": episode_num}
        for a in agents:
            record[f"{a}_mp_reward"] = rewards.get(a, 0.0)
        record["total_mp_reward"] = sum(rewards.get(a, 0.0) for a in agents)
        records.append(record)

        if step_count % 100 == 0:
            total_so_far = sum(r['total_mp_reward'] for r in records)
            print(f"  Step {step_count * DELTA_TIME}s, cumulative reward={total_so_far:.1f}")

    par_env.close()
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    for ep in range(1, NUM_EPISODES + 1):
        print(f"Running MP episode {ep}/{NUM_EPISODES}...")
        records = run_mp_episode(ep)
        all_records.extend(records)
        total_r = sum(r['total_mp_reward'] for r in records)
        print(f"  Episode {ep} done: {len(records)} steps, total reward = {total_r:.1f}")

    output_file = os.path.join(OUTPUT_DIR, "mp_reference.csv")
    if all_records:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
    print(f"\nSaved: {output_file}")

    print("\n" + "=" * 60)
    rewards_per_ep = []
    for ep in range(1, NUM_EPISODES + 1):
        ep_records = [r for r in all_records if r["episode"] == ep]
        total_r = sum(r["total_mp_reward"] for r in ep_records)
        per_agent = {a: sum(r[f"{a}_mp_reward"] for r in ep_records) for a in ["1", "2", "5", "6"]}
        print(f"  Ep {ep}: total={total_r:.1f}, per_agent={per_agent}")
        rewards_per_ep.append(total_r)
    print(f"\n  Mean +/- Std: {np.mean(rewards_per_ep):.1f} +/- {np.std(rewards_per_ep):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()