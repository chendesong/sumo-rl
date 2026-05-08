"""
Test script: verify pedestrian data retrieval from SUMO via traci.
Runs one 3600s episode, prints per-crossing pedestrian queue and waiting time
every 100 seconds.

Usage: run in PyCharm or from conda env:
  python experiments/test_pedestrian.py
"""

import os
import sys
import math

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib

BASE_DIR = "C:/Users/ucemdc3/PycharmProjects/sumo-rl"
NET_FILE = os.path.join(BASE_DIR, "nets/2x2grid/01.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, "nets/2x2grid/02.rou.xml")

# ── Crossing IDs for each intersection ──
# From net.xml: each intersection has 4 crossings (c0, c1, c2, c3)
# State string positions 16,17,18,19 correspond to c0,c1,c2,c3
TLS_IDS = ["1", "2", "5", "6"]
CROSSING_IDS = {
    ts: [f":{ts}_c0", f":{ts}_c1", f":{ts}_c2", f":{ts}_c3"]
    for ts in TLS_IDS
}

# ── Jaywalking model parameters (paper Eq. 8) ──
ALPHA_P = 0.1  # sensitivity
BETA_P = 60.0  # half-violation threshold (seconds)


def jaywalking_probability(wait_time):
    """Paper Eq. (8): P_viol = sigmoid(alpha * (wait - beta))"""
    return 1.0 / (1.0 + math.exp(-ALPHA_P * (wait_time - BETA_P)))


def get_pedestrian_data(sumo, ts_id):
    """
    For a given intersection, get per-crossing pedestrian data.
    Returns dict: crossing_id -> {queue, max_wait, total_wait, p_viol, expected_violations}

    Logic: a pedestrian is "waiting at crossing X" if:
      - they are on a walkingArea (edge starts with ":" and contains "_w")
      - their next edge (getNextEdge) is the crossing edge
      - their speed is ~0 (waiting)
    """
    crossing_ids = CROSSING_IDS[ts_id]
    data = {}

    for c_id in crossing_ids:
        data[c_id] = {"queue": 0, "max_wait": 0.0, "total_wait": 0.0}

    # Iterate all persons in simulation
    for pid in sumo.person.getIDList():
        next_edge = sumo.person.getNextEdge(pid)
        if next_edge in crossing_ids:
            wait = sumo.person.getWaitingTime(pid)
            speed = sumo.person.getSpeed(pid)
            # Count as queued if waiting (speed < 0.1 m/s)
            if speed < 0.1 and wait > 0:
                data[next_edge]["queue"] += 1
                data[next_edge]["total_wait"] += wait
                data[next_edge]["max_wait"] = max(data[next_edge]["max_wait"], wait)

    # Add jaywalking metrics
    for c_id in crossing_ids:
        d = data[c_id]
        d["p_viol"] = jaywalking_probability(d["max_wait"])
        d["expected_violations"] = d["queue"] * jaywalking_probability(d["max_wait"]) if d["queue"] > 0 else 0.0

    return data


def main():
    # Start SUMO
    sumo_cmd = [
        sumolib.checkBinary("sumo"),
        "-n", NET_FILE,
        "-r", ROUTE_FILE,
        "--no-warnings",
        "--random",
    ]
    traci.start(sumo_cmd)

    print("=" * 100)
    print(
        f"{'step':>6} | {'TLS':>3} | {'crossing':>8} | {'queue':>5} | {'max_wait':>9} | {'total_wait':>10} | {'P_viol':>7} | {'E[viol]':>8}")
    print("=" * 100)

    step = 0
    while step <= 3600:
        traci.simulationStep()
        step = traci.simulation.getTime()

        # Print every 100 seconds
        if step % 100 == 0 and step > 0:
            total_peds = len(traci.person.getIDList())
            print(f"\n--- Step {step:.0f} | Total persons in sim: {total_peds} ---")

            for ts_id in TLS_IDS:
                ped_data = get_pedestrian_data(traci, ts_id)
                for c_id, d in ped_data.items():
                    if d["queue"] > 0 or step % 500 == 0:  # print non-zero or every 500s
                        print(
                            f"{step:6.0f} | {ts_id:>3} | {c_id:>8} | {d['queue']:5d} | {d['max_wait']:9.1f} | {d['total_wait']:10.1f} | {d['p_viol']:7.4f} | {d['expected_violations']:8.3f}")

            # Summary
            all_queued = sum(
                get_pedestrian_data(traci, ts)[c]["queue"]
                for ts in TLS_IDS
                for c in CROSSING_IDS[ts]
            )
            all_violations = sum(
                get_pedestrian_data(traci, ts)[c]["expected_violations"]
                for ts in TLS_IDS
                for c in CROSSING_IDS[ts]
            )
            print(f"       TOTAL queued pedestrians: {all_queued} | Expected violations: {all_violations:.2f}")

    traci.close()
    print("\n" + "=" * 100)
    print("Test complete. If you saw pedestrian queues > 0 at crossings, the data pipeline works.")
    print("If all queues are 0, check your .rou.xml has personFlow definitions.")


if __name__ == "__main__":
    main()