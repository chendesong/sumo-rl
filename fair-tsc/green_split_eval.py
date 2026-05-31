"""Evaluate a trained Fair-TSC/MAPPO checkpoint and report green splits.

The script is intentionally separate from training and from
``run_comparison.py``.  It runs one deterministic policy rollout, writes
green-split CSVs, parses SUMO tripinfo for vehicle travel time and
throughput, and saves lightweight plots for paper figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from evaluate import MetricsCollector, parse_tripinfo
from risk_aware_sim import _jsonable, _load_actor
from sumo_env import FairTSCEnv


PHASE_LABELS = {
    0: "NS vehicle",
    1: "EW vehicle",
    2: "pedestrian",
}

PHASE_COLORS = {
    0: "#4C78A8",
    1: "#54A24B",
    2: "#F58518",
}


def _route_for_demand(demand: str) -> str:
    return os.path.join(C.BASE_DIR, "nets", "4x4grid", f"4x4_{demand}.rou.xml")


def _phase_label(phase: int) -> str:
    return PHASE_LABELS.get(int(phase), f"phase {phase}")


def _phase_color(phase: int) -> str:
    return PHASE_COLORS.get(int(phase), "#9D9DA1")


def green_split_from_phase_starts(
    phase_start_log: Dict[str, Dict[int, List[float]]],
    horizon_s: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Convert phase activation starts into per-intersection/network splits.

    Durations are measured between consecutive green activation starts.
    SUMO inserts yellow internally, so the split is best interpreted as
    service-cycle allocation associated with each controlled green phase.
    """
    rows: List[Dict] = []
    network_seconds: Dict[int, float] = {}

    for agent_id, phase_map in sorted(phase_start_log.items()):
        events: List[Tuple[float, int]] = []
        for phase, starts in phase_map.items():
            for start in starts:
                start_f = float(start)
                if 0.0 <= start_f <= horizon_s:
                    events.append((start_f, int(phase)))
        events.sort(key=lambda x: (x[0], x[1]))

        phase_seconds = {int(phase): 0.0 for phase in phase_map.keys()}
        for idx, (start, phase) in enumerate(events):
            end = events[idx + 1][0] if idx + 1 < len(events) else horizon_s
            phase_seconds[phase] = phase_seconds.get(phase, 0.0) + max(0.0, end - start)

        total = sum(phase_seconds.values())
        denom = max(total, 1e-9)
        for phase in sorted(phase_seconds):
            seconds = float(phase_seconds[phase])
            network_seconds[phase] = network_seconds.get(phase, 0.0) + seconds
            rows.append(
                {
                    "agent_id": agent_id,
                    "phase": int(phase),
                    "phase_label": _phase_label(phase),
                    "green_seconds": seconds,
                    "green_split": seconds / denom,
                }
            )

    network_rows: List[Dict] = []
    total_network = sum(network_seconds.values())
    denom = max(total_network, 1e-9)
    for phase in sorted(network_seconds):
        seconds = float(network_seconds[phase])
        network_rows.append(
            {
                "phase": int(phase),
                "phase_label": _phase_label(phase),
                "green_seconds": seconds,
                "green_split": seconds / denom,
            }
        )
    return rows, network_rows


def write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(out_dir: str, per_agent_rows: List[Dict], network_rows: List[Dict]) -> Dict[str, str]:
    """Save green split pie and heatmap.  Plotting is optional at runtime."""
    paths: Dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting dependency is optional
        print(f"[green_split_eval] matplotlib unavailable, skipping plots: {exc}")
        return paths

    if network_rows:
        labels = [r["phase_label"] for r in network_rows]
        values = [r["green_seconds"] for r in network_rows]
        colors = [_phase_color(int(r["phase"])) for r in network_rows]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
        ax.set_title("Network Green Split")
        fig.tight_layout()
        path = os.path.join(out_dir, "green_split_network_pie.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["network_pie"] = path

    if per_agent_rows:
        agents = sorted({r["agent_id"] for r in per_agent_rows})
        phases = sorted({int(r["phase"]) for r in per_agent_rows})
        mat = np.zeros((len(agents), len(phases)), dtype=np.float64)
        a_idx = {a: i for i, a in enumerate(agents)}
        p_idx = {p: i for i, p in enumerate(phases)}
        for r in per_agent_rows:
            mat[a_idx[r["agent_id"]], p_idx[int(r["phase"])]] = float(r["green_split"])

        fig_w = max(7.0, 0.42 * len(agents))
        fig, ax = plt.subplots(figsize=(fig_w, 4.6))
        im = ax.imshow(mat.T, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(float(mat.max()), 1e-6))
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=90, fontsize=8)
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels([_phase_label(p) for p in phases])
        ax.set_xlabel("Intersection")
        ax.set_title("Green Split by Intersection")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("share")
        fig.tight_layout()
        path = os.path.join(out_dir, "green_split_heatmap.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["heatmap"] = path

    return paths


def run(args: argparse.Namespace) -> Dict:
    import torch

    route_file = args.route_file or _route_for_demand(args.demand)
    if not os.path.exists(route_file):
        raise FileNotFoundError(route_file)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(C.BASE_DIR, "outputs", f"green_split_{args.demand}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    tripinfo_path = os.path.join(out_dir, "tripinfo.xml")
    additional = f"--tripinfo-output {tripinfo_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=route_file,
        out_csv_name=None,
        num_seconds=args.num_seconds,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
        additional_sumo_cmd=additional,
    )

    try:
        obs = env.reset(seed=args.seed)
        actor = _load_actor(env, args.ckpt, actor_key=args.actor_key, device=device)
        coll = MetricsCollector()

        done = False
        while not done:
            local_b = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
            idx_b = torch.arange(env.num_agents, device=device)
            with torch.no_grad():
                action, _logp = actor.act(local_b, idx_b, deterministic=True)
            action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}

            next_obs, reward, _cp, _cs, done, info = env.step(action_dict)
            rewards_array = np.array([reward[a] for a in env.agent_ids], dtype=np.float32)
            coll.add(info, mean_reward=float(rewards_array.mean()) if rewards_array.size else 0.0)
            obs = next_obs

        phase_log = env.get_phase_start_log()
        env_metrics = coll.finalize(env)
    finally:
        env.close()

    per_agent_rows, network_rows = green_split_from_phase_starts(phase_log, horizon_s=args.num_seconds)
    per_agent_csv = os.path.join(out_dir, "green_split_by_intersection.csv")
    network_csv = os.path.join(out_dir, "green_split_network.csv")
    write_csv(per_agent_csv, per_agent_rows)
    write_csv(network_csv, network_rows)

    efficiency = parse_tripinfo(tripinfo_path, horizon_s=args.num_seconds)
    efficiency.update(
        {
            "mean_system_total_waiting_time_s": float(np.mean(env_metrics.get("system_total_waiting_time_series", [0.0]))),
            "mean_ped_waiting_time_s": float(np.mean(env_metrics.get("agents_total_ped_waiting_time_series", [0.0]))),
            "mean_expected_ped_violations": float(np.mean(env_metrics.get("agents_total_expected_violations_series", [0.0]))),
        }
    )
    for key in (
        "departed_total",
        "arrived_total",
        "loaded_total",
        "teleported_total",
        "active_vehicle_count_end",
        "pending_vehicle_count_end",
        "min_expected_number_end",
        "total_vehicle_demand",
        "unfinished_vehicle_demand",
        "completion_rate_departed",
        "completion_rate_demand",
    ):
        if key in env_metrics:
            efficiency[key] = env_metrics[key]

    plot_paths = save_plots(out_dir, per_agent_rows, network_rows)
    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "demand": args.demand,
        "route_file": os.path.abspath(route_file),
        "seed": args.seed,
        "num_seconds": args.num_seconds,
        "green_split_by_intersection_csv": per_agent_csv,
        "green_split_network_csv": network_csv,
        "tripinfo_xml": tripinfo_path,
        "plots": plot_paths,
        "efficiency": efficiency,
        "network_green_split": network_rows,
    }

    summary_path = os.path.join(out_dir, "green_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)

    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    print(f"[green_split_eval] summary={summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint containing actor_marl")
    parser.add_argument("--demand", choices=sorted(C.DEMAND_LEVELS), default=C.DEMAND_LEVEL)
    parser.add_argument("--route-file", default=None)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--actor-key", default="actor_marl")
    parser.add_argument("--num-seconds", type=int, default=C.NUM_SECONDS)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
