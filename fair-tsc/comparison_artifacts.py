"""Shared artifacts for formal baseline comparison runs.

This module keeps the comparison outputs aligned across rule-based,
MARL, and fairness baselines:

* final metrics are enriched with the same composite Theil cost;
* every method can write episode-level green split rows;
* learned baselines can write a compact training log.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import config as C


PHASE_LABELS = {
    0: "NS vehicle",
    1: "EW vehicle",
    2: "pedestrian",
}

GREEN_SPLIT_FIELDS = [
    "method",
    "seed",
    "stage",
    "episode",
    "scope",
    "agent_id",
    "phase",
    "phase_label",
    "green_seconds",
    "green_split",
]

TRAIN_LOG_FIELDS = [
    "method",
    "seed",
    "episode",
    "reward_mean",
    "reward_std",
    "theil_intra",
    "max_phase_interval",
    "phase_service_mean_interval",
]


def _phase_label(phase: int) -> str:
    return PHASE_LABELS.get(int(phase), f"phase {phase}")


def append_csv_rows(path: str, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def green_split_rows_from_phase_log(
    phase_start_log: Dict[str, Dict[int, List[float]]],
    horizon_s: float,
    method: str,
    episode: int,
    stage: str,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Convert per-agent phase start logs into green-split rows.

    Durations are measured between consecutive green activation starts.
    Yellow time is attributed to the preceding selected green phase, which
    matches how the decision policy allocates service phases.
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
                    "method": method,
                    "seed": "" if seed is None else int(seed),
                    "stage": stage,
                    "episode": int(episode),
                    "scope": "intersection",
                    "agent_id": agent_id,
                    "phase": int(phase),
                    "phase_label": _phase_label(phase),
                    "green_seconds": seconds,
                    "green_split": seconds / denom,
                }
            )

    network_total = sum(network_seconds.values())
    network_denom = max(network_total, 1e-9)
    for phase in sorted(network_seconds):
        seconds = float(network_seconds[phase])
        rows.append(
            {
                "method": method,
                "seed": "" if seed is None else int(seed),
                "stage": stage,
                "episode": int(episode),
                "scope": "network",
                "agent_id": "network",
                "phase": int(phase),
                "phase_label": _phase_label(phase),
                "green_seconds": seconds,
                "green_split": seconds / network_denom,
            }
        )

    return rows


def write_green_split_episode(
    artifact_dir: Optional[str],
    method: str,
    env,
    episode: int,
    stage: str = "train",
    seed: Optional[int] = None,
) -> None:
    if not artifact_dir:
        return
    rows = green_split_rows_from_phase_log(
        env.get_phase_start_log(),
        horizon_s=float(getattr(env, "num_seconds", C.NUM_SECONDS)),
        method=method,
        episode=episode,
        stage=stage,
        seed=seed,
    )
    append_csv_rows(
        os.path.join(artifact_dir, f"{method}_green_split_episode.csv"),
        rows,
        GREEN_SPLIT_FIELDS,
    )


def write_last_n_green_split_summary(
    artifact_dir: Optional[str],
    method: str,
    last_n: int = 30,
    stage: str = "train",
) -> str:
    """Average green split over the last N episodes for plotting."""
    if not artifact_dir:
        return ""
    source = os.path.join(artifact_dir, f"{method}_green_split_episode.csv")
    if not os.path.exists(source):
        return ""
    with open(source, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("stage") == stage]
    if not rows:
        return ""
    episodes = sorted({int(float(r["episode"])) for r in rows if r.get("episode") not in (None, "")})
    keep = set(episodes[-int(last_n):])
    grouped: Dict[Tuple[str, str, int, str], Dict[str, List[float]]] = {}
    for row in rows:
        try:
            ep = int(float(row["episode"]))
            phase = int(float(row["phase"]))
            seconds = float(row["green_seconds"])
            split = float(row["green_split"])
        except (KeyError, TypeError, ValueError):
            continue
        if ep not in keep:
            continue
        key = (row.get("scope", ""), row.get("agent_id", ""), phase, row.get("phase_label", _phase_label(phase)))
        grouped.setdefault(key, {"seconds": [], "split": []})
        grouped[key]["seconds"].append(seconds)
        grouped[key]["split"].append(split)

    out_rows = []
    for (scope, agent_id, phase, phase_label), values in sorted(grouped.items()):
        out_rows.append(
            {
                "method": method,
                "stage": stage,
                "last_n": int(len(keep)),
                "scope": scope,
                "agent_id": agent_id,
                "phase": phase,
                "phase_label": phase_label,
                "green_seconds_mean": float(np.mean(values["seconds"])) if values["seconds"] else 0.0,
                "green_split_mean": float(np.mean(values["split"])) if values["split"] else 0.0,
                "green_split_std": float(np.std(values["split"])) if values["split"] else 0.0,
            }
        )
    out_path = os.path.join(artifact_dir, f"{method}_green_split_last{len(keep)}.csv")
    if os.path.exists(out_path):
        os.remove(out_path)
    append_csv_rows(
        out_path,
        out_rows,
        [
            "method",
            "stage",
            "last_n",
            "scope",
            "agent_id",
            "phase",
            "phase_label",
            "green_seconds_mean",
            "green_split_mean",
            "green_split_std",
        ],
    )
    return out_path


def write_train_episode(
    artifact_dir: Optional[str],
    method: str,
    env,
    episode: int,
    reward_by_agent: Dict[str, float],
    seed: Optional[int] = None,
) -> None:
    if not artifact_dir:
        return
    rewards = np.asarray(list(reward_by_agent.values()), dtype=np.float64)
    phase = env.get_phase_service_summary()
    row = {
        "method": method,
        "seed": "" if seed is None else int(seed),
        "episode": int(episode),
        "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "reward_std": float(rewards.std()) if rewards.size else 0.0,
        "theil_intra": float(phase.get("theil_intra", 0.0) or 0.0),
        "max_phase_interval": float(phase.get("max_phase_interval", 0.0) or 0.0),
        "phase_service_mean_interval": float(phase.get("phase_service_mean_interval", 0.0) or 0.0),
    }
    append_csv_rows(
        os.path.join(artifact_dir, f"{method}_train_log.csv"),
        [row],
        TRAIN_LOG_FIELDS,
    )


def composite_theil(result: Dict, alpha: Optional[float] = None,
                    t_inter_0: Optional[float] = None,
                    t_intra_0: Optional[float] = None) -> Dict[str, float]:
    alpha = C.FAIR_ALPHA if alpha is None else float(alpha)
    t_inter_0 = C.T_INTER_0 if t_inter_0 is None else float(t_inter_0)
    t_intra_0 = C.T_INTRA_0 if t_intra_0 is None else float(t_intra_0)
    t_inter = float(result.get("theil_inter", result.get("theil_raw", 0.0)) or 0.0)
    t_intra = float(result.get("theil_intra", 0.0) or 0.0)
    raw = alpha * t_inter + (1.0 - alpha) * t_intra
    norm = (
        alpha * t_inter / max(t_inter_0, C.THEIL_EPS)
        + (1.0 - alpha) * t_intra / max(t_intra_0, C.THEIL_EPS)
    )
    return {
        "Tdual_raw": float(raw),
        "Cfair_composite": float(norm),
        "fair_alpha": float(alpha),
        "T_inter_0": float(t_inter_0),
        "T_intra_0": float(t_intra_0),
    }


def enrich_result(result: Dict, method: str, group: str, seed: int,
                  episodes: int, artifact_dir: str) -> Dict:
    out = {k: v for k, v in result.items() if not str(k).startswith("_")}
    out.update(composite_theil(out))
    out.update(
        {
            "method": method,
            "group": group,
            "seed": int(seed),
            "episodes": int(episodes),
            "artifact_dir": os.path.abspath(artifact_dir),
            "ped_reward_mode": C.PED_REWARD_MODE,
            "omega_p": float(C.OMEGA_P),
            "omega_ped_wait": float(C.OMEGA_PED_WAIT),
            "delta_time": int(C.DELTA_TIME),
            "route_file": C.ROUTE_FILE,
            "net_file": C.NET_FILE,
        }
    )
    return out
