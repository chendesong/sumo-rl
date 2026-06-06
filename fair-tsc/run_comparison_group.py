"""Run formal comparison baselines in three server-friendly groups.

Groups:
  1 / rule      fixed_time, max_pressure
  2 / local     ma2c, fairsignal
  3 / graph     colight, sociallight

All methods use the current config.py environment: same route, seed,
delta-time, pedestrian reward mode, and IPPO/UE reference for the
unified sacrifice-gap Theil metric.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

import config as C
from comparison_artifacts import enrich_result, write_last_n_green_split_summary
from evaluate import attach_tripinfo_metrics, load_shared_ue_critic, make_tripinfo_sumo_cmd
from sumo_env import FairTSCEnv


GROUPS = {
    "1": ["fixed_time", "max_pressure"],
    "rule": ["fixed_time", "max_pressure"],
    "2": ["ma2c", "fairsignal"],
    "local": ["ma2c", "fairsignal"],
    "3": ["colight", "sociallight"],
    "graph": ["colight", "sociallight"],
}

METHOD_MODULES = {
    "fixed_time": "baselines.fixed_time",
    "max_pressure": "baselines.max_pressure",
    "ma2c": "baselines.ma2c",
    "fairsignal": "baselines.fairsignal",
    "colight": "baselines.colight",
    "sociallight": "baselines.sociallight",
}


def _jsonable_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return ""
    return value


def _write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable_value(v) for k, v in row.items()})


def _mean_result(results: Iterable[Dict]) -> Dict:
    results = list(results)
    if not results:
        return {}
    out: Dict = {}
    keys = sorted({k for r in results for k in r if not str(k).startswith("_")})
    for key in keys:
        vals = []
        for r in results:
            try:
                vals.append(float(r[key]))
            except (KeyError, TypeError, ValueError):
                pass
        if vals:
            out[key] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
        elif key in results[-1]:
            out[key] = results[-1][key]
    return out


def _train_last_n_summary(artifact_dir: str, method: str, last_n: int) -> Dict:
    path = os.path.join(artifact_dir, f"{method}_train_log.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    tail = rows[-int(last_n):]

    def mean_col(name: str) -> float:
        vals = []
        for row in tail:
            try:
                vals.append(float(row.get(name, "")))
            except ValueError:
                pass
        return float(np.mean(vals)) if vals else 0.0

    return {
        "train_last_n": int(len(tail)),
        "train_last_reward_mean": mean_col("reward_mean"),
        "train_last_reward_std_mean": mean_col("reward_std"),
        "train_last_theil_intra": mean_col("theil_intra"),
        "train_last_max_phase_interval": mean_col("max_phase_interval"),
    }


def _default_out_dir(group: str, seed: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M")
    return os.path.join(
        C.BASE_DIR,
        "outputs",
        "comparison_formal",
        f"group_{group}_s{seed}_{stamp}",
    )


def _load_v_ue():
    if not C.UE_CKPT:
        raise ValueError(
            "Formal comparison requires FAIR_TSC_UE_CKPT to point at the shared IPPO/UE reference checkpoint."
        )
    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
    )
    try:
        return load_shared_ue_critic(ckpt_path=C.UE_CKPT, env=env)
    finally:
        env.close()


def _run_rule_method(method: str, v_ue, out_dir: str, group: str, seed: int, eval_episodes: int) -> Dict:
    module = importlib.import_module(METHOD_MODULES[method])
    results = []
    artifact_dir = os.path.join(out_dir, method)
    os.makedirs(artifact_dir, exist_ok=True)
    for ep in range(1, eval_episodes + 1):
        tripinfo = os.path.join(artifact_dir, f"tripinfo_eval_ep{ep:03d}.xml")
        result = module.main(
            v_ue=v_ue,
            additional_sumo_cmd=make_tripinfo_sumo_cmd(tripinfo),
            artifact_dir=artifact_dir,
            episode=ep,
            stage="eval",
            seed=seed + ep - 1,
        )
        result = attach_tripinfo_metrics(result, tripinfo, horizon_s=C.NUM_SECONDS)
        results.append(result)
    result = _mean_result(results)
    result["eval_episodes"] = int(eval_episodes)
    result["green_split_last_n_csv"] = write_last_n_green_split_summary(
        artifact_dir, method, last_n=eval_episodes, stage="eval"
    )
    return enrich_result(result, method=method, group=group, seed=seed, episodes=0, artifact_dir=artifact_dir)


def _run_learned_method(method: str, v_ue, out_dir: str, group: str, seed: int, episodes: int, last_n: int) -> Dict:
    module = importlib.import_module(METHOD_MODULES[method])
    artifact_dir = os.path.join(out_dir, method)
    os.makedirs(artifact_dir, exist_ok=True)
    tripinfo = os.path.join(artifact_dir, "tripinfo_final_eval.xml")
    result = module.main(
        v_ue=v_ue,
        additional_sumo_cmd=make_tripinfo_sumo_cmd(tripinfo),
        artifact_dir=artifact_dir,
        num_episodes=episodes,
        seed=seed,
    )
    result = attach_tripinfo_metrics(result, tripinfo, horizon_s=C.NUM_SECONDS)
    result.update(_train_last_n_summary(artifact_dir, method, last_n=last_n))
    result["green_split_last_n_csv"] = write_last_n_green_split_summary(
        artifact_dir, method, last_n=last_n, stage="train"
    )
    return enrich_result(result, method=method, group=group, seed=seed, episodes=episodes, artifact_dir=artifact_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=sorted(GROUPS.keys()))
    parser.add_argument("--methods", default="", help="Comma-separated override, e.g. ma2c,fairsignal")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("FAIR_TSC_COMPARISON_EPISODES", "300")))
    parser.add_argument("--eval-episodes", type=int, default=int(os.environ.get("FAIR_TSC_COMPARISON_EVAL_EPISODES", "30")))
    parser.add_argument("--last-n", type=int, default=int(os.environ.get("FAIR_TSC_COMPARISON_LAST_N", "30")))
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or GROUPS[args.group]
    unknown = [m for m in methods if m not in METHOD_MODULES]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    out_dir = args.out_dir or _default_out_dir(args.group, args.seed)
    os.makedirs(out_dir, exist_ok=True)
    print("===== formal comparison group =====")
    print(f"group={args.group} methods={methods}")
    print(f"seed={args.seed} episodes={args.episodes} eval_episodes={args.eval_episodes}")
    print(f"route={C.ROUTE_FILE}")
    print(f"net={C.NET_FILE}")
    print(f"delta={C.DELTA_TIME} ped_mode={C.PED_REWARD_MODE} omega_p={C.OMEGA_P}")
    if C.PED_REWARD_MODE != "queue" or abs(float(C.OMEGA_P) - 1.0) > 1e-9:
        print("[warning] formal setting should use FAIR_TSC_PED_REWARD_MODE=queue and FAIR_TSC_OMEGA_P=1.0")
    print(f"out_dir={out_dir}")
    print("===== loading shared IPPO/UE reference =====")
    v_ue = _load_v_ue()

    rows = []
    for method in methods:
        print(f"\n===== {method} =====")
        if method in {"fixed_time", "max_pressure"}:
            result = _run_rule_method(method, v_ue, out_dir, args.group, args.seed, args.eval_episodes)
        else:
            result = _run_learned_method(method, v_ue, out_dir, args.group, args.seed, args.episodes, args.last_n)
        rows.append(result)
        print(
            f"[{method}] eff={result.get('efficiency')} "
            f"Tinter={result.get('theil_inter')} Tintra={result.get('theil_intra')} "
            f"Cfair={result.get('Cfair_composite')}"
        )

    summary_path = os.path.join(out_dir, "comparison_group_summary.csv")
    _write_csv(summary_path, rows)
    print(f"\n[run_comparison_group] wrote {summary_path}")


if __name__ == "__main__":
    main()
