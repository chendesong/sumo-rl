"""COMA-style counterfactual diagnostic probe.

This script is a post-hoc probe, not a training algorithm.  It estimates
whether local flow/queue is aligned with each intersection's one-step
counterfactual contribution to short-horizon network efficiency:

    A_i = Q(s, a) - sum_a' pi_i(a'|o_i) Q(s, a_-i, a')

Q is estimated by perturbing one agent's current action, then letting all
agents continue with the loaded policy.  The return is locked to raw
efficiency only: negative system total waiting time.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config as C
from networks import SharedActor
from sumo_env import FairTSCEnv


@dataclass
class BranchSnapshot:
    state_path: str
    wrapper_state: dict
    rng_state: dict


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _copy_np_dict(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32).copy() for k, v in d.items()}


def _info_probe(info: dict) -> dict:
    if isinstance(info, dict) and info:
        if all(isinstance(v, dict) for v in info.values()):
            return next(iter(info.values()))
        return info
    return {}


def efficiency_from_info(info: dict) -> float:
    probe = _info_probe(info)
    if "system_total_waiting_time" in probe:
        return -float(probe.get("system_total_waiting_time", 0.0))
    if "agents_total_reward" in probe:
        return float(probe.get("agents_total_reward", 0.0))
    return 0.0


def snapshot_wrapper_state(env: FairTSCEnv) -> dict:
    inner = env._walk_to_sumo_env()
    ts_state = {}
    for agent, ts in inner.traffic_signals.items():
        ts_state[agent] = {
            "green_phase": int(getattr(ts, "green_phase", 0)),
            "is_yellow": bool(getattr(ts, "is_yellow", False)),
            "time_since_last_phase_change": float(getattr(ts, "time_since_last_phase_change", 0.0)),
            "next_action_time": float(getattr(ts, "next_action_time", 0.0)),
            "last_ts_waiting_time": float(getattr(ts, "last_ts_waiting_time", 0.0)),
            "last_ped_waiting_time": float(getattr(ts, "last_ped_waiting_time", 0.0)),
            "last_reward": getattr(ts, "last_reward", None),
        }
    return {
        "observations": _copy_np_dict(getattr(inner, "observations", {})),
        "rewards": dict(getattr(inner, "rewards", {})),
        "metrics": list(getattr(inner, "metrics", [])),
        "vehicles": dict(getattr(inner, "vehicles", {})),
        "num_arrived_vehicles": int(getattr(inner, "num_arrived_vehicles", 0)),
        "num_departed_vehicles": int(getattr(inner, "num_departed_vehicles", 0)),
        "num_teleported_vehicles": int(getattr(inner, "num_teleported_vehicles", 0)),
        "phase_start_log": env.get_phase_start_log(),
        "last_phase": dict(env._last_phase),
        "phase_count": dict(env._phase_count),
        "pending_phase_start": dict(env._pending_phase_start),
        "traffic_signals": ts_state,
    }


def restore_wrapper_state(env: FairTSCEnv, state: dict):
    inner = env._walk_to_sumo_env()
    inner.observations = _copy_np_dict(state.get("observations", {}))
    inner.rewards = dict(state.get("rewards", {}))
    inner.metrics = list(state.get("metrics", []))
    inner.vehicles = dict(state.get("vehicles", {}))
    inner.num_arrived_vehicles = int(state.get("num_arrived_vehicles", 0))
    inner.num_departed_vehicles = int(state.get("num_departed_vehicles", 0))
    inner.num_teleported_vehicles = int(state.get("num_teleported_vehicles", 0))
    env._phase_start_log = {
        a: {int(p): list(vals) for p, vals in phase_map.items()}
        for a, phase_map in state.get("phase_start_log", {}).items()
    }
    env._last_phase = dict(state.get("last_phase", {}))
    env._phase_count = dict(state.get("phase_count", {}))
    env._pending_phase_start = dict(state.get("pending_phase_start", {}))
    for agent, values in state.get("traffic_signals", {}).items():
        ts = inner.traffic_signals.get(agent)
        if ts is None:
            continue
        for key, value in values.items():
            setattr(ts, key, value)


def save_branch_snapshot(env: FairTSCEnv, path: str, rng_state: dict) -> BranchSnapshot:
    inner = env._walk_to_sumo_env()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    inner.sumo.simulation.saveState(path)
    return BranchSnapshot(
        state_path=path,
        wrapper_state=snapshot_wrapper_state(env),
        rng_state=rng_state,
    )


def load_branch_snapshot(env: FairTSCEnv, snap: BranchSnapshot):
    inner = env._walk_to_sumo_env()
    inner.sumo.simulation.loadState(snap.state_path)
    restore_wrapper_state(env, snap.wrapper_state)
    restore_rng_state(snap.rng_state)


def make_env(additional_sumo_cmd: Optional[str] = None) -> FairTSCEnv:
    return FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
        use_gui=False,
        additional_sumo_cmd=additional_sumo_cmd,
    )


def find_default_ckpt() -> str:
    patterns = [
        os.path.join(C.BASE_DIR, "checkpoints", f"*4x4_{C.DEMAND_LEVEL}*", "final.pt"),
        os.path.join(C.BASE_DIR, "checkpoints", f"*4x4_{C.DEMAND_LEVEL}*", "ep_*.pt"),
        os.path.join(C.BASE_DIR, "checkpoints", "*4x4*", "final.pt"),
        os.path.join(C.BASE_DIR, "checkpoints", "*4x4*", "ep_*.pt"),
        os.path.join(C.BASE_DIR, "checkpoints", "*", "final.pt"),
        os.path.join(C.BASE_DIR, "checkpoints", "*", "ep_*.pt"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = [p for p in files if os.path.exists(p)]
    if not files:
        raise FileNotFoundError("No checkpoint found. Pass --ckpt explicitly.")
    return max(files, key=os.path.getmtime)


def load_actor(ckpt_path: str, env: FairTSCEnv, device: torch.device, actor_key: str) -> SharedActor:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if actor_key not in ckpt:
        raise KeyError(f"Checkpoint missing {actor_key!r}. Keys: {sorted(ckpt.keys())}")
    actor = SharedActor(
        local_obs_dim=env.local_obs_dim,
        num_agents=env.num_agents,
        action_dim=env.action_dim,
        hidden=C.ACTOR_HIDDEN,
    ).to(device)
    actor.load_state_dict(ckpt[actor_key])
    actor.eval()
    return actor


@torch.no_grad()
def policy_actions_and_probs(
    actor: SharedActor,
    obs: Dict[str, np.ndarray],
    agent_ids: Sequence[str],
    device: torch.device,
    deterministic: bool,
) -> Tuple[Dict[str, int], np.ndarray]:
    local_b = torch.from_numpy(np.stack([obs[a] for a in agent_ids])).to(device)
    idx_b = torch.arange(len(agent_ids), device=device)
    dist = actor(local_b, idx_b)
    probs = dist.probs.detach().cpu().numpy()
    if deterministic:
        action = probs.argmax(axis=1)
    else:
        action_t = dist.sample()
        action = action_t.detach().cpu().numpy()
    action_dict = {a: int(action[i]) for i, a in enumerate(agent_ids)}
    return action_dict, probs


def collect_agent_load(env: FairTSCEnv) -> Dict[str, Dict[str, float]]:
    inner = env._walk_to_sumo_env()
    out = {}
    for agent in env.agent_ids:
        ts = inner.traffic_signals[agent]
        incoming = float(sum(inner.sumo.lane.getLastStepVehicleNumber(lane) for lane in ts.lanes))
        queue = float(sum(inner.sumo.lane.getLastStepHaltingNumber(lane) for lane in ts.lanes))
        outgoing = float(sum(inner.sumo.lane.getLastStepVehicleNumber(lane) for lane in ts.out_lanes))
        pressure = incoming - outgoing
        wait = 0.0
        for lane in ts.lanes:
            for veh in inner.sumo.lane.getLastStepVehicleIDs(lane):
                wait += float(inner.sumo.vehicle.getWaitingTime(veh))
        out[agent] = {
            "flow": incoming,
            "queue": queue,
            "outflow": outgoing,
            "pressure": pressure,
            "waiting": wait,
        }
    return out


def rollout_q_from_current_state(
    env: FairTSCEnv,
    actor: SharedActor,
    obs: Dict[str, np.ndarray],
    first_action: Dict[str, int],
    horizon_steps: int,
    gamma: float,
    device: torch.device,
    deterministic_continuation: bool,
) -> float:
    total = 0.0
    discount = 1.0
    current_obs = obs
    action_dict = first_action
    for h in range(max(1, horizon_steps)):
        next_obs, _rewards, _cp, _cs, done, info = env.step(action_dict)
        total += discount * efficiency_from_info(info)
        if done or not next_obs:
            break
        current_obs = next_obs
        action_dict, _ = policy_actions_and_probs(
            actor, current_obs, env.agent_ids, device, deterministic=deterministic_continuation
        )
        discount *= gamma
    return float(total)


def estimate_counterfactual_save_load(
    env: FairTSCEnv,
    actor: SharedActor,
    obs: Dict[str, np.ndarray],
    joint_action: Dict[str, int],
    probs: np.ndarray,
    snap: BranchSnapshot,
    horizon_steps: int,
    gamma: float,
    device: torch.device,
    deterministic_continuation: bool,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    action_dim = env.action_dim
    for i, agent in enumerate(env.agent_ids):
        q_values = []
        for action in range(action_dim):
            load_branch_snapshot(env, snap)
            perturbed = dict(joint_action)
            perturbed[agent] = int(action)
            q = rollout_q_from_current_state(
                env,
                actor,
                _copy_np_dict(obs),
                perturbed,
                horizon_steps,
                gamma,
                device,
                deterministic_continuation,
            )
            q_values.append(q)
        q_arr = np.asarray(q_values, dtype=np.float64)
        baseline = float(np.dot(probs[i, :action_dim], q_arr))
        actual_action = int(joint_action[agent])
        actual_q = float(q_arr[actual_action])
        out[agent] = {
            "q_values": q_arr,
            "q_baseline": baseline,
            "q_actual": actual_q,
            "coma_advantage": actual_q - baseline,
            "pi_actual": float(probs[i, actual_action]),
        }
    load_branch_snapshot(env, snap)
    return out


def replay_to_state(
    seed: int,
    action_history: Sequence[Dict[str, int]],
    actor: SharedActor,
    device: torch.device,
    deterministic_continuation: bool,
    first_action: Dict[str, int],
    horizon_steps: int,
    gamma: float,
) -> float:
    env = make_env()
    try:
        obs = env.reset(seed=seed)
        for action_dict in action_history:
            obs, _r, _cp, _cs, done, _info = env.step(dict(action_dict))
            if done:
                return 0.0
        return rollout_q_from_current_state(
            env,
            actor,
            obs,
            first_action,
            horizon_steps,
            gamma,
            device,
            deterministic_continuation,
        )
    finally:
        env.close()


def estimate_counterfactual_replay(
    env: FairTSCEnv,
    actor: SharedActor,
    seed: int,
    action_history: Sequence[Dict[str, int]],
    joint_action: Dict[str, int],
    probs: np.ndarray,
    horizon_steps: int,
    gamma: float,
    device: torch.device,
    deterministic_continuation: bool,
    rng_state: dict,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    action_dim = env.action_dim
    for i, agent in enumerate(env.agent_ids):
        q_values = []
        for action in range(action_dim):
            restore_rng_state(rng_state)
            perturbed = dict(joint_action)
            perturbed[agent] = int(action)
            q = replay_to_state(
                seed,
                action_history,
                actor,
                device,
                deterministic_continuation,
                perturbed,
                horizon_steps,
                gamma,
            )
            q_values.append(q)
        q_arr = np.asarray(q_values, dtype=np.float64)
        baseline = float(np.dot(probs[i, :action_dim], q_arr))
        actual_action = int(joint_action[agent])
        actual_q = float(q_arr[actual_action])
        out[agent] = {
            "q_values": q_arr,
            "q_baseline": baseline,
            "q_actual": actual_q,
            "coma_advantage": actual_q - baseline,
            "pi_actual": float(probs[i, actual_action]),
        }
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "seed",
        "sample_id",
        "decision_step",
        "sim_time",
        "agent",
        "action_actual",
        "pi_actual",
        "flow",
        "queue",
        "outflow",
        "pressure",
        "waiting",
        "q_actual",
        "q_baseline",
        "coma_advantage",
        "q_values_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_summary(path: str, rows: List[dict], args: argparse.Namespace, ckpt_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    metrics = {}
    if rows:
        adv = np.asarray([float(r["coma_advantage"]) for r in rows], dtype=np.float64)
        for key in ("flow", "queue", "pressure", "waiting"):
            x = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
            metrics[f"pearson_{key}_adv"] = pearson(x, adv)
            metrics[f"spearman_{key}_adv"] = spearman(x, adv)
        metrics["adv_mean"] = float(np.mean(adv))
        metrics["adv_std"] = float(np.std(adv))
        metrics["num_points"] = len(rows)
    summary = {
        "ckpt": ckpt_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def make_plot(path: str, rows: List[dict], summary: dict):
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flow = np.asarray([float(r["flow"]) for r in rows], dtype=np.float64)
    adv = np.asarray([float(r["coma_advantage"]) for r in rows], dtype=np.float64)
    colors = np.asarray([float(r["queue"]) for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    sc = ax.scatter(flow, adv, c=colors, cmap="viridis", s=24, alpha=0.8, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Incoming vehicle count at intersection")
    ax.set_ylabel("COMA-style counterfactual advantage")
    rho = summary.get("metrics", {}).get("spearman_flow_adv", float("nan"))
    r = summary.get("metrics", {}).get("pearson_flow_adv", float("nan"))
    ax.set_title(f"Flow vs counterfactual advantage (Pearson={r:.3f}, Spearman={rho:.3f})")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Queue count")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_seed(args: argparse.Namespace, seed: int, ckpt_path: str, device: torch.device, rows: List[dict]):
    set_seed(seed)
    env = make_env()
    state_path = os.path.join(args.out_dir, "states", f"coma_state_seed{seed}.xml")
    action_history: List[Dict[str, int]] = []
    try:
        obs = env.reset(seed=seed)
        actor = load_actor(ckpt_path, env, device, args.actor_key)
        sample_id = 0
        decision_step = 0
        done = False
        save_load_available: Optional[bool] = None

        while not done and decision_step < args.max_decision_steps:
            joint_action, probs = policy_actions_and_probs(
                actor, obs, env.agent_ids, device, deterministic=args.deterministic_policy
            )
            should_probe = (
                decision_step >= args.burn_in_steps
                and (decision_step - args.burn_in_steps) % args.sample_every == 0
                and sample_id < args.max_samples
            )

            rng_after_action = capture_rng_state()
            if should_probe:
                agent_load = collect_agent_load(env)
                mode = args.state_mode
                cf = None
                if mode in ("auto", "save-load"):
                    try:
                        snap = save_branch_snapshot(env, state_path, rng_after_action)
                        cf = estimate_counterfactual_save_load(
                            env,
                            actor,
                            _copy_np_dict(obs),
                            joint_action,
                            probs,
                            snap,
                            args.horizon_steps,
                            args.gamma,
                            device,
                            args.deterministic_continuation,
                        )
                        save_load_available = True
                    except Exception as exc:
                        if mode == "save-load":
                            raise
                        print(f"[seed={seed}] save/load failed, falling back to replay: {type(exc).__name__}: {exc}")
                        save_load_available = False
                        cf = None

                if cf is None:
                    restore_rng_state(rng_after_action)
                    cf = estimate_counterfactual_replay(
                        env,
                        actor,
                        seed,
                        action_history,
                        joint_action,
                        probs,
                        args.horizon_steps,
                        args.gamma,
                        device,
                        args.deterministic_continuation,
                        rng_after_action,
                    )

                sim_time = env._sim_time()
                for agent in env.agent_ids:
                    item = cf[agent]
                    load = agent_load[agent]
                    rows.append(
                        {
                            "seed": seed,
                            "sample_id": sample_id,
                            "decision_step": decision_step,
                            "sim_time": sim_time,
                            "agent": agent,
                            "action_actual": joint_action[agent],
                            "pi_actual": item["pi_actual"],
                            "flow": load["flow"],
                            "queue": load["queue"],
                            "outflow": load["outflow"],
                            "pressure": load["pressure"],
                            "waiting": load["waiting"],
                            "q_actual": item["q_actual"],
                            "q_baseline": item["q_baseline"],
                            "coma_advantage": item["coma_advantage"],
                            "q_values_json": json.dumps(item["q_values"].tolist()),
                        }
                    )
                print(
                    f"[seed={seed}] sample={sample_id} step={decision_step} "
                    f"mode={'save-load' if save_load_available else 'replay'} rows={len(rows)}"
                )
                sample_id += 1
                restore_rng_state(rng_after_action)

            next_obs, _rewards, _cp, _cs, done, _info = env.step(joint_action)
            action_history.append(dict(joint_action))
            if next_obs:
                obs = next_obs
            decision_step += 1
            if sample_id >= args.max_samples and args.stop_after_samples:
                break
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="COMA-style post-hoc counterfactual probe.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path. Defaults to newest 4x4 checkpoint.")
    parser.add_argument("--actor-key", default="actor_marl", help="Actor key inside checkpoint.")
    parser.add_argument("--seeds", default=str(C.SEED), help="Comma-separated seeds, e.g. 42,43,44.")
    parser.add_argument("--max-samples", type=int, default=8, help="Sampled decision states per seed.")
    parser.add_argument("--sample-every", type=int, default=20, help="Decision-step interval between samples.")
    parser.add_argument("--burn-in-steps", type=int, default=20, help="Decision steps before first sample.")
    parser.add_argument("--horizon-steps", type=int, default=12, help="Short-horizon continuation length.")
    parser.add_argument("--max-decision-steps", type=int, default=C.STEPS_PER_EPISODE, help="Probe rollout cap.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Efficiency-return discount.")
    parser.add_argument("--state-mode", choices=["auto", "save-load", "replay"], default="auto")
    parser.add_argument("--deterministic-policy", action="store_true", help="Use greedy action for the probed policy.")
    parser.add_argument("--deterministic-continuation", action="store_true", help="Use greedy continuation policy.")
    parser.add_argument("--no-stop-after-samples", dest="stop_after_samples", action="store_false")
    parser.set_defaults(stop_after_samples=True)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(C.BASE_DIR, "outputs", f"coma_probe_{time.strftime('%Y%m%d_%H%M')}"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = args.ckpt or find_default_ckpt()
    seeds = parse_seeds(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: List[dict] = []
    print(f"[coma_probe] ckpt={ckpt_path}")
    print(f"[coma_probe] out_dir={args.out_dir}")
    print(f"[coma_probe] device={device} seeds={seeds}")
    for seed in seeds:
        run_seed(args, seed, ckpt_path, device, rows)

    csv_path = os.path.join(args.out_dir, "coma_probe.csv")
    summary_path = os.path.join(args.out_dir, "coma_probe_summary.json")
    plot_path = os.path.join(args.out_dir, "coma_flow_advantage.png")
    write_csv(csv_path, rows)
    summary = write_summary(summary_path, rows, args, ckpt_path)
    make_plot(plot_path, rows, summary)

    print(f"[coma_probe] rows={len(rows)}")
    print(f"[coma_probe] csv={csv_path}")
    print(f"[coma_probe] summary={summary_path}")
    print(f"[coma_probe] plot={plot_path}")
    print(json.dumps(summary.get("metrics", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
