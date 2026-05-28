"""Risk-aware SUMO evaluation with wait-time-driven pedestrian events.

This script is intentionally outside training.  It turns the existing
Cox-Weibull pedestrian violation proxy into stochastic SUMO disruptions:

    pedestrian wait -> P(violation event) -> nearby vehicles slow down

The resulting efficiency loss is then measured from SUMO dynamics rather than
being added as a post-hoc scalar penalty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

import config as C
from evaluate import (
    MetricsCollector,
    compute_deltas_from_rollout,
    evaluate_run,
    load_shared_ue_critic,
)
from networks import SharedActor
from sumo_env import FairTSCEnv


@dataclass
class RiskConfig:
    """Parameters for converting violation risk into vehicle disruptions."""

    event_scale: float = 0.10
    cooldown_s: float = 10.0
    slow_duration_s: float = 5.0
    target_speed: float = 0.1
    upstream_distance_m: float = 45.0
    max_vehicles_per_event: int = 8


class RiskEventInjector:
    """Inject wait-time-driven pedestrian disruption events into SUMO."""

    def __init__(self, env: FairTSCEnv, cfg: RiskConfig, seed: int):
        self.env = env
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.cooldown_until: Dict[str, float] = {}
        self.events: List[Dict] = []
        self.num_events = 0
        self.num_vehicle_slowdowns = 0

    def apply(self) -> None:
        """Sample pedestrian events and slow nearby vehicles if triggered."""
        sumo_env = self.env._walk_to_sumo_env()
        now = float(getattr(sumo_env, "sim_step", 0.0))

        for agent_id, ts in sumo_env.traffic_signals.items():
            try:
                crossing_risk = ts.get_jaywalking_per_crossing()
            except Exception:
                continue

            for crossing_id, data in crossing_risk.items():
                queue = int(data.get("queue", 0) or 0)
                p_viol = float(data.get("p_viol", 0.0) or 0.0)
                expected = float(data.get("expected_violations", queue * p_viol) or 0.0)
                if queue <= 0 or expected <= 0.0:
                    continue
                if now < self.cooldown_until.get(crossing_id, -1.0):
                    continue

                p_event = 1.0 - math.exp(-max(expected, 0.0) * self.cfg.event_scale)
                p_event = min(max(p_event, 0.0), 1.0)
                if self.rng.random() >= p_event:
                    continue

                affected = self._slow_nearby_vehicles(ts)
                self.cooldown_until[crossing_id] = now + self.cfg.cooldown_s
                self.num_events += 1
                self.num_vehicle_slowdowns += len(affected)
                self.events.append(
                    {
                        "time": now,
                        "agent_id": agent_id,
                        "crossing_id": crossing_id,
                        "queue": queue,
                        "max_wait": float(data.get("max_wait", 0.0) or 0.0),
                        "p_viol": p_viol,
                        "expected_violations": expected,
                        "p_event": p_event,
                        "vehicles_slowed": len(affected),
                        "vehicle_ids": " ".join(affected),
                    }
                )

    def _slow_nearby_vehicles(self, ts) -> List[str]:
        """Slow vehicles close to the stop line on the intersection approaches."""
        sumo = ts.sumo
        candidates = []
        for lane_id in getattr(ts, "lanes", []):
            try:
                lane_len = float(getattr(ts, "lanes_length", {}).get(lane_id, sumo.lane.getLength(lane_id)))
                vehicle_ids = list(sumo.lane.getLastStepVehicleIDs(lane_id))
            except Exception:
                continue
            for veh_id in vehicle_ids:
                try:
                    pos = float(sumo.vehicle.getLanePosition(veh_id))
                except Exception:
                    continue
                dist_to_stop = lane_len - pos
                if 0.0 <= dist_to_stop <= self.cfg.upstream_distance_m:
                    candidates.append((dist_to_stop, veh_id))

        affected: List[str] = []
        seen = set()
        for _dist, veh_id in sorted(candidates, key=lambda x: x[0]):
            if veh_id in seen:
                continue
            seen.add(veh_id)
            try:
                sumo.vehicle.slowDown(
                    veh_id,
                    float(self.cfg.target_speed),
                    float(self.cfg.slow_duration_s),
                )
                affected.append(veh_id)
            except Exception:
                continue
            if len(affected) >= self.cfg.max_vehicles_per_event:
                break
        return affected

    def summary(self) -> Dict[str, float]:
        return {
            "risk_events": int(self.num_events),
            "risk_vehicle_slowdowns": int(self.num_vehicle_slowdowns),
            "risk_events_per_hour": float(self.num_events) * 3600.0 / max(float(C.NUM_SECONDS), 1.0),
            "risk_slowdowns_per_event": (
                float(self.num_vehicle_slowdowns) / float(self.num_events)
                if self.num_events
                else 0.0
            ),
        }


def _route_for_demand(demand: str) -> str:
    return os.path.join(C.BASE_DIR, "nets", "4x4grid", f"4x4_{demand}.rou.xml")


def _load_actor(env: FairTSCEnv, ckpt_path: str, actor_key: str, device):
    import torch

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if actor_key not in ckpt:
        raise KeyError(f"Checkpoint missing {actor_key}; keys={sorted(ckpt.keys())}")

    actor = SharedActor(
        local_obs_dim=env.local_obs_dim,
        num_agents=env.num_agents,
        action_dim=env.action_dim,
        hidden=C.ACTOR_HIDDEN,
    ).to(device)
    actor.load_state_dict(ckpt[actor_key])
    actor.eval()
    return actor


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "_ema_next"}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_policy_episode(
    ckpt_path: str,
    route_file: str,
    seed: int,
    risk_cfg: Optional[RiskConfig] = None,
    actor_key: str = "actor_marl",
) -> Dict:
    """Run one deterministic policy rollout, optionally with risk injection."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=route_file,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
    )
    injector = RiskEventInjector(env, risk_cfg, seed=seed + 100_003) if risk_cfg else None

    try:
        obs = env.reset(seed=seed)
        actor = _load_actor(env, ckpt_path, actor_key=actor_key, device=device)
        v_ue = load_shared_ue_critic(ckpt_path=ckpt_path, env=env, device=device)

        coll = MetricsCollector()
        rollout = []
        done = False
        while not done:
            global_obs = env.get_global_obs(obs)
            local_b = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
            idx_b = torch.arange(env.num_agents, device=device)
            with torch.no_grad():
                action, _logp = actor.act(local_b, idx_b, deterministic=True)
            action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}

            next_obs, reward, _cp, _cs, done, info = env.step(action_dict)
            if injector is not None and not done:
                injector.apply()

            rewards_array = np.array([reward[a] for a in env.agent_ids], dtype=np.float32)
            coll.add(info, mean_reward=float(rewards_array.mean()) if rewards_array.size else 0.0)
            rollout.append({"global_obs": global_obs, "rewards_array": rewards_array})
            obs = next_obs

        env_metrics = coll.finalize(env)
        deltas_tn = compute_deltas_from_rollout(
            rollout,
            v_ue=v_ue,
            num_agents=env.num_agents,
            gamma=C.GAMMA,
        )
        result = evaluate_run(deltas_tn, env_metrics, delta_valid=True)
        result.pop("_ema_next", None)
        result["ckpt_path"] = ckpt_path
        result["route_file"] = route_file
        result["seed"] = seed
        result["risk_enabled"] = bool(risk_cfg)
        if injector is not None:
            result.update(injector.summary())
            result["_risk_events"] = injector.events
        return result
    finally:
        env.close()


def write_event_csv(events: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "time",
        "agent_id",
        "crossing_id",
        "queue",
        "max_wait",
        "p_viol",
        "expected_violations",
        "p_event",
        "vehicles_slowed",
        "vehicle_ids",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in events:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Fair-TSC/MAPPO checkpoint path")
    parser.add_argument("--demand", choices=["low", "medium", "high"], default=C.DEMAND_LEVEL)
    parser.add_argument("--route-file", default=None, help="Override route file")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--actor-key", default="actor_marl")
    parser.add_argument("--risk-only", action="store_true")
    parser.add_argument("--event-scale", type=float, default=0.10)
    parser.add_argument("--cooldown-s", type=float, default=10.0)
    parser.add_argument("--slow-duration-s", type=float, default=5.0)
    parser.add_argument("--target-speed", type=float, default=0.1)
    parser.add_argument("--upstream-distance-m", type=float, default=45.0)
    parser.add_argument("--max-vehicles-per-event", type=int, default=8)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    route_file = args.route_file or _route_for_demand(args.demand)
    if not os.path.exists(route_file):
        raise FileNotFoundError(route_file)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(C.BASE_DIR, "outputs", f"risk_aware_eval_{args.demand}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    risk_cfg = RiskConfig(
        event_scale=args.event_scale,
        cooldown_s=args.cooldown_s,
        slow_duration_s=args.slow_duration_s,
        target_speed=args.target_speed,
        upstream_distance_m=args.upstream_distance_m,
        max_vehicles_per_event=args.max_vehicles_per_event,
    )

    report: Dict = {
        "demand": args.demand,
        "route_file": route_file,
        "ckpt": args.ckpt,
        "seed": args.seed,
        "risk_config": asdict(risk_cfg),
    }

    if not args.risk_only:
        report["baseline"] = run_policy_episode(
            ckpt_path=args.ckpt,
            route_file=route_file,
            seed=args.seed,
            risk_cfg=None,
            actor_key=args.actor_key,
        )

    risk = run_policy_episode(
        ckpt_path=args.ckpt,
        route_file=route_file,
        seed=args.seed,
        risk_cfg=risk_cfg,
        actor_key=args.actor_key,
    )
    events = list(risk.pop("_risk_events", []))
    report["risk_aware"] = risk

    if "baseline" in report:
        base = report["baseline"]
        report["efficiency_loss"] = float(base.get("efficiency", 0.0)) - float(risk.get("efficiency", 0.0))
        report["waiting_time_increase"] = (
            -float(risk.get("efficiency", 0.0)) - -float(base.get("efficiency", 0.0))
        )
        report["ped_risk_change"] = float(risk.get("ped_risk", 0.0)) - float(base.get("ped_risk", 0.0))

    json_path = os.path.join(out_dir, "risk_aware_eval.json")
    events_path = os.path.join(out_dir, "risk_events.csv")
    write_event_csv(events, events_path)
    report["event_csv"] = events_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2, sort_keys=True)

    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    print(f"[risk_aware_eval] json={json_path}")
    print(f"[risk_aware_eval] events={events_path}")


if __name__ == "__main__":
    main()
