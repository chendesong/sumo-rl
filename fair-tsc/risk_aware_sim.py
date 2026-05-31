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
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config as C
from evaluate import (
    MetricsCollector,
    attach_tripinfo_metrics,
    compute_deltas_from_rollout,
    evaluate_run,
    load_shared_ue_critic,
    make_tripinfo_sumo_cmd,
    merge_sumo_cmds,
)
from networks import SharedActor
from safety_eval import make_fcd_sumo_cmd, parse_fcd_vehicle_count
from sumo_env import FairTSCEnv


@dataclass
class RiskConfig:
    """Parameters for converting violation risk into vehicle disruptions."""

    hazard_multiplier: float = 1.0
    cooldown_s: float = 10.0
    upstream_distance_m: float = 45.0
    conflict_distance_m: float = 6.0
    comfort_decel_mps2: float = 3.0
    stop_buffer_m: float = 2.0
    min_brake_speed_mps: float = 0.5
    max_vehicles_per_event: int = 4
    disruption_duration_s: float = 5.0


class RiskEventInjector:
    """Inject wait-time-driven pedestrian disruption events into SUMO."""

    def __init__(
        self,
        env: FairTSCEnv,
        cfg: RiskConfig,
        seed: int,
        event_plan: Optional[set[Tuple[int, str]]] = None,
        horizon_seconds: float = C.NUM_SECONDS,
    ):
        self.env = env
        self.cfg = cfg
        self.seed = int(seed)
        self.event_plan = event_plan
        self.horizon_seconds = float(horizon_seconds)
        self.cooldown_until: Dict[str, float] = {}
        self.events: List[Dict] = []
        self._conflict_lane_cache: Dict[Tuple[str, str], List[str]] = {}
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
                max_wait = float(data.get("max_wait", 0.0) or 0.0)
                if queue <= 0 or max_wait <= 0.0:
                    continue
                if now < self.cooldown_until.get(crossing_id, -1.0):
                    continue

                p_step = self._per_person_step_probability(ts, max_wait)
                p_event = 1.0 - (1.0 - p_step) ** queue
                p_event = min(max(p_event, 0.0), 1.0)
                step_key = self._step_key(now)
                u_event = self._common_uniform(step_key, crossing_id)
                if self.event_plan is None:
                    triggered = u_event < p_event
                else:
                    triggered = (step_key, crossing_id) in self.event_plan
                if not triggered:
                    continue

                affected = self._slow_conflict_lead_vehicles(ts, crossing_id)
                self.cooldown_until[crossing_id] = now + self.cfg.cooldown_s
                self.num_events += 1
                self.num_vehicle_slowdowns += len(affected)
                self.events.append(
                    {
                        "time": now,
                        "agent_id": agent_id,
                        "crossing_id": crossing_id,
                        "queue": queue,
                        "max_wait": max_wait,
                        "p_step": p_step,
                        "p_event": p_event,
                        "u_event": u_event,
                        "vehicles_slowed": len(affected),
                        "vehicle_ids": " ".join(v["vehicle_id"] for v in affected),
                        "slowdown_commands": json.dumps(affected, sort_keys=True),
                    }
                )

    def _per_person_step_probability(self, ts, wait_time: float) -> float:
        """Convert the Cox-Weibull cumulative hazard into a one-step risk."""
        dt = float(getattr(ts, "delta_time", getattr(self.env, "delta_time", C.DELTA_TIME)) or C.DELTA_TIME)
        prev_wait = max(float(wait_time) - dt, 0.0)
        try:
            flow = float(ts._conflicting_flow_approx())
        except Exception:
            flow = 0.0
        h_now = self._integrated_hazard(ts, wait_time, flow)
        h_prev = self._integrated_hazard(ts, prev_wait, flow)
        delta_h = max(h_now - h_prev, 0.0) * max(float(self.cfg.hazard_multiplier), 0.0)
        return 1.0 - math.exp(-delta_h)

    @staticmethod
    def _integrated_hazard(ts, wait_time: float, conflicting_flow: float) -> float:
        if wait_time <= 0.0:
            return 0.0
        lambda_w = max(float(getattr(ts, "lambda_w", 60.0)), 1e-9)
        k_w = max(float(getattr(ts, "k_w", 2.0)), 1e-9)
        beta_f = float(getattr(ts, "beta_f", 0.1))
        deterrence = math.exp(-beta_f * max(float(conflicting_flow), 0.0))
        return (float(wait_time) / lambda_w) ** k_w * deterrence

    def _common_uniform(self, step: int, crossing_id: str) -> float:
        """Common random number keyed by seed, time step, and crossing id."""
        key = f"{self.seed}|{int(step)}|{crossing_id}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)

    def _step_key(self, sim_time: float) -> int:
        return int(round(float(sim_time) / max(float(self.env.delta_time), 1e-9)))

    def _slow_conflict_lead_vehicles(self, ts, crossing_id: str) -> List[Dict]:
        """Slow only lead vehicles on lanes whose paths conflict with a crossing."""
        sumo = ts.sumo
        lanes = self._conflict_lanes_for_crossing(ts, crossing_id)
        candidates = []
        for lane_id in lanes:
            try:
                lane_len = float(getattr(ts, "lanes_length", {}).get(lane_id, sumo.lane.getLength(lane_id)))
                vehicle_ids = list(sumo.lane.getLastStepVehicleIDs(lane_id))
            except Exception:
                continue
            lead = None
            for veh_id in vehicle_ids:
                try:
                    pos = float(sumo.vehicle.getLanePosition(veh_id))
                    speed = float(sumo.vehicle.getSpeed(veh_id))
                except Exception:
                    continue
                dist_to_stop = lane_len - pos
                if 0.0 <= dist_to_stop <= self.cfg.upstream_distance_m and speed >= self.cfg.min_brake_speed_mps:
                    if lead is None or dist_to_stop < lead[0]:
                        lead = (dist_to_stop, lane_id, veh_id, speed)
            if lead is not None:
                candidates.append(lead)

        affected: List[Dict] = []
        seen = set()
        for dist_to_stop, lane_id, veh_id, speed in sorted(candidates, key=lambda x: x[0]):
            if veh_id in seen:
                continue
            seen.add(veh_id)
            command = self._brake_command(speed=speed, dist_to_stop=dist_to_stop)
            if command is None:
                continue
            target_speed, duration, decel = command
            try:
                sumo.vehicle.slowDown(
                    veh_id,
                    float(target_speed),
                    float(duration),
                )
                affected.append(
                    {
                        "vehicle_id": veh_id,
                        "lane_id": lane_id,
                        "distance_to_stop_m": float(dist_to_stop),
                        "speed_mps": float(speed),
                        "target_speed_mps": float(target_speed),
                        "duration_s": float(duration),
                        "comfort_decel_mps2": float(decel),
                    }
                )
            except Exception:
                continue
            if len(affected) >= self.cfg.max_vehicles_per_event:
                break
        return affected

    def _brake_command(self, speed: float, dist_to_stop: float) -> Optional[Tuple[float, float, float]]:
        """Compute a fixed disruption command for violation-induced blockage."""
        speed = max(float(speed), 0.0)
        if speed < self.cfg.min_brake_speed_mps:
            return None
        decel = max(float(self.cfg.comfort_decel_mps2), 1e-6)
        duration = max(float(self.cfg.disruption_duration_s), 1e-3)
        target_speed = 0.0
        if duration <= 1e-6:
            return None
        return target_speed, duration, decel

    def _conflict_lanes_for_crossing(self, ts, crossing_id: str) -> List[str]:
        key = (str(ts.id), str(crossing_id))
        cached = self._conflict_lane_cache.get(key)
        if cached is not None:
            return cached

        sumo = ts.sumo
        crossing_shape = self._lane_shape(sumo, f"{crossing_id}_0") or self._lane_shape(sumo, crossing_id)
        lanes = set()
        if crossing_shape:
            try:
                controlled_links = sumo.trafficlight.getControlledLinks(ts.id)
            except Exception:
                controlled_links = []
            for link_group in controlled_links:
                for link in link_group:
                    if not link:
                        continue
                    in_lane = link[0] if len(link) > 0 else ""
                    out_lane = link[1] if len(link) > 1 else ""
                    via_lane = link[2] if len(link) > 2 else ""
                    path_shape = self._lane_shape(sumo, via_lane)
                    if not path_shape:
                        path_shape = self._lane_shape(sumo, in_lane) + self._lane_shape(sumo, out_lane)
                    if in_lane and path_shape and self._polyline_distance(path_shape, crossing_shape) <= self.cfg.conflict_distance_m:
                        lanes.add(in_lane)

        if not lanes:
            lanes.update(getattr(ts, "lanes", []))
        ordered = [lane for lane in getattr(ts, "lanes", []) if lane in lanes]
        self._conflict_lane_cache[key] = ordered
        return ordered

    @staticmethod
    def _lane_shape(sumo, lane_id: str) -> List[Tuple[float, float]]:
        if not lane_id:
            return []
        try:
            return [(float(x), float(y)) for x, y in sumo.lane.getShape(lane_id)]
        except Exception:
            return []

    @classmethod
    def _polyline_distance(
        cls,
        a: Sequence[Tuple[float, float]],
        b: Sequence[Tuple[float, float]],
    ) -> float:
        if not a or not b:
            return float("inf")
        if len(a) == 1 and len(b) == 1:
            return math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1])
        if len(a) == 1:
            return min(cls._point_segment_distance(a[0], b[j], b[j + 1]) for j in range(len(b) - 1))
        if len(b) == 1:
            return min(cls._point_segment_distance(b[0], a[i], a[i + 1]) for i in range(len(a) - 1))
        return min(
            cls._segment_distance(a[i], a[i + 1], b[j], b[j + 1])
            for i in range(len(a) - 1)
            for j in range(len(b) - 1)
        )

    @classmethod
    def _segment_distance(cls, a, b, c, d) -> float:
        if cls._segments_intersect(a, b, c, d):
            return 0.0
        return min(
            cls._point_segment_distance(a, c, d),
            cls._point_segment_distance(b, c, d),
            cls._point_segment_distance(c, a, b),
            cls._point_segment_distance(d, a, b),
        )

    @classmethod
    def _segments_intersect(cls, a, b, c, d) -> bool:
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
            )

        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)
        if o1 * o2 < 0.0 and o3 * o4 < 0.0:
            return True
        if abs(o1) <= 1e-9 and on_segment(a, c, b):
            return True
        if abs(o2) <= 1e-9 and on_segment(a, d, b):
            return True
        if abs(o3) <= 1e-9 and on_segment(c, a, d):
            return True
        if abs(o4) <= 1e-9 and on_segment(c, b, d):
            return True
        return False

    @staticmethod
    def _point_segment_distance(p, a, b) -> float:
        px, py = p
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        qx, qy = ax + t * dx, ay + t * dy
        return math.hypot(px - qx, py - qy)

    def summary(self) -> Dict[str, float]:
        return {
            "risk_events": int(self.num_events),
            "risk_vehicle_slowdowns": int(self.num_vehicle_slowdowns),
            "risk_events_per_hour": float(self.num_events) * 3600.0 / max(self.horizon_seconds, 1.0),
            "risk_slowdowns_per_event": (
                float(self.num_vehicle_slowdowns) / float(self.num_events)
                if self.num_events
                else 0.0
            ),
        }


def _route_for_demand(demand: str) -> str:
    return os.path.join(C.BASE_DIR, "nets", "4x4grid", f"4x4_{demand}.rou.xml")


def read_event_plan(path: str, delta_time: float = C.DELTA_TIME) -> set[Tuple[int, str]]:
    """Read a prior risk_events.csv as a replayable event schedule."""
    plan: set[Tuple[int, str]] = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            crossing_id = row.get("crossing_id")
            if not crossing_id:
                continue
            try:
                step = int(round(float(row.get("time", 0.0)) / max(float(delta_time), 1e-9)))
            except (TypeError, ValueError):
                continue
            plan.add((step, crossing_id))
    return plan


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


def _resolve_fcd_output(base_path: Optional[str], label: str) -> Optional[str]:
    """Resolve one FCD path per rollout for external SSAM processing."""
    if not base_path:
        return None
    expanded = os.path.abspath(os.path.expanduser(base_path))
    root, ext = os.path.splitext(expanded)
    if ext.lower() == ".xml":
        return f"{root}_{label}{ext}"
    os.makedirs(expanded, exist_ok=True)
    return os.path.join(expanded, f"{label}.fcd.xml")


def run_policy_episode(
    ckpt_path: str,
    route_file: str,
    seed: int,
    risk_cfg: Optional[RiskConfig] = None,
    actor_key: str = "actor_marl",
    event_plan: Optional[set[Tuple[int, str]]] = None,
    num_seconds: Optional[int] = None,
    additional_sumo_cmd: Optional[str] = None,
) -> Dict:
    """Run one deterministic policy rollout, optionally with risk injection."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=route_file,
        out_csv_name=None,
        num_seconds=int(num_seconds or C.NUM_SECONDS),
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    injector = (
        RiskEventInjector(
            env,
            risk_cfg,
            seed=seed,
            event_plan=event_plan,
            horizon_seconds=int(num_seconds or C.NUM_SECONDS),
        )
        if risk_cfg
        else None
    )

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
        result["num_seconds"] = int(num_seconds or C.NUM_SECONDS)
        result["risk_enabled"] = bool(risk_cfg)
        result["event_plan_replay"] = bool(event_plan)
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
        "p_step",
        "p_event",
        "u_event",
        "vehicles_slowed",
        "vehicle_ids",
        "slowdown_commands",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in events:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Fair-TSC/MAPPO checkpoint path")
    parser.add_argument("--demand", choices=sorted(C.DEMAND_LEVELS), default=C.DEMAND_LEVEL)
    parser.add_argument("--route-file", default=None, help="Override route file")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--actor-key", default="actor_marl")
    parser.add_argument("--num-seconds", type=int, default=C.NUM_SECONDS)
    parser.add_argument("--risk-only", action="store_true")
    parser.add_argument("--hazard-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--event-scale",
        type=float,
        default=None,
        help="Deprecated alias for --hazard-multiplier.",
    )
    parser.add_argument("--cooldown-s", type=float, default=10.0)
    parser.add_argument("--upstream-distance-m", type=float, default=45.0)
    parser.add_argument("--conflict-distance-m", type=float, default=6.0)
    parser.add_argument("--comfort-decel-mps2", type=float, default=3.0)
    parser.add_argument("--stop-buffer-m", type=float, default=2.0)
    parser.add_argument("--min-brake-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-vehicles-per-event", type=int, default=4)
    parser.add_argument(
        "--disruption-duration-s",
        type=float,
        default=5.0,
        help="Duration used by slowDown(..., 0, duration) after each violation event.",
    )
    parser.add_argument(
        "--event-plan-in",
        default=None,
        help="Replay a prior risk_events.csv schedule instead of sampling events.",
    )
    parser.add_argument(
        "--fcd-output",
        default=None,
        help=(
            "Export SUMO FCD trajectories for external SSAM. If this is an .xml path, "
            "the script writes *_baseline.xml and *_risk_aware.xml; otherwise it is "
            "treated as a directory containing baseline.fcd.xml and risk_aware.fcd.xml."
        ),
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    route_file = args.route_file or _route_for_demand(args.demand)
    if not os.path.exists(route_file):
        raise FileNotFoundError(route_file)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(C.BASE_DIR, "outputs", f"risk_aware_sim_{args.demand}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    baseline_fcd = _resolve_fcd_output(args.fcd_output, "baseline")
    risk_fcd = _resolve_fcd_output(args.fcd_output, "risk_aware")
    baseline_tripinfo = os.path.join(out_dir, "baseline.tripinfo.xml")
    risk_tripinfo = os.path.join(out_dir, "risk_aware.tripinfo.xml")

    risk_cfg = RiskConfig(
        hazard_multiplier=args.event_scale if args.event_scale is not None else args.hazard_multiplier,
        cooldown_s=args.cooldown_s,
        upstream_distance_m=args.upstream_distance_m,
        conflict_distance_m=args.conflict_distance_m,
        comfort_decel_mps2=args.comfort_decel_mps2,
        stop_buffer_m=args.stop_buffer_m,
        min_brake_speed_mps=args.min_brake_speed_mps,
        max_vehicles_per_event=args.max_vehicles_per_event,
        disruption_duration_s=args.disruption_duration_s,
    )

    report: Dict = {
        "demand": args.demand,
        "route_file": route_file,
        "ckpt": args.ckpt,
        "seed": args.seed,
        "num_seconds": args.num_seconds,
        "risk_config": asdict(risk_cfg),
        "common_random_field": "seed+time_step+crossing_id",
        "event_plan_in": os.path.abspath(args.event_plan_in) if args.event_plan_in else None,
        "ssam_note": "FCD trajectories are exported here; SSAM conflict counts are computed post-hoc.",
    }

    event_plan = read_event_plan(args.event_plan_in, delta_time=C.DELTA_TIME) if args.event_plan_in else None

    if not args.risk_only:
        report["baseline"] = run_policy_episode(
            ckpt_path=args.ckpt,
            route_file=route_file,
            seed=args.seed,
            risk_cfg=None,
            actor_key=args.actor_key,
            num_seconds=args.num_seconds,
            additional_sumo_cmd=merge_sumo_cmds(
                make_fcd_sumo_cmd(baseline_fcd) if baseline_fcd else None,
                make_tripinfo_sumo_cmd(baseline_tripinfo),
            ),
        )
        attach_tripinfo_metrics(report["baseline"], baseline_tripinfo, horizon_s=args.num_seconds)
        if baseline_fcd:
            report["baseline"]["fcd_output"] = baseline_fcd
            report["baseline"]["fcd_vehicle_count"] = parse_fcd_vehicle_count(baseline_fcd)

    risk = run_policy_episode(
        ckpt_path=args.ckpt,
        route_file=route_file,
        seed=args.seed,
        risk_cfg=risk_cfg,
        actor_key=args.actor_key,
        event_plan=event_plan,
        num_seconds=args.num_seconds,
        additional_sumo_cmd=merge_sumo_cmds(
            make_fcd_sumo_cmd(risk_fcd) if risk_fcd else None,
            make_tripinfo_sumo_cmd(risk_tripinfo),
        ),
    )
    events = list(risk.pop("_risk_events", []))
    attach_tripinfo_metrics(risk, risk_tripinfo, horizon_s=args.num_seconds)
    if risk_fcd:
        risk["fcd_output"] = risk_fcd
        risk["fcd_vehicle_count"] = parse_fcd_vehicle_count(risk_fcd)
    report["risk_aware"] = risk

    if "baseline" in report:
        base = report["baseline"]
        report["efficiency_loss"] = float(base.get("efficiency", 0.0)) - float(risk.get("efficiency", 0.0))
        report["travel_time_increase_s"] = (
            float(risk.get("total_travel_time_s", 0.0)) - float(base.get("total_travel_time_s", 0.0))
        )
        report["time_loss_increase_s"] = (
            float(risk.get("total_time_loss_s", 0.0)) - float(base.get("total_time_loss_s", 0.0))
        )
        report["vehicle_waiting_time_increase_s"] = (
            float(risk.get("total_vehicle_waiting_time_s", 0.0))
            - float(base.get("total_vehicle_waiting_time_s", 0.0))
        )
        report["throughput_drop_veh_per_hour"] = (
            float(base.get("throughput_veh_per_hour", 0.0)) - float(risk.get("throughput_veh_per_hour", 0.0))
        )
        report["completion_rate_departed_drop"] = (
            float(base.get("completion_rate_departed", 0.0)) - float(risk.get("completion_rate_departed", 0.0))
        )
        report["completion_rate_demand_drop"] = (
            float(base.get("completion_rate_demand", 0.0)) - float(risk.get("completion_rate_demand", 0.0))
        )
        report["unfinished_vehicle_demand_increase"] = (
            float(risk.get("unfinished_vehicle_demand", 0.0)) - float(base.get("unfinished_vehicle_demand", 0.0))
        )
        report["pending_vehicle_count_increase"] = (
            float(risk.get("pending_vehicle_count_end", 0.0)) - float(base.get("pending_vehicle_count_end", 0.0))
        )
        report["active_vehicle_count_increase"] = (
            float(risk.get("active_vehicle_count_end", 0.0)) - float(base.get("active_vehicle_count_end", 0.0))
        )
        report["queue_efficiency_loss"] = (
            float(base.get("queue_efficiency", 0.0)) - float(risk.get("queue_efficiency", 0.0))
        )
        report["waiting_time_increase"] = report["vehicle_waiting_time_increase_s"]
        report["ped_risk_change"] = float(risk.get("ped_risk", 0.0)) - float(base.get("ped_risk", 0.0))

    json_path = os.path.join(out_dir, "risk_aware_sim.json")
    events_path = os.path.join(out_dir, "risk_events.csv")
    write_event_csv(events, events_path)
    report["event_csv"] = events_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2, sort_keys=True)

    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    print(f"[risk_aware_sim] json={json_path}")
    print(f"[risk_aware_sim] events={events_path}")


if __name__ == "__main__":
    main()
