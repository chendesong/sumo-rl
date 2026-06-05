"""DO NOT USE in the current Cox-merged Fair-TSC workflow.

This SSAM/FCD proxy is parked for now.  The active safety/risk path is the
Cox training-time SUMO intervention in `train_fair_tsc_cox_merged.py`.

SSAM-style vehicle conflict screening from SUMO FCD trajectories.

This is a lightweight post-process for large SUMO FCD files when the FHWA
SSAM desktop application is not available on the server. It is intentionally
named as a proxy: it detects vehicle-vehicle time-to-collision conflicts from
trajectories, but it is not the official FHWA SSAM classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class VehicleState:
    vid: str
    x: float
    y: float
    vx: float
    vy: float
    speed: float


@dataclass
class ConflictStats:
    label: str
    fcd_path: str
    timesteps: int = 0
    vehicle_count: int = 0
    conflict_count: int = 0
    serious_conflict_count: int = 0
    min_ttc: Optional[float] = None
    ttc_sum: float = 0.0
    max_active_vehicles: int = 0
    elapsed_s: float = 0.0

    @property
    def mean_ttc(self) -> Optional[float]:
        if self.conflict_count <= 0:
            return None
        return self.ttc_sum / self.conflict_count

    @property
    def conflict_rate_per_1000veh(self) -> Optional[float]:
        if self.vehicle_count <= 0:
            return None
        return 1000.0 * self.conflict_count / self.vehicle_count

    @property
    def serious_rate_per_1000veh(self) -> Optional[float]:
        if self.vehicle_count <= 0:
            return None
        return 1000.0 * self.serious_conflict_count / self.vehicle_count

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "fcd_path": self.fcd_path,
            "timesteps": self.timesteps,
            "vehicle_count": self.vehicle_count,
            "conflict_count": self.conflict_count,
            "serious_conflict_count": self.serious_conflict_count,
            "conflict_rate_per_1000veh": self.conflict_rate_per_1000veh,
            "serious_rate_per_1000veh": self.serious_rate_per_1000veh,
            "min_ttc": self.min_ttc,
            "mean_ttc": self.mean_ttc,
            "max_active_vehicles": self.max_active_vehicles,
            "elapsed_s": self.elapsed_s,
        }


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _velocity_from_sumo_angle(speed: float, angle_deg: float) -> Tuple[float, float]:
    # SUMO angles are navigation-style: 0 deg points north, increasing clockwise.
    theta = math.radians(angle_deg)
    return speed * math.sin(theta), speed * math.cos(theta)


def _parse_vehicle(elem: ET.Element) -> Optional[VehicleState]:
    try:
        vid = elem.attrib["id"]
        x = float(elem.attrib["x"])
        y = float(elem.attrib["y"])
        speed = float(elem.attrib.get("speed", "0"))
        angle = float(elem.attrib.get("angle", "0"))
    except (KeyError, TypeError, ValueError):
        return None
    vx, vy = _velocity_from_sumo_angle(speed, angle)
    return VehicleState(vid=vid, x=x, y=y, vx=vx, vy=vy, speed=speed)


def _iter_timesteps(fcd_path: str) -> Iterable[Tuple[float, List[VehicleState]]]:
    for _event, elem in ET.iterparse(fcd_path, events=("end",)):
        if _strip_ns(elem.tag) != "timestep":
            continue
        try:
            sim_t = float(elem.attrib.get("time", "0"))
        except ValueError:
            sim_t = 0.0
        vehicles: List[VehicleState] = []
        for child in list(elem):
            if _strip_ns(child.tag) != "vehicle":
                continue
            state = _parse_vehicle(child)
            if state is not None:
                vehicles.append(state)
        yield sim_t, vehicles
        elem.clear()


def _heading_angle_deg(a: VehicleState, b: VehicleState) -> float:
    dot = a.vx * b.vx + a.vy * b.vy
    na = math.hypot(a.vx, a.vy)
    nb = math.hypot(b.vx, b.vy)
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, dot / (na * nb)))
    return math.degrees(math.acos(c))


def _is_conflict(
    a: VehicleState,
    b: VehicleState,
    max_distance_m: float,
    ttc_threshold_s: float,
    conflict_radius_m: float,
    min_speed_mps: float,
) -> Optional[float]:
    if a.speed < min_speed_mps and b.speed < min_speed_mps:
        return None
    dx = b.x - a.x
    dy = b.y - a.y
    dist2 = dx * dx + dy * dy
    if dist2 <= 1e-9 or dist2 > max_distance_m * max_distance_m:
        return None

    rvx = b.vx - a.vx
    rvy = b.vy - a.vy
    rel_speed2 = rvx * rvx + rvy * rvy
    if rel_speed2 <= 1e-9:
        return None

    closing = -(dx * rvx + dy * rvy)
    if closing <= 0.0:
        return None
    ttc = closing / rel_speed2
    if ttc < 0.0 or ttc > ttc_threshold_s:
        return None

    miss_x = dx + rvx * ttc
    miss_y = dy + rvy * ttc
    if miss_x * miss_x + miss_y * miss_y > conflict_radius_m * conflict_radius_m:
        return None
    return float(ttc)


def analyze_fcd(
    label: str,
    fcd_path: str,
    *,
    max_distance_m: float = 20.0,
    ttc_threshold_s: float = 1.5,
    serious_ttc_s: float = 1.0,
    conflict_radius_m: float = 2.5,
    min_speed_mps: float = 0.5,
    cooldown_s: float = 5.0,
    progress_steps: int = 300,
) -> ConflictStats:
    fcd_path = os.path.abspath(os.path.expanduser(fcd_path))
    if not os.path.exists(fcd_path):
        raise FileNotFoundError(fcd_path)

    start = time.time()
    stats = ConflictStats(label=label, fcd_path=fcd_path)
    vehicle_ids = set()
    last_conflict_time: Dict[Tuple[str, str], float] = {}
    cell_size = max_distance_m
    neighbor_offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    for sim_t, vehicles in _iter_timesteps(fcd_path):
        stats.timesteps += 1
        stats.max_active_vehicles = max(stats.max_active_vehicles, len(vehicles))
        for v in vehicles:
            vehicle_ids.add(v.vid)

        grid: Dict[Tuple[int, int], List[VehicleState]] = defaultdict(list)
        for v in vehicles:
            grid[(math.floor(v.x / cell_size), math.floor(v.y / cell_size))].append(v)

        seen_pairs = set()
        for (cx, cy), bucket in grid.items():
            for ox, oy in neighbor_offsets:
                other = grid.get((cx + ox, cy + oy))
                if not other:
                    continue
                for a in bucket:
                    for b in other:
                        if a.vid >= b.vid:
                            continue
                        key = (a.vid, b.vid)
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        if sim_t - last_conflict_time.get(key, -1e18) < cooldown_s:
                            continue
                        ttc = _is_conflict(
                            a,
                            b,
                            max_distance_m=max_distance_m,
                            ttc_threshold_s=ttc_threshold_s,
                            conflict_radius_m=conflict_radius_m,
                            min_speed_mps=min_speed_mps,
                        )
                        if ttc is None:
                            continue
                        stats.conflict_count += 1
                        if ttc <= serious_ttc_s:
                            stats.serious_conflict_count += 1
                        stats.ttc_sum += ttc
                        stats.min_ttc = ttc if stats.min_ttc is None else min(stats.min_ttc, ttc)
                        last_conflict_time[key] = sim_t

        if progress_steps and stats.timesteps % progress_steps == 0:
            rate = stats.conflict_rate_per_1000veh
            print(
                f"[{label}] t={sim_t:.0f} steps={stats.timesteps} "
                f"veh={len(vehicle_ids)} conflicts={stats.conflict_count} "
                f"rate={rate if rate is not None else 0:.2f}/1000veh",
                file=sys.stderr,
                flush=True,
            )

    stats.vehicle_count = len(vehicle_ids)
    stats.elapsed_s = time.time() - start
    return stats


def _write_csv(rows: List[Dict[str, object]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="SSAM-style TTC conflict screening for SUMO FCD files.")
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        metavar=("LABEL", "FCD_XML"),
        required=True,
        help="Add one FCD file to analyze.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--max-distance-m", type=float, default=20.0)
    parser.add_argument("--ttc-threshold-s", type=float, default=1.5)
    parser.add_argument("--serious-ttc-s", type=float, default=1.0)
    parser.add_argument("--conflict-radius-m", type=float, default=2.5)
    parser.add_argument("--min-speed-mps", type=float, default=0.5)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--progress-steps", type=int, default=300)
    args = parser.parse_args()

    rows = []
    for label, fcd_path in args.case:
        stats = analyze_fcd(
            label,
            fcd_path,
            max_distance_m=args.max_distance_m,
            ttc_threshold_s=args.ttc_threshold_s,
            serious_ttc_s=args.serious_ttc_s,
            conflict_radius_m=args.conflict_radius_m,
            min_speed_mps=args.min_speed_mps,
            cooldown_s=args.cooldown_s,
            progress_steps=args.progress_steps,
        )
        row = stats.as_dict()
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "SSAM-style TTC proxy from SUMO FCD; not the official FHWA SSAM classifier.",
                "parameters": {
                    "max_distance_m": args.max_distance_m,
                    "ttc_threshold_s": args.ttc_threshold_s,
                    "serious_ttc_s": args.serious_ttc_s,
                    "conflict_radius_m": args.conflict_radius_m,
                    "min_speed_mps": args.min_speed_mps,
                    "cooldown_s": args.cooldown_s,
                },
                "rows": rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    if args.out_csv:
        _write_csv(rows, args.out_csv)
    print(f"[ssam_fcd_proxy] wrote {args.out_json}")
    if args.out_csv:
        print(f"[ssam_fcd_proxy] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
