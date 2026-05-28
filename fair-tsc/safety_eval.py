"""Post-hoc safety evaluation helpers for Fair-TSC.

This module keeps safety measurement outside the training objective:

* S_ped is a pedestrian violation-risk proxy computed from the existing
  Cox-Weibull/Phi-style SUMO info stream:

      S_ped = mean_t total_expected_violations(t) / (4 * N)

  It is not an observed violation count and is not injected back into SUMO.

* S_veh is an SSAM-style vehicle conflict rate. SUMO can export FCD
  trajectories, but SSAM conflict classification is an external post-process.
  This file therefore stores/combines SSAM conflict counts once they are
  available, instead of pretending SUMO has produced them directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional

import numpy as np

import config as C


DEFAULT_CROSSINGS_PER_AGENT = 4


def normalize_pedestrian_risk(
    expected_violations: float,
    num_agents: Optional[int] = None,
    crossings_per_agent: int = DEFAULT_CROSSINGS_PER_AGENT,
) -> float:
    """Normalize the per-step expected pedestrian violations to S_ped.

    The upstream SUMO wrapper reports a step-level network total:
    ``agents_total_expected_violations``.  The paper-level proxy averages it
    over intersections and crossings, hence the denominator ``4 * N`` for the
    4x4 grid with four pedestrian crossings per intersection.
    """
    if num_agents is None:
        num_agents = int(getattr(C, "NUM_AGENTS", 0) or 0)
    denom = max(float(num_agents) * float(crossings_per_agent), 1.0)
    return float(expected_violations) / denom


def veh_conflict_rate(num_conflicts: Optional[float], num_vehicles: Optional[float]) -> Optional[float]:
    """Return SSAM conflict rate per 1000 vehicles, or None if unavailable."""
    if num_conflicts is None or num_vehicles is None:
        return None
    num_vehicles = float(num_vehicles)
    if num_vehicles <= 0:
        return None
    return 1000.0 * float(num_conflicts) / num_vehicles


def safety_adjusted_efficiency(
    efficiency: float,
    s_veh: Optional[float] = None,
    s_ped: Optional[float] = None,
    beta_veh: float = 0.0,
    beta_ped: float = 0.0,
    veh_ref: float = 1.0,
    ped_ref: float = 1.0,
) -> float:
    """Efficiency minus optional normalized safety penalties.

    Coefficients default to zero intentionally: the script records the safety
    exposures first, and only computes a nontrivial SAE once the experimenter
    explicitly chooses beta weights.
    """
    score = float(efficiency)
    if s_veh is not None and beta_veh:
        score -= float(beta_veh) * float(s_veh) / max(float(veh_ref), 1e-12)
    if s_ped is not None and beta_ped:
        score -= float(beta_ped) * float(s_ped) / max(float(ped_ref), 1e-12)
    return float(score)


def parse_fcd_vehicle_count(fcd_path: str) -> int:
    """Count unique vehicles in a SUMO FCD trajectory file."""
    if not fcd_path or not os.path.exists(fcd_path):
        return 0
    vehicle_ids = set()
    for _event, elem in ET.iterparse(fcd_path, events=("end",)):
        if elem.tag == "vehicle":
            vid = elem.attrib.get("id")
            if vid:
                vehicle_ids.add(vid)
        elem.clear()
    return len(vehicle_ids)


def make_fcd_sumo_cmd(fcd_output: str) -> str:
    """SUMO additional command for exporting trajectories for SSAM."""
    fcd_output = os.path.abspath(fcd_output)
    os.makedirs(os.path.dirname(fcd_output), exist_ok=True)
    return f"--fcd-output {fcd_output}"


def _float_or_none(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def augment_result_with_safety(
    result: Dict,
    veh_conflicts: Optional[float] = None,
    veh_count: Optional[float] = None,
    beta_veh: float = 0.0,
    beta_ped: float = 0.0,
    veh_ref: float = 1.0,
    ped_ref: float = 1.0,
) -> Dict:
    """Attach S_veh, S_ped, and optional safety-adjusted efficiency."""
    out = dict(result)
    s_ped = _float_or_none(out.get("ped_risk"))
    s_veh = veh_conflict_rate(veh_conflicts, veh_count)
    out["S_ped"] = s_ped
    out["S_veh"] = s_veh
    out["veh_conflicts"] = veh_conflicts
    out["veh_count"] = veh_count
    out["beta_veh"] = beta_veh
    out["beta_ped"] = beta_ped
    out["safety_adjusted_efficiency"] = safety_adjusted_efficiency(
        efficiency=float(out.get("efficiency", 0.0) or 0.0),
        s_veh=s_veh,
        s_ped=s_ped,
        beta_veh=beta_veh,
        beta_ped=beta_ped,
        veh_ref=veh_ref,
        ped_ref=ped_ref,
    )
    return out


def augment_comparison_csv(
    comparison_csv: str,
    out_csv: str,
    veh_conflicts: Optional[float] = None,
    veh_count: Optional[float] = None,
    beta_veh: float = 0.0,
    beta_ped: float = 0.0,
    veh_ref: float = 1.0,
    ped_ref: float = 1.0,
) -> List[Dict]:
    """Read a comparison CSV and write one with safety columns attached."""
    with open(comparison_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    augmented = [
        augment_result_with_safety(
            row,
            veh_conflicts=veh_conflicts,
            veh_count=veh_count,
            beta_veh=beta_veh,
            beta_ped=beta_ped,
            veh_ref=veh_ref,
            ped_ref=ped_ref,
        )
        for row in rows
    ]

    fieldnames = list(rows[0].keys()) if rows else []
    for key in (
        "S_ped",
        "S_veh",
        "veh_conflicts",
        "veh_count",
        "beta_veh",
        "beta_ped",
        "safety_adjusted_efficiency",
    ):
        if key not in fieldnames:
            fieldnames.append(key)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in augmented:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return augmented


def run_fair_tsc_safety_eval(
    ckpt_path: Optional[str] = None,
    seed: Optional[int] = None,
    fcd_output: Optional[str] = None,
    out_json: Optional[str] = None,
    veh_conflicts: Optional[float] = None,
    veh_count: Optional[float] = None,
    beta_veh: float = 0.0,
    beta_ped: float = 0.0,
    veh_ref: float = 1.0,
    ped_ref: float = 1.0,
) -> Dict:
    """Run one Fair-TSC rollout and emit post-hoc safety metrics."""
    from run_comparison import fair_tsc_fresh_eval

    additional_sumo_cmd = make_fcd_sumo_cmd(fcd_output) if fcd_output else None
    result = fair_tsc_fresh_eval(
        ckpt_path=ckpt_path,
        seed=seed,
        additional_sumo_cmd=additional_sumo_cmd,
    )

    if veh_count is None and fcd_output and os.path.exists(fcd_output):
        veh_count = float(parse_fcd_vehicle_count(fcd_output))

    report = augment_result_with_safety(
        result,
        veh_conflicts=veh_conflicts,
        veh_count=veh_count,
        beta_veh=beta_veh,
        beta_ped=beta_ped,
        veh_ref=veh_ref,
        ped_ref=ped_ref,
    )
    report["fcd_output"] = os.path.abspath(fcd_output) if fcd_output else None
    report["ckpt_path"] = ckpt_path
    report["seed"] = seed

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return report


def _default_safety_dir() -> str:
    return os.path.join(C.BASE_DIR, "outputs", f"safety_eval_{time.strftime('%Y%m%d_%H%M')}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc safety evaluation for Fair-TSC.")
    parser.add_argument("--ckpt", default=None, help="Fair-TSC checkpoint path. Defaults to latest configured ckpt.")
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed.")
    parser.add_argument("--fcd-output", default=None, help="Write SUMO FCD trajectories for external SSAM.")
    parser.add_argument("--out-json", default=None, help="JSON report path for one Fair-TSC rollout.")
    parser.add_argument("--comparison-csv", default=None, help="Existing run_comparison CSV to augment.")
    parser.add_argument("--out-csv", default=None, help="Augmented comparison CSV path.")
    parser.add_argument("--veh-conflicts", type=float, default=None, help="SSAM conflict count from external post-process.")
    parser.add_argument("--veh-count", type=float, default=None, help="Vehicle count for S_veh. Can be inferred from FCD.")
    parser.add_argument("--beta-veh", type=float, default=0.0, help="SAE vehicle-safety penalty weight.")
    parser.add_argument("--beta-ped", type=float, default=0.0, help="SAE pedestrian-risk penalty weight.")
    parser.add_argument("--veh-ref", type=float, default=1.0, help="S_veh normalization reference.")
    parser.add_argument("--ped-ref", type=float, default=1.0, help="S_ped normalization reference.")
    args = parser.parse_args()

    out_dir = _default_safety_dir()
    if args.comparison_csv:
        out_csv = args.out_csv or os.path.join(out_dir, "comparison_safety.csv")
        rows = augment_comparison_csv(
            args.comparison_csv,
            out_csv,
            veh_conflicts=args.veh_conflicts,
            veh_count=args.veh_count,
            beta_veh=args.beta_veh,
            beta_ped=args.beta_ped,
            veh_ref=args.veh_ref,
            ped_ref=args.ped_ref,
        )
        print(f"[safety_eval] wrote {out_csv} ({len(rows)} rows)")
        return

    fcd_output = args.fcd_output
    if fcd_output is None:
        fcd_output = os.path.join(out_dir, "fair_tsc.fcd.xml")
    out_json = args.out_json or os.path.join(out_dir, "fair_tsc_safety.json")
    report = run_fair_tsc_safety_eval(
        ckpt_path=args.ckpt,
        seed=args.seed,
        fcd_output=fcd_output,
        out_json=out_json,
        veh_conflicts=args.veh_conflicts,
        veh_count=args.veh_count,
        beta_veh=args.beta_veh,
        beta_ped=args.beta_ped,
        veh_ref=args.veh_ref,
        ped_ref=args.ped_ref,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[safety_eval] wrote {out_json}")


if __name__ == "__main__":
    main()
