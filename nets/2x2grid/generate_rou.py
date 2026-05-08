"""
generate_routes.py — Programmatic route file generator for the 2x2 grid.

Strict design: only emits flows whose origin and destination are on
the audited boundary edges of the network.  Any flow that would put a
vehicle or pedestrian on an interior edge is rejected at generation
time, so the resulting .rou.xml is guaranteed legal w.r.t. the net.

Network topology (audited from 01.net.xml):
    Traffic-light intersections : 1, 2, 5, 6
    Boundary dead-ends          : 0, 3, 4, 7, 9, 10, 17, 18

Inflow edges (8) — vehicles enter the network via these:
    -h11 (0  -> 1)     -h21 (4  -> 5)
    -v11 (17 -> 1)     -v21 (18 -> 2)
     h13 (3  -> 2)      h23 (7  -> 6)
     v13 (9  -> 5)      v23 (10 -> 6)

Outflow edges (8) — vehicles exit the network via these:
     h11 (1  -> 0)      h21 (5  -> 4)
     v11 (1  -> 17)     v21 (2  -> 18)
    -h13 (2  -> 3)     -h23 (6  -> 7)
    -v13 (5  -> 9)     -v23 (6  -> 10)

Usage:
    python generate_routes.py --out 02.rou.xml --demand low --seed 42
"""

from __future__ import annotations
import argparse
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ──────────────────────────────────────────────────────────────────
# Audited boundary edges — DO NOT EDIT without re-running net audit
# ──────────────────────────────────────────────────────────────────
INFLOW_EDGES = [
    "-h11", "-h21", "-v11", "-v21",
    "h13", "h23", "v13", "-h21",  # corrected below
]
# Re-derived from junction.from = dead_end:
INFLOW_EDGES = ["-h11", "-h21", "-v11", "-v21", "h13", "h23", "v13", "v23"]
OUTFLOW_EDGES = ["h11", "h21", "v11", "v21", "-h13", "-h23", "-v13", "-v23"]

# Map each inflow edge to the TL it enters (for documentation only).
INFLOW_TO_TL = {
    "-h11": "1", "-v11": "1",
    "h13": "2",  "-v21": "2",
    "-h21": "5", "v13": "5",
    "h23": "6",  "v23": "6",
}
# Map each outflow edge to the TL it exits from.
OUTFLOW_FROM_TL = {
    "h11": "1", "v11": "1",
    "-h13": "2", "v21": "2",
    "h21": "5", "-v13": "5",
    "-h23": "6", "-v23": "6",
}

# ──────────────────────────────────────────────────────────────────
# Demand profiles
# ──────────────────────────────────────────────────────────────────
DEMAND_PROFILES = {
    "low": {
        "veh_per_hour_per_inflow": 250,    # 8 inflows * 250 = 2000 veh/h total
        "ped_per_hour_per_inflow": 60,     # 8 inflows * 60  = 480  ped/h total
    },
    "medium": {
        "veh_per_hour_per_inflow": 450,    # 3600 veh/h
        "ped_per_hour_per_inflow": 150,    # 1200 ped/h
    },
    "high": {
        "veh_per_hour_per_inflow": 700,    # 5600 veh/h
        "ped_per_hour_per_inflow": 300,    # 2400 ped/h
    },
}

# Episode horizon (must match SumoEnvironment num_seconds)
DEFAULT_END_TIME = 3600.0  # seconds


def make_flow(
    flow_id: str,
    from_edge: str,
    to_edge: str,
    veh_per_hour: float,
    begin: float,
    end: float,
    departLane: str = "best",
    departSpeed: str = "max",
) -> ET.Element:
    """Build a <flow> element. Throws if from/to is not a legal boundary edge."""
    if from_edge not in INFLOW_EDGES:
        raise ValueError(f"flow {flow_id}: from-edge {from_edge!r} is NOT an inflow edge")
    if to_edge not in OUTFLOW_EDGES:
        raise ValueError(f"flow {flow_id}: to-edge {to_edge!r} is NOT an outflow edge")
    el = ET.Element("flow", {
        "id":         flow_id,
        "begin":      f"{begin:.2f}",
        "end":        f"{end:.2f}",
        "perHour":    f"{veh_per_hour:.2f}",
        "from":       from_edge,
        "to":         to_edge,
        "departLane": departLane,
        "departSpeed": departSpeed,
    })
    return el


def make_person_flow(
    pf_id: str,
    from_edge: str,
    to_edge: str,
    ped_per_hour: float,
    begin: float,
    end: float,
) -> ET.Element:
    """Build a <personFlow> element with a <personTrip>. Same boundary check."""
    if from_edge not in INFLOW_EDGES:
        raise ValueError(f"personFlow {pf_id}: from-edge {from_edge!r} is NOT an inflow edge")
    if to_edge not in OUTFLOW_EDGES:
        raise ValueError(f"personFlow {pf_id}: to-edge {to_edge!r} is NOT an outflow edge")
    pf = ET.Element("personFlow", {
        "id":      pf_id,
        "begin":   f"{begin:.2f}",
        "end":     f"{end:.2f}",
        "perHour": f"{ped_per_hour:.2f}",
    })
    pt = ET.SubElement(pf, "personTrip", {
        "from": from_edge,
        "to":   to_edge,
    })
    return pf


def generate_symmetric_demand(
    demand: str = "low",
    end_time: float = DEFAULT_END_TIME,
    seed: int = 42,
) -> ET.Element:
    """Generate symmetric demand: every inflow has equal weight, destinations
    chosen uniformly at random over outflows (excluding U-turn back through
    the same TL it entered)."""
    rng = random.Random(seed)
    profile = DEMAND_PROFILES[demand]
    veh_rate = profile["veh_per_hour_per_inflow"]
    ped_rate = profile["ped_per_hour_per_inflow"]

    routes = ET.Element("routes")
    routes.append(ET.Comment(
        f" Generated by generate_routes.py (seed={seed}, demand={demand}). "
        f"Total veh: {veh_rate * len(INFLOW_EDGES)} veh/h, "
        f"total ped: {ped_rate * len(INFLOW_EDGES)} ped/h. "
    ))

    # ── Vehicle flows ────────────────────────────────────────────
    # For each inflow, pick 1 destination, ensuring every outflow is
    # used at least once (round-robin assignment from the candidate
    # pool, shuffled to avoid bias).  Avoids U-turn back through
    # the same TL the vehicle entered.
    shuffled_outflows = list(OUTFLOW_EDGES)
    rng.shuffle(shuffled_outflows)
    used_outflows = set()
    for src in INFLOW_EDGES:
        src_tl = INFLOW_TO_TL[src]
        # Prefer outflows not yet used; among those, only different-TL ones
        priority = [o for o in shuffled_outflows
                    if o not in used_outflows and OUTFLOW_FROM_TL[o] != src_tl]
        if not priority:
            # All same-TL-allowed outflows used; fall back to any different-TL
            priority = [o for o in shuffled_outflows
                        if OUTFLOW_FROM_TL[o] != src_tl]
        dst = priority[0]
        used_outflows.add(dst)
        fid = f"f_{src}_to_{dst}"
        routes.append(make_flow(
            flow_id=fid, from_edge=src, to_edge=dst,
            veh_per_hour=veh_rate, begin=0.0, end=end_time,
        ))

    # ── Pedestrian flows ─────────────────────────────────────────
    # Same logic: each inflow generates one personFlow, with round-robin
    # destination assignment over outflows.
    shuffled_ped = list(OUTFLOW_EDGES)
    rng.shuffle(shuffled_ped)
    used_ped = set()
    for src in INFLOW_EDGES:
        src_tl = INFLOW_TO_TL[src]
        priority = [o for o in shuffled_ped
                    if o not in used_ped and OUTFLOW_FROM_TL[o] != src_tl]
        if not priority:
            priority = [o for o in shuffled_ped
                        if OUTFLOW_FROM_TL[o] != src_tl]
        dst = priority[0]
        used_ped.add(dst)
        pid = f"pf_{src}_to_{dst}"
        routes.append(make_person_flow(
            pf_id=pid, from_edge=src, to_edge=dst,
            ped_per_hour=ped_rate, begin=0.0, end=end_time,
        ))

    return routes


def prettify(el: ET.Element) -> str:
    """Pretty-print an XML tree, with the SUMO routes header."""
    rough = ET.tostring(el, encoding="unicode")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="    ")
    # minidom adds an XML declaration; replace with the standard SUMO one
    lines = pretty.splitlines()
    body = "\n".join(line for line in lines[1:] if line.strip())
    header = ('<?xml version="1.0" encoding="UTF-8"?>\n\n'
              '<!-- generated by generate_routes.py -->\n')
    return header + body + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate 2x2 grid route file")
    parser.add_argument("--out", default="02.rou.xml", help="output path")
    parser.add_argument("--demand", choices=list(DEMAND_PROFILES.keys()),
                        default="low", help="demand level")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--end", type=float, default=DEFAULT_END_TIME,
                        help="episode end time in seconds")
    args = parser.parse_args()

    routes = generate_symmetric_demand(
        demand=args.demand, end_time=args.end, seed=args.seed,
    )
    xml_str = prettify(routes)
    with open(args.out, "w") as f:
        f.write(xml_str)

    # Verify by re-parsing and counting
    tree = ET.parse(args.out)
    n_flow = len(tree.findall("flow"))
    n_pflow = len(tree.findall("personFlow"))
    print(f"[OK] wrote {args.out}")
    print(f"     {n_flow} vehicle flows, {n_pflow} person flows")
    print(f"     demand profile: {args.demand}")
    p = DEMAND_PROFILES[args.demand]
    print(f"     vehicle total: {p['veh_per_hour_per_inflow'] * 8} veh/h")
    print(f"     pedestrian total: {p['ped_per_hour_per_inflow'] * 8} ped/h")


if __name__ == "__main__":
    main()