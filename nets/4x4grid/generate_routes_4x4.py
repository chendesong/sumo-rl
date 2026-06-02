"""
Generate route files for the 4x4 grid net.

Style follows the original 2x2 generate_routes.py:
- 16 vehicle <flow>s (one per entry edge -> randomly chosen exit edge)
- 16 pedestrian <personFlow>s (one per entry edge -> randomly chosen exit edge)
- begin=0, end=3600 (1-hour horizon)
- Same OD pairs across the three levels (only base perHour scales)

Per-entry heterogeneity (NEW):
    Entries are split into two tiers and weighted to keep the network total
    identical to a uniform run, while creating intrinsic demand asymmetry
    across intersections. This gives the IPPO baseline a non-trivial Theil
    index even before MARL coordination kicks in, which is what the
    inter-intersection fairness motivation in the paper needs.

      Arterial (8 entries, weight 1.5):
          - East-West arterials through middle rows: -h21 -h31  h25 h35
          - North-South arterials through middle cols: -v21 -v31  v25 v35
      Local    (8 entries, weight 0.5):
          - All corner entries on the outer rows / cols:
            -h11 -h41  h15 h45  -v11 -v41  v15 v45

    Weighted sum = 8 * 1.5 + 8 * 0.5 = 16, so total network demand equals
    16 * base_perHour (same as the previous uniform version).

Base perHour per entry edge (BEFORE the tier weight is applied):
                low      medium     high
    vehicles    250      500        750    veh/h
    pedestrians 60       120        180    ped/h

Actual per-entry perHour = base * weight (1.5 arterial / 0.5 local).
At "high":  arterial = 750 * 1.5 = 1125 veh/h  (below ~1296 veh/h per-direction
            green-time capacity for a 2-lane edge with 37s green in ~104s cycle),
            local    = 750 * 0.5 = 375 veh/h.
Network total per level is unchanged from the uniform setting (16 * base).

Ultra-stress demand:
    The "ultra_stress" file is intentionally not total-demand preserving.  It
    concentrates demand on the middle two rows and columns, while keeping all
    outer entries alive at a low rate.  The OD pairs are deterministic
    corridor/cross-corridor trips through the central 2x2 region, so multiple
    intersections can become bottlenecks instead of only one or two extremes.

Curriculum demand:
    The curriculum files keep the ultra-stress OD pairs fixed and only ramp
    demand intensity inside one episode. This keeps the bottleneck locations
    consistent across stages, so the policy learns the same corridors under
    increasing load:

      curriculum_lmh: low-like -> medium-like -> high-like
      curriculum_lmhu: low-like -> medium-like -> high-like -> ultra-like,
                       with a stronger pedestrian stream
      curriculum_mhu: medium-like -> high-like -> ultra-stress
"""

import argparse
import os
import random

GRID = 4
SIM_BEGIN = 0.0
SIM_END = 3600.0  # 1 hour
PED_TYPE_ID = "ped_wait_hazard"

DEMAND = {
    "low":    {"veh": 250,  "ped": 60},
    "medium": {"veh": 500,  "ped": 120},
    "high":   {"veh": 750,  "ped": 180},
    "ultra_stress": {"veh": 650, "ped": 65},
}

CURRICULUMS = {
    "curriculum_lmh": [
        ("low", 0.0, 1200.0, 250.0, 25.0),
        ("medium", 1200.0, 2400.0, 350.0, 35.0),
        ("high", 2400.0, 3600.0, 450.0, 45.0),
    ],
    "curriculum_lmhu": [
        ("low", 0.0, 900.0, 250.0, 40.0),
        ("medium", 900.0, 1800.0, 350.0, 55.0),
        ("high", 1800.0, 2700.0, 450.0, 70.0),
        ("ultra_stress", 2700.0, 3600.0, 550.0, 85.0),
    ],
    "curriculum_mhu": [
        ("medium", 0.0, 1200.0, 450.0, 45.0),
        ("high", 1200.0, 2400.0, 550.0, 55.0),
        ("ultra_stress", 2400.0, 3600.0, 650.0, 65.0),
    ],
}

# ---------------------------------------------------------------------------
# Per-entry tier weights
# ---------------------------------------------------------------------------
ARTERIAL_WEIGHT = 1.5
LOCAL_WEIGHT    = 0.5
ULTRA_ARTERIAL_WEIGHT = 1.46
ULTRA_LOCAL_WEIGHT = 0.46

# Arterial entries: middle 2 rows (r=2,3) for E-W flow + middle 2 cols (c=2,3) for N-S flow
ARTERIAL_ENTRIES = set()
for r in (2, 3):
    ARTERIAL_ENTRIES.add(f"-h{r}1")  # West entry
    ARTERIAL_ENTRIES.add(f"h{r}5")   # East entry
for c in (2, 3):
    ARTERIAL_ENTRIES.add(f"-v{c}1")  # North entry
    ARTERIAL_ENTRIES.add(f"v{c}5")   # South entry


def entry_weight(entry_edge: str, level: str = "high") -> float:
    if level == "ultra_stress":
        return ULTRA_ARTERIAL_WEIGHT if entry_edge in ARTERIAL_ENTRIES else ULTRA_LOCAL_WEIGHT
    return ARTERIAL_WEIGHT if entry_edge in ARTERIAL_ENTRIES else LOCAL_WEIGHT


# ---------------------------------------------------------------------------
# Entry / exit edges (must match the convention in generate_4x4_net.py)
# ---------------------------------------------------------------------------
def get_entries_exits():
    """Return (entries, exits) where each item is (side, edge_id)."""
    entries, exits = [], []
    for r in range(1, GRID + 1):
        entries.append(("W", f"-h{r}1"))
        exits.append(("W",   f"h{r}1"))
    for r in range(1, GRID + 1):
        entries.append(("E", f"h{r}5"))
        exits.append(("E",   f"-h{r}5"))
    for c in range(1, GRID + 1):
        entries.append(("N", f"-v{c}1"))
        exits.append(("N",   f"v{c}1"))
    for c in range(1, GRID + 1):
        entries.append(("S", f"v{c}5"))
        exits.append(("S",   f"-v{c}5"))
    return entries, exits


def reverse_edge(eid: str) -> str:
    """`-h11` <-> `h11`,  `v25` <-> `-v25`."""
    return eid[1:] if eid.startswith("-") else "-" + eid


def sample_od(entries, exits, rng):
    """For each entry, pick a random exit avoiding the self-reverse exit."""
    exit_edges = [e for _, e in exits]
    od = []
    for _, e_in in entries:
        cand = [e for e in exit_edges if e != reverse_edge(e_in)]
        e_out = rng.choice(cand)
        od.append((e_in, e_out))
    return od


def ultra_stress_od():
    """Deterministic OD pairs that stress non-central corner-side bottlenecks.

    The heavy corridors are moved away from the central 2x2 to avoid whole-grid
    lockup under fixed-time control.  The intended high-pressure intersections
    are J44, J42, J11, and J13, with feeder flows still present elsewhere.
    """
    veh_od = [
        ("-h11", "-h15"),  # west row 1 -> east row 1, stresses J11/J13
        ("-h21", "-h15"),  # west feeder -> row 1 corridor
        ("-h31", "-h45"),  # west feeder -> row 4 corridor
        ("-h41", "-h45"),  # west row 4 -> east row 4, stresses J42/J44
        ("h15", "h11"),    # east row 1 -> west row 1
        ("h25", "h11"),    # east feeder -> row 1 corridor
        ("h35", "h41"),    # east feeder -> row 4 corridor
        ("h45", "h41"),    # east row 4 -> west row 4
        ("-v11", "-v15"),  # north col 1 -> south col 1, stresses J11
        ("-v21", "-v45"),  # north feeder -> col 4 corridor
        ("-v31", "-v15"),  # north col 3 -> south col 1/3 side
        ("-v41", "-v45"),  # north col 4 -> south col 4, stresses J44
        ("v15", "v11"),    # south col 1 -> north col 1
        ("v25", "v41"),    # south feeder -> col 4 corridor
        ("v35", "v11"),    # south col 3 -> north col 1/3 side
        ("v45", "v41"),    # south col 4 -> north col 4
    ]

    ped_od = [
        ("-h11", "-v15"),
        ("-h21", "-v15"),
        ("-h31", "-v45"),
        ("-h41", "-v45"),
        ("h15", "v11"),
        ("h25", "v11"),
        ("h35", "v41"),
        ("h45", "v41"),
        ("-v11", "-h15"),
        ("-v21", "-h45"),
        ("-v31", "-h15"),
        ("-v41", "-h45"),
        ("v15", "h11"),
        ("v25", "h41"),
        ("v35", "h11"),
        ("v45", "h41"),
    ]
    return veh_od, ped_od


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_rou(out_dir: str, level: str, veh_od, ped_od) -> str:
    base_v = DEMAND[level]["veh"]
    base_p = DEMAND[level]["ped"]
    total_v = sum(base_v * entry_weight(e_in, level) for e_in, _ in veh_od)
    total_p = sum(base_p * entry_weight(e_in, level) for e_in, _ in ped_od)

    path = os.path.join(out_dir, f"4x4_{level}.rou.xml")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '',
        '<!-- generated by generate_routes_4x4.py -->',
        '<routes>',
        f'    <vType id="{PED_TYPE_ID}" vClass="pedestrian" '
        'impatience="0.0" jmIgnoreFoeProb="0.0" jmDriveAfterRedTime="-1"/>',
        f'    <!-- demand={level}: arterial weight {entry_weight(next(iter(ARTERIAL_ENTRIES)), level)}, '
        f'local weight {entry_weight("__local__", level)}, total {int(total_v)} veh/h + '
        f'{int(total_p)} ped/h -->',
    ]

    for e_in, e_out in veh_od:
        per_v = base_v * entry_weight(e_in, level)
        lines.append(
            f'    <flow id="f_{e_in}_to_{e_out}" '
            f'begin="{SIM_BEGIN:.2f}" end="{SIM_END:.2f}" '
            f'perHour="{per_v:.2f}" from="{e_in}" to="{e_out}" '
            f'departLane="best" departSpeed="max"/>'
        )

    for e_in, e_out in ped_od:
        per_p = base_p * entry_weight(e_in, level)
        lines.append(
            f'    <personFlow id="pf_{e_in}_to_{e_out}" '
            f'type="{PED_TYPE_ID}" '
            f'begin="{SIM_BEGIN:.2f}" end="{SIM_END:.2f}" '
            f'perHour="{per_p:.2f}">'
        )
        lines.append(f'        <personTrip from="{e_in}" to="{e_out}"/>')
        lines.append('    </personFlow>')

    lines.append('</routes>')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}  ({int(total_v)} veh/h + {int(total_p)} ped/h)")
    return path


def write_curriculum_rou(out_dir: str, level: str, veh_od, ped_od) -> str:
    """Write one route file with fixed OD pairs and staged demand intensity."""
    segments = CURRICULUMS[level]
    path = os.path.join(out_dir, f"4x4_{level}.rou.xml")
    total_v = 0.0
    total_p = 0.0
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '',
        '<!-- generated by generate_routes_4x4.py -->',
        '<routes>',
        f'    <vType id="{PED_TYPE_ID}" vClass="pedestrian" '
        'impatience="0.0" jmIgnoreFoeProb="0.0" jmDriveAfterRedTime="-1"/>',
        f'    <!-- demand={level}: same ultra-stress OD, staged '
        + ', '.join(f'{int(begin)}-{int(end)} {label}' for label, begin, end, _base_v, _base_p in segments)
        + ' -->',
    ]

    for label, begin, end, base_v, base_p in segments:
        duration_h = max(end - begin, 0.0) / 3600.0
        seg_v = sum(base_v * entry_weight(e_in, "ultra_stress") for e_in, _ in veh_od)
        seg_p = sum(base_p * entry_weight(e_in, "ultra_stress") for e_in, _ in ped_od)
        total_v += seg_v * duration_h
        total_p += seg_p * duration_h
        lines.append(
            f'    <!-- segment={label}: begin={begin:.0f}, end={end:.0f}, '
            f'{int(seg_v)} veh/h + {int(seg_p)} ped/h -->'
        )

        for e_in, e_out in veh_od:
            per_v = base_v * entry_weight(e_in, "ultra_stress")
            lines.append(
                f'    <flow id="f_{label}_{e_in}_to_{e_out}" '
                f'begin="{begin:.2f}" end="{end:.2f}" '
                f'perHour="{per_v:.2f}" from="{e_in}" to="{e_out}" '
                f'departLane="best" departSpeed="max"/>'
            )

        for e_in, e_out in ped_od:
            per_p = base_p * entry_weight(e_in, "ultra_stress")
            lines.append(
                f'    <personFlow id="pf_{label}_{e_in}_to_{e_out}" '
                f'type="{PED_TYPE_ID}" '
                f'begin="{begin:.2f}" end="{end:.2f}" '
                f'perHour="{per_p:.2f}">'
            )
            lines.append(f'        <personTrip from="{e_in}" to="{e_out}"/>')
            lines.append('    </personFlow>')

    lines.append(
        f'    <!-- expected total over 3600s: {int(total_v)} vehicles + {int(total_p)} pedestrians -->'
    )
    lines.append('</routes>')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}  ({int(total_v)} vehicles + {int(total_p)} pedestrians over 3600s)")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir: str = ".", seed: int = 42, levels=None):
    if levels is None:
        levels = ("low", "medium", "high", "ultra_stress")
    os.makedirs(out_dir, exist_ok=True)
    entries, exits = get_entries_exits()
    rng = random.Random(seed)
    veh_od = sample_od(entries, exits, rng)
    ped_od = sample_od(entries, exits, rng)

    print(f"OD seed={seed}")
    print(f"Arterial entries (weight {ARTERIAL_WEIGHT}): "
          f"{sorted(ARTERIAL_ENTRIES)}")
    local_entries = [e for _, e in entries if e not in ARTERIAL_ENTRIES]
    print(f"Local    entries (weight {LOCAL_WEIGHT}): {sorted(local_entries)}")
    print()

    allowed = sorted([*DEMAND.keys(), *CURRICULUMS.keys()])
    for level in levels:
        if level not in allowed:
            raise ValueError(f"Unknown level {level!r}; choose from {allowed}")
        if level == "ultra_stress":
            level_veh_od, level_ped_od = ultra_stress_od()
        elif level in CURRICULUMS:
            level_veh_od, level_ped_od = ultra_stress_od()
            write_curriculum_rou(out_dir, level, level_veh_od, level_ped_od)
            continue
        else:
            level_veh_od, level_ped_od = veh_od, ped_od
        write_rou(out_dir, level, level_veh_od, level_ped_od)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=[
            "low",
            "medium",
            "high",
            "ultra_stress",
            "curriculum_lmh",
            "curriculum_lmhu",
            "curriculum_mhu",
        ],
        help=f"Demand levels to generate. Choices: {', '.join([*DEMAND.keys(), *CURRICULUMS.keys()])}",
    )
    args = parser.parse_args()
    main(args.out_dir, args.seed, args.levels)
