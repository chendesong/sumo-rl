"""
Generate a 4x4 SUMO grid network matching the original 2x2 logic.

What this produces:
- 4x4.nod.xml : 16 traffic-light intersections + 16 dead-ends
- 4x4.edg.xml : 80 vehicle edges (2 lanes + sidewalk via sidewalkWidth)
- 4x4.tll.xml : 16 traffic-light programs, all cloned from original TL "1"
- 4x4.net.xml : compiled by netconvert (auto crossings + sidewalks)

Naming convention (extends the original h{r}{s} / v{c}{s} scheme):
    TL nodes      : J{r}{c}   for r,c in 1..4  (r=1 top, r=4 bottom)
    Dead-ends     : N{c} S{c} W{r} E{r}
    Horizontal    : h{r}{s}   westbound,  s=1..5 (W -> J1 -> J2 -> J3 -> J4 -> E)
                    -h{r}{s}  eastbound
    Vertical      : v{c}{s}   northbound, s=1 nearest N,  s=5 nearest S
                    -v{c}{s}  southbound

Direction matches the original 2x2: h11 = J1->W0 (westbound) and v11 = J1->N17 (northbound, topmost segment).

Run on the same machine where SUMO is installed (so `netconvert` is on PATH).
Tested target: SUMO 1.20+.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Network geometry
# ---------------------------------------------------------------------------
GRID = 4                # 4x4 traffic-light intersections
SPACING = 150           # metres between adjacent intersections
LANE_SPEED = 13.89      # m/s (~50 km/h), same as original 2x2
SIDEWALK_WIDTH = 2.0    # metres, same as original 2x2 (_0 lane width)
NUM_VEH_LANES = 2       # vehicle lanes per direction (NOT counting sidewalk)
PRIORITY = 1

# Origin chosen so the existing 2x2 (junctions 1,2,5,6 at 300,450,600,450)
# would lie in the top-left 2x2 block of this 4x4 grid.
# Row r=1 -> y = 750; r=4 -> y = 300.   Col c=1 -> x = 300; c=4 -> x = 750.
def tl_x(c: int) -> int: return 300 + (c - 1) * SPACING
def tl_y(r: int) -> int: return 750 - (r - 1) * SPACING

# Dead-end positions (one SPACING beyond the outermost TL on each side)
NORTH_Y = tl_y(1) + SPACING   # 900
SOUTH_Y = tl_y(GRID) - SPACING  # 150
WEST_X  = tl_x(1) - SPACING   # 150
EAST_X  = tl_x(GRID) + SPACING  # 900


# ---------------------------------------------------------------------------
# Original tlLogic from 2x2 net (TL id "1") -- copied verbatim
# ---------------------------------------------------------------------------
ORIGINAL_PHASES = [
    ("37", "gGGgrrrrgGGgrrrrrrrr"),  # NS vehicle green
    ("37", "rrrrgGGgrrrrgGGgrrrr"),  # EW vehicle green
    ("15", "rrrrrrrrrrrrrrrrGGGG"),  # exclusive pedestrian green
]


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------
def write_nodes(path: str) -> None:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    # Traffic-light intersections
    for r in range(1, GRID + 1):
        for c in range(1, GRID + 1):
            lines.append(
                f'  <node id="J{r}{c}" x="{tl_x(c)}" y="{tl_y(r)}" '
                f'type="traffic_light"/>'
            )
    # North & south dead-ends (per column)
    for c in range(1, GRID + 1):
        lines.append(f'  <node id="N{c}" x="{tl_x(c)}" y="{NORTH_Y}" type="dead_end"/>')
        lines.append(f'  <node id="S{c}" x="{tl_x(c)}" y="{SOUTH_Y}" type="dead_end"/>')
    # West & east dead-ends (per row)
    for r in range(1, GRID + 1):
        lines.append(f'  <node id="W{r}" x="{WEST_X}" y="{tl_y(r)}" type="dead_end"/>')
        lines.append(f'  <node id="E{r}" x="{EAST_X}" y="{tl_y(r)}" type="dead_end"/>')
    lines.append("</nodes>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_edges(path: str) -> None:
    """
    Each row has 5 horizontal segments (W -> J1 -> J2 -> J3 -> J4 -> E).
    Each column has 5 vertical segments (S -> J4 -> J3 -> J2 -> J1 -> N).
    Each segment is bidirectional -> two edges (forward + reversed with '-').

    Direction convention matches the original 2x2:
        h{r}{s}  goes "westward" (toward smaller x):
                   - h{r}1 : J{r}1  -> W{r}        (matches original h11: 1->0)
                   - h{r}2 : J{r}2  -> J{r}1
                   - ...
                   - h{r}5 : E{r}   -> J{r}4
        -h{r}{s} goes "eastward" (toward larger x).

        v{c}{s}  goes "northward" (toward larger y):
                   - v{c}1 : J1{c}  -> N{c}        (matches original v11: 1->17)
                   - v{c}2 : J2{c}  -> J1{c}
                   - ...
                   - v{c}5 : S{c}   -> J4{c}
        -v{c}{s} goes "southward" (toward smaller y).
    """
    edges = []

    # Horizontal edges
    for r in range(1, GRID + 1):
        nodes_west_to_east = [f"W{r}"] + [f"J{r}{c}" for c in range(1, GRID + 1)] + [f"E{r}"]
        for s in range(1, GRID + 2):
            left_node  = nodes_west_to_east[s - 1]
            right_node = nodes_west_to_east[s]
            edges.append((f"h{r}{s}",  right_node, left_node))   # westbound
            edges.append((f"-h{r}{s}", left_node,  right_node))  # eastbound

    # Vertical edges. Segment 1 is the TOPMOST (between J1{c} and N{c}),
    # segment 5 is the BOTTOMMOST (between J4{c} and S{c}), matching the
    # original 2x2 where v11 = J1 -> N17.
    for c in range(1, GRID + 1):
        nodes_north_to_south = [f"N{c}"] + [f"J{r}{c}" for r in range(1, GRID + 1)] + [f"S{c}"]
        for s in range(1, GRID + 2):
            top_node    = nodes_north_to_south[s - 1]
            bottom_node = nodes_north_to_south[s]
            edges.append((f"v{c}{s}",  bottom_node, top_node))   # northbound (toward larger y)
            edges.append((f"-v{c}{s}", top_node,    bottom_node))

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for eid, frm, to in edges:
        # numLanes = 2 vehicle lanes; sidewalkWidth makes netconvert add a sidewalk lane (index 0)
        lines.append(
            f'  <edge id="{eid}" from="{frm}" to="{to}" '
            f'priority="{PRIORITY}" numLanes="{NUM_VEH_LANES}" '
            f'speed="{LANE_SPEED}" sidewalkWidth="{SIDEWALK_WIDTH}"/>'
        )
    lines.append("</edges>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_tllogic(path: str) -> None:
    """All 16 TLs clone the original TL '1' phase pattern."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<additional>",
    ]
    for r in range(1, GRID + 1):
        for c in range(1, GRID + 1):
            lines.append(
                f'  <tlLogic id="J{r}{c}" type="static" programID="0" offset="0">'
            )
            for dur, state in ORIGINAL_PHASES:
                lines.append(f'    <phase duration="{dur}" state="{state}"/>')
            lines.append("  </tlLogic>")
    lines.append("</additional>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# netconvert call + sanity check
# ---------------------------------------------------------------------------
def run_netconvert(nod: str, edg: str, tll: str, out_net: str) -> int:
    cmd = [
        "netconvert",
        "--node-files", nod,
        "--edge-files", edg,
        "--tllogic-files", tll,
        "--crossings.guess",                  # auto crosswalks at all TLs
        "--no-turnarounds", "true",           # match original (no U-turns)
        "--no-internal-links", "false",
        "--offset.disable-normalization", "true",
        "--output-file", out_net,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def sanity_check(net_path: str) -> None:
    """Confirm every TL still has 20-character phase states (16 veh + 4 ped)."""
    tree = ET.parse(net_path)
    root = tree.getroot()
    tls = root.findall("tlLogic")
    print(f"\nFound {len(tls)} tlLogic blocks in {net_path}")
    bad = []
    for tl in tls:
        for phase in tl.findall("phase"):
            state = phase.get("state", "")
            if len(state) != 20:
                bad.append((tl.get("id"), len(state), state))
    if bad:
        print("WARNING: some phase states are not 20 characters:")
        for tlid, n, s in bad[:5]:
            print(f"  TL {tlid}: len={n}  state={s!r}")
        print(
            "If lengths differ from 20, the connection ordering produced by "
            "netconvert does not match the original 2x2. Open the .net.xml in "
            "netedit, copy a TL's auto-generated phase state, and regenerate."
        )
    else:
        print("All TLs have 20-character phase states (matches original 2x2).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir: str = ".") -> int:
    os.makedirs(out_dir, exist_ok=True)
    nod = os.path.join(out_dir, "4x4.nod.xml")
    edg = os.path.join(out_dir, "4x4.edg.xml")
    tll = os.path.join(out_dir, "4x4.tll.xml")
    net = os.path.join(out_dir, "4x4.net.xml")

    print(f"Writing {nod}")
    write_nodes(nod)
    print(f"Writing {edg}")
    write_edges(edg)
    print(f"Writing {tll}")
    write_tllogic(tll)

    rc = run_netconvert(nod, edg, tll, net)
    if rc != 0:
        print(f"netconvert returned {rc} -- check the error above.")
        return rc
    sanity_check(net)
    return 0


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(out_dir))