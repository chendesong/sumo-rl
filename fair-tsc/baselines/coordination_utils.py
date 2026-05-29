"""Small helpers for coordination-oriented TSC baselines."""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


def grid_coord(agent_id: str) -> Tuple[int, int] | None:
    """Parse common grid ids such as J11, J23, tls_3_4."""
    nums = re.findall(r"\d+", str(agent_id))
    if not nums:
        return None
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    token = nums[-1]
    if len(token) >= 2:
        return int(token[-2]), int(token[-1])
    return None


def build_neighbor_map(agent_ids: List[str], max_neighbors: int = 4) -> Dict[str, List[str]]:
    """Return nearest grid-neighbor ids for each agent.

    The 4x4 grid ids parse as J11..J44. For non-grid ids, fall back to
    adjacent sorted ids so the baselines remain runnable on other maps.
    """
    coords = {a: grid_coord(a) for a in agent_ids}
    if all(c is not None for c in coords.values()):
        out = {}
        for a in agent_ids:
            ar, ac = coords[a]
            adjacent = []
            fallback = []
            for b in agent_ids:
                if a == b:
                    continue
                br, bc = coords[b]
                dist = abs(ar - br) + abs(ac - bc)
                if dist <= 0:
                    continue
                if dist == 1:
                    adjacent.append(b)
                fallback.append((dist, b))
            out[a] = sorted(adjacent)[:max_neighbors]
            if not out[a]:
                out[a] = [b for _, b in sorted(fallback)[:max_neighbors]]
        return out

    out = {}
    n = len(agent_ids)
    for i, a in enumerate(agent_ids):
        candidates = []
        if i > 0:
            candidates.append(agent_ids[i - 1])
        if i + 1 < n:
            candidates.append(agent_ids[i + 1])
        out[a] = candidates[:max_neighbors]
    return out


def graph_distances(agent_ids: List[str], neighbor_map: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Unweighted shortest-path distances over the neighbor map."""
    distances: Dict[str, Dict[str, int]] = {}
    for src in agent_ids:
        dist = {src: 0}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nxt in neighbor_map.get(cur, []):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        distances[src] = dist
    return distances


def spatially_discounted_rewards(
    rewards: Dict[str, float],
    agent_ids: List[str],
    distances: Dict[str, Dict[str, int]],
    gamma: float = 0.9,
) -> Dict[str, float]:
    """MA2C-style reward sharing with spatial discount."""
    out = {}
    for a in agent_ids:
        total = 0.0
        norm = 0.0
        dist_a = distances.get(a, {})
        for b in agent_ids:
            if b not in dist_a:
                continue
            w = float(gamma) ** float(dist_a[b])
            total += w * float(rewards.get(b, 0.0))
            norm += w
        out[a] = total / max(norm, 1e-8)
    return out


def mean_or_zeros(values: Iterable[np.ndarray], shape_like: np.ndarray) -> np.ndarray:
    values = list(values)
    if not values:
        return np.zeros_like(shape_like, dtype=np.float32)
    return np.mean(np.stack(values), axis=0).astype(np.float32)
