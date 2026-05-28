"""Dual-level fairness utilities for Fair-TSC.

This module replaces the old multi-constraint Lagrangian code.  It keeps
the inter-intersection sacrifice-gap Theil metric, adds intra-intersection
phase-service fairness, and controls one adaptive fairness weight with a
PID-style controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


def theil_t_index(values: np.ndarray, eps: float = 1e-6) -> float:
    """Return Theil-T over non-negative values."""
    x = np.maximum(np.asarray(values, dtype=np.float64), 0.0) + eps
    if x.size == 0:
        return 0.0
    mean = float(x.mean())
    if mean <= 0.0:
        return 0.0
    ratio = x / mean
    return float(np.mean(ratio * np.log(ratio)))


def theil_t_contributions(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-item Theil terms whose mean equals :func:`theil_t_index`."""
    x = np.maximum(np.asarray(values, dtype=np.float64), 0.0) + eps
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    mean = float(x.mean())
    if mean <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    ratio = x / mean
    return (ratio * np.log(ratio)).astype(np.float32)


def theil_t_batch(values_per_step: np.ndarray, eps: float = 1e-6) -> float:
    """Diagnostic: average per-step Theil over a [T, N] array."""
    arr = np.asarray(values_per_step, dtype=np.float32)
    if arr.ndim == 1:
        return theil_t_index(arr, eps)
    return float(np.mean([theil_t_index(arr[t], eps) for t in range(arr.shape[0])]))


def theil_t_episode(values_per_step: np.ndarray, eps: float = 1e-6) -> float:
    """Theil on per-agent episode means."""
    arr = np.asarray(values_per_step, dtype=np.float32)
    if arr.ndim == 1:
        return theil_t_index(arr, eps)
    return theil_t_index(arr.mean(axis=0), eps)


def compute_sacrifice_gaps(
    buffer,
    ue_critic,
    marl_critic,
    device,
):
    """Compute delta_i(t) = [V^UE(s,i) - V^MARL(s,i)]_+ for a rollout."""
    import torch

    del device  # tensors already live on the right device after compute_gae()
    assert buffer.flat is not None, "call buffer.compute_gae() before sacrifice gaps"
    global_obs = buffer.flat["global_obs"]
    agent_idx = buffer.flat["agent_idx"]
    with torch.no_grad():
        v_ue = ue_critic(global_obs, agent_idx)
        v_marl = marl_critic(global_obs, agent_idx)
    return (v_ue - v_marl).clamp(min=0.0)


def reshape_deltas_to_step_agent(
    delta_flat,
    agent_idx_flat,
    num_agents: int,
) -> np.ndarray:
    """Reshape flat rollout deltas into [T, N] by agent index."""
    batch = int(delta_flat.shape[0])
    if batch % num_agents != 0:
        raise ValueError(f"buffer size {batch} not divisible by num_agents {num_agents}")
    steps = batch // num_agents
    delta_np = delta_flat.detach().cpu().numpy()
    agent_np = agent_idx_flat.detach().cpu().numpy()

    out = np.zeros((steps, num_agents), dtype=np.float32)
    for idx in range(num_agents):
        vals = delta_np[agent_np == idx]
        if len(vals) < steps:
            vals = np.pad(vals, (0, steps - len(vals)), mode="constant")
        out[:, idx] = vals[:steps]
    return out


def compute_inter_fairness(
    delta_agent_mean: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray]:
    """Return network inter-Theil and per-agent contribution terms."""
    contrib = theil_t_contributions(delta_agent_mean, eps=eps)
    return float(contrib.mean()) if contrib.size else 0.0, contrib


def phase_service_theil_from_intervals(
    intervals_by_agent: Mapping[str, Mapping[int, Iterable[float]]],
    agent_ids: Optional[List[str]] = None,
    eps: float = 1e-6,
) -> Tuple[Dict[str, float], float, float]:
    """Compute intra-intersection Theil from phase service intervals.

    Args:
        intervals_by_agent: {agent: {phase_idx: [ell_1, ell_2, ...]}}.
            The caller should already include the full-horizon fallback
            for phases activated fewer than twice.

    Returns:
        per_agent: {agent: T_i^intra}
        mean_intra: mean_i T_i^intra
        max_interval: largest interval observed across all agents/phases
    """
    if agent_ids is None:
        agent_ids = list(intervals_by_agent.keys())

    per_agent: Dict[str, float] = {}
    max_interval = 0.0
    for agent in agent_ids:
        values: List[float] = []
        for phase_intervals in intervals_by_agent.get(agent, {}).values():
            for interval in phase_intervals:
                val = float(interval)
                values.append(val)
                max_interval = max(max_interval, val)
        per_agent[agent] = theil_t_index(np.asarray(values, dtype=np.float32), eps=eps) if values else 0.0

    mean_intra = float(np.mean(list(per_agent.values()))) if per_agent else 0.0
    return per_agent, mean_intra, float(max_interval)


def build_per_agent_fair_cost(
    agent_ids: List[str],
    inter_contrib: np.ndarray,
    intra_by_agent: Mapping[str, float],
    *,
    alpha: float,
    t_inter_0: float,
    t_intra_0: float,
    num_agents: int,
    eps: float = 1e-6,
) -> Tuple[Dict[str, float], float]:
    """Build c_i^fair and the network-level C_fair.

    c_i^fair = alpha * T_i^inter / T_inter_0
             + (1-alpha) * (T_i^intra / N) / T_intra_0

    C_fair = alpha * T_inter / T_inter_0
           + (1-alpha) * T_intra / T_intra_0
    """
    inter_ref = max(float(t_inter_0), eps)
    intra_ref = max(float(t_intra_0), eps)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    costs: Dict[str, float] = {}
    n = max(num_agents, 1)
    for idx, agent in enumerate(agent_ids):
        inter_i = float(inter_contrib[idx]) / n if idx < len(inter_contrib) else 0.0
        intra_i = float(intra_by_agent.get(agent, 0.0))
        costs[agent] = (
            alpha * inter_i / inter_ref
            + (1.0 - alpha) * (intra_i / n) / intra_ref
        )
    t_inter = float(np.mean(inter_contrib)) if len(inter_contrib) else 0.0
    t_intra = float(np.mean([float(intra_by_agent.get(a, 0.0)) for a in agent_ids])) if agent_ids else 0.0
    c_fair = alpha * t_inter / inter_ref + (1.0 - alpha) * t_intra / intra_ref
    return costs, c_fair


def apply_fair_advantage(
    buffer,
    per_agent_cost: Mapping[str, float],
    agent_idx_to_id: Mapping[int, str],
    lambda_fair: float,
):
    """In-place PPO advantage shaping: A_i^fair = A_i - lambda_k c_i^fair."""
    import torch

    assert buffer.flat is not None
    device = buffer.flat["advantage"].device
    agent_idx = buffer.flat["agent_idx"].detach().cpu().numpy()
    costs = np.asarray(
        [float(per_agent_cost.get(agent_idx_to_id[int(idx)], 0.0)) for idx in agent_idx],
        dtype=np.float32,
    )
    cost_t = torch.from_numpy(costs).to(device)
    buffer.flat["advantage"] = buffer.flat["advantage"] - float(lambda_fair) * cost_t


@dataclass
class PIDFairnessController:
    """PID-style controller for one adaptive fairness weight."""

    target: float
    kp: float
    ki: float
    kd: float
    lambda_max: float
    integral_max: float
    ema_beta: float = 0.9
    lambda_value: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0
    cost_ema: Optional[float] = None

    def update(self, cost: float) -> Dict[str, float]:
        raw_cost = float(cost)
        if self.cost_ema is None:
            self.cost_ema = raw_cost
        else:
            self.cost_ema = self.ema_beta * self.cost_ema + (1.0 - self.ema_beta) * raw_cost

        error = float(self.cost_ema - self.target)
        self.integral = float(np.clip(self.integral + error, -self.integral_max, self.integral_max))
        derivative = error - self.prev_error
        self.prev_error = error

        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.lambda_value = float(np.clip(control, 0.0, self.lambda_max))
        return self.stats(raw_cost=raw_cost, derivative=derivative)

    def stats(self, raw_cost: Optional[float] = None, derivative: float = 0.0) -> Dict[str, float]:
        return {
            "C_fair_raw": float(0.0 if raw_cost is None else raw_cost),
            "C_fair_ema": float(0.0 if self.cost_ema is None else self.cost_ema),
            "fair_target": float(self.target),
            "pid_error": float(0.0 if self.cost_ema is None else self.cost_ema - self.target),
            "pid_integral": float(self.integral),
            "pid_derivative": float(derivative),
            "lambda_fair": float(self.lambda_value),
        }

    def state_dict(self) -> Dict[str, float]:
        return {
            "lambda_value": self.lambda_value,
            "integral": self.integral,
            "prev_error": self.prev_error,
            "cost_ema": self.cost_ema,
        }

    def load_state_dict(self, state: Mapping[str, float]):
        self.lambda_value = float(state.get("lambda_value", 0.0))
        self.integral = float(state.get("integral", 0.0))
        self.prev_error = float(state.get("prev_error", 0.0))
        cost_ema = state.get("cost_ema", None)
        self.cost_ema = None if cost_ema is None else float(cost_ema)
