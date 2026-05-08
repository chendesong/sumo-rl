"""
Lagrangian multipliers + Theil index for Fair-TSC.

Implements paper §II.E + §II.D:
  - Theil-T index over sacrifice gaps:                 Eq. (18)
  - Sacrifice gap δ_i = [V^UE(s,i) - V^MARL(s,i)]_+ :  Eq. (17)
  - Fair advantage A^fair_i = A_i - λ^p C_p - λ^s C_s - (µ/N) δ_i : Eq. (28)
  - Dual updates for λ^p_i, λ^s_i, µ:                  Eq. (30), (31), (32)
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════════
# Theil-T index — paper Eq. (18)
# ════════════════════════════════════════════════════════════════════

def theil_t_index(deltas: np.ndarray, eps: float = 1e-6) -> float:
    """Theil-T index over per-agent sacrifice gaps.

    T = (1/N) Σ_i (δ̃_i / δ̄) · ln(δ̃_i / δ̄)

    where δ̃_i = δ_i + eps  (ensure positivity for log)
          δ̄   = mean(δ̃)

    Args:
        deltas: array shape [N], non-negative sacrifice gaps for one timestep
        eps:    smoothing constant (paper uses 1e-6)

    Returns:
        T ≥ 0.  T = 0 ⇔ all gaps equal.
    """
    d = np.maximum(deltas, 0.0) + eps
    mean = d.mean()
    if mean <= 0:
        return 0.0
    ratio = d / mean
    return float((ratio * np.log(ratio)).mean())


def theil_t_batch(deltas_per_step: np.ndarray, eps: float = 1e-6) -> float:
    """Average Theil-T over a batch of timesteps.

    Args:
        deltas_per_step: shape [T, N]  (T timesteps × N agents)

    Returns:
        T̄_B = (1/T) Σ_t T(t)   — paper Eq. (32) batch-averaged
    """
    if deltas_per_step.ndim == 1:
        return theil_t_index(deltas_per_step, eps)
    T_vals = [theil_t_index(deltas_per_step[t], eps) for t in range(deltas_per_step.shape[0])]
    return float(np.mean(T_vals))


# ════════════════════════════════════════════════════════════════════
# Sacrifice gap — paper Eq. (17)
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_sacrifice_gaps(
    buffer,
    ue_critic: nn.Module,
    marl_critic: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute δ_i(t) = [V^UE(s,i) - V^MARL(s,i)]_+ for every (t, i) in buffer.

    Args:
        buffer:      RolloutBuffer with compute_gae() already called (uses .flat)
        ue_critic:   frozen V^UE network
        marl_critic: V^MARL network (used for difference; gradients NOT propagated here)

    Returns:
        deltas: float tensor [B] aligned with buffer.flat["agent_idx"]
    """
    assert buffer.flat is not None, "must call buffer.compute_gae() before compute_sacrifice_gaps()"

    g = buffer.flat["global_obs"]
    a = buffer.flat["agent_idx"]

    v_ue   = ue_critic(g, a)
    v_marl = marl_critic(g, a)

    delta = (v_ue - v_marl).clamp(min=0.0)
    return delta


def reshape_deltas_to_step_agent(
    delta_flat: torch.Tensor,
    agent_idx_flat: torch.Tensor,
    num_agents: int,
) -> np.ndarray:
    """Reshape flat δ into [T, N] for Theil-T computation.

    Buffer layout: agents are concatenated as [agent_0_traj, agent_1_traj, ...].
    Each agent has the same trajectory length T = total / N.
    """
    B = delta_flat.shape[0]
    assert B % num_agents == 0, f"buffer size {B} not divisible by num_agents {num_agents}"
    T = B // num_agents

    delta_np = delta_flat.detach().cpu().numpy()
    agent_idx_np = agent_idx_flat.detach().cpu().numpy()

    # Build [T, N] by sorting: for each (t, n), find the entry with that index
    out = np.zeros((T, num_agents), dtype=np.float32)
    # The buffer stores agent_0's T transitions first (order [0,0,...,0, 1,1,...,1, ...])
    # so we can simply reshape per-agent
    for n in range(num_agents):
        mask = agent_idx_np == n
        agent_deltas = delta_np[mask]
        if len(agent_deltas) != T:
            # Trajectory lengths differ — fall back to truncating
            agent_deltas = agent_deltas[:T]
            if len(agent_deltas) < T:
                # Pad with zero
                agent_deltas = np.concatenate([agent_deltas, np.zeros(T - len(agent_deltas), dtype=np.float32)])
        out[:, n] = agent_deltas
    return out


# ════════════════════════════════════════════════════════════════════
# LagrangianMultipliers — paper Eq. (30), (31), (32)
# ════════════════════════════════════════════════════════════════════

class LagrangianMultipliers:
    """Maintains and updates dual variables: λ^p_i, λ^s_i (per-agent), µ (network-level).

    Paper §III.C.3 specifies a TWO-TIMESCALE update: η_λ, η_µ ≪ η_θ, η_φ.
    All three are projected to be non-negative via [·]_+.

    Initial values: all zero (paper Algorithm 1 line 1).
    """

    def __init__(
        self,
        agent_ids: List[str],
        d_p: float,
        d_s: float,
        t_max: float,
        eta_lambda: float,
        eta_mu: float,
    ):
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.d_p = d_p
        self.d_s = d_s
        self.t_max = t_max
        self.eta_lambda = eta_lambda
        self.eta_mu = eta_mu

        # All multipliers init to 0 (paper line 1)
        self.lambda_p: Dict[str, float] = {a: 0.0 for a in agent_ids}
        self.lambda_s: Dict[str, float] = {a: 0.0 for a in agent_ids}
        self.mu: float = 0.0

    # ─────────────────────────────────────────────────────────────────
    # Dual updates (slow timescale) — paper Eq. (30), (31), (32)
    # ─────────────────────────────────────────────────────────────────

    def update_intra(self, mean_c_p: Dict[str, float], mean_c_s: Dict[str, float]):
        """Update λ^p_i, λ^s_i with batch-averaged constraint costs.

        Args:
            mean_c_p: {agent_id: (1/|B|) Σ_t C^p_i(t)}  (paper Eq. 30)
            mean_c_s: {agent_id: (1/|B|) Σ_t C^s_i(t)}  (paper Eq. 31)
        """
        for a in self.agent_ids:
            grad_p = mean_c_p.get(a, 0.0) - self.d_p
            grad_s = mean_c_s.get(a, 0.0) - self.d_s
            self.lambda_p[a] = max(0.0, self.lambda_p[a] + self.eta_lambda * grad_p)
            self.lambda_s[a] = max(0.0, self.lambda_s[a] + self.eta_lambda * grad_s)

    def update_inter(self, theil_avg: float):
        """Update µ with batch-averaged Theil index.  Paper Eq. (32)."""
        grad = theil_avg - self.t_max
        self.mu = max(0.0, self.mu + self.eta_mu * grad)

    # ─────────────────────────────────────────────────────────────────
    # Apply to advantage — paper Eq. (28)
    # ─────────────────────────────────────────────────────────────────

    def apply_fair_advantage(
        self,
        buffer,
        deltas: torch.Tensor,
        agent_idx_to_id: Dict[int, str],
    ):
        """In-place: replace buffer.flat['advantage'] with A^fair (paper Eq. 28).

        A^fair_i = A_i - λ^p_i C^p_i - λ^s_i C^s_i - (µ/N) δ_i

        Args:
            buffer:          RolloutBuffer with compute_gae() done
            deltas:          tensor [B] from compute_sacrifice_gaps()
            agent_idx_to_id: {0: agent_id, 1: agent_id, ...}
        """
        assert buffer.flat is not None
        device = buffer.flat["advantage"].device

        # Build per-transition λ^p, λ^s tensors
        agent_idx = buffer.flat["agent_idx"].cpu().numpy()
        lam_p = np.array([self.lambda_p[agent_idx_to_id[i]] for i in agent_idx], dtype=np.float32)
        lam_s = np.array([self.lambda_s[agent_idx_to_id[i]] for i in agent_idx], dtype=np.float32)
        lam_p_t = torch.from_numpy(lam_p).to(device)
        lam_s_t = torch.from_numpy(lam_s).to(device)

        c_p = buffer.flat["c_p"]
        c_s = buffer.flat["c_s"]

        adv = buffer.flat["advantage"]
        adv_fair = adv - lam_p_t * c_p - lam_s_t * c_s - (self.mu / self.num_agents) * deltas

        buffer.flat["advantage"] = adv_fair

    # ─────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, float]:
        return {
            "lambda_p_mean": float(np.mean(list(self.lambda_p.values()))),
            "lambda_p_max":  float(np.max(list(self.lambda_p.values()))),
            "lambda_s_mean": float(np.mean(list(self.lambda_s.values()))),
            "lambda_s_max":  float(np.max(list(self.lambda_s.values()))),
            "mu":            float(self.mu),
        }