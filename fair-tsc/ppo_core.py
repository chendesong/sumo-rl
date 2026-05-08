"""
Generic PPO update for Fair-TSC.

Used twice (with different networks + buffers):
  - Stage 1 / UE rollouts:  update π_UE, V^UE on D_UE      (no Lagrangian terms)
  - Stage 2 / MARL rollouts: update π_θ, V^MARL on D_MARL  (with fair advantage)

Standard PPO-clip + value-loss + entropy bonus. Advantage normalisation
is applied per minibatch. No target network (standard PPO; paper Eq. 33's
target network is a non-standard MAPPO addition we omit for v1).

The "fair advantage" augmentation (paper Eq. 28) is NOT applied here —
it is computed in train.py BEFORE handing the buffer to ppo_update(),
so this function stays a clean reusable PPO core.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ppo_update(
    actor: nn.Module,
    critic: nn.Module,
    actor_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer,
    buffer,
    *,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    entropy_coeff: float,
    vf_coeff: float,
    grad_clip: float,
    normalize_adv: bool = True,
) -> Dict[str, float]:
    """One PPO update over the rollout buffer.

    Args:
        actor:        SharedActor (returns Categorical via .evaluate(local_obs, agent_idx, action))
        critic:       SharedCritic (returns scalar V via __call__(global_obs, agent_idx))
        actor_optim:  optimiser for actor.parameters()
        critic_optim: optimiser for critic.parameters()
        buffer:       RolloutBuffer with compute_gae() already called

    Returns:
        dict of training stats (mean over all minibatches × ppo_epochs)
    """
    assert buffer.flat is not None, "must call buffer.compute_gae() before ppo_update()"

    stats = {
        "policy_loss":     [],
        "value_loss":      [],
        "entropy":         [],
        "approx_kl":       [],
        "clip_frac":       [],
        "explained_var":   [],
    }

    for _ in range(ppo_epochs):
        for mb in buffer.iter_minibatches(minibatch_size, shuffle=True):
            local_obs  = mb["local_obs"]
            global_obs = mb["global_obs"]
            agent_idx  = mb["agent_idx"]
            action     = mb["action"]
            old_logp   = mb["logprob"]
            adv        = mb["advantage"]
            ret        = mb["return"]
            old_val    = mb["value"]

            if normalize_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ── Actor: PPO-clip surrogate ────────────────────
            new_logp, entropy = actor.evaluate(local_obs, agent_idx, action)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_bonus = entropy.mean()

            actor_loss_total = policy_loss - entropy_coeff * entropy_bonus

            actor_optim.zero_grad()
            actor_loss_total.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
            actor_optim.step()

            # ── Critic: MSE on returns ───────────────────────
            new_val = critic(global_obs, agent_idx)
            value_loss = F.mse_loss(new_val, ret)
            critic_loss_total = vf_coeff * value_loss

            critic_optim.zero_grad()
            critic_loss_total.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            critic_optim.step()

            # ── Diagnostics ──────────────────────────────────
            with torch.no_grad():
                approx_kl = (old_logp - new_logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                # explained variance: 1 - var(ret - val) / var(ret)
                var_ret = ret.var().item()
                ev = 1.0 - ((ret - new_val).var().item() / (var_ret + 1e-8))

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy_bonus.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)
            stats["explained_var"].append(ev)

    return {k: float(np.mean(v)) for k, v in stats.items()}


# ════════════════════════════════════════════════════════════════════
# Convenience: collect bootstrap value for last state when episode ends
# mid-rollout (we always finish a full episode here, so done=True and
# bootstrap = 0; kept as helper in case future versions truncate)
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def bootstrap_last_values(
    critic: nn.Module,
    final_global_obs: np.ndarray,
    agent_ids: list,
    num_agents: int,
    device: torch.device,
) -> Dict[str, float]:
    """Compute V(s_T, i) for each agent — used as GAE bootstrap when T_ep is reached."""
    global_batch = torch.from_numpy(np.tile(final_global_obs, (num_agents, 1))).to(device)
    idx_batch = torch.arange(num_agents, device=device)
    v = critic(global_batch, idx_batch).cpu().numpy()
    return {a: float(v[i]) for i, a in enumerate(agent_ids)}