"""FairSignal baseline — faithful implementation.

Faithful to:  Raeis & Leon-Garcia, "A Deep Reinforcement Learning Approach
for Fair Traffic Signal Control", ITSC 2021 (arXiv:2107.10146), §III-A.
FairSignal (Cai et al., IEEE T-ITS) extends this single-intersection
delay-based fairness reward to multi-intersection via COMA.  Here we
adapt the SAME reward to the project's PPO + shared-actor/shared-critic
backbone — keeps the architecture identical to the IPPO baseline so the
gap to Fair-TSC is purely the fairness mechanism.

──────────────────────────────────────────────────────────────────────
Reward (Raeis Eq. 4, per intersection i, per decision step t):

    r_i(t) = - Σ_{n ∈ N_{i,t}}  (1 + α · (2 · d_n(t) − 1))

  N_{i,t}  : vehicles on intersection i's incoming approaches at time t.
  d_n(t)   : accumulated waiting time of vehicle n up to time t (seconds).
  α        : fairness/throughput trade-off.  α = 2.0 (paper default).

Why this is "fair":  the expected cumulative reward (Raeis Eq. 6) is

    E[Σ_t r_i(t)] = -E[Σ_n w_n] - α · E[Σ_n w_n²]

The Σw_n² term mirrors the denominator of Jain's fairness index and
penalises extreme per-vehicle waits — vehicle-level fairness.

──────────────────────────────────────────────────────────────────────
Difference vs Fair-TSC (the paper's intended contrast):

  - FairSignal/DFC:  vehicle-level fairness via quadratic-in-wait penalty
                     (per-intersection, absolute, no counterfactual).
  - Fair-TSC (ours): intersection-level fairness via Theil-T over the
                     counterfactual sacrifice gap δ = [V^UE − V^MARL]_+.

The two mechanisms answer different fairness questions; the comparison
plot uses the unified δ formula to put them on the same axis.

──────────────────────────────────────────────────────────────────────
δ semantics (UNIFIED across all methods):

    δ_i(t) = max( V^UE(s_t, i) − G_t(i), 0 )

G_t(i) uses the RAW env reward, NOT the FairSignal-shaped surrogate.
The fairness metric must measure true reward space, not the optimiser's
internal proxy.  FairSignal's own trained critic is NOT used for δ —
it is kept for reproducibility only.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from sumo_env import FairTSCEnv
from networks import SharedActor, SharedCritic
from rollout_buffer import RolloutBuffer
from ppo_core import ppo_update, bootstrap_last_values
from evaluate import (
    MetricsCollector,
    compute_deltas_from_rollout,
    evaluate_run,
    load_shared_ue_critic,
)


FAIRSIGNAL_ALPHA = 2.0   # Raeis 2021 paper default (Eq. 4); higher α → more fairness weight.


def _agent_incoming_lanes(sumo_inner, agent_id) -> List[str]:
    """Return list of incoming lane IDs for a TrafficSignal.

    sumo-rl's TrafficSignal historically exposes `lanes` (incoming).
    Defensive: try common attribute names, return [] on failure.
    """
    ts = sumo_inner.traffic_signals.get(agent_id) if hasattr(sumo_inner, "traffic_signals") else None
    if ts is None:
        return []
    for attr in ("lanes", "in_lanes", "incoming_lanes"):
        v = getattr(ts, attr, None)
        if v:
            return list(v)
    return []


def _vehicle_wait_times_on_lanes(lane_ids: List[str]) -> List[float]:
    """Query per-vehicle accumulated waiting times on the given lanes.

    Uses traci API (which routes to libsumo because the env sets
    LIBSUMO_AS_TRACI=1 in os.environ before importing sumo_rl).
    Returns a flat list of float waiting times, one per vehicle currently
    on any of the given lanes.
    """
    try:
        import traci
    except ImportError:
        return []
    waits: List[float] = []
    for lane_id in lane_ids:
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        except Exception:
            continue
        for veh in veh_ids:
            try:
                d = float(traci.vehicle.getAccumulatedWaitingTime(veh))
            except Exception:
                continue
            waits.append(d)
    return waits


def compute_fairsignal_rewards(sumo_inner, agent_ids: List[str],
                                alpha: float = FAIRSIGNAL_ALPHA) -> Dict[str, float]:
    """FairSignal/DFC reward per intersection (Raeis Eq. 4).

    r_i = - Σ_{n ∈ N_i}  (1 + α · (2 · d_n − 1))

    Scaled by C.REWARD_SCALE for parity with the env's queue-ped reward.
    """
    out: Dict[str, float] = {}
    for a in agent_ids:
        lanes = _agent_incoming_lanes(sumo_inner, a)
        waits = _vehicle_wait_times_on_lanes(lanes)
        if not waits:
            out[a] = 0.0
            continue
        total = sum(1.0 + alpha * (2.0 * d - 1.0) for d in waits)
        out[a] = -total / C.REWARD_SCALE
    return out


def collect_episode_fairsignal(env, actor, critic, buffer, device, seed=None,
                                coll: Optional[MetricsCollector] = None,
                                rollout: Optional[list] = None,
                                alpha: float = FAIRSIGNAL_ALPHA):
    """Rollout one episode under FairSignal delay-based reward.

    Critically:
    - buffer.add receives the FAIRSIGNAL-SHAPED reward (what the policy learns).
    - rollout records the RAW env reward (what feeds G for the unified δ).
    - coll receives env info as-is for efficiency / ped_wait metrics.
    """
    obs = env.reset(seed=seed)
    sumo_inner = env._walk_to_sumo_env()   # so we can query traffic_signals.lanes + libsumo
    done = False
    ep_R = {a: 0.0 for a in env.agent_ids}    # report RAW env reward
    n = 0
    while not done:
        g = env.get_global_obs(obs)
        local_b  = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
        global_b = torch.from_numpy(np.tile(g, (env.num_agents, 1))).to(device)
        idx_b    = torch.arange(env.num_agents, device=device)
        with torch.no_grad():
            action, logprob = actor.act(local_b, idx_b)
            value = critic(global_b, idx_b)
        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, R, Cp, Cs, done, info = env.step(action_dict)

        # FairSignal delay-based reward — Raeis Eq. 4, queried from libsumo
        # AFTER the step (so d_n reflects the post-step waiting times).
        r_fairsig = compute_fairsignal_rewards(sumo_inner, env.agent_ids, alpha=alpha)

        # RAW env reward vector (for δ / G computation in evaluate.py)
        r_vec_raw = np.array([R[a] for a in env.agent_ids], dtype=np.float32)

        for i, a in enumerate(env.agent_ids):
            shaped = float(r_fairsig.get(a, 0.0))
            buffer.add(
                agent_id=a, local_obs=obs[a], global_obs=g,
                action=int(action[i].item()), logprob=float(logprob[i].item()),
                reward=shaped, value=float(value[i].item()), done=done,
                c_p=0.0, c_s=0.0,
            )
            ep_R[a] += R[a]   # report RAW env reward in the log

        if coll is not None:
            coll.add(info, mean_reward=float(r_vec_raw.mean()))
        if rollout is not None:
            # Use the RAW r_vec for G (unified δ formula).
            rollout.append({"global_obs": g, "rewards_array": r_vec_raw.copy()})

        obs = next_obs
        n += 1

    last_v = bootstrap_last_values(critic, env.get_global_obs(obs),
                                   env.agent_ids, env.num_agents, device)
    buffer.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    return ep_R, n


def train_fairsignal(num_episodes: int = 50, seed: Optional[int] = None,
                     v_ue=None,
                     alpha: float = FAIRSIGNAL_ALPHA,
                     save_critic: bool = True) -> Dict:
    """Train FairSignal for `num_episodes`, then run one eval.

    Args:
        num_episodes: training episode count.
        seed:         RNG / env seed.
        v_ue:         pre-loaded shared V^UE.  None → lazy-load on ep 0.
        alpha:        FairSignal fairness coefficient (Raeis Eq. 4 α).
        save_critic:  if True, dump trained critic to
                      `<BASE_DIR>/outputs/fairsignal_critic.pt`.

    Returns the dict from `evaluate_run` (delta_valid=True).
    """
    if seed is None:
        seed = C.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
    )
    try:
        actor  = SharedActor (env.local_obs_dim,  env.num_agents, env.action_dim, C.ACTOR_HIDDEN ).to(device)
        critic = SharedCritic(env.global_obs_dim, env.num_agents,                  C.CRITIC_HIDDEN).to(device)
        o_a = torch.optim.Adam(actor.parameters(),  lr=C.ACTOR_LR)
        o_c = torch.optim.Adam(critic.parameters(), lr=C.CRITIC_LR)

        for ep in range(num_episodes):
            buf = RolloutBuffer(env.agent_ids, env.num_agents)
            ep_R, n = collect_episode_fairsignal(
                env, actor, critic, buf, device, seed=seed + ep, coll=None,
                alpha=alpha,
            )
            if v_ue is None and ep == 0:
                v_ue = load_shared_ue_critic(env=env, device=device)
            st = ppo_update(
                actor=actor, critic=critic,
                actor_optim=o_a, critic_optim=o_c, buffer=buf,
                ppo_epochs=C.PPO_EPOCHS, minibatch_size=C.MINIBATCH_SIZE,
                clip_eps=C.CLIP_EPS, entropy_coeff=C.ENTROPY_COEFF,
                vf_coeff=C.VF_COEFF, grad_clip=C.GRAD_CLIP,
            )
            rR = np.array(list(ep_R.values()))
            print(f"[FairSignal] ep={ep+1:3d}/{num_episodes} "
                  f"R̄(raw)={rR.mean():+.1f} ploss={st['policy_loss']:+.4f} "
                  f"H={st['entropy']:.3f} α={alpha}")

        # Save the trained (FairSignal-shaped reward) critic for reproducibility.
        # NOT used for δ — δ uses realized G from raw env reward.
        if save_critic:
            out_dir = os.path.join(C.BASE_DIR, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "fairsignal_critic.pt")
            torch.save({
                "critic": critic.state_dict(),
                "global_obs_dim": env.global_obs_dim,
                "num_agents":     env.num_agents,
                "hidden":         C.CRITIC_HIDDEN,
                "num_episodes":   num_episodes,
                "seed":           seed,
                "fairsignal_alpha": alpha,
            }, ckpt_path)
            print(f"[FairSignal] saved trained critic → {ckpt_path}")

        # ── Final eval episode ───────────────────────────────────
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)

        coll = MetricsCollector()
        buf = RolloutBuffer(env.agent_ids, env.num_agents)
        rollout = []
        _ = collect_episode_fairsignal(
            env, actor, critic, buf, device,
            seed=seed + num_episodes, coll=coll, rollout=rollout,
            alpha=alpha,
        )
        env_metrics = coll.finalize(env)

        if v_ue is None:
            v_ue = load_shared_ue_critic(env=env, device=device)

        # δ uses realized discounted return G from the RAW env rewards
        # (NOT the FairSignal-shaped reward, NOT FairSignal's own critic).
        if len(rollout) == 0:
            deltas_TN = np.zeros((1, env.num_agents), dtype=np.float32)
        else:
            deltas_TN = compute_deltas_from_rollout(
                rollout, v_ue=v_ue, num_agents=env.num_agents, gamma=C.GAMMA,
            )

        result = evaluate_run(deltas_TN, env_metrics, delta_valid=True)
        print(f"[FairSignal eval] {result}")
        return result
    finally:
        env.close()


def main(v_ue=None, **_unused):
    return train_fairsignal(num_episodes=50, v_ue=v_ue)


if __name__ == "__main__":
    main()
