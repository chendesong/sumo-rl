"""FairSignal baseline, matched to Cai et al. (IEEE IoT-J 2025).

The FairSignal paper defines fairness over intersections rather than over
individual vehicles. With q_i(t) denoting the queue length at intersection
i, its reward is:

    r_E(t) = - sum_i q_i(t)
    r_F(t) = - alpha * sum_i q_i(t)^2
    r(t)   = r_E(t) + r_F(t)

This mirrors the denominator of Jain's fairness index over intersection
traffic conditions: large imbalance in intersection queues receives a
quadratic penalty.

Implementation note:
The paper trains the reward with Advanced-COMA. In this codebase we keep
the same PPO/shared-network training scaffold used by the other learned
baselines, but the shaped reward now follows FairSignal Eq. 15/16/18.
Evaluation still uses the raw environment reward for the unified delta
metric, so the comparison is not biased by FairSignal's own reward scale.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

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


FAIRSIGNAL_ALPHA = float(os.environ.get("FAIR_TSC_FAIRSIGNAL_ALPHA", "2.0"))


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


def _lane_halting_count(lane_id: str) -> float:
    try:
        import traci
    except ImportError:
        return 0.0
    try:
        return float(traci.lane.getLastStepHaltingNumber(lane_id))
    except Exception:
        return 0.0


def _intersection_queue(sumo_inner, agent_id: str) -> float:
    """Queue length q_i(t) for FairSignal Eq. 15/16.

    Prefer sumo-rl's raw `get_total_queued` when available. Fall back to
    summing halting vehicles on incoming lanes.
    """
    ts = sumo_inner.traffic_signals.get(agent_id) if hasattr(sumo_inner, "traffic_signals") else None
    if ts is not None:
        try:
            return float(ts.get_total_queued())
        except Exception:
            pass
    lanes = _agent_incoming_lanes(sumo_inner, agent_id)
    return float(sum(_lane_halting_count(lane_id) for lane_id in lanes))


def compute_fairsignal_rewards(sumo_inner, agent_ids: List[str],
                                alpha: float = FAIRSIGNAL_ALPHA) -> Dict[str, float]:
    """FairSignal global reward broadcast to all intersections.

    Cai et al. Eq. 15/16/18:

        r = -sum_i q_i - alpha * sum_i q_i^2

    Scaled by C.REWARD_SCALE for parity with the env rewards.
    """
    queues = np.asarray([_intersection_queue(sumo_inner, a) for a in agent_ids], dtype=np.float32)
    reward = -(float(queues.sum()) + float(alpha) * float(np.square(queues).sum()))
    shaped = reward / C.REWARD_SCALE
    return {a: shaped for a in agent_ids}


def compute_fairsignal_components(sumo_inner, agent_ids: List[str],
                                  alpha: float = FAIRSIGNAL_ALPHA) -> Tuple[float, float, float]:
    """Return (efficiency term, fairness term, total), already scaled."""
    queues = np.asarray([_intersection_queue(sumo_inner, a) for a in agent_ids], dtype=np.float32)
    r_eff = -float(queues.sum()) / C.REWARD_SCALE
    r_fair = -float(alpha) * float(np.square(queues).sum()) / C.REWARD_SCALE
    return r_eff, r_fair, r_eff + r_fair


def collect_episode_fairsignal(env, actor, critic, buffer, device, seed=None,
                                coll: Optional[MetricsCollector] = None,
                                rollout: Optional[list] = None,
                                alpha: float = FAIRSIGNAL_ALPHA):
    """Rollout one episode under FairSignal intersection-queue reward.

    Critically:
    - buffer.add receives the FAIRSIGNAL-SHAPED reward (what the policy learns).
    - rollout records the RAW env reward (what feeds G for the unified delta).
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

        # FairSignal Eq. 15/16/18, queried after the step so queues reflect
        # the post-action state.
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
                     save_critic: bool = True,
                     additional_sumo_cmd: Optional[str] = None) -> Dict:
    """Train FairSignal for `num_episodes`, then run one eval.

    Args:
        num_episodes: training episode count.
        seed:         RNG / env seed.
        v_ue:         pre-loaded shared V^UE.  None → lazy-load on ep 0.
        alpha:        FairSignal fairness coefficient in Cai et al. Eq. 18.
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
        additional_sumo_cmd=additional_sumo_cmd,
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
                "fairsignal_reward": "r=-sum_i q_i-alpha*sum_i q_i^2",
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


def main(v_ue=None, additional_sumo_cmd: Optional[str] = None, **_unused):
    return train_fairsignal(num_episodes=50, v_ue=v_ue, additional_sumo_cmd=additional_sumo_cmd)


if __name__ == "__main__":
    main()
