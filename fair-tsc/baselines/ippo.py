"""Independent PPO baseline (IPPO).

Same actor / critic architecture as Fair-TSC and the same PPO update,
but:
  - NO PID fairness penalty
  - NO sacrifice gap δ computed during training
  - NO UE warm-up stage
  - Critic still uses the global obs (the SharedCritic interface) so
    that the network shapes match — this is technically "centralised
    critic, independent execution", which is the most common form of
    IPPO in TSC literature.  Switching to a strictly local critic would
    require a new network class; keeping CC keeps the comparison fair
    on the *value-function side* and isolates the fairness-penalty effect.

After 50 training episodes, ONE final eval rollout is collected and
δ is computed under the unified formula

    δ_i(t) = max( V^UE(s_t, i) − G_t(i), 0 )

where V^UE is the SHARED Fair-TSC critic_ue and G_t(i) is the realized
discounted return of agent i from raw env rewards on this rollout.
IPPO's own trained critic is NOT used for δ — but we still save it for
reproducibility.
"""

import os
import sys
from typing import Dict, Optional

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


def collect_episode_with_metrics(env, actor, critic, buffer, device, seed=None,
                                  coll: Optional[MetricsCollector] = None,
                                  rollout: Optional[list] = None):
    """Mirror of train.collect_one_episode but also pumps env info into
    a MetricsCollector when one is supplied (used at eval time).

    If `rollout` is a non-None list, append per-step
    {"global_obs": g, "rewards_array": np.array[N]} dicts (rewards are
    the RAW env rewards in env.agent_ids order) — used downstream by
    `compute_deltas_from_rollout` for the unified G-based δ formula.
    """
    obs = env.reset(seed=seed)
    done = False
    ep_R = {a: 0.0 for a in env.agent_ids}
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
        for i, a in enumerate(env.agent_ids):
            buffer.add(
                agent_id=a, local_obs=obs[a], global_obs=g,
                action=int(action[i].item()), logprob=float(logprob[i].item()),
                reward=R[a], value=float(value[i].item()), done=done,
                c_p=0.0, c_s=0.0,   # IPPO ignores constraint costs
            )
            ep_R[a] += R[a]
        if coll is not None:
            mean_r = float(np.mean(list(R.values()))) if R else 0.0
            coll.add(info, mean_reward=mean_r)
        if rollout is not None:
            r_arr = np.array([R[a] for a in env.agent_ids], dtype=np.float32)
            rollout.append({"global_obs": g, "rewards_array": r_arr})
        obs = next_obs
        n += 1

    last_v = bootstrap_last_values(critic, env.get_global_obs(obs),
                                   env.agent_ids, env.num_agents, device)
    buffer.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    return ep_R, n


def train_ippo(num_episodes: int = 50, seed: Optional[int] = None,
               v_ue=None,
               save_critic: bool = True) -> Dict:
    """Train IPPO for `num_episodes`, then run one eval episode.

    Args:
        num_episodes: number of training episodes (default 50).
        seed:         RNG / env seed.  None → C.SEED.
        v_ue:         pre-loaded shared V^UE `SharedCritic`. None → load
                      lazily from default ckpt path after first reset.
        save_critic:  if True, save IPPO's trained critic to
                      `<BASE_DIR>/outputs/ippo_critic.pt` for repro.

    Returns the dict from evaluate_run on the eval episode (with
    `delta_valid=True`).
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
            ep_R, n = collect_episode_with_metrics(
                env, actor, critic, buf, device, seed=seed + ep, coll=None,
            )
            if v_ue is None and ep == 0:
                # env is now post-reset; safe to load V^UE.
                v_ue = load_shared_ue_critic(env=env, device=device)
            st = ppo_update(
                actor=actor, critic=critic,
                actor_optim=o_a, critic_optim=o_c, buffer=buf,
                ppo_epochs=C.PPO_EPOCHS, minibatch_size=C.MINIBATCH_SIZE,
                clip_eps=C.CLIP_EPS, entropy_coeff=C.ENTROPY_COEFF,
                vf_coeff=C.VF_COEFF, grad_clip=C.GRAD_CLIP,
            )
            rR = np.array(list(ep_R.values()))
            print(f"[IPPO] ep={ep+1:3d}/{num_episodes} "
                  f"R̄={rR.mean():+.1f} ploss={st['policy_loss']:+.4f} "
                  f"H={st['entropy']:.3f}")

        # Save the trained critic for reproducibility.
        if save_critic:
            out_dir = os.path.join(C.BASE_DIR, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "ippo_critic.pt")
            torch.save({
                "critic": critic.state_dict(),
                "global_obs_dim": env.global_obs_dim,
                "num_agents":     env.num_agents,
                "hidden":         C.CRITIC_HIDDEN,
                "num_episodes":   num_episodes,
                "seed":           seed,
            }, ckpt_path)
            print(f"[IPPO] saved trained critic → {ckpt_path}")

        # ── Final eval episode (collect rollout + env metrics) ───────
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)

        coll = MetricsCollector()
        buf = RolloutBuffer(env.agent_ids, env.num_agents)
        rollout = []
        _ = collect_episode_with_metrics(
            env, actor, critic, buf, device,
            seed=seed + num_episodes, coll=coll, rollout=rollout,
        )
        env_metrics = coll.finalize(env)

        if v_ue is None:
            v_ue = load_shared_ue_critic(env=env, device=device)

        # δ uses realized discounted return G (raw env rewards), NOT
        # IPPO's own trained critic.
        if len(rollout) == 0:
            deltas_TN = np.zeros((1, env.num_agents), dtype=np.float32)
        else:
            deltas_TN = compute_deltas_from_rollout(
                rollout, v_ue=v_ue, num_agents=env.num_agents, gamma=C.GAMMA,
            )

        result = evaluate_run(deltas_TN, env_metrics, delta_valid=True)
        print(f"[IPPO eval] {result}")
        return result
    finally:
        env.close()


def main(v_ue=None, **_unused):
    """Entry point. `v_ue` may be a pre-loaded SharedCritic.  Also
    accepts legacy `v_ue_fn=` from old callers (ignored)."""
    return train_ippo(num_episodes=50, v_ue=v_ue)


if __name__ == "__main__":
    main()
