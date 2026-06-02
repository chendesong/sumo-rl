"""Standalone IPPO baseline training.

This is the long-run version of the IPPO baseline used by run_comparison.py.
It writes a train_log.csv and Fair-TSC-compatible checkpoints so later
evaluation scripts can reuse the trained policy instead of retraining IPPO
inside a serial comparison run.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

import config as C
from baselines.ippo import collect_episode_with_metrics
from evaluate import MetricsCollector
from networks import SharedActor, SharedCritic
from ppo_core import ppo_update
from rollout_buffer import RolloutBuffer
from sumo_env import FairTSCEnv


def _run_name() -> str:
    stamp = time.strftime("%Y%m%d_%H%M")
    return f"ippo_4x4_{C.DEMAND_LEVEL}_s{C.SEED}_{stamp}"


def _mean(seq) -> float:
    arr = np.asarray(list(seq), dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _save_ckpt(path, actor, critic, env, episode, global_step):
    torch.save(
        {
            "episode": int(episode),
            "global_step": int(global_step),
            "baseline": "ippo",
            "fairness_enabled": False,
            "demand": C.DEMAND_LEVEL,
            "seed": C.SEED,
            "actor_marl": actor.state_dict(),
            "critic_marl": critic.state_dict(),
            "actor_ippo": actor.state_dict(),
            "critic_ippo": critic.state_dict(),
            "local_obs_dim": env.local_obs_dim,
            "global_obs_dim": env.global_obs_dim,
            "num_agents": env.num_agents,
            "action_dim": env.action_dim,
            "actor_hidden": C.ACTOR_HIDDEN,
            "critic_hidden": C.CRITIC_HIDDEN,
        },
        path,
    )


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = _run_name()
    out_dir = os.path.join(C.BASE_DIR, "outputs", run_name)
    ckpt_dir = os.path.join(C.BASE_DIR, "checkpoints", run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.csv")
    log_file = None

    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
    )

    try:
        obs = env.reset(seed=C.SEED)
        actor = SharedActor(env.local_obs_dim, env.num_agents, env.action_dim, C.ACTOR_HIDDEN).to(device)
        critic = SharedCritic(env.global_obs_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=C.ACTOR_LR)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=C.CRITIC_LR)

        log_fields = [
            "stage",
            "episode",
            "global_step",
            "wall_time_s",
            "fairness_enabled",
            "credit_mode",
            "reward_mean",
            "reward_min",
            "reward_max",
            *[f"reward_{a}" for a in env.agent_ids],
            "theil_inter",
            "theil_intra",
            "max_phase_interval",
            "system_wait_mean",
            "ped_wait",
            "ped_risk",
            "time_to_teleport",
            "teleported_total",
            "departed_total",
            "arrived_total",
            "completion_rate_departed",
            "completion_rate_demand",
            "unfinished_vehicle_demand",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            "explained_var",
        ]

        log_file = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        print(f"[IPPO standalone] run={run_name}")
        print(f"[IPPO standalone] log={log_path}")
        print(f"[IPPO standalone] target_steps={C.TOTAL_STEPS}")

        t0 = time.time()
        episode = 0
        global_step = 0
        while global_step < C.TOTAL_STEPS:
            episode += 1
            buffer = RolloutBuffer(env.agent_ids, env.num_agents)
            coll = MetricsCollector()
            ep_reward, n_steps = collect_episode_with_metrics(
                env,
                actor,
                critic,
                buffer,
                device,
                seed=C.SEED + episode,
                coll=coll,
            )
            global_step += n_steps

            stats = ppo_update(
                actor=actor,
                critic=critic,
                actor_optim=actor_optim,
                critic_optim=critic_optim,
                buffer=buffer,
                ppo_epochs=C.PPO_EPOCHS,
                minibatch_size=C.MINIBATCH_SIZE,
                clip_eps=C.CLIP_EPS,
                entropy_coeff=C.ENTROPY_COEFF,
                vf_coeff=C.VF_COEFF,
                grad_clip=C.GRAD_CLIP,
            )

            rewards = np.asarray(list(ep_reward.values()), dtype=np.float64)
            env_metrics = coll.finalize(env)
            completion_rate_departed = (
                float(env_metrics.get("arrived_total", 0.0) or 0.0)
                / max(float(env_metrics.get("departed_total", 0.0) or 0.0), 1.0)
            )
            elapsed = time.time() - t0
            row = {
                "stage": 2,
                "episode": episode,
                "global_step": global_step,
                "wall_time_s": elapsed,
                "fairness_enabled": 0,
                "credit_mode": "ippo",
                "reward_mean": float(rewards.mean()),
                "reward_min": float(rewards.min()),
                "reward_max": float(rewards.max()),
                **{f"reward_{a}": float(ep_reward[a]) for a in env.agent_ids},
                "theil_inter": np.nan,
                "theil_intra": float(env_metrics.get("theil_intra", 0.0) or 0.0),
                "max_phase_interval": float(env_metrics.get("max_phase_interval", 0.0) or 0.0),
                "system_wait_mean": _mean(env_metrics.get("system_total_waiting_time_series", [])),
                "ped_wait": _mean(env_metrics.get("agents_total_ped_waiting_time_series", [])),
                "ped_risk": float(env_metrics.get("ped_risk", 0.0) or 0.0),
                "time_to_teleport": float(C.TIME_TO_TELEPORT),
                "teleported_total": float(env_metrics.get("teleported_total", 0.0) or 0.0),
                "departed_total": float(env_metrics.get("departed_total", 0.0) or 0.0),
                "arrived_total": float(env_metrics.get("arrived_total", 0.0) or 0.0),
                "completion_rate_departed": completion_rate_departed,
                "completion_rate_demand": float(env_metrics.get("completion_rate_demand", 0.0) or 0.0),
                "unfinished_vehicle_demand": float(env_metrics.get("unfinished_vehicle_demand", 0.0) or 0.0),
                **stats,
            }
            writer.writerow(row)
            log_file.flush()

            print(
                f"[IPPO] ep={episode:4d} step={global_step:6d}/{C.TOTAL_STEPS} "
                f"R={rewards.mean():+.1f} Tintra={row['theil_intra']:.4f} "
                f"maxPhase={row['max_phase_interval']:.1f} "
                f"H={stats.get('entropy', 0.0):.3f} tel={row['teleported_total']:.0f} t={elapsed:.0f}s"
            )

            if episode % C.SAVE_CKPT_EVERY_N == 0:
                ckpt_path = os.path.join(ckpt_dir, f"ep_{episode:04d}.pt")
                _save_ckpt(ckpt_path, actor, critic, env, episode, global_step)
                print(f"   [ckpt] saved {ckpt_path}")

        final_path = os.path.join(ckpt_dir, "final.pt")
        _save_ckpt(final_path, actor, critic, env, episode, global_step)
        print(f"[done] {episode} episodes, {global_step} steps, {(time.time() - t0) / 60:.1f} min.")
        print(f"Final ckpt: {final_path}")
    finally:
        try:
            if log_file is not None:
                log_file.close()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
