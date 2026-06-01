"""Fair-TSC training loop.

Stage 1 warms up the selfish UE actor/critic.  Stage 2 trains MAPPO.
When FAIRNESS_ENABLED=False, Stage 2 is vanilla MAPPO and only logs
T_inter/T_intra for calibration.  When FAIRNESS_ENABLED=True, Stage 2
uses the dual-level fairness cost and one PID-controlled fairness weight.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict

import numpy as np
import torch

import config as C
from fairness import (
    PIDFairnessController,
    apply_fair_advantage,
    build_per_agent_fair_cost,
    compute_inter_fairness,
    compute_sacrifice_gaps,
    phase_service_theil_from_intervals,
    reshape_deltas_to_step_agent,
)
from networks import SharedActor, SharedCritic
from ppo_core import bootstrap_last_values, ppo_update
from rollout_buffer import RolloutBuffer
from safety_eval import normalize_pedestrian_risk
from sumo_env import FairTSCEnv


def _info_probe(info: Dict) -> Dict:
    if isinstance(info, dict) and info:
        if all(isinstance(v, dict) for v in info.values()):
            return next(iter(info.values()))
        return info
    return {}


def _mean_or_zero(values) -> float:
    if not values:
        return 0.0
    return float(np.asarray(values, dtype=np.float64).mean())


def collect_one_episode(env, actor, critic, buffer, device, seed=None):
    obs = env.reset(seed=seed)
    done = False
    ep_reward = {a: 0.0 for a in env.agent_ids}
    n_steps = 0
    ped_wait_series = []
    ped_expected_series = []
    vehicle_queue_series = []
    ped_queue_series = []

    while not done:
        global_obs = env.get_global_obs(obs)
        local_b = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
        global_b = torch.from_numpy(np.tile(global_obs, (env.num_agents, 1))).to(device)
        idx_b = torch.arange(env.num_agents, device=device)

        with torch.no_grad():
            action, logprob = actor.act(local_b, idx_b)
            value = critic(global_b, idx_b)

        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, reward, _cp, _cs, done, info = env.step(action_dict)
        probe = _info_probe(info)
        if probe:
            ped_wait_series.append(float(probe.get("agents_total_ped_waiting_time", 0.0)))
            ped_expected_series.append(float(probe.get("agents_total_expected_violations", 0.0)))
            vehicle_queue_series.append(float(probe.get("agents_total_stopped", 0.0)))
            ped_queue_series.append(float(probe.get("agents_total_ped_queued", 0.0)))

        for i, agent in enumerate(env.agent_ids):
            buffer.add(
                agent_id=agent,
                local_obs=obs[agent],
                global_obs=global_obs,
                action=int(action[i].item()),
                logprob=float(logprob[i].item()),
                reward=float(reward[agent]),
                value=float(value[i].item()),
                done=done,
            )
            ep_reward[agent] += float(reward[agent])

        if next_obs:
            obs = next_obs
        n_steps += 1

    if C.REWARD_NORMALIZE:
        reward_norm = buffer.normalize_rewards(
            center=C.REWARD_NORM_CENTER,
            clip=C.REWARD_NORM_CLIP,
            eps=C.REWARD_NORM_EPS,
        )
    else:
        reward_norm = {
            "reward_norm_enabled": 0,
            "reward_norm_mean": 0.0,
            "reward_norm_std": 1.0,
        }

    last_v = bootstrap_last_values(critic, env.get_global_obs(obs), env.agent_ids, env.num_agents, device)
    buffer.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    ped_expected = _mean_or_zero(ped_expected_series)
    safety = {
        "ped_wait": _mean_or_zero(ped_wait_series),
        "ped_expected_violations": ped_expected,
        "ped_risk": normalize_pedestrian_risk(ped_expected, num_agents=env.num_agents),
    }
    denom = max(float(env.num_agents) * float(C.REWARD_SCALE), 1e-9)
    reward_components = {
        "reward_vehicle_component": -float(np.sum(vehicle_queue_series)) / denom,
        "reward_ped_component": -float(C.OMEGA_P) * float(np.sum(ped_queue_series)) / denom,
        "vehicle_queue_mean": _mean_or_zero(vehicle_queue_series),
        "ped_queue_mean": _mean_or_zero(ped_queue_series),
    }
    reward_components["reward_env_component_sum"] = (
        reward_components["reward_vehicle_component"] + reward_components["reward_ped_component"]
    )
    return ep_reward, n_steps, safety, reward_norm, reward_components


def compute_dual_level_fairness(env, buffer, deltas):
    deltas_tn = reshape_deltas_to_step_agent(deltas, buffer.flat["agent_idx"], env.num_agents)
    delta_agent_mean = deltas_tn.mean(axis=0)
    theil_inter, inter_contrib = compute_inter_fairness(delta_agent_mean, eps=C.THEIL_EPS)

    phase_intervals = env.get_phase_service_intervals(include_unserved=True)
    intra_by_agent, theil_intra, max_phase_interval = phase_service_theil_from_intervals(
        phase_intervals, env.agent_ids, eps=C.THEIL_EPS
    )

    per_agent_cost, c_fair = build_per_agent_fair_cost(
        env.agent_ids,
        inter_contrib,
        intra_by_agent,
        alpha=C.FAIR_ALPHA,
        t_inter_0=C.T_INTER_0,
        t_intra_0=C.T_INTRA_0,
        num_agents=env.num_agents,
        eps=C.FAIR_EPS,
    )

    return {
        "deltas_TN": deltas_tn,
        "delta_agent_mean": delta_agent_mean,
        "theil_inter": float(theil_inter),
        "theil_intra": float(theil_intra),
        "max_phase_interval": float(max_phase_interval),
        "inter_contrib": inter_contrib,
        "intra_by_agent": intra_by_agent,
        "per_agent_cost": per_agent_cost,
        "C_fair": float(c_fair),
    }


def disabled_pid_stats(c_fair: float) -> Dict[str, float]:
    return {
        "C_fair_raw": float(c_fair),
        "C_fair_ema": float(c_fair),
        "fair_target": float(C.FAIR_C_TARGET),
        "pid_error": 0.0,
        "pid_integral": 0.0,
        "pid_derivative": 0.0,
        "lambda_fair": 0.0,
    }


def select_fair_credit(fair: Dict, agent_ids) -> Dict[str, float]:
    """Return the advantage-level fairness credit for the selected ablation."""
    if C.FAIR_CREDIT_MODE == "per_agent":
        return fair["per_agent_cost"]
    if C.FAIR_CREDIT_MODE == "global":
        share = float(fair["C_fair"]) / max(len(agent_ids), 1)
        return {agent: share for agent in agent_ids}
    if C.FAIR_CREDIT_MODE == "none":
        return {agent: 0.0 for agent in agent_ids}
    raise ValueError(f"Unknown FAIR_TSC_CREDIT_MODE={C.FAIR_CREDIT_MODE}")


def fair_penalty_summary(fair_credit: Dict[str, float], lambda_fair: float) -> Dict[str, float]:
    vals = np.asarray(
        [float(lambda_fair) * float(v) for v in fair_credit.values()],
        dtype=np.float64,
    )
    if vals.size == 0:
        return {"fair_penalty_mean": 0.0, "fair_penalty_max": 0.0}
    return {
        "fair_penalty_mean": float(vals.mean()),
        "fair_penalty_max": float(vals.max()),
    }


def write_row(writer, base, stats):
    row = dict(base)
    row.update(stats)
    writer.writerow(row)


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"mode = {'Fair-TSC PID' if C.FAIRNESS_ENABLED else 'vanilla MAPPO calibration'}")
    print(f"T_INTER_0={C.T_INTER_0:.6f}  T_INTRA_0={C.T_INTRA_0:.6f}")
    print(f"fair_credit_mode={C.FAIR_CREDIT_MODE}")
    print(
        f"reward_norm={int(C.REWARD_NORMALIZE)} center={int(C.REWARD_NORM_CENTER)} "
        f"clip={C.REWARD_NORM_CLIP:g}"
    )

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    log_path = os.path.join(C.OUTPUT_DIR, "train_log.csv")
    per_agent_path = os.path.join(C.OUTPUT_DIR, "per_agent_log.csv")
    print(f"output dir : {C.OUTPUT_DIR}")
    print(f"ckpt dir   : {C.CKPT_DIR}")
    print(f"log file   : {log_path}")

    env = FairTSCEnv(
        net_file=C.NET_FILE,
        route_file=C.ROUTE_FILE,
        out_csv_name=os.path.join(C.OUTPUT_DIR, "ep"),
        num_seconds=C.NUM_SECONDS,
        delta_time=C.DELTA_TIME,
        min_green=C.MIN_GREEN,
        use_gui=C.USE_GUI,
    )
    print(
        f"agents={env.agent_ids}  N={env.num_agents}  "
        f"D_l={env.local_obs_dim}  D_g={env.global_obs_dim}  A={env.action_dim}"
    )

    actor_marl = SharedActor(env.local_obs_dim, env.num_agents, env.action_dim, C.ACTOR_HIDDEN).to(device)
    critic_marl = SharedCritic(env.global_obs_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)
    actor_ue = SharedActor(env.local_obs_dim, env.num_agents, env.action_dim, C.ACTOR_HIDDEN).to(device)
    critic_ue = SharedCritic(env.global_obs_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)

    opt_actor_marl = torch.optim.Adam(actor_marl.parameters(), lr=C.ACTOR_LR)
    opt_critic_marl = torch.optim.Adam(critic_marl.parameters(), lr=C.CRITIC_LR)
    opt_actor_ue = torch.optim.Adam(actor_ue.parameters(), lr=C.ACTOR_LR)
    opt_critic_ue = torch.optim.Adam(critic_ue.parameters(), lr=C.CRITIC_LR)

    pid = PIDFairnessController(
        target=C.FAIR_C_TARGET,
        kp=C.PID_KP,
        ki=C.PID_KI,
        kd=C.PID_KD,
        lambda_max=C.PID_LAMBDA_MAX,
        integral_max=C.PID_INTEGRAL_MAX,
        ema_beta=C.PID_EMA_BETA,
    )

    log_fields = [
        "stage",
        "episode",
        "global_step",
        "wall_time_s",
        "fairness_enabled",
        "fair_credit_mode",
        "reward_mean",
        "reward_min",
        "reward_max",
        "reward_vehicle_component",
        "reward_ped_component",
        "reward_env_component_sum",
        "vehicle_queue_mean",
        "ped_queue_mean",
        "fair_penalty_mean",
        "fair_penalty_max",
        "reward_after_fair_proxy",
        *[f"reward_{a}" for a in env.agent_ids],
        "reward_norm_enabled",
        "reward_norm_mean",
        "reward_norm_std",
        "delta_mean",
        "delta_max",
        "theil_inter",
        "theil_intra",
        "max_phase_interval",
        "ped_wait",
        "ped_risk",
        "ped_expected_violations",
        "C_fair_raw",
        "C_fair_ema",
        "fair_target",
        "lambda_fair",
        "pid_error",
        "pid_integral",
        "pid_derivative",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "explained_var",
    ]
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    per_agent_fields = [
        "episode",
        "stage",
        "agent",
        "delta_mean",
        "T_inter_i",
        "T_intra_i",
        "c_fair_i",
    ]
    per_agent_file = open(per_agent_path, "w", newline="", encoding="utf-8")
    per_agent_writer = csv.DictWriter(per_agent_file, fieldnames=per_agent_fields)
    per_agent_writer.writeheader()

    t0 = time.time()
    global_step = 0
    episode = 0
    agent_idx_to_id = {i: a for i, a in enumerate(env.agent_ids)}

    print(f"\n{'=' * 70}\nSTAGE 1: UE warm-up   target={C.T_WARM} steps\n{'=' * 70}")
    while global_step < C.T_WARM:
        buffer = RolloutBuffer(env.agent_ids, env.num_agents)
        ep_reward, n, safety, reward_norm, reward_components = collect_one_episode(
            env, actor_ue, critic_ue, buffer, device, seed=C.SEED + episode
        )
        global_step += n
        episode += 1

        ppo_stats = ppo_update(
            actor=actor_ue,
            critic=critic_ue,
            actor_optim=opt_actor_ue,
            critic_optim=opt_critic_ue,
            buffer=buffer,
            ppo_epochs=C.PPO_EPOCHS,
            minibatch_size=C.MINIBATCH_SIZE,
            clip_eps=C.CLIP_EPS,
            entropy_coeff=C.ENTROPY_COEFF,
            vf_coeff=C.VF_COEFF,
            grad_clip=C.GRAD_CLIP,
        )

        rewards = np.asarray(list(ep_reward.values()), dtype=np.float32)
        penalty_stats = fair_penalty_summary({}, 0.0)
        elapsed = time.time() - t0
        print(
            f"[STAGE1] ep={episode:3d} step={global_step:6d}/{C.T_WARM} "
            f"R={rewards.mean():+.1f} H={ppo_stats['entropy']:.3f} t={elapsed:.0f}s"
        )

        write_row(
            log_writer,
            {
                "stage": 1,
                "episode": episode,
                "global_step": global_step,
                "wall_time_s": elapsed,
                "fairness_enabled": int(C.FAIRNESS_ENABLED),
                "fair_credit_mode": C.FAIR_CREDIT_MODE,
                "reward_mean": float(rewards.mean()),
                "reward_min": float(rewards.min()),
                "reward_max": float(rewards.max()),
                **reward_components,
                **penalty_stats,
                "reward_after_fair_proxy": float(rewards.mean()) - penalty_stats["fair_penalty_mean"],
                **{f"reward_{a}": float(ep_reward[a]) for a in env.agent_ids},
                **reward_norm,
                "delta_mean": 0.0,
                "delta_max": 0.0,
                "theil_inter": 0.0,
                "theil_intra": 0.0,
                "max_phase_interval": 0.0,
                "ped_wait": safety["ped_wait"],
                "ped_risk": safety["ped_risk"],
                "ped_expected_violations": safety["ped_expected_violations"],
            },
            {**disabled_pid_stats(0.0), **ppo_stats},
        )
        log_file.flush()

    for p in actor_ue.parameters():
        p.requires_grad = False
    for p in critic_ue.parameters():
        p.requires_grad = False
    actor_ue.eval()
    critic_ue.eval()
    print(f"\n[STAGE1 done] global_step={global_step}, UE frozen.\n")

    print(f"{'=' * 70}\nSTAGE 2: MARL training   target={C.TOTAL_STEPS} total steps\n{'=' * 70}")
    while global_step < C.TOTAL_STEPS:
        buffer = RolloutBuffer(env.agent_ids, env.num_agents)
        ep_reward, n, safety, reward_norm, reward_components = collect_one_episode(
            env, actor_marl, critic_marl, buffer, device, seed=C.SEED + episode
        )
        global_step += n
        episode += 1

        deltas = compute_sacrifice_gaps(buffer, critic_ue, critic_marl, device)
        fair = compute_dual_level_fairness(env, buffer, deltas)

        if C.FAIRNESS_ENABLED:
            pid_stats = pid.update(fair["C_fair"])
            fair_credit = select_fair_credit(fair, env.agent_ids)
            apply_fair_advantage(buffer, fair_credit, agent_idx_to_id, pid.lambda_value)
        else:
            pid_stats = disabled_pid_stats(fair["C_fair"])
            fair_credit = select_fair_credit(fair, env.agent_ids)
        penalty_stats = fair_penalty_summary(fair_credit, pid_stats["lambda_fair"])

        ppo_stats = ppo_update(
            actor=actor_marl,
            critic=critic_marl,
            actor_optim=opt_actor_marl,
            critic_optim=opt_critic_marl,
            buffer=buffer,
            ppo_epochs=C.PPO_EPOCHS,
            minibatch_size=C.MINIBATCH_SIZE,
            clip_eps=C.CLIP_EPS,
            entropy_coeff=C.ENTROPY_COEFF,
            vf_coeff=C.VF_COEFF,
            grad_clip=C.GRAD_CLIP,
        )

        rewards = np.asarray(list(ep_reward.values()), dtype=np.float32)
        delta_agent_mean = fair["delta_agent_mean"]
        worst_idx = int(delta_agent_mean.argmax()) if len(delta_agent_mean) else 0
        best_idx = int(delta_agent_mean.argmin()) if len(delta_agent_mean) else 0
        elapsed = time.time() - t0

        print(
            f"[STAGE2] ep={episode:3d} step={global_step:6d}/{C.TOTAL_STEPS} "
            f"R={rewards.mean():+.1f} Tinter={fair['theil_inter']:.4f} "
            f"Tintra={fair['theil_intra']:.4f} Cfair={fair['C_fair']:.4f} "
            f"lambda={pid_stats['lambda_fair']:.4f} H={ppo_stats['entropy']:.3f} t={elapsed:.0f}s"
        )
        print(
            f"   worst-delta: {env.agent_ids[worst_idx]}={delta_agent_mean[worst_idx]:.3f}  "
            f"best-delta: {env.agent_ids[best_idx]}={delta_agent_mean[best_idx]:.3f}"
        )

        for i, agent in enumerate(env.agent_ids):
            per_agent_writer.writerow(
                {
                    "episode": episode,
                    "stage": 2,
                    "agent": agent,
                    "delta_mean": float(delta_agent_mean[i]),
                    "T_inter_i": float(fair["inter_contrib"][i]),
                    "T_intra_i": float(fair["intra_by_agent"].get(agent, 0.0)),
                    "c_fair_i": float(fair_credit.get(agent, 0.0)),
                }
            )
        per_agent_file.flush()

        write_row(
            log_writer,
            {
                "stage": 2,
                "episode": episode,
                "global_step": global_step,
                "wall_time_s": elapsed,
                "fairness_enabled": int(C.FAIRNESS_ENABLED),
                "fair_credit_mode": C.FAIR_CREDIT_MODE,
                "reward_mean": float(rewards.mean()),
                "reward_min": float(rewards.min()),
                "reward_max": float(rewards.max()),
                **reward_components,
                **penalty_stats,
                "reward_after_fair_proxy": float(rewards.mean()) - penalty_stats["fair_penalty_mean"],
                **{f"reward_{a}": float(ep_reward[a]) for a in env.agent_ids},
                **reward_norm,
                "delta_mean": float(deltas.mean().item()),
                "delta_max": float(deltas.max().item()),
                "theil_inter": float(fair["theil_inter"]),
                "theil_intra": float(fair["theil_intra"]),
                "max_phase_interval": float(fair["max_phase_interval"]),
                "ped_wait": safety["ped_wait"],
                "ped_risk": safety["ped_risk"],
                "ped_expected_violations": safety["ped_expected_violations"],
            },
            {**pid_stats, **ppo_stats},
        )
        log_file.flush()

        if episode % C.SAVE_CKPT_EVERY_N == 0:
            ckpt_path = os.path.join(C.CKPT_DIR, f"ep_{episode:04d}.pt")
            torch.save(
                {
                    "episode": episode,
                    "global_step": global_step,
                    "fairness_enabled": C.FAIRNESS_ENABLED,
                    "actor_marl": actor_marl.state_dict(),
                    "critic_marl": critic_marl.state_dict(),
                    "actor_ue": actor_ue.state_dict(),
                    "critic_ue": critic_ue.state_dict(),
                    "pid": pid.state_dict(),
                    "T_INTER_0": C.T_INTER_0,
                    "T_INTRA_0": C.T_INTRA_0,
                    "FAIR_ALPHA": C.FAIR_ALPHA,
                    "FAIR_CREDIT_MODE": C.FAIR_CREDIT_MODE,
                    "REWARD_NORMALIZE": C.REWARD_NORMALIZE,
                    "REWARD_NORM_CENTER": C.REWARD_NORM_CENTER,
                    "REWARD_NORM_CLIP": C.REWARD_NORM_CLIP,
                },
                ckpt_path,
            )
            print(f"   [ckpt] saved {ckpt_path}")

    final_path = os.path.join(C.CKPT_DIR, "final.pt")
    torch.save(
        {
            "episode": episode,
            "global_step": global_step,
            "fairness_enabled": C.FAIRNESS_ENABLED,
            "actor_marl": actor_marl.state_dict(),
            "critic_marl": critic_marl.state_dict(),
            "actor_ue": actor_ue.state_dict(),
            "critic_ue": critic_ue.state_dict(),
            "pid": pid.state_dict(),
            "T_INTER_0": C.T_INTER_0,
            "T_INTRA_0": C.T_INTRA_0,
            "FAIR_ALPHA": C.FAIR_ALPHA,
            "FAIR_CREDIT_MODE": C.FAIR_CREDIT_MODE,
            "REWARD_NORMALIZE": C.REWARD_NORMALIZE,
            "REWARD_NORM_CENTER": C.REWARD_NORM_CENTER,
            "REWARD_NORM_CLIP": C.REWARD_NORM_CLIP,
        },
        final_path,
    )
    print(f"\n[done] {episode} episodes, {global_step} steps, {(time.time() - t0) / 60:.1f} min.")
    print(f"Final ckpt: {final_path}")

    log_file.close()
    per_agent_file.close()
    env.close()


if __name__ == "__main__":
    main()
