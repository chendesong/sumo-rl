"""
Stage-1 (UE warm-up) convergence diagnostic.

Runs only Stage 1 of the Fair-TSC training, for a configurable number of
episodes, logging per-episode metrics so you can decide whether the current
T_WARM (2000 steps ~ 3 episodes) is enough or needs to grow.

Usage:
    python fair-tsc/diagnose_ue.py                 # default 30 episodes
    python fair-tsc/diagnose_ue.py --episodes 50
    python fair-tsc/diagnose_ue.py --plot          # also dumps PNG at the end

Reads config from config.py (same as train.py). Outputs:
    <OUTPUT_DIR>/ue_diagnostic.csv
    <OUTPUT_DIR>/ue_diagnostic.png   (if --plot)
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from sumo_env import FairTSCEnv
from networks import SharedActor, SharedCritic
from rollout_buffer import RolloutBuffer
from ppo_core import ppo_update, bootstrap_last_values
import config as C


def collect_one_episode(env, actor, critic, buffer, device, seed=None):
    obs = env.reset(seed=seed)
    done = False
    ep_R = {a: 0.0 for a in env.agent_ids}
    n = 0
    while not done:
        g = env.get_global_obs(obs)
        local_b = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
        global_b = torch.from_numpy(np.tile(g, (env.num_agents, 1))).to(device)
        idx_b = torch.arange(env.num_agents, device=device)
        with torch.no_grad():
            action, logprob = actor.act(local_b, idx_b)
            value = critic(global_b, idx_b)
        action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
        next_obs, R, Cp, Cs, done, _ = env.step(action_dict)
        for i, a in enumerate(env.agent_ids):
            buffer.add(
                agent_id=a, local_obs=obs[a], global_obs=g,
                action=int(action[i].item()), logprob=float(logprob[i].item()),
                reward=R[a], value=float(value[i].item()), done=done,
                c_p=Cp[a], c_s=Cs[a],
            )
            ep_R[a] += R[a]
        obs = next_obs
        n += 1
    last_v = bootstrap_last_values(critic, env.get_global_obs(obs),
                                   env.agent_ids, env.num_agents, device)
    buffer.compute_gae(last_v, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA, device=device)
    return ep_R, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30,
                        help="how many UE warm-up episodes to run (default 30 ~ 21600 steps)")
    parser.add_argument("--plot", action="store_true",
                        help="save a PNG of convergence curves at the end")
    parser.add_argument("--seed", type=int, default=C.SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device       : {device}")
    print(f"episodes     : {args.episodes}")
    print(f"steps target : {args.episodes * C.ROLLOUT_LENGTH}")
    print(f"NET_FILE     : {C.NET_FILE}")
    print(f"ROUTE_FILE   : {C.ROUTE_FILE}")
    print(f"current T_WARM (in config.py) : {C.T_WARM}")

    out_dir = os.path.join(C.OUTPUT_DIR + "_ue_diag")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "ue_diagnostic.csv")
    print(f"output dir   : {out_dir}\n")

    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
    )

    actor_ue = SharedActor(env.local_obs_dim, env.num_agents, env.action_dim, C.ACTOR_HIDDEN).to(device)
    critic_ue = SharedCritic(env.global_obs_dim, env.num_agents, C.CRITIC_HIDDEN).to(device)

    opt_a = torch.optim.Adam(actor_ue.parameters(), lr=C.ACTOR_LR)
    opt_c = torch.optim.Adam(critic_ue.parameters(), lr=C.CRITIC_LR)

    fields = [
        "episode", "global_step", "wall_time_s",
        "reward_mean", "reward_min", "reward_max", "reward_std",
        "policy_loss", "value_loss", "entropy", "approx_kl", "explained_var",
    ]
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=fields)
    writer.writeheader()

    t0 = time.time()
    gstep = 0
    history = []

    for ep in range(1, args.episodes + 1):
        buf = RolloutBuffer(env.agent_ids, env.num_agents)
        ep_R, n = collect_one_episode(env, actor_ue, critic_ue, buf, device, seed=args.seed + ep)
        gstep += n

        st = ppo_update(
            actor=actor_ue, critic=critic_ue,
            actor_optim=opt_a, critic_optim=opt_c, buffer=buf,
            ppo_epochs=C.PPO_EPOCHS, minibatch_size=C.MINIBATCH_SIZE,
            clip_eps=C.CLIP_EPS, entropy_coeff=C.ENTROPY_COEFF,
            vf_coeff=C.VF_COEFF, grad_clip=C.GRAD_CLIP,
        )

        rR = np.array(list(ep_R.values()))
        et = time.time() - t0
        row = {
            "episode": ep, "global_step": gstep, "wall_time_s": et,
            "reward_mean": float(rR.mean()), "reward_min": float(rR.min()),
            "reward_max": float(rR.max()), "reward_std": float(rR.std()),
            "policy_loss": st["policy_loss"], "value_loss": st["value_loss"],
            "entropy": st["entropy"], "approx_kl": st["approx_kl"],
            "explained_var": st["explained_var"],
        }
        writer.writerow(row)
        log_file.flush()
        history.append(row)

        print(f"ep={ep:3d} step={gstep:6d} t={et:5.0f}s "
              f"R̄={rR.mean():+8.1f} (min={rR.min():+8.1f} max={rR.max():+8.1f} std={rR.std():6.1f}) "
              f"ploss={st['policy_loss']:+.4f} vloss={st['value_loss']:8.0f} "
              f"H={st['entropy']:.3f} KL={st['approx_kl']:.4f} EV={st['explained_var']:+.3f}")

    log_file.close()
    print(f"\nCSV saved: {log_path}")

    # Quick convergence verdict
    if len(history) >= 10:
        recent = history[-5:]
        earlier = history[-10:-5]
        recent_R = np.mean([r["reward_mean"] for r in recent])
        earlier_R = np.mean([r["reward_mean"] for r in earlier])
        recent_EV = np.mean([r["explained_var"] for r in recent])
        rel_change = abs(recent_R - earlier_R) / (abs(earlier_R) + 1e-6)
        print(f"\n[verdict] last-5 vs prev-5 R̄ rel change = {rel_change*100:.2f}%   "
              f"recent EV = {recent_EV:+.3f}")
        if rel_change < 0.05 and recent_EV > 0.3:
            print("[verdict] looks CONVERGED — current T_WARM may be adequate "
                  "if it reaches this point.")
        else:
            print("[verdict] NOT YET CONVERGED — bump T_WARM higher than where R̄ "
                  "stops drifting >5% per 5-episode window.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot")
            return
        eps = [r["episode"] for r in history]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes[0, 0].plot(eps, [r["reward_mean"] for r in history]); axes[0, 0].set_title("reward_mean")
        axes[0, 1].plot(eps, [r["reward_std"]  for r in history]); axes[0, 1].set_title("reward_std (cross-agent)")
        axes[0, 2].plot(eps, [r["value_loss"]  for r in history]); axes[0, 2].set_title("value_loss"); axes[0, 2].set_yscale("log")
        axes[1, 0].plot(eps, [r["explained_var"] for r in history]); axes[1, 0].set_title("explained_var")
        axes[1, 1].plot(eps, [r["entropy"]       for r in history]); axes[1, 1].set_title("entropy")
        axes[1, 2].plot(eps, [r["approx_kl"]     for r in history]); axes[1, 2].set_title("approx_kl")
        for ax in axes.flat:
            ax.set_xlabel("episode"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(out_dir, "ue_diagnostic.png")
        plt.savefig(png_path, dpi=120)
        print(f"PNG saved: {png_path}")


if __name__ == "__main__":
    main()
