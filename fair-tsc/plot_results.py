"""Plot Fair-TSC training curves from train_log.csv.

Usage:
    python plot_results.py <path_to_train_log.csv>
    python plot_results.py                          # auto-pick latest run
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pull current budgets so annotations match the run that was actually performed.
sys.path.insert(0, os.path.dirname(__file__))
import config as C


def find_latest_log():
    base = os.path.join(os.path.dirname(__file__), "..", "outputs")
    candidates = sorted(glob.glob(os.path.join(base, "*fair_tsc_*", "train_log.csv")))
    if not candidates:
        raise FileNotFoundError(f"No train_log.csv under {base}")
    return candidates[-1]


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    print(f"Reading {log_path}")
    df = pd.read_csv(log_path)

    s1 = df[df.stage == 1]
    s2 = df[df.stage == 2]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    # 1) Reward
    ax = axes[0, 0]
    ax.plot(df.episode, df.reward_mean, label="mean", lw=1.2)
    ax.fill_between(df.episode, df.reward_min, df.reward_max, alpha=0.2, label="min-max")
    if len(s1):
        ax.axvspan(s1.episode.min() - 0.5, s1.episode.max() + 0.5, color="orange", alpha=0.15, label="Stage 1 (UE)")
    ax.set_title("Reward (per episode)"); ax.set_xlabel("episode"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # 2) Per-agent reward (Stage 2 only, smoothed)
    ax = axes[0, 1]
    agent_cols = [c for c in df.columns if c.startswith("reward_") and c not in ("reward_mean", "reward_min", "reward_max")]
    win = max(1, len(s2) // 30)
    for c in agent_cols:
        ax.plot(s2.episode, s2[c].rolling(win, min_periods=1).mean(), label=c.replace("reward_", "agent "))
    ax.set_title(f"Per-agent reward (Stage 2, smoothed w={win})"); ax.set_xlabel("episode"); ax.legend(); ax.grid(alpha=0.3)

    # 3) Theil index
    ax = axes[0, 2]
    ax.plot(s2.episode, s2.theil, lw=1)
    ax.axhline(C.T_MAX, ls="--", c="r", label=f"T_MAX={C.T_MAX}")
    ax.set_title("Theil-T index (fairness)"); ax.set_xlabel("episode"); ax.legend(); ax.grid(alpha=0.3)

    # 4) Sacrifice gap
    ax = axes[1, 0]
    ax.plot(s2.episode, s2.delta_mean, label="mean")
    ax.plot(s2.episode, s2.delta_max, label="max", alpha=0.6)
    ax.set_title("Sacrifice gap δ = [V^UE - V^MARL]_+"); ax.set_xlabel("episode"); ax.legend(); ax.grid(alpha=0.3)

    # 5) Lagrangians
    ax = axes[1, 1]
    ax.plot(s2.episode, s2.lambda_p_mean, label="λ_p mean")
    ax.plot(s2.episode, s2.lambda_s_mean, label="λ_s mean")
    ax.plot(s2.episode, s2.mu, label="µ")
    ax.set_title("Lagrangian multipliers (non-zero ⇒ constraint binding)")
    ax.set_xlabel("episode"); ax.legend(); ax.grid(alpha=0.3)

    # 6) Constraint costs (per-step means; budgets are per-step too)
    ax = axes[1, 2]
    ax.plot(df.episode, df.C_p_mean / C.ROLLOUT_LENGTH, label="C_p (per-step mean)")
    ax.plot(df.episode, df.C_s_mean / C.ROLLOUT_LENGTH, label="C_s (per-step mean)")
    ax.axhline(C.D_P, ls="--", c="C0", alpha=0.4, label=f"D_P={C.D_P}")
    ax.axhline(C.D_S, ls="--", c="C1", alpha=0.4, label=f"D_S={C.D_S}")
    ax.set_title("Constraint costs vs. budgets")
    ax.set_xlabel("episode"); ax.legend(); ax.grid(alpha=0.3)

    # 7) Entropy
    ax = axes[2, 0]
    ax.plot(df.episode, df.entropy, lw=1)
    ax.set_title("Policy entropy (action: 3 phases)"); ax.set_xlabel("episode"); ax.grid(alpha=0.3)

    # 8) Losses
    ax = axes[2, 1]
    ax.plot(df.episode, df.policy_loss, label="policy_loss")
    ax2 = ax.twinx()
    ax2.plot(df.episode, df.value_loss, c="orange", label="value_loss", alpha=0.7)
    ax.set_title("Policy / Value loss"); ax.set_xlabel("episode")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=0.3)

    # 9) Diagnostics
    ax = axes[2, 2]
    ax.plot(df.episode, df.approx_kl, label="approx_kl")
    ax.plot(df.episode, df.explained_var, label="explained_var")
    ax.set_title("PPO diagnostics"); ax.set_xlabel("episode"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"Fair-TSC training curves — {os.path.basename(os.path.dirname(log_path))}", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(log_path), "training_curves.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Print summary numbers
    print("\n=== Summary ===")
    print(f"Total episodes: {len(df)}  (Stage 1: {len(s1)}, Stage 2: {len(s2)})")
    print(f"Total steps:    {df.global_step.iloc[-1]}")
    print(f"Wall time:      {df.wall_time_s.iloc[-1] / 60:.1f} min")
    if len(s2):
        print(f"Reward (S2 start → end):        {s2.reward_mean.iloc[0]:+.1f}  →  {s2.reward_mean.iloc[-1]:+.1f}")
        print(f"Reward (last 20-ep mean):       {s2.reward_mean.tail(20).mean():+.1f}")
        theil_end = s2.theil.tail(20).mean()
        print(f"Theil  (last 20-ep mean):       {theil_end:.4f}  (budget T_MAX={C.T_MAX})  "
              f"{'BINDS' if theil_end > C.T_MAX else 'satisfied'}")
    cp_step = df.C_p_mean.tail(20).mean() / C.ROLLOUT_LENGTH
    cs_step = df.C_s_mean.tail(20).mean() / C.ROLLOUT_LENGTH
    print(f"C_p    (last 20-ep per-step):   {cp_step:.4f}  (budget D_P={C.D_P})  "
          f"{'BINDS' if cp_step > C.D_P else 'satisfied'}")
    print(f"C_s    (last 20-ep per-step):   {cs_step:.4f}  (budget D_S={C.D_S})  "
          f"{'BINDS' if cs_step > C.D_S else 'satisfied'}")
    print(f"Entropy (start → end):          {df.entropy.iloc[0]:.3f} → {df.entropy.iloc[-1]:.3f}")
    if len(s2):
        print(f"λ_p / λ_s / µ (end):            "
              f"{s2.lambda_p_mean.iloc[-1]:.3f} / {s2.lambda_s_mean.iloc[-1]:.3f} / {s2.mu.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
