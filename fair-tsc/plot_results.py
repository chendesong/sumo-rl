"""Plot Fair-TSC / MAPPO-calibration training curves."""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_log():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    patterns = ["fair_tsc_*", "mappo_calib_*"]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(base, pattern, "train_log.csv")))
    if not candidates:
        raise FileNotFoundError(f"No train_log.csv under {base}")
    return sorted(candidates)[-1]


def _plot_if_present(ax, df, x, columns):
    for col, label in columns:
        if col in df.columns:
            ax.plot(df[x], df[col], label=label)


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    print(f"Reading {log_path}")
    df = pd.read_csv(log_path)
    s1 = df[df.stage == 1]
    s2 = df[df.stage == 2]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    ax = axes[0, 0]
    ax.plot(df.episode, df.reward_mean, label="mean", lw=1.2)
    ax.fill_between(df.episode, df.reward_min, df.reward_max, alpha=0.2, label="min-max")
    if len(s1):
        ax.axvspan(s1.episode.min() - 0.5, s1.episode.max() + 0.5, color="orange", alpha=0.15, label="Stage 1")
    ax.set_title("Reward")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    agent_cols = [c for c in df.columns if c.startswith("reward_") and c not in ("reward_mean", "reward_min", "reward_max")]
    win = max(1, len(s2) // 30) if len(s2) else 1
    for col in agent_cols:
        ax.plot(s2.episode, s2[col].rolling(win, min_periods=1).mean(), lw=0.8)
    ax.set_title(f"Per-Agent Reward (Stage 2, w={win})")
    ax.set_xlabel("episode")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    _plot_if_present(ax, s2, "episode", [("theil_inter", "T_inter"), ("theil_intra", "T_intra")])
    ax.set_title("Dual-Level Fairness")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    _plot_if_present(ax, s2, "episode", [("delta_mean", "delta mean"), ("delta_max", "delta max")])
    ax.set_title("Sacrifice Gap")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    _plot_if_present(ax, s2, "episode", [("lambda_fair", "lambda_fair"), ("C_fair_ema", "C_fair EMA"), ("fair_target", "target")])
    ax.set_title("PID Fairness Weight")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    _plot_if_present(ax, s2, "episode", [("max_phase_interval", "max phase interval")])
    ax.set_title("Phase Service Interval")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    _plot_if_present(ax, df, "episode", [("entropy", "entropy")])
    ax.set_title("Policy Entropy")
    ax.set_xlabel("episode")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    _plot_if_present(ax, df, "episode", [("policy_loss", "policy loss")])
    ax2 = ax.twinx()
    _plot_if_present(ax2, df, "episode", [("value_loss", "value loss")])
    ax.set_title("Losses")
    ax.set_xlabel("episode")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    _plot_if_present(ax, df, "episode", [("approx_kl", "approx KL"), ("explained_var", "explained var"), ("clip_frac", "clip frac")])
    ax.set_title("PPO Diagnostics")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(os.path.basename(os.path.dirname(log_path)), fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(log_path), "training_curves.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved {out_path}")

    print("\n=== Summary ===")
    print(f"Total episodes: {len(df)}  (Stage 1: {len(s1)}, Stage 2: {len(s2)})")
    print(f"Total steps:    {df.global_step.iloc[-1]}")
    print(f"Wall time:      {df.wall_time_s.iloc[-1] / 60:.1f} min")
    if len(s2):
        tail = s2.tail(min(20, len(s2)))
        print(f"Reward last mean:       {tail.reward_mean.mean():+.2f}")
        print(f"T_inter last mean:      {tail.theil_inter.mean():.6f}")
        print(f"T_intra last mean:      {tail.theil_intra.mean():.6f}")
        print(f"Max phase interval:     {tail.max_phase_interval.mean():.2f}")
        if "lambda_fair" in tail.columns:
            print(f"lambda_fair end:        {tail.lambda_fair.iloc[-1]:.6f}")


if __name__ == "__main__":
    main()
