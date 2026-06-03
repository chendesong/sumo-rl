"""Plot Fair-TSC / MAPPO-calibration training curves."""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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


def _rolling_window(df, default=20):
    if len(df) <= 1:
        return 1
    return max(1, min(default, len(df) // 5))


def _fairness_enabled(df):
    if "fairness_enabled" not in df.columns:
        return not bool(len(df)) or "mappo_calib" not in str(df.attrs.get("run_name", ""))
    values = pd.to_numeric(df["fairness_enabled"], errors="coerce").dropna()
    return bool(len(values) and values.max() > 0)


REWARD_DIAGNOSTIC_COLS = {
    "reward_vehicle_component",
    "reward_ped_component",
    "reward_env_component_sum",
    "fair_penalty_mean",
    "fair_penalty_max",
    "reward_after_fair_proxy",
    "reward_norm_enabled",
    "reward_norm_mean",
    "reward_norm_var",
}


def _print_window_stats(df, window, columns):
    tail = df.tail(min(window, len(df)))
    print(f"\nLast{len(tail)} mean +/- std:")
    for col, label in columns:
        if col not in tail.columns:
            continue
        values = pd.to_numeric(tail[col], errors="coerce").dropna()
        if len(values) == 0:
            continue
        print(f"  {label:24s} {values.mean():+10.4f} +/- {values.std(ddof=0):.4f}")


def _mean_num(df, col):
    if col not in df.columns or len(df) == 0:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def _fmt(value, digits=4):
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _print_run_context(df, log_path, is_fair_run):
    run_name, demand, seed = _run_context_from_log(log_path)
    print("\n=== Run Context ===")
    print(f"run_name:          {run_name}")
    print(f"demand:            {demand}")
    print(f"seed:              {seed}")
    print(f"fairness_enabled:  {int(is_fair_run)}")
    for col, label in [
        ("fair_credit_mode", "credit_mode"),
        ("fair_target", "fair_target"),
        ("reward_norm_enabled", "reward_norm"),
        ("reward_norm_mean", "reward_norm_mean_last"),
        ("reward_norm_var", "reward_norm_var_last"),
    ]:
        if col not in df.columns:
            continue
        value = df[col].dropna().iloc[-1] if len(df[col].dropna()) else ""
        print(f"{label:18s} {value}")


def _print_baseline_comparison(log_path, tail):
    baselines, baseline_path = _load_rule_based_baselines(log_path)
    if baselines is None or baselines.empty:
        print("\n=== Rule Baseline Comparison ===")
        print("No matching rule_based_baselines.csv found.")
        return

    print("\n=== Rule Baseline Comparison ===")
    print(f"baseline_csv: {baseline_path}")
    run_means = {
        "vehicle_queue_mean": _mean_num(tail, "vehicle_queue_mean"),
        "ped_queue_mean": _mean_num(tail, "ped_queue_mean"),
        "teleported_total": _mean_num(tail, "teleported_total"),
        "completion_rate_departed": _mean_num(tail, "completion_rate_departed"),
        "max_phase_interval": _mean_num(tail, "max_phase_interval"),
    }
    print(
        "run last-window: "
        f"vehQ={_fmt(run_means['vehicle_queue_mean'], 2)} "
        f"pedQ={_fmt(run_means['ped_queue_mean'], 2)} "
        f"tele={_fmt(run_means['teleported_total'], 2)} "
        f"completion={_fmt(run_means['completion_rate_departed'], 4)} "
        f"maxPhase={_fmt(run_means['max_phase_interval'], 1)}"
    )
    for _, row in baselines.iterrows():
        method = row.get("method", "baseline")
        parts = [f"{method}:"]
        for col, label, digits in [
            ("vehicle_queue_mean", "vehQ", 2),
            ("ped_queue_mean", "pedQ", 2),
            ("teleported_total", "tele", 2),
            ("completion_rate_departed", "completion", 4),
            ("max_phase_interval", "maxPhase", 1),
        ]:
            if col not in row or pd.isna(row[col]):
                continue
            val = float(row[col])
            delta = run_means[col] - val if col in run_means and not pd.isna(run_means[col]) else np.nan
            parts.append(f"{label}={_fmt(val, digits)} (run-baseline {delta:+.{digits}f})")
        print("  " + "  ".join(parts))


def _print_fairness_effect(s2):
    if len(s2) == 0:
        return
    head = s2.head(min(20, len(s2)))
    tail = s2.tail(min(20, len(s2)))
    print("\n=== Fairness Effect First-vs-Last ===")
    for col, label in [
        ("theil_inter", "T_inter"),
        ("theil_intra", "T_intra"),
        ("C_fair_raw", "C_fair_raw"),
        ("C_fair_ema", "C_fair_ema"),
        ("lambda_fair", "lambda_fair"),
        ("fair_penalty_mean", "fair_penalty_mean"),
        ("fair_penalty_max", "fair_penalty_max"),
        ("max_phase_interval", "max_phase_interval"),
        ("delta_max", "delta_max"),
        ("delta_mean", "delta_mean"),
    ]:
        if col not in s2.columns:
            continue
        first = _mean_num(head, col)
        last = _mean_num(tail, col)
        print(f"{label:20s} first={_fmt(first)}  last={_fmt(last)}  change={_fmt(last - first)}")


def _print_per_agent_fairness(log_path, last_n=20):
    path = os.path.join(os.path.dirname(log_path), "per_agent_log.csv")
    print("\n=== Per-Agent Fairness Diagnostics ===")
    if not os.path.exists(path):
        print("No per_agent_log.csv found.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("per_agent_log.csv is empty.")
        return
    episodes = sorted(pd.to_numeric(df["episode"], errors="coerce").dropna().unique())
    if not episodes:
        print("No valid per-agent episodes.")
        return
    keep = set(episodes[-min(last_n, len(episodes)):])
    tail = df[df["episode"].isin(keep)].copy()
    numeric_cols = [c for c in ["delta_mean", "T_inter_i", "T_intra_i", "c_fair_i"] if c in tail.columns]
    for col in numeric_cols:
        tail[col] = pd.to_numeric(tail[col], errors="coerce")
    grouped = tail.groupby("agent", as_index=False)[numeric_cols].mean()
    if grouped.empty:
        print("No grouped per-agent data.")
        return
    for col, title in [
        ("c_fair_i", "highest fairness credit"),
        ("T_inter_i", "highest inter contribution"),
        ("T_intra_i", "highest intra contribution"),
        ("delta_mean", "largest sacrifice gap"),
    ]:
        if col not in grouped.columns:
            continue
        print(f"\nTop agents by {title}:")
        for _, row in grouped.sort_values(col, ascending=False).head(5).iterrows():
            details = " ".join(f"{c}={_fmt(row[c])}" for c in numeric_cols)
            print(f"  {row['agent']}: {details}")


def _run_context_from_log(log_path):
    run_name = os.path.basename(os.path.dirname(log_path))
    demand = None
    seed = None
    marker = "_4x4_"
    if marker in run_name:
        suffix = run_name.split(marker, 1)[1]
        if "_s" in suffix:
            demand = suffix.rsplit("_s", 1)[0]
            seed_part = suffix.rsplit("_s", 1)[1].split("_", 1)[0]
            if seed_part.isdigit():
                seed = seed_part
    return run_name, demand, seed


def _load_rule_based_baselines(log_path):
    env_csv = os.environ.get("FAIR_TSC_BASELINE_CSV")
    candidates = []
    if env_csv:
        candidates.append(env_csv)

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    _run_name, demand, seed = _run_context_from_log(log_path)
    if demand and seed:
        candidates.extend(
            glob.glob(os.path.join(base, f"rule_based_{demand}_s{seed}_*", "rule_based_baselines.csv"))
        )
    if demand:
        candidates.extend(glob.glob(os.path.join(base, f"rule_based_{demand}_*", "rule_based_baselines.csv")))
    if os.environ.get("FAIR_TSC_ALLOW_ANY_BASELINE", "0").lower() in {"1", "true", "yes"}:
        candidates.extend(glob.glob(os.path.join(base, "rule_based_*", "rule_based_baselines.csv")))

    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        return pd.DataFrame(), None

    baseline_path = sorted(set(existing), key=os.path.getmtime)[-1]
    try:
        return pd.read_csv(baseline_path), baseline_path
    except Exception as exc:
        print(f"Warning: failed to read baseline CSV {baseline_path}: {exc}")
        return pd.DataFrame(), None


def _overlay_baseline_lines(ax, baselines, column, transform=None):
    if baselines is None or baselines.empty or column is None:
        return
    transform = transform or (lambda value, _row: value)
    styles = {
        "fixed_time": ("#7a7a7a", "--"),
        "max_pressure": ("#c44e52", ":"),
    }
    for _, row in baselines.iterrows():
        method = str(row.get("method", "baseline"))
        if column not in row or pd.isna(row[column]):
            continue
        value = transform(float(row[column]), row)
        color, linestyle = styles.get(method, ("#555555", "-."))
        ax.axhline(value, color=color, linestyle=linestyle, lw=1.6, alpha=0.9, label=f"{method} baseline")


def _plot_efficiency_components(df, log_path, reward_win):
    baselines, baseline_path = _load_rule_based_baselines(log_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    plotted = False
    for col, label in [
        ("reward_vehicle_component", "vehicle term"),
        ("reward_ped_component", "pedestrian term"),
        ("reward_env_component_sum", "vehicle + pedestrian"),
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            ax.plot(df.episode, values.rolling(reward_win, min_periods=1).mean(), label=label, lw=1.8)
            plotted = True
    _overlay_baseline_lines(ax, baselines, "reward_env_component_sum")
    ax.set_title("Reward Components (rolling)")
    ax.set_xlabel("episode")
    if plotted or not baselines.empty:
        ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    plotted = False
    for col, label in [
        ("vehicle_queue_mean", "vehicle queue mean"),
        ("ped_queue_mean", "pedestrian queue mean"),
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            ax.plot(df.episode, values.rolling(reward_win, min_periods=1).mean(), label=label, lw=1.8)
            plotted = True
    _overlay_baseline_lines(ax, baselines, "vehicle_queue_mean")
    ax.set_title("Queue Components (rolling)")
    ax.set_xlabel("episode")
    if plotted or not baselines.empty:
        ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    plotted = False
    for col, label in [
        ("teleported_total", "teleported total"),
        ("pending_vehicle_count", "pending vehicles"),
        ("active_vehicle_count", "active vehicles"),
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            ax.plot(df.episode, values.rolling(reward_win, min_periods=1).mean(), label=label, lw=1.5)
            plotted = True
    _overlay_baseline_lines(ax, baselines, "teleported_total")
    ax.set_title("Gridlock / Teleport Diagnostics")
    ax.set_xlabel("episode")
    if plotted or not baselines.empty:
        ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    plotted = False
    for col, label in [
        ("completion_rate_departed", "completion / departed"),
        ("completion_rate_demand", "completion / demand"),
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            ax.plot(df.episode, values.rolling(reward_win, min_periods=1).mean(), label=label, lw=1.8)
            plotted = True
    _overlay_baseline_lines(ax, baselines, "completion_rate_departed")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Completion Rate")
    ax.set_xlabel("episode")
    if plotted or not baselines.empty:
        ax.legend()
    ax.grid(alpha=0.3)

    suffix = ""
    if baseline_path:
        suffix = f"\nrule baselines: {os.path.basename(os.path.dirname(baseline_path))}"
        print(f"Using rule-based baselines from {baseline_path}")
    fig.suptitle(os.path.basename(os.path.dirname(log_path)) + " - Efficiency Components" + suffix, fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(log_path), "efficiency_components.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def main_one(log_path):
    print(f"Reading {log_path}")
    df = pd.read_csv(log_path)
    df.attrs["run_name"] = os.path.basename(os.path.dirname(log_path))
    s1 = df[df.stage == 1]
    s2 = df[df.stage == 2]
    is_fair_run = _fairness_enabled(df)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    ax = axes[0, 0]
    reward_win = _rolling_window(s2 if len(s2) else df, default=20)
    ax.plot(df.episode, df.reward_mean, label="raw mean", lw=0.8, alpha=0.45)
    ax.plot(
        df.episode,
        df.reward_mean.rolling(reward_win, min_periods=1).mean(),
        label=f"rolling mean (w={reward_win})",
        lw=2.0,
    )
    for col, label in [
        ("reward_vehicle_component", "vehicle queue term"),
        ("reward_ped_component", "ped queue term"),
        ("reward_after_fair_proxy", "after fairness proxy"),
    ]:
        if col in df.columns:
            ax.plot(
                df.episode,
                df[col].rolling(reward_win, min_periods=1).mean(),
                label=label,
                lw=1.2,
                ls="--",
            )
    ax.fill_between(df.episode, df.reward_min, df.reward_max, alpha=0.2, label="min-max")
    if len(s1):
        ax.axvspan(s1.episode.min() - 0.5, s1.episode.max() + 0.5, color="orange", alpha=0.15, label="Stage 1")
    ax.set_title("Reward")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    agent_cols = [
        c
        for c in df.columns
        if c.startswith("reward_J")
        and c not in ("reward_mean", "reward_min", "reward_max")
        and c not in REWARD_DIAGNOSTIC_COLS
    ]
    win = max(1, len(s2) // 30) if len(s2) else 1
    for col in agent_cols:
        ax.plot(s2.episode, s2[col].rolling(win, min_periods=1).mean(), lw=0.8)
    ax.set_title(f"Per-Agent Reward (Stage 2, w={win})")
    ax.set_xlabel("episode")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    fair_win = _rolling_window(s2, default=20)
    if is_fair_run:
        for col, label in [("theil_inter", "T_inter"), ("theil_intra", "T_intra")]:
            if col in s2.columns:
                ax.plot(s2.episode, s2[col], lw=0.7, alpha=0.35)
                ax.plot(
                    s2.episode,
                    s2[col].rolling(fair_win, min_periods=1).mean(),
                    label=f"{label} rolling",
                    lw=1.8,
                )
        ax.set_title("Dual-Level Fairness")
        ax.set_xlabel("episode")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    ax = axes[1, 0]
    if is_fair_run:
        _plot_if_present(ax, s2, "episode", [("delta_mean", "delta mean"), ("delta_max", "delta max")])
        ax.set_title("Sacrifice Gap")
        ax.set_xlabel("episode")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    if is_fair_run:
        _plot_if_present(
            ax,
            s2,
            "episode",
            [("lambda_fair", "lambda_fair"), ("C_fair_ema", "C_fair EMA"), ("fair_target", "target")],
        )
        ax.set_title("PID Fairness Weight")
    else:
        for col, label in [
            ("vehicle_queue_mean", "vehicle queue"),
            ("ped_queue_mean", "pedestrian queue"),
        ]:
            if col in s2.columns:
                ax.plot(
                    s2.episode,
                    s2[col].rolling(fair_win, min_periods=1).mean(),
                    label=f"{label} rolling",
                    lw=1.8,
                )
        ax.set_title("Queue Components")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    if "max_phase_interval" in s2.columns:
        ax.plot(s2.episode, s2.max_phase_interval, lw=0.7, alpha=0.35)
        ax.plot(
            s2.episode,
            s2.max_phase_interval.rolling(fair_win, min_periods=1).mean(),
            label=f"max phase interval rolling (w={fair_win})",
            lw=1.8,
        )
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
    _plot_efficiency_components(df, log_path, reward_win)

    print("\n=== Summary ===")
    _print_run_context(df, log_path, is_fair_run)
    print(f"Total episodes: {len(df)}  (Stage 1: {len(s1)}, Stage 2: {len(s2)})")
    print(f"Total steps:    {df.global_step.iloc[-1]}")
    print(f"Wall time:      {df.wall_time_s.iloc[-1] / 60:.1f} min")
    if len(s2):
        tail = s2.tail(min(20, len(s2)))
        head = s2.head(min(20, len(s2)))
        print(f"Reward last mean:       {tail.reward_mean.mean():+.2f}")
        print(f"Reward first mean:      {head.reward_mean.mean():+.2f}")
        print(f"Reward last-first:      {tail.reward_mean.mean() - head.reward_mean.mean():+.2f}")
        print(f"T_inter last mean:      {tail.theil_inter.mean():.6f}")
        print(f"T_intra last mean:      {tail.theil_intra.mean():.6f}")
        print(f"Max phase interval:     {tail.max_phase_interval.mean():.2f}")
        if is_fair_run and "lambda_fair" in tail.columns:
            print(f"lambda_fair end:        {tail.lambda_fair.iloc[-1]:.6f}")
        diagnostic_cols = [
            ("reward_mean", "reward mean"),
            ("theil_inter", "T_inter"),
            ("theil_intra", "T_intra"),
            ("max_phase_interval", "max phase interval"),
            ("reward_vehicle_component", "vehicle queue term"),
            ("reward_ped_component", "ped queue term"),
            ("reward_after_fair_proxy", "after fairness proxy"),
            ("vehicle_queue_mean", "vehicle queue mean"),
            ("ped_queue_mean", "ped queue mean"),
            ("fair_penalty_mean", "fair penalty mean"),
            ("fair_penalty_max", "fair penalty max"),
            ("teleported_total", "teleported total"),
            ("completion_rate_departed", "completion departed"),
        ]
        _print_window_stats(s2, 20, diagnostic_cols)
        _print_window_stats(s2, 50, diagnostic_cols)
        _print_baseline_comparison(log_path, tail)
        if is_fair_run:
            _print_fairness_effect(s2)
            _print_per_agent_fairness(log_path, last_n=20)


def main_multi(log_paths):
    dfs = []
    labels = []
    for path in log_paths:
        df = pd.read_csv(path)
        s2 = df[df.stage == 2].copy()
        if len(s2) == 0:
            continue
        s2 = s2.reset_index(drop=True)
        dfs.append(s2)
        labels.append(os.path.basename(os.path.dirname(path)))

    if not dfs:
        raise ValueError("No Stage-2 rows found in the supplied logs.")

    n = min(len(df) for df in dfs)
    xs = np.arange(1, n + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    cols = [
        ("reward_mean", "Reward Mean"),
        ("theil_inter", "T_inter"),
        ("theil_intra", "T_intra"),
        ("teleported_total", "Teleported Vehicles"),
    ]
    for ax, (col, title) in zip(axes.ravel(), cols):
        present = [pd.to_numeric(df[col].iloc[:n], errors="coerce").to_numpy(dtype=float)
                   for df in dfs if col in df.columns]
        if not present:
            ax.set_title(f"{title} (missing)")
            continue
        mat = np.vstack(present)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        win = max(1, min(20, n // 5))
        mean_s = pd.Series(mean).rolling(win, min_periods=1).mean().to_numpy()
        std_s = pd.Series(std).rolling(win, min_periods=1).mean().to_numpy()
        ax.plot(xs, mean_s, lw=2.0, label=f"mean rolling (w={win})")
        ax.fill_between(xs, mean_s - std_s, mean_s + std_s, alpha=0.2, label="+/- std")
        ax.set_title(title)
        ax.set_xlabel("Stage-2 episode")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Multi-Seed Training Curves", fontsize=14)
    fig.tight_layout()
    out_dir = os.path.commonpath([os.path.dirname(p) for p in log_paths])
    if not os.path.isdir(out_dir):
        out_dir = os.path.dirname(log_paths[0])
    out_path = os.path.join(out_dir, "multi_seed_training_curves.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")
    print("Logs:")
    for label, path in zip(labels, log_paths):
        print(f"  {label}: {path}")


def main():
    if len(sys.argv) > 2:
        main_multi(sys.argv[1:])
        return
    log_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    main_one(log_path)


if __name__ == "__main__":
    main()
