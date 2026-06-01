#!/bin/bash
# Check training progress. Run from anywhere: bash ~/sumo-rl/fair-tsc/check.sh
set -e
shopt -s nullglob

find_log_from_pid() {
    local pid="$1"
    local fd target
    for fd in /proc/"$pid"/fd/*; do
        target=$(readlink "$fd" 2>/dev/null || true)
        case "$target" in
            *"/outputs/"*"train_log.csv")
                echo "$target"
                return 0
                ;;
        esac
    done
    return 1
}

PIDS=$(pgrep -u "$USER" -f "train.py" || true)
LATEST=""

if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        if LOG_FROM_PID=$(find_log_from_pid "$pid"); then
            LATEST="$LOG_FROM_PID"
            break
        fi
    done
fi

if [ -z "$LATEST" ]; then
    logs=(~/sumo-rl/outputs/fair_tsc_4x4_*/train_log.csv ~/sumo-rl/outputs/mappo_calib_4x4_*/train_log.csv)
    if [ ${#logs[@]} -gt 0 ]; then
        LATEST=$(ls -t "${logs[@]}" 2>/dev/null | head -1)
    fi
fi

if [ -z "$LATEST" ]; then
    echo "No train_log.csv found under ~/sumo-rl/outputs/{fair_tsc,mappo_calib}_4x4_*"
    exit 1
fi

echo "log: $LATEST"
echo "rows: $(wc -l < "$LATEST")"
echo ""
echo "=== process ==="
if [ -n "$PIDS" ]; then
    ps -o pid,ppid,etime,time,pcpu,pmem,cmd -p $(echo "$PIDS" | tr '\n' ' ')
    echo ""
    for pid in $PIDS; do
        cwd=$(readlink -f /proc/"$pid"/cwd 2>/dev/null || true)
        [ -n "$cwd" ] && echo "pid $pid cwd: $cwd"
    done
else
    echo "NOT RUNNING (process not found)"
fi

echo ""
echo "=== last Stage-2 episodes ==="
python - <<PY
import pandas as pd
from pathlib import Path

path = Path("$LATEST")
try:
    df = pd.read_csv(path)
except pd.errors.EmptyDataError:
    print("Log file exists but is empty; first episode has not flushed yet.")
    raise SystemExit(0)

s2 = df[df.stage == 2]
print(f"Stage 1: {(df.stage == 1).sum()} ep, Stage 2: {len(s2)} ep")
print()
if len(s2):
    cols = [
        "episode", "global_step", "reward_mean", "theil_inter", "theil_intra",
        "max_phase_interval", "C_fair_raw", "lambda_fair",
        "reward_vehicle_component", "reward_ped_component", "fair_penalty_mean",
        "vehicle_queue_mean", "ped_queue_mean",
        "explained_var", "approx_kl",
    ]
    cols = [c for c in cols if c in s2.columns]
    print(s2[cols].tail(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
else:
    cols = [c for c in ["stage", "episode", "global_step", "reward_mean", "explained_var", "entropy"] if c in df.columns]
    print("No Stage 2 episodes yet; Stage 1 in progress.")
    print(df[cols].tail(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
PY
