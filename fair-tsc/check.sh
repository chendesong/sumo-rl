#!/bin/bash
# Check training progress. Run from anywhere:  bash ~/sumo-rl/fair-tsc/check.sh
set -e

LATEST=$(ls -td ~/sumo-rl/outputs/fair_tsc_4x4_high_*/train_log.csv 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    echo "No train_log.csv found under ~/sumo-rl/outputs/fair_tsc_4x4_high_*"
    exit 1
fi

echo "log: $LATEST"
echo "rows: $(wc -l < "$LATEST")"
echo ""
echo "=== process ==="
PID=$(pgrep -f "fair-tsc/train" || true)
if [ -n "$PID" ]; then
    ps -o pid,etime,pcpu,pmem,cmd -p "$PID"
else
    echo "NOT RUNNING (process not found)"
fi
echo ""
echo "=== last 20 Stage-2 episodes ==="
python -c "
import pandas as pd
df = pd.read_csv('$LATEST')
s2 = df[df.stage==2]
print(f'Stage 1: {(df.stage==1).sum()} ep, Stage 2: {len(s2)} ep')
print()
if len(s2):
    cols = ['episode','reward_mean','explained_var','theil','mu','lambda_p_mean','approx_kl']
    print(s2[cols].tail(20).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
else:
    print('No Stage 2 episodes yet — Stage 1 in progress.')
    print()
    print(df[['stage','episode','reward_mean','explained_var','entropy']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
"
