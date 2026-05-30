#!/usr/bin/env bash
set -euo pipefail

# Minimal Pareto sweep for the Fair-TSC paper.
# Run inside a tmux session on the server:
#   bash run_hyperparam_sweep.sh
#
# Main axis: FAIR_TSC_C_TARGET.
# Lower target = stricter fairness pressure; higher target = looser,
# usually better efficiency. Keep alpha fixed for the main Pareto plot.

cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/tmp/sumo_fake}"
mkdir -p "$SUMO_HOME/tools"
export LIBSUMO_AS_TRACI="${LIBSUMO_AS_TRACI:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

export FAIR_TSC_FAIRNESS_ENABLED=1
export FAIR_TSC_CREDIT_MODE="${FAIR_TSC_CREDIT_MODE:-per_agent}"
export FAIR_TSC_DEMAND="${FAIR_TSC_DEMAND:-high}"
export FAIR_TSC_ALPHA="${FAIR_TSC_ALPHA:-0.5}"

# Vanilla MAPPO calibration refs from the completed high-demand run.
export FAIR_TSC_T_INTER_0="${FAIR_TSC_T_INTER_0:-0.28911206051707267}"
export FAIR_TSC_T_INTRA_0="${FAIR_TSC_T_INTRA_0:-0.14331125381978083}"

# For quick screening, override before running:
#   FAIR_TSC_TOTAL_STEPS=100000 bash run_hyperparam_sweep.sh
export FAIR_TSC_TOTAL_STEPS="${FAIR_TSC_TOTAL_STEPS:-300000}"

TARGETS=(${FAIR_TSC_SWEEP_TARGETS:-0.8 1.0 1.2})

for target in "${TARGETS[@]}"; do
  export FAIR_TSC_C_TARGET="$target"
  echo "=== Fair-TSC sweep: demand=$FAIR_TSC_DEMAND alpha=$FAIR_TSC_ALPHA C_target=$FAIR_TSC_C_TARGET steps=$FAIR_TSC_TOTAL_STEPS ==="
  python -u train.py 2>&1 | tee "sweep_alpha${FAIR_TSC_ALPHA}_ctarget${FAIR_TSC_C_TARGET}_$(date +%Y%m%d_%H%M).log"
done
