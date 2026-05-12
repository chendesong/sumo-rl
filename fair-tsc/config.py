"""
Fair-TSC global configuration.

ONE place to tune everything. All hyperparameters from the paper live here
with explicit references to the equation / section that motivates them.
"""

import os
import time

# ═══════════════════════════════════════════════════════════════════════
# Paths (auto-detected by platform: Windows local vs. Linux monaco)
# Override with env var FAIR_TSC_BASE_DIR if needed.
# ═══════════════════════════════════════════════════════════════════════
if os.environ.get("FAIR_TSC_BASE_DIR"):
    BASE_DIR = os.environ["FAIR_TSC_BASE_DIR"]
elif os.name == "nt":
    BASE_DIR = "C:/Users/ucemdc3/PycharmProjects/sumo-rl"
else:
    BASE_DIR = os.path.expanduser("~/sumo-rl")

NET_FILE   = os.path.join(BASE_DIR, "nets/2x2grid/01.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, "nets/2x2grid/02.rou.xml")

_TS = time.strftime("%Y%m%d_%H%M")
RUN_NAME   = f"fair_tsc_{_TS}"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", RUN_NAME)
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints", RUN_NAME)

# ═══════════════════════════════════════════════════════════════════════
# SUMO simulation
# ═══════════════════════════════════════════════════════════════════════
NUM_SECONDS = 3600    # episode length (seconds of simulated time, T_ep)
DELTA_TIME  = 5       # seconds between agent decisions (Δt)
MIN_GREEN   = 5       # minimum green time before phase change allowed
USE_GUI     = False
LIBSUMO     = True    # set LIBSUMO_AS_TRACI=1 in env

STEPS_PER_EPISODE = NUM_SECONDS // DELTA_TIME  # = 720

# ═══════════════════════════════════════════════════════════════════════
# Reward — paper Eq. (15)
# R_i(t) = -Σ q^v_{i,k} - ω_p · Σ q^p_{i,c}
# ═══════════════════════════════════════════════════════════════════════
OMEGA_P = 1.0   # pedestrian queue weight — equal to vehicle queue weight (1:1)

# ═══════════════════════════════════════════════════════════════════════
# Soft constraint budgets — paper Eq. (7), (11), (19)
# ═══════════════════════════════════════════════════════════════════════
D_P    = 0.05    # moderate (1.0 → 0.02 was too tight, drove policy to extremes)
D_S    = 0.02    # moderate (C_s ≡ 0 in 2x2grid anyway)
T_MAX  = 0.02    # moderate (observed Theil ≈ 0.025; constraint will lightly bind)

# ═══════════════════════════════════════════════════════════════════════
# Training schedule — paper §III.D
# ═══════════════════════════════════════════════════════════════════════
T_WARM            = 2000     # Stage 1: UE warm-up steps (paper default)
TOTAL_STEPS       = 100_000  # short tuning run (was 300k); ~140 stage-2 episodes
ROLLOUT_LENGTH    = 720      # one episode per rollout = 720 decision steps
PPO_EPOCHS        = 10
MINIBATCH_SIZE    = 256
BATCH_SIZE        = ROLLOUT_LENGTH * 4  # 4 agents × 720 steps = 2880 transitions

# ═══════════════════════════════════════════════════════════════════════
# Optimisation — paper §III.D ("two-timescale": η_λ, η_µ ≪ η_θ, η_φ)
# ═══════════════════════════════════════════════════════════════════════
ACTOR_LR    = 3e-4
CRITIC_LR   = 1e-3
ETA_LAMBDA  = 3e-3   # moderate (1e-3 → 1e-2 too aggressive)
ETA_MU      = 3e-4   # moderate (1e-4 → 1e-3 too aggressive)

GAMMA       = 0.99
GAE_LAMBDA  = 0.95
CLIP_EPS    = 0.2
ENTROPY_COEFF = 0.01
VF_COEFF      = 0.5
GRAD_CLIP     = 0.5
TAU_TGT       = 0.005   # Polyak target-network update (paper §III.D Eq. 33)

# ═══════════════════════════════════════════════════════════════════════
# Network architecture
# ═══════════════════════════════════════════════════════════════════════
ACTOR_HIDDEN  = [256, 256]
CRITIC_HIDDEN = [256, 256]
NUM_AGENTS    = 4   # 2x2grid (will be re-checked at runtime)

# ═══════════════════════════════════════════════════════════════════════
# Theil index numerical safety — paper Eq. (18)
# ═══════════════════════════════════════════════════════════════════════
THEIL_EPS = 1e-6

# ═══════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════
LOG_EVERY_N_EPISODES = 1
SAVE_CKPT_EVERY_N    = 50  # save model every 50 episodes
SEED                 = 42