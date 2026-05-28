"""
Fair-TSC global configuration.

Set FAIR_TSC_FAIRNESS_ENABLED=0 to run the vanilla MAPPO calibration pass.
That pass logs T_inter and T_intra; copy the final-episode means into
T_INTER_0 and T_INTRA_0 before the formal Fair-TSC run.
"""

import os
import time


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() not in {
        "0", "false", "no", "off"
    }


# Paths
if os.environ.get("FAIR_TSC_BASE_DIR"):
    BASE_DIR = os.environ["FAIR_TSC_BASE_DIR"]
elif os.name == "nt":
    BASE_DIR = "C:/Users/ucemdc3/PycharmProjects/sumo-rl"
else:
    BASE_DIR = os.path.expanduser("~/sumo-rl")


# Network and demand
DEMAND_LEVEL = "high"            # "low" / "medium" / "high"
NET_FILE = os.path.join(BASE_DIR, "nets/4x4grid/4x4.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, f"nets/4x4grid/4x4_{DEMAND_LEVEL}.rou.xml")


# Training mode
_fairness_env = os.environ.get("FAIR_TSC_FAIRNESS_ENABLED", os.environ.get("FAIRNESS_ENABLED", "1"))
FAIRNESS_ENABLED = _env_bool("FAIR_TSC_FAIRNESS_ENABLED", _fairness_env)

_TS = time.strftime("%Y%m%d_%H%M")
RUN_PREFIX = "fair_tsc" if FAIRNESS_ENABLED else "mappo_calib"
RUN_NAME = f"{RUN_PREFIX}_4x4_{DEMAND_LEVEL}_{_TS}"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", RUN_NAME)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", RUN_NAME)


# SUMO simulation
NUM_SECONDS = 3600
DELTA_TIME = 5
MIN_GREEN = 5
USE_GUI = False
LIBSUMO = True
STEPS_PER_EPISODE = NUM_SECONDS // DELTA_TIME


# Reward
OMEGA_P = 1.0
REWARD_SCALE = 30.0
REWARD_NORMALIZE = _env_bool("FAIR_TSC_REWARD_NORMALIZE", "1")
REWARD_NORM_CENTER = _env_bool("FAIR_TSC_REWARD_NORM_CENTER", "0")
REWARD_NORM_CLIP = float(os.environ.get("FAIR_TSC_REWARD_NORM_CLIP", "10.0"))
REWARD_NORM_EPS = 1e-8


# Dual-level fairness and PID adaptive penalty
FAIR_ALPHA = float(os.environ.get("FAIR_TSC_ALPHA", "0.5"))

# Placeholders for calibration. Run vanilla MAPPO with
# FAIRNESS_ENABLED=False, then replace these with the final-episode means.
T_INTER_0 = float(os.environ.get("FAIR_TSC_T_INTER_0", "1.0"))
T_INTRA_0 = float(os.environ.get("FAIR_TSC_T_INTRA_0", "1.0"))

FAIR_C_TARGET = float(os.environ.get("FAIR_TSC_C_TARGET", "1.0"))
FAIR_EPS = 1e-6

PID_KP = float(os.environ.get("FAIR_TSC_PID_KP", "0.50"))
PID_KI = float(os.environ.get("FAIR_TSC_PID_KI", "0.02"))
PID_KD = float(os.environ.get("FAIR_TSC_PID_KD", "0.10"))
PID_LAMBDA_MAX = float(os.environ.get("FAIR_TSC_PID_LAMBDA_MAX", "5.0"))
PID_INTEGRAL_MAX = float(os.environ.get("FAIR_TSC_PID_INTEGRAL_MAX", "20.0"))
PID_EMA_BETA = float(os.environ.get("FAIR_TSC_PID_EMA_BETA", "0.9"))

# If a phase is activated fewer than twice in one episode, treat its
# service interval as the full episode horizon.
PHASE_UNSERVED_INTERVAL = NUM_SECONDS


# Training schedule
T_WARM = 7200
TOTAL_STEPS = 300_000
ROLLOUT_LENGTH = 720
PPO_EPOCHS = 10
MINIBATCH_SIZE = 512
BATCH_SIZE = ROLLOUT_LENGTH * 16


# Optimization
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.01
VF_COEFF = 0.5
GRAD_CLIP = 0.5
TAU_TGT = 0.005


# Network architecture
ACTOR_HIDDEN = [256, 256]
CRITIC_HIDDEN = [256, 256]
NUM_AGENTS = 16


# Theil numerical safety
THEIL_EPS = 1e-6
THEIL_EMA_BETA = 0.9


# Logging
LOG_EVERY_N_EPISODES = 1
SAVE_CKPT_EVERY_N = 50
SEED = 42
