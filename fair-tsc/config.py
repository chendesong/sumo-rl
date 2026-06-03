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
DEMAND_LEVELS = {
    "low",
    "medium",
    "high",
    "ultra_stress",
    "curriculum_lmh",
    "curriculum_lmhu",
    "curriculum_lmhu_ped",
    "curriculum_mhu",
}
DEMAND_LEVEL = os.environ.get("FAIR_TSC_DEMAND", "curriculum_mhu").lower()
if DEMAND_LEVEL not in DEMAND_LEVELS:
    raise ValueError(f"FAIR_TSC_DEMAND must be one of: {', '.join(sorted(DEMAND_LEVELS))}")

_NET_NAME = (
    "4x4_ped_yellow.net.xml"
    if DEMAND_LEVEL in {
        "ultra_stress",
        "curriculum_lmh",
        "curriculum_lmhu",
        "curriculum_lmhu_ped",
        "curriculum_mhu",
    }
    else "4x4.net.xml"
)
NET_FILE = os.environ.get(
    "FAIR_TSC_NET_FILE",
    os.path.join(BASE_DIR, "nets/4x4grid", _NET_NAME),
)
ROUTE_FILE = os.environ.get(
    "FAIR_TSC_ROUTE_FILE",
    os.path.join(BASE_DIR, f"nets/4x4grid/4x4_{DEMAND_LEVEL}.rou.xml"),
)


# Training mode
_fairness_env = os.environ.get("FAIR_TSC_FAIRNESS_ENABLED", os.environ.get("FAIRNESS_ENABLED", "1"))
FAIRNESS_ENABLED = _env_bool("FAIR_TSC_FAIRNESS_ENABLED", _fairness_env)
SEED = int(os.environ.get("FAIR_TSC_SEED", "42"))

_TS = time.strftime("%Y%m%d_%H%M")
RUN_PREFIX = "fair_tsc" if FAIRNESS_ENABLED else "mappo_calib"
RUN_TAG = os.environ.get("FAIR_TSC_RUN_TAG", "").strip()
_RUN_TAG_PART = f"_{RUN_TAG}" if RUN_TAG else ""
RUN_NAME = f"{RUN_PREFIX}_4x4_{DEMAND_LEVEL}_s{SEED}{_RUN_TAG_PART}_{_TS}"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", RUN_NAME)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", RUN_NAME)


# SUMO simulation
NUM_SECONDS = int(os.environ.get("FAIR_TSC_NUM_SECONDS", "3600"))
DELTA_TIME = int(os.environ.get("FAIR_TSC_DELTA_TIME", "5"))
MIN_GREEN = int(os.environ.get("FAIR_TSC_MIN_GREEN", "5"))
TIME_TO_TELEPORT = int(os.environ.get("FAIR_TSC_TIME_TO_TELEPORT", "600"))
USE_GUI = False
LIBSUMO = True
STEPS_PER_EPISODE = NUM_SECONDS // DELTA_TIME


# Reward
PED_REWARD_MODE = os.environ.get("FAIR_TSC_PED_REWARD_MODE", "queue").lower()
if PED_REWARD_MODE not in {"queue", "wait", "queue_wait"}:
    raise ValueError("FAIR_TSC_PED_REWARD_MODE must be one of: queue, wait, queue_wait")
OMEGA_P = float(os.environ.get("FAIR_TSC_OMEGA_P", "1.0"))
OMEGA_PED_WAIT = float(os.environ.get("FAIR_TSC_OMEGA_PED_WAIT", "0.02"))
REWARD_SCALE = 30.0
REWARD_NORMALIZE = _env_bool("FAIR_TSC_REWARD_NORMALIZE", "1")
REWARD_NORM_CENTER = _env_bool("FAIR_TSC_REWARD_NORM_CENTER", "0")
REWARD_NORM_CLIP = float(os.environ.get("FAIR_TSC_REWARD_NORM_CLIP", "10.0"))
REWARD_NORM_EPS = 1e-8


# Dual-level fairness and PID adaptive penalty
FAIR_ALPHA = float(os.environ.get("FAIR_TSC_ALPHA", "0.5"))
FAIR_CREDIT_MODE = os.environ.get("FAIR_TSC_CREDIT_MODE", "per_agent").lower()
if FAIR_CREDIT_MODE not in {"per_agent", "global", "none"}:
    raise ValueError("FAIR_TSC_CREDIT_MODE must be one of: per_agent, global, none")

# Calibration references. Environment variables still take precedence; the
# LMHU intra reference is fixed from the seed-45/46 intra-only sensitivity run.
_DEFAULT_T_INTER_0 = {
    "curriculum_lmhu": "0.15",
}
_DEFAULT_T_INTRA_0 = {
    "curriculum_lmhu": "0.219",
}
T_INTER_0 = float(os.environ.get("FAIR_TSC_T_INTER_0", _DEFAULT_T_INTER_0.get(DEMAND_LEVEL, "1.0")))
T_INTRA_0 = float(os.environ.get("FAIR_TSC_T_INTRA_0", _DEFAULT_T_INTRA_0.get(DEMAND_LEVEL, "1.0")))
UE_CKPT = os.environ.get("FAIR_TSC_UE_CKPT", "").strip()

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
T_WARM = int(os.environ.get("FAIR_TSC_T_WARM", "7200"))
TOTAL_STEPS = int(os.environ.get("FAIR_TSC_TOTAL_STEPS", "300000"))
ROLLOUT_LENGTH = int(os.environ.get("FAIR_TSC_ROLLOUT_LENGTH", "720"))
PPO_EPOCHS = 10
MINIBATCH_SIZE = int(os.environ.get("FAIR_TSC_MINIBATCH_SIZE", "1024"))
BATCH_SIZE = ROLLOUT_LENGTH * 16


# Optimization
ACTOR_LR = float(os.environ.get("FAIR_TSC_ACTOR_LR", "2e-4"))
CRITIC_LR = float(os.environ.get("FAIR_TSC_CRITIC_LR", "5e-4"))
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
