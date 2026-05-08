"""
MAPPO (Multi-Agent PPO with Centralised Critic) for 2x2 SUMO Grid
WITH pedestrian observation + jaywalking reward
──────────────────────────────────────────────────────────────
Difference from IPPO:
  - Critic sees global state (all 4 agents' obs concatenated)
  - Actor sees only local obs (same as IPPO)
  - Uses custom CentralisedCriticModel + MAPPOEnvWrapper
──────────────────────────────────────────────────────────────
Ray/RLlib 2.7.x  |  Single-core  |  Torch
"""

import os
import sys
import time

# ── Monkey-patch os.replace for Windows Defender ─────────────
_original_replace = os.replace

def _safe_replace(src, dst):
    for attempt in range(10):
        try:
            return _original_replace(src, dst)
        except PermissionError:
            time.sleep(0.5)
    return _original_replace(src, dst)

os.replace = _safe_replace

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = "C:/Users/ucemdc3/PycharmProjects/sumo-rl"
_timestamp = time.strftime("%Y%m%d_%H%M")
OUTPUT_DIR = os.path.join(BASE_DIR, f"outputs/2x2grid_mappo_{_timestamp}")
RAY_RESULTS = os.path.join(BASE_DIR, "ray_results")

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Add experiments dir to path so we can import the wrapper and model
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import sumo_rl
from sumo_rl.environment.observations import PedestrianObservationFunction

# Import MAPPO components
from centralised_critic import CentralisedCriticModel
from mappo_env_wrapper import MAPPOEnvWrapper

# ── Register custom model ────────────────────────────────────
ModelCatalog.register_custom_model("cc_model", CentralisedCriticModel)

# ── Simulation parameters ────────────────────────────────────
NUM_SECONDS = 3600
DELTA_TIME = 5
MIN_GREEN = 5
TOTAL_TIMESTEPS = 300_000

# ── IPPO baseline for sacrifice gap ──────────────────────────
IPPO_BASELINE_DIR = os.path.join(BASE_DIR, "outputs/ippo_ped_server_20260419")
IPPO_CSV_PREFIX = "ippo_ped"


def create_mappo_env(_):
    """Create the MAPPO environment with centralised obs."""
    par_env = sumo_rl.parallel_env(
        net_file=os.path.join(BASE_DIR, "nets/2x2grid/01.net.xml"),
        route_file=os.path.join(BASE_DIR, "nets/2x2grid/02.rou.xml"),
        out_csv_name=os.path.join(OUTPUT_DIR, "mappo_ped"),
        use_gui=False,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        min_green=MIN_GREEN,
        reward_fn="queue-ped",
        observation_class=PedestrianObservationFunction,
    )
    return MAPPOEnvWrapper(
        par_env,
        ippo_csv_dir=IPPO_BASELINE_DIR,
        ippo_csv_prefix=IPPO_CSV_PREFIX,
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ray.init(num_cpus=1, num_gpus=0)

    env_name = "2x2grid_mappo"
    register_env(env_name, create_mappo_env)

    # Create a temp env to get obs/action spaces
    temp_env = create_mappo_env(None)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    agent_ids = temp_env.agents
    temp_env.par_env.close()

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            disable_env_checking=True,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=128,
            sample_timeout_s=300,
        )
        .training(
            train_batch_size=2048,
            lr=5e-4,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            minibatch_size=256,
            num_epochs=10,
            model={
                "custom_model": "cc_model",
                "fcnet_hiddens": [256, 256],
            },
        )
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .debugging(log_level="WARN")
        .framework(framework="torch")
        .resources(num_gpus=0)
    )

    tune.run(
        "PPO",
        name="MAPPO_2x2_ped",
        stop={"timesteps_total": TOTAL_TIMESTEPS},
        checkpoint_freq=0,
        checkpoint_at_end=True,
        storage_path=os.path.join(RAY_RESULTS, env_name),
        config=config,
        verbose=1,
    )

    ray.shutdown()