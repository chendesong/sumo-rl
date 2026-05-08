"""
IPPO (Parameter-Sharing Independent PPO) for 2x2 SUMO Grid
WITH pedestrian observation + jaywalking reward
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
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/2x2grid_ippo")
RAY_RESULTS = os.path.join(BASE_DIR, "ray_results")

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import sumo_rl
from sumo_rl.environment.observations import PedestrianObservationFunction

# ── Simulation parameters ────────────────────────────────────
NUM_SECONDS = 3600
DELTA_TIME = 5
YELLOW_TIME = 5      # pedestrian clearance time; must be >= walk time
MIN_GREEN = 5
TOTAL_TIMESTEPS = 300_000

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ray.init(num_cpus=1, num_gpus=0)

    env_name = "2x2grid_ped"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=os.path.join(BASE_DIR, "nets/2x2grid/01.net.xml"),
                route_file=os.path.join(BASE_DIR, "nets/2x2grid/02.rou.xml"),
                out_csv_name=os.path.join(OUTPUT_DIR, "ippo_ped"),
                use_gui=False,
                num_seconds=NUM_SECONDS,
                delta_time=DELTA_TIME,
                yellow_time=YELLOW_TIME,
                min_green=MIN_GREEN,
                reward_fn="queue-ped",
                observation_class=PedestrianObservationFunction,
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
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
        )
        .debugging(log_level="WARN")
        .framework(framework="torch")
        .resources(num_gpus=0)
    )

    tune.run(
        "PPO",
        name="PPO_2x2_ped",
        stop={"timesteps_total": TOTAL_TIMESTEPS},
        checkpoint_freq=0,
        checkpoint_at_end=True,
        storage_path=os.path.join(RAY_RESULTS, env_name),
        config=config,
        verbose=1,
    )

    ray.shutdown()