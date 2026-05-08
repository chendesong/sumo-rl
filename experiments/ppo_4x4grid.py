import os
import sys

BASE_DIR = "C:/Users/ucemdc3/PycharmProjects/sumo-rl"
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import sumo_rl


if __name__ == "__main__":
    # Use:
    # ray[rllib]==2.7.0
    # numpy == 1.23.4
    # Pillow>=9.4.0
    # ray[rllib]==2.7.0
    # SuperSuit>=3.9.0
    # torch>=1.13.1
    # tensorflow-probability>=0.19.0
    ray.init()

    env_name = "4x4grid"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                # 使用 os.path.join 确保是绝对路径
                net_file=os.path.join(BASE_DIR, "sumo_rl/nets/4x4-Lucas/4x4.net.xml"),
                route_file=os.path.join(BASE_DIR, "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"),
                out_csv_name=os.path.join(BASE_DIR, "outputs/4x4grid/ppo"),
                use_gui=False,
                num_seconds=2000,
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        # 1. 核心修改：关闭不成熟的新 API 堆栈，回到经典稳定模式
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        # 2. 采样配置（在经典模式下，这里依然可以用 env_runners）
        .env_runners(
            num_env_runners=4,
            rollout_fragment_length=128
        )
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            minibatch_size=64,
            num_epochs=5,  # 确保这里用的是 num_epochs
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # 3. 修改 tune.run 调用的地方（Ray 2.x 建议直接传入 config 对象，不转 dict）
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 1000},
        checkpoint_freq=1,
        storage_path="C:/Users/ucemdc3/ray_results/" + env_name,
        config=config,
    )