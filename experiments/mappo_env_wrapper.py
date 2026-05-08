"""
MAPPO Environment Wrapper for sumo-rl.

Wraps the PettingZoo ParallelEnv to convert each agent's observation
from a flat Box to a Dict containing:
    "local_obs": the agent's own observation (same as before)
    "global_obs": concatenation of ALL agents' observations (centralised critic input)

Also loads pre-trained IPPO baseline rewards and injects them into
each TrafficSignal for sacrifice gap computation (paper Eq. 24).
"""

import glob
import os

import numpy as np
import pandas as pd
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MAPPOEnvWrapper(MultiAgentEnv):
    """Wraps a PettingZoo ParallelEnv for MAPPO with centralised critic.

    Args:
        par_env: PettingZoo ParallelEnv instance.
        ippo_csv_dir: Directory containing IPPO CSV result files
                      (e.g. "outputs/ippo_ped_server_20260419/").
                      If None, sacrifice gap defaults to 0.
        ippo_csv_prefix: Filename prefix for IPPO CSVs (default "ippo_ped").
    """

    def __init__(self, par_env, ippo_csv_dir=None, ippo_csv_prefix="ippo_ped"):
        super().__init__()
        self.par_env = par_env
        self.ippo_csv_dir = ippo_csv_dir
        self.ippo_csv_prefix = ippo_csv_prefix

        # Get agent IDs and spaces from the parallel env
        self.par_env.reset()
        self._agent_ids = set(self.par_env.agents)
        self.agents = list(self.par_env.agents)

        # Get local obs space from any agent (all agents share the same space)
        sample_agent = self.agents[0]
        local_obs_space = self.par_env.observation_space(sample_agent)
        local_obs_dim = local_obs_space.shape[0]

        # Global obs = concatenation of all agents' obs
        num_agents = len(self.agents)
        global_obs_dim = local_obs_dim * num_agents

        # Dict observation space for MAPPO
        self._obs_space = spaces.Dict({
            "local_obs": spaces.Box(
                low=np.zeros(local_obs_dim, dtype=np.float32),
                high=np.ones(local_obs_dim, dtype=np.float32),
            ),
            "global_obs": spaces.Box(
                low=np.zeros(global_obs_dim, dtype=np.float32),
                high=np.ones(global_obs_dim, dtype=np.float32),
            ),
        })

        # Action space (same for all agents)
        self._action_space = self.par_env.action_space(sample_agent)

        # Store last observations for global state construction
        self._last_obs = {}

        # ── Load IPPO baseline rewards ──────────────────────────
        self._ippo_baselines = self._load_ippo_baselines()

    def _load_ippo_baselines(self) -> dict:
        """Load converged IPPO results and compute per-agent episode-total reward.

        Takes the last `converged_frac` of episodes (default: last 10%),
        sums each agent's reward over the full episode, then averages
        across those converged episodes.  This gives a stable, single
        number per agent representing "what this intersection achieves
        when acting independently."

        Returns:
            dict: {agent_id_str: mean_episode_total_reward}
                  e.g. {"1": -52.3, "2": -41.7, "5": -68.1, "6": -35.9}
        """
        if self.ippo_csv_dir is None or not os.path.isdir(self.ippo_csv_dir):
            print("[FairShapley] No IPPO baseline dir provided; sacrifice gap = 0")
            return {}

        pattern = os.path.join(self.ippo_csv_dir, f"{self.ippo_csv_prefix}*.csv")
        csv_files = sorted(glob.glob(pattern))

        if not csv_files:
            print(f"[FairShapley] No IPPO CSVs found matching {pattern}; sacrifice gap = 0")
            return {}

        # Use the last 10% of episodes as converged results
        converged_frac = 0.1
        n_converged = max(1, int(len(csv_files) * converged_frac))
        converged_files = csv_files[-n_converged:]

        # For each converged episode, sum per-agent reward over the episode
        agent_episode_totals = {a: [] for a in self.agents}

        for f in converged_files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"[FairShapley] Warning: could not read {f}: {e}")
                continue

            for agent_id in self.agents:
                col = f"{agent_id}_reward"
                if col not in df.columns:
                    continue
                episode_total = df[col].dropna().sum()
                agent_episode_totals[agent_id].append(episode_total)

        baselines = {}
        for agent_id in self.agents:
            totals = agent_episode_totals[agent_id]
            if totals:
                baselines[agent_id] = sum(totals) / len(totals)
            else:
                baselines[agent_id] = 0.0

        print(f"[FairShapley] IPPO baselines from last {n_converged}/{len(csv_files)} converged episodes:")
        for a, v in baselines.items():
            print(f"  Agent {a}: mean episode-total reward = {v:.4f}")

        return baselines

    def _inject_ippo_baselines(self):
        """Inject IPPO baseline rewards into TrafficSignal objects.

        Converts episode-total baseline to per-step by dividing by the
        number of steps per episode (num_seconds / delta_time).
        Called after each reset since TrafficSignal objects are rebuilt.
        """
        if not self._ippo_baselines:
            return

        # Walk the full PZ wrapper chain to find SumoEnvironment
        # Chain: par_env → .aec_env → .env → .env → ... → SumoEnvironment
        sumo_env = None
        obj = self.par_env

        # Try .aec_env first (PZ parallel-to-AEC wrapper)
        if hasattr(obj, 'aec_env'):
            obj = obj.aec_env

        # Walk .env until we find traffic_signals
        for _ in range(10):  # safety limit
            if hasattr(obj, 'traffic_signals'):
                sumo_env = obj
                break
            elif hasattr(obj, 'env'):
                obj = obj.env
            elif hasattr(obj, 'unwrapped'):
                obj = obj.unwrapped
            else:
                break

        if sumo_env is None:
            print("[FairShapley] WARNING: could not find SumoEnvironment in wrapper chain")
            return

        # Number of decision steps per episode
        steps_per_ep = sumo_env.sim_max_time / sumo_env.delta_time

        for ts_id, ts_obj in sumo_env.traffic_signals.items():
            if ts_id in self._ippo_baselines:
                # Convert episode-total to per-step average
                per_step = self._ippo_baselines[ts_id] / max(steps_per_ep, 1.0)
                ts_obj.set_ippo_reference(per_step)

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def _build_mappo_obs(self, obs_dict):
        """Convert {agent: local_obs_array} to {agent: {"local_obs": ..., "global_obs": ...}}."""
        # Build global obs: concatenate all agents' obs in fixed order
        global_obs = np.concatenate(
            [obs_dict.get(a, np.zeros_like(next(iter(obs_dict.values())))) for a in self.agents],
            dtype=np.float32,
        )

        mappo_obs = {}
        for agent_id, local_obs in obs_dict.items():
            mappo_obs[agent_id] = {
                "local_obs": local_obs.astype(np.float32),
                "global_obs": global_obs.copy(),
            }
        return mappo_obs

    def reset(self, *, seed=None, options=None):
        obs, infos = self.par_env.reset(seed=seed, options=options)
        self._last_obs = obs



        mappo_obs = self._build_mappo_obs(obs)
        return mappo_obs, infos

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = self.par_env.step(action_dict)
        self._last_obs.update(obs)
        mappo_obs = self._build_mappo_obs(self._last_obs)

        # RLlib expects "__all__" key for done signals
        terminateds["__all__"] = all(terminateds.get(a, False) for a in self.agents)
        truncateds["__all__"] = all(truncateds.get(a, False) for a in self.agents)

        return mappo_obs, rewards, terminateds, truncateds, infos