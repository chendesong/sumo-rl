"""
Fair-TSC environment wrapper.

Wraps sumo_rl.parallel_env to expose, ON EVERY STEP, the four quantities
the training loop needs:

    obs[i]      : local observation of agent i  (paper Eq. 13)
    reward[i]   : queue-based local reward      (paper Eq. 15)
    C_p[i]      : pedestrian non-compliance     (paper Eq. 6)
    C_s[i]      : spillback cost                (paper Eq. 10)

Plus a method for global state (concatenation of all agents' local states)
needed by the centralised critics V^MARL and V^UE.

Unlike the previous MAPPOEnvWrapper, this is NOT an RLlib MultiAgentEnv —
this is a thin functional wrapper for our hand-written PPO loop.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Tell sumo-rl to use libsumo; must be set BEFORE importing sumo_rl
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

# fake SUMO_HOME (libsumo is the actual backend)
if "SUMO_HOME" not in os.environ:
    fake = "/tmp/sumo_fake" if os.name != "nt" else os.path.join(os.getenv("TEMP", "."), "sumo_fake")
    os.makedirs(os.path.join(fake, "tools"), exist_ok=True)
    os.environ["SUMO_HOME"] = fake

import sumo_rl
from sumo_rl.environment.observations import PedestrianObservationFunction


class FairTSCEnv:
    """Thin functional wrapper around sumo_rl.parallel_env.

    Exposes per-agent (obs, reward, C_p, C_s) plus a concatenated global
    state for the centralised critics. Maintains agent ordering across
    resets so policy networks see consistent indexing.
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: str = None,
        num_seconds: int = 3600,
        delta_time: int = 5,
        min_green: int = 5,
        use_gui: bool = False,
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.out_csv_name = out_csv_name
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.min_green = min_green
        self.use_gui = use_gui

        # Defer construction until reset()
        self._par_env = None
        self.agent_ids: List[str] = []
        self.num_agents: int = 0
        self.local_obs_dim: int = 0
        self.global_obs_dim: int = 0
        self.action_dim: int = 0

        # Initialise once to fix dimensions
        self._build_env()
        _ = self.reset()

    # ─────────────────────────────────────────────────────────────────
    # Internal: build / rebuild the underlying sumo_rl ParallelEnv
    # ─────────────────────────────────────────────────────────────────

    def _build_env(self):
        if self._par_env is not None:
            try:
                self._par_env.close()
            except Exception:
                pass
            self._par_env = None

        self._par_env = sumo_rl.parallel_env(
            net_file=self.net_file,
            route_file=self.route_file,
            out_csv_name=self.out_csv_name,
            use_gui=self.use_gui,
            num_seconds=self.num_seconds,
            delta_time=self.delta_time,
            min_green=self.min_green,
            reward_fn="queue-ped",
            observation_class=PedestrianObservationFunction,
        )

    def _walk_to_sumo_env(self):
        """Walk PettingZoo wrapper chain to find the underlying SumoEnvironment.

        Needed so we can access traffic_signals[i] for C^p / C^s.
        """
        obj = self._par_env
        if hasattr(obj, "aec_env"):
            obj = obj.aec_env
        for _ in range(10):
            if hasattr(obj, "traffic_signals"):
                return obj
            elif hasattr(obj, "env"):
                obj = obj.env
            elif hasattr(obj, "unwrapped"):
                obj = obj.unwrapped
            else:
                break
        raise RuntimeError("Could not locate SumoEnvironment in PettingZoo wrapper chain")

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def reset(self, seed: int = None) -> Dict[str, np.ndarray]:
        """Reset and return {agent_id: local_obs}.

        Side effect: caches agent ordering and obs/action dims on first call.
        """
        if self._par_env is None:
            self._build_env()

        obs_dict, _ = self._par_env.reset(seed=seed)

        # Cache agent ordering (sorted for determinism)
        if not self.agent_ids:
            self.agent_ids = sorted(self._par_env.agents)
            self.num_agents = len(self.agent_ids)
            self.local_obs_dim = self._par_env.observation_space(self.agent_ids[0]).shape[0]
            self.global_obs_dim = self.local_obs_dim * self.num_agents
            self.action_dim = self._par_env.action_space(self.agent_ids[0]).n

        return {a: obs_dict[a].astype(np.float32) for a in self.agent_ids}

    def step(
        self, action_dict: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],   # next_obs
        Dict[str, float],        # rewards
        Dict[str, float],        # C_p (per agent)
        Dict[str, float],        # C_s (per agent)
        bool,                    # done (any-agent-done OR all-agents-done; we use all)
        dict,                    # info
    ]:
        """One environment step.

        Returns per-agent dicts plus C^p_i / C^s_i (paper Eq. 6, 10).
        """
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self._par_env.step(action_dict)

        # Pull C^p, C^s from each TrafficSignal
        sumo_env = self._walk_to_sumo_env()
        c_p = {}
        c_s = {}
        for a in self.agent_ids:
            ts = sumo_env.traffic_signals.get(a)
            if ts is None:
                c_p[a] = 0.0
                c_s[a] = 0.0
                continue
            try:
                c_p[a] = float(ts.get_total_expected_violations())
            except Exception:
                c_p[a] = 0.0
            try:
                c_s[a] = float(ts.get_spillback_cost())
            except Exception:
                c_s[a] = 0.0

        # Cast obs / rewards to clean float32 keyed by sorted agent order
        next_obs = {a: obs_dict[a].astype(np.float32) for a in self.agent_ids if a in obs_dict}
        rewards  = {a: float(reward_dict.get(a, 0.0)) for a in self.agent_ids}

        # Done = all agents done (episode ends when SUMO horizon reached)
        done_all = all(term_dict.get(a, False) or trunc_dict.get(a, False) for a in self.agent_ids)

        return next_obs, rewards, c_p, c_s, done_all, info_dict

    def get_global_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concat per-agent obs in fixed order to form global state for critics."""
        return np.concatenate(
            [obs_dict[a] for a in self.agent_ids], axis=0
        ).astype(np.float32)

    def close(self):
        if self._par_env is not None:
            try:
                self._par_env.close()
            except Exception:
                pass
            self._par_env = None