"""Fair-TSC environment wrapper.

The wrapper keeps the hand-written PPO loop independent from PettingZoo
details and records phase activation start times for intra-intersection
fairness:

    ell_{i,sigma,m} = t_start_{i,sigma,m+1} - t_start_{i,sigma,m}

Legacy C_p/C_s return slots are preserved as zeros so older baselines do
not need signature changes, but they are no longer training constraints.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

if "SUMO_HOME" not in os.environ:
    fake = "/tmp/sumo_fake" if os.name != "nt" else os.path.join(os.getenv("TEMP", "."), "sumo_fake")
    os.makedirs(os.path.join(fake, "tools"), exist_ok=True)
    os.environ["SUMO_HOME"] = fake

import sumo_rl
from sumo_rl.environment.observations import PedestrianObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

import config as C
from fairness import phase_service_theil_from_intervals


TrafficSignal.omega_p = C.OMEGA_P


class FairTSCEnv:
    """Thin functional wrapper around ``sumo_rl.parallel_env``."""

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        num_seconds: int = 3600,
        delta_time: int = 5,
        min_green: int = 5,
        use_gui: bool = False,
        additional_sumo_cmd: Optional[str] = None,
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.out_csv_name = out_csv_name
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.min_green = min_green
        self.use_gui = use_gui
        self.additional_sumo_cmd = additional_sumo_cmd

        self._par_env = None
        self.agent_ids: List[str] = []
        self.num_agents = 0
        self.local_obs_dim = 0
        self.global_obs_dim = 0
        self.action_dim = 0

        self._phase_start_log: Dict[str, Dict[int, List[float]]] = {}
        self._last_phase: Dict[str, int] = {}
        self._phase_count: Dict[str, int] = {}
        self._pending_phase_start: Dict[str, Tuple[int, float]] = {}

        self._build_env()
        _ = self.reset()

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
            additional_sumo_cmd=self.additional_sumo_cmd,
        )

    def _walk_to_sumo_env(self):
        obj = self._par_env
        if hasattr(obj, "aec_env"):
            obj = obj.aec_env
        for _ in range(10):
            if hasattr(obj, "traffic_signals"):
                return obj
            if hasattr(obj, "env"):
                obj = obj.env
                continue
            if hasattr(obj, "unwrapped"):
                obj = obj.unwrapped
                continue
            break
        raise RuntimeError("Could not locate SumoEnvironment in PettingZoo wrapper chain")

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if self._par_env is None:
            self._build_env()

        obs_dict, _ = self._par_env.reset(seed=seed)

        if not self.agent_ids:
            self.agent_ids = sorted(self._par_env.agents)
            self.num_agents = len(self.agent_ids)
            self.local_obs_dim = self._par_env.observation_space(self.agent_ids[0]).shape[0]
            self.global_obs_dim = self.local_obs_dim * self.num_agents
            self.action_dim = self._par_env.action_space(self.agent_ids[0]).n

        self._reset_phase_log()
        return {a: obs_dict[a].astype(np.float32) for a in self.agent_ids}

    def step(
        self, action_dict: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        bool,
        dict,
    ]:
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self._par_env.step(action_dict)
        self._record_phase_activation_changes()

        next_obs = {a: obs_dict[a].astype(np.float32) for a in self.agent_ids if a in obs_dict}
        rewards = {a: float(reward_dict.get(a, 0.0)) / C.REWARD_SCALE for a in self.agent_ids}

        c_p = {a: 0.0 for a in self.agent_ids}
        c_s = {a: 0.0 for a in self.agent_ids}
        done_all = all(term_dict.get(a, False) or trunc_dict.get(a, False) for a in self.agent_ids)

        if done_all:
            self._inject_info_metrics(info_dict, self.get_phase_service_summary())

        return next_obs, rewards, c_p, c_s, done_all, info_dict

    def get_global_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs_dict[a] for a in self.agent_ids], axis=0).astype(np.float32)

    def _sim_time(self) -> float:
        try:
            return float(self._walk_to_sumo_env().sim_step)
        except Exception:
            return 0.0

    def _reset_phase_log(self):
        sumo_env = self._walk_to_sumo_env()
        now = float(getattr(sumo_env, "sim_step", 0.0))
        self._phase_start_log = {}
        self._last_phase = {}
        self._phase_count = {}
        self._pending_phase_start = {}

        for agent in self.agent_ids:
            ts = sumo_env.traffic_signals.get(agent)
            phase_count = int(getattr(ts, "num_green_phases", self.action_dim)) if ts is not None else self.action_dim
            current_phase = int(getattr(ts, "green_phase", 0)) if ts is not None else 0
            self._phase_count[agent] = phase_count
            self._phase_start_log[agent] = {p: [] for p in range(phase_count)}
            self._append_phase_start(agent, current_phase, now)
            self._last_phase[agent] = current_phase

    def _append_phase_start(self, agent: str, phase: int, start_time: float):
        log = self._phase_start_log.setdefault(agent, {})
        starts = log.setdefault(int(phase), [])
        start_time = float(start_time)
        if not starts or abs(starts[-1] - start_time) > 1e-6:
            starts.append(start_time)

    def _record_phase_activation_changes(self):
        sumo_env = self._walk_to_sumo_env()
        now = float(getattr(sumo_env, "sim_step", self._sim_time()))

        for agent in self.agent_ids:
            pending = self._pending_phase_start.get(agent)
            if pending is not None:
                phase, start_time = pending
                if now >= start_time:
                    self._append_phase_start(agent, phase, start_time)
                    self._pending_phase_start.pop(agent, None)

            ts = sumo_env.traffic_signals.get(agent)
            if ts is None:
                continue

            current_phase = int(getattr(ts, "green_phase", 0))
            previous_phase = self._last_phase.get(agent)
            if previous_phase is None:
                self._last_phase[agent] = current_phase
                self._append_phase_start(agent, current_phase, now)
                continue
            if current_phase == previous_phase:
                continue

            phase_age = float(getattr(ts, "time_since_last_phase_change", 0.0))
            yellow_time = float(getattr(ts, "yellow_time", 0.0))
            transition_start = max(0.0, now - phase_age)
            green_start = transition_start + yellow_time

            if now >= green_start:
                self._append_phase_start(agent, current_phase, green_start)
            else:
                self._pending_phase_start[agent] = (current_phase, green_start)
            self._last_phase[agent] = current_phase

    def get_phase_start_log(self) -> Dict[str, Dict[int, List[float]]]:
        return copy.deepcopy(self._phase_start_log)

    def get_phase_service_intervals(self, include_unserved: bool = True) -> Dict[str, Dict[int, List[float]]]:
        intervals: Dict[str, Dict[int, List[float]]] = {}
        fallback = float(getattr(C, "PHASE_UNSERVED_INTERVAL", self.num_seconds))
        for agent in self.agent_ids:
            phase_count = self._phase_count.get(agent, self.action_dim)
            intervals[agent] = {}
            for phase in range(phase_count):
                starts = sorted(self._phase_start_log.get(agent, {}).get(phase, []))
                diffs = [float(starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]
                if include_unserved and not diffs:
                    diffs = [fallback]
                intervals[agent][phase] = diffs
        return intervals

    def get_phase_service_summary(self) -> Dict[str, float]:
        intervals = self.get_phase_service_intervals(include_unserved=True)
        intra_by_agent, theil_intra, max_interval = phase_service_theil_from_intervals(
            intervals, self.agent_ids, eps=C.THEIL_EPS
        )
        mean_interval = 0.0
        flat = [
            interval
            for phase_map in intervals.values()
            for phase_intervals in phase_map.values()
            for interval in phase_intervals
        ]
        if flat:
            mean_interval = float(np.mean(flat))
        return {
            "theil_intra": float(theil_intra),
            "max_phase_interval": float(max_interval),
            "phase_service_mean_interval": float(mean_interval),
            **{f"theil_intra_{agent}": float(intra_by_agent.get(agent, 0.0)) for agent in self.agent_ids},
        }

    @staticmethod
    def _inject_info_metrics(info: dict, metrics: Dict[str, float]):
        if isinstance(info, dict) and info and all(isinstance(v, dict) for v in info.values()):
            for per_agent in info.values():
                per_agent.update(metrics)
        elif isinstance(info, dict):
            info.update(metrics)

    def close(self):
        if self._par_env is not None:
            try:
                self._par_env.close()
            except Exception:
                pass
            self._par_env = None
