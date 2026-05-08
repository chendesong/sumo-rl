"""Observation functions for traffic signals."""

from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )


class PedestrianObservationFunction(ObservationFunction):
    """Observation function that includes pedestrian features.

    Extends the default observation with per-crossing pedestrian data:
        obs = [phase_one_hot, min_green,
               lane_densities, lane_queues,           ← vehicle (same as default)
               ped_queue_c0..c3, ped_wait_c0..c3]     ← pedestrian (new)

    Pedestrian queue is normalised by MAX_PED_QUEUE (default 20).
    Pedestrian wait is normalised by MAX_PED_WAIT (default 120s).
    """

    MAX_PED_QUEUE = 20.0   # normalisation ceiling for pedestrian queue
    MAX_PED_WAIT = 120.0   # normalisation ceiling for pedestrian wait (seconds)

    def __init__(self, ts: TrafficSignal):
        """Initialize pedestrian observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return observation with pedestrian features."""
        # ── Vehicle features (same as default) ──
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()

        # ── Pedestrian features (new) ──
        ped_queues = self.ts.get_pedestrian_queue_per_crossing()     # [q_c0, q_c1, q_c2, q_c3]
        ped_waits = self.ts.get_pedestrian_wait_per_crossing()       # [w_c0, w_c1, w_c2, w_c3]

        # Normalise to [0, 1]
        ped_queues_norm = [min(q / self.MAX_PED_QUEUE, 1.0) for q in ped_queues]
        ped_waits_norm = [min(w / self.MAX_PED_WAIT, 1.0) for w in ped_waits]

        observation = np.array(
            phase_id + min_green + density + queue + ped_queues_norm + ped_waits_norm,
            dtype=np.float32,
        )
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space including pedestrian dimensions."""
        num_ped_features = self.ts.num_crossings * 2  # 4 queues + 4 waits = 8
        total_dim = self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + num_ped_features
        return spaces.Box(
            low=np.zeros(total_dim, dtype=np.float32),
            high=np.ones(total_dim, dtype=np.float32),
        )