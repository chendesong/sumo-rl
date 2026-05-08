"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""

import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import math
import numpy as np
from gymnasium import spaces


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: List[float],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo = sumo

        if type(self.reward_fn) is list:
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn) for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        # ── Pedestrian infrastructure ────────────────────────
        # Each intersection has 4 crossings: :id_c0, :id_c1, :id_c2, :id_c3
        self.crossing_ids = [f":{self.id}_c0", f":{self.id}_c1", f":{self.id}_c2", f":{self.id}_c3"]
        self.num_crossings = len(self.crossing_ids)

        # Cox-Weibull non-compliance model parameters (paper Eq. 3-4).
        # h(w) = (k/lambda) * (w/lambda)^(k-1) * exp(-beta_f * f)
        # P_viol = 1 - exp[ -(w/lambda)^k * exp(-beta_f * f) ]
        #
        # Default values from paper Sec II.A.3 (calibrated against simulator):
        #   lambda_w  : Weibull scale, ~ median violation wait without deterrence (s)
        #   k_w       : Weibull shape (>1, hazard accelerates with wait)
        #   beta_f    : flow deterrence coefficient ((veh/min)^-1)
        self.lambda_w = 60.0   # paper default
        self.k_w      = 2.0    # paper default
        self.beta_f   = 0.1    # paper default ((veh/min)^-1)

        # Spillback / store-and-forward parameters (paper Eq. 8-10).
        # N_l = kappa_l * L_l * lambda_l : per-lane jam capacity (veh)
        # We treat each SUMO lane as one "directed link" so lambda_l = 1.
        # kappa_jam (veh/m) is the jam density factor; standard urban value
        # is ~0.18 veh/m (i.e. ~5.5 m per stopped vehicle including gaps).
        self.kappa_jam = 0.18  # veh/m, industry standard

        # Track last total pedestrian waiting time (kept for optional diff-based rewards)
        self.last_ped_waiting_time = 0.0

        self.observation_space = self.observation_fn.observation_space()
        # Action space: Discrete(N) — select next green phase
        # For the 3-phase design in this work, N=3:
        #   0 = NS vehicle service, 1 = EW vehicle service, 2 = exclusive pedestrian
        # Yellow transitions are auto-generated by sumo-rl and inserted
        # on phase changes; their duration is the env parameter yellow_time.
        self.action_space = spaces.Discrete(self.num_green_phases)

    def _get_reward_fn_from_string(self, reward_fn):
        if type(reward_fn) is str:
            if reward_fn in TrafficSignal.reward_fns.keys():
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        return reward_fn

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        self.num_total_phases = len(phases)
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    def get_phase_info(self):
        """Returns (num_controlled_green_phases, num_total_phases_in_net_xml)."""
        return self.num_green_phases, self.num_total_phases

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        """Sets the next green phase.

        Action encoding:
            0 .. N-1  : select green phase to execute next.

        If the agent selects a different phase from the current one,
        a yellow transition is inserted automatically (duration =
        self.yellow_time) before switching. If the agent selects the
        current phase, service continues without transition.

        Minimum-green is enforced: if time_since_last_phase_change
        is below (yellow_time + min_green), the current phase is
        held regardless of the action.

        Args:
            new_phase (int): integer in [0, num_green_phases - 1].
        """
        new_phase = int(new_phase)

        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self) -> Union[float, np.ndarray]:
        """Computes the reward of the traffic signal. If it is a list of rewards, it returns a numpy array."""
        if self.reward_dim == 1:
            self.last_reward = self.reward_list[0](self)
        else:
            self.last_reward = np.array([reward_fn(self) for reward_fn in self.reward_list], dtype=np.float32)
            if self.reward_weights is not None:
                self.last_reward = np.dot(self.last_reward, self.reward_weights)  # Linear combination of rewards

        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _co2_reward(self):
        return -self.get_total_co2()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_ts_waiting_time - ts_wait
        self.last_ts_waiting_time = ts_wait
        return reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def get_total_co2(self) -> float:
        """Returns the total CO2 emissions (mg/s) of the vehicles in the incoming lanes of the intersection."""
        return sum(self.sumo.lane.getCO2Emission(lane) for lane in self.lanes)

    # ══════════════════════════════════════════════════════════
    # Step 2: Pedestrian data retrieval per crossing
    # ══════════════════════════════════════════════════════════

    def get_pedestrian_data_per_crossing(self) -> dict:
        """Returns per-crossing pedestrian data for this intersection.

        For each crossing, returns:
            queue (int): number of pedestrians waiting (speed < 0.1 m/s)
            max_wait (float): maximum waiting time among queued pedestrians
            total_wait (float): sum of waiting times of all queued pedestrians

        Maps to paper variables:
            queue    -> q^p_{i,c}(t)  (Eq. 9, 16)
            max_wait -> w^p_{i,c}(t)  (Eq. 6, 8)
        """
        data = {c_id: {"queue": 0, "max_wait": 0.0, "total_wait": 0.0} for c_id in self.crossing_ids}

        for pid in self.sumo.person.getIDList():
            next_edge = self.sumo.person.getNextEdge(pid)
            if next_edge in data:
                wait = self.sumo.person.getWaitingTime(pid)
                speed = self.sumo.person.getSpeed(pid)
                if speed < 0.1 and wait > 0:
                    data[next_edge]["queue"] += 1
                    data[next_edge]["total_wait"] += wait
                    data[next_edge]["max_wait"] = max(data[next_edge]["max_wait"], wait)

        return data

    def get_pedestrian_queue_per_crossing(self) -> List[int]:
        """Returns [q_c0, q_c1, q_c2, q_c3] — pedestrian queue at each crossing."""
        data = self.get_pedestrian_data_per_crossing()
        return [data[c]["queue"] for c in self.crossing_ids]

    def get_pedestrian_wait_per_crossing(self) -> List[float]:
        """Returns [w_c0, w_c1, w_c2, w_c3] — max pedestrian waiting time at each crossing."""
        data = self.get_pedestrian_data_per_crossing()
        return [data[c]["max_wait"] for c in self.crossing_ids]

    def get_total_pedestrian_waiting_time(self) -> float:
        """Returns sum of all pedestrian waiting times across all crossings."""
        data = self.get_pedestrian_data_per_crossing()
        return sum(data[c]["total_wait"] for c in self.crossing_ids)

    def get_total_pedestrian_queued(self) -> int:
        """Returns total number of queued pedestrians at this intersection."""
        data = self.get_pedestrian_data_per_crossing()
        return sum(data[c]["queue"] for c in self.crossing_ids)

    # ══════════════════════════════════════════════════════════
    # Step 3: Cox-Weibull non-compliance model — Paper Eq. (3)(4)(5)
    # ══════════════════════════════════════════════════════════

    def _conflicting_flow_approx(self) -> float:
        """Approximate conflicting vehicle flow at this intersection (veh/min).

        Approximation: total vehicles on incoming lanes / 4 crosswalks,
        normalised to per-minute units. This is the simple network-average
        estimator. A per-crossing version (querying SUMO net's
        ``crossingEdges`` attribute) is possible but adds boilerplate; the
        averaged estimator captures the relevant deterrence signal.

        Returns:
            float: conflicting flow estimate in veh/min
        """
        # Total vehicles on all incoming lanes
        total_veh = sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
        # Convert to per-minute by scaling: assume agents step at delta_time
        # so this is a snapshot count; scale by 60 / delta_time to approximate
        # vehicles-per-minute throughput at the crosswalk.
        veh_per_min = total_veh * (60.0 / max(self.delta_time, 1.0))
        # Average across 4 crosswalks (per-crossing conflicting flow share)
        return veh_per_min / max(self.num_crossings, 1)

    def _cox_weibull_violation_probability(self, wait_time: float, conflicting_flow: float) -> float:
        """Cox proportional-hazards model with Weibull baseline (paper Eq. 4).

        P_viol(w, f) = 1 - exp[ -(w / lambda)^k * exp(-beta_f * f) ]

        Properties:
          - Monotonically increasing in w (longer wait -> higher violation prob)
          - Monotonically decreasing in f (more cars -> deterrence)
          - Reduces to pure Weibull when f = 0
          - Bounded in [0, 1]

        Args:
            wait_time: cumulative pedestrian waiting time at the crosswalk (s)
            conflicting_flow: vehicle flow crossing this crosswalk (veh/min)
        Returns:
            float in [0, 1]
        """
        if wait_time <= 0.0:
            return 0.0
        # Hazard integrand
        scale_term = (wait_time / self.lambda_w) ** self.k_w
        deterrence = math.exp(-self.beta_f * max(conflicting_flow, 0.0))
        return 1.0 - math.exp(-scale_term * deterrence)

    def get_jaywalking_per_crossing(self) -> dict:
        """Returns per-crossing non-compliance metrics (paper Eq. 4-5).

        For each crossing:
            p_viol (float): Cox-Weibull violation probability
            expected_violations (float): q^p_{i,c} * P_viol_{i,c} — paper Eq. (5)
        """
        data = self.get_pedestrian_data_per_crossing()
        f_ic = self._conflicting_flow_approx()  # shared across crossings (averaged)
        result = {}
        for c_id in self.crossing_ids:
            d = data[c_id]
            p_viol = self._cox_weibull_violation_probability(d["max_wait"], f_ic)
            expected_viol = d["queue"] * p_viol if d["queue"] > 0 else 0.0
            result[c_id] = {
                "queue": d["queue"],
                "max_wait": d["max_wait"],
                "total_wait": d["total_wait"],
                "p_viol": p_viol,
                "expected_violations": expected_viol,
            }
        return result

    def get_total_expected_violations(self) -> float:
        """Paper C^p_i(t) = sum_c V_{i,c}(t) — total expected violations."""
        jw = self.get_jaywalking_per_crossing()
        return sum(jw[c]["expected_violations"] for c in self.crossing_ids)

    # ══════════════════════════════════════════════════════════
    # Step 4: Spillback cost — Paper Eq. (8)-(10)
    # ══════════════════════════════════════════════════════════

    def get_spillback_cost(self) -> float:
        """Paper Eq. (10): C^s_i(t) = sum_l max(n_l(t) - N_l, 0).

        Spillback cost: total over-capacity vehicles across all incoming
        lanes of intersection i. Only the EXCESS over jam capacity counts;
        a lane below capacity contributes zero.

        Per-lane capacity: N_l = kappa_jam * L_l (with lambda_l=1 since
        each SUMO lane is treated as its own directed link).

        Returns:
            float: total over-capacity vehicle count, in veh.  Units match
                   paper Eq. (10).  Always >= 0.
        """
        cost = 0.0
        for lane in self.lanes:
            n_l = self.sumo.lane.getLastStepVehicleNumber(lane)
            L_l = self.lanes_length[lane]
            N_l = self.kappa_jam * L_l  # max capacity (veh)
            excess = n_l - N_l
            if excess > 0.0:
                cost += excess
        return cost

    # ══════════════════════════════════════════════════════════
    # Step 5: Full local state vector — Paper Eq. (12)
    # s_i = (q^v, q^p, n^in, n^out, sigma, tau, w^p)
    # ══════════════════════════════════════════════════════════

    def get_local_state(self) -> dict:
        """Paper Eq. (12): full per-intersection state vector (raw, unnormalised).

        Returns a dict (not flat array) so RL code can pick & normalise
        per its own scheme.  Each entry is a python list / scalar.

        Keys:
            q_v       : list[int], length = len(self.lanes).  Halting vehicle
                        count per incoming lane.
            q_p       : list[int], length 4.  Pedestrian queue per crosswalk.
            n_in      : list[int], length = len(self.lanes).  Total vehicle
                        count per incoming lane.
            n_out     : list[int], length = len(self.out_lanes).  Total vehicle
                        count per outgoing lane.
            sigma     : int in {0, 1, 2}.  Active green phase index.
            tau       : float, seconds since last phase change.
            w_p       : list[float], length 4.  Max pedestrian waiting time per
                        crosswalk (s).

        Notes:
          - Paper notation maps: q^v -> q_v, q^p -> q_p, n^in -> n_in,
            n^out -> n_out, sigma -> sigma, tau -> tau, w^p -> w_p.
          - For the 2x2grid network in this work, len(self.lanes) = 12 and
            len(self.out_lanes) = 12, but the actual count is whatever SUMO
            reports for the controlled lanes of this intersection.
          - This method is decoupled from any observation function: callers
            (RL critic, fairness module, logging) all use the same source.
        """
        # Vehicle queues per incoming lane (halting cars only)
        q_v = [self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]
        # Total vehicle counts per incoming/outgoing lane
        n_in = [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
        n_out = [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes]

        # Pedestrian queue + max wait per crosswalk
        ped_data = self.get_pedestrian_data_per_crossing()
        q_p = [ped_data[c]["queue"] for c in self.crossing_ids]
        w_p = [ped_data[c]["max_wait"] for c in self.crossing_ids]

        return {
            "q_v":   q_v,
            "q_p":   q_p,
            "n_in":  n_in,
            "n_out": n_out,
            "sigma": int(self.green_phase),
            "tau":   float(self.time_since_last_phase_change),
            "w_p":   w_p,
        }

    def get_local_state_flat(self) -> np.ndarray:
        """Flat numpy version of get_local_state(), for direct critic input.

        Layout (concatenation order, matches paper Eq. 12):
            [q_v..., q_p..., n_in..., n_out..., sigma_one_hot..., tau, w_p...]

        sigma is one-hot encoded over num_green_phases entries (default 3)
        so the vector has fixed dimensionality once num lanes is fixed.

        Returns:
            np.ndarray (float32) of length:
                len(lanes) + 4 + len(lanes) + len(out_lanes)
                + num_green_phases + 1 + 4
        """
        s = self.get_local_state()
        sigma_onehot = [1.0 if s["sigma"] == i else 0.0 for i in range(self.num_green_phases)]
        flat = (
            list(map(float, s["q_v"]))
            + list(map(float, s["q_p"]))
            + list(map(float, s["n_in"]))
            + list(map(float, s["n_out"]))
            + sigma_onehot
            + [float(s["tau"])]
            + list(map(float, s["w_p"]))
        )
        return np.array(flat, dtype=np.float32)

    def get_local_state_dim(self) -> int:
        """Returns the dimensionality of get_local_state_flat() output."""
        return (
            len(self.lanes)
            + self.num_crossings
            + len(self.lanes)
            + len(self.out_lanes)
            + self.num_green_phases
            + 1
            + self.num_crossings
        )

    # ══════════════════════════════════════════════════════════
    # Reward function — Paper Eq. (19)
    # R_i(t) = -sum_k q_v(t) - omega_p * sum_c q_p(t)
    # ══════════════════════════════════════════════════════════

    def _queue_ped_reward(self):
        """Paper Eq. (19): queue-based local reward.

        R_i(t) = - sum_k q_{i,k}^v(t) - omega_p * sum_c q_{i,c}^p(t)

        Vehicle queue: total halting vehicles across all incoming lanes.
        Pedestrian queue: total waiting pedestrians across all 4 crossings.
        omega_p (>0) weighs pedestrian queues relative to vehicle queues.

        This reward is dense, low-variance, and well suited to on-policy
        MARL training.  Soft constraints (vehicle unpatience, pedestrian
        non-compliance, spillback) enter the main-method training via
        Lagrange multipliers, not through this reward.
        """
        omega_p = 1.0  # hyperparameter; Section IV reports the value used
        veh_queue = self.get_total_queued()
        ped_queue = self.get_total_pedestrian_queued()
        return -float(veh_queue) - omega_p * float(ped_queue)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "queue-ped": _queue_ped_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
        "co2": _co2_reward,
    }