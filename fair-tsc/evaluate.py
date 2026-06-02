"""
Unified evaluation layer for Fair-TSC algorithm comparison.

SINGLE SOURCE OF TRUTH for the comparison metrics. Every baseline /
algorithm finishes its run by:

    1. Building a `deltas_TN` array of shape [T, N] via the SHARED V^UE
       (loaded from a Fair-TSC checkpoint, identical for every method)
       and the realized discounted return G_t^method(i):

           δ_i(t) = [ V^UE(s_t, i)  −  G_t^method(i) ]_+

       V^UE        : shared, frozen, identical for ALL methods.
       G_t^method  : realized discounted return for agent i from
                     timestep t under method M, computed by backward
                     γ-sweep over the RAW env rewards captured during
                     the eval rollout.  γ = config.GAMMA.

       For FairLight, "raw" specifically means the env-returned reward,
       NOT the variance-shaped surrogate the FairLight policy optimises.
       The fairness metric must measure true reward space.

    2. Collecting raw per-step env metrics (system_total_waiting_time,
       agents_total_ped_waiting_time, mean reward, ...) into a dict.

    3. Calling `evaluate_run(deltas_TN, env_metrics, delta_valid=...) -> dict`.

NO baseline is allowed to compute Theil internally. NO baseline is
allowed to define its own "fairness" scalar. The whole point of this
file is to keep the comparison apples-to-apples.

Returned keys:
    theil_ema      :  Theil-T on EMA(per-agent δ̄_i) with β = config.THEIL_EMA_BETA
                      (same smoothing rule used by the Fair-TSC trainer)
    theil_raw      :  Single-episode Theil-T on per-agent δ̄_i (diagnostic)
    efficiency     :  -mean(system_total_waiting_time)  — higher is better.
                      Falls back to mean(reward) if waiting-time series absent.
    ped_wait       :  mean(agents_total_ped_waiting_time)
    ped_risk       :  mean expected pedestrian violation proxy / (4 * N)
    delta_max      :  max_i δ̄_i  (worst-agent sacrifice gap, time-averaged)
    delta_valid    :  bool — kept for forward compatibility; under the
                      G-based δ formula every method has a real number,
                      so all callers pass True.
"""

import os
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

import numpy as np

import config as C
from fairness import phase_service_theil_from_intervals, theil_t_index
from safety_eval import normalize_pedestrian_risk


# ────────────────────────────────────────────────────────────────────
# Module-level EMA store
# ────────────────────────────────────────────────────────────────────
# evaluate_run() supports an optional `ema_state` arg so the caller can
# carry EMA across multiple eval calls (e.g. across episodes for a
# learning baseline).  For one-shot evals (fixed-time, MP), pass None
# and the function returns theil_ema = theil_raw (no smoothing history).

def update_ema(prev_ema: Optional[np.ndarray],
               delta_agent_mean: np.ndarray,
               beta: float = None) -> np.ndarray:
    """Pure-function EMA: ema_t = β·ema_{t-1} + (1-β)·x_t.

    Fair-TSC and baselines use the exact same smoothing rule.
    """
    if beta is None:
        beta = C.THEIL_EMA_BETA
    if prev_ema is None:
        return delta_agent_mean.astype(np.float32).copy()
    return (beta * prev_ema + (1.0 - beta) * delta_agent_mean).astype(np.float32)


def _safe_mean(seq) -> float:
    if seq is None:
        return 0.0
    arr = np.asarray(list(seq), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def parse_tripinfo(path: str, horizon_s: float) -> Dict[str, float]:
    """Parse SUMO tripinfo output into episode-level efficiency metrics."""
    durations = []
    waiting_times = []
    time_losses = []

    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            iterator = ET.iterparse(path, events=("end",))
            for _event, elem in iterator:
                if elem.tag == "tripinfo":
                    for key, store in (
                        ("duration", durations),
                        ("waitingTime", waiting_times),
                        ("timeLoss", time_losses),
                    ):
                        try:
                            store.append(float(elem.attrib.get(key, "0")))
                        except (TypeError, ValueError):
                            store.append(0.0)
                    elem.clear()
        except ET.ParseError:
            # A partially flushed tripinfo file should not crash a long
            # comparison run; keep whatever complete tripinfo rows were read.
            pass

    completed = len(durations)
    hours = max(float(horizon_s) / 3600.0, 1e-9)
    return {
        "completed_vehicles": int(completed),
        "throughput_veh_per_hour": float(completed / hours),
        "total_travel_time_s": float(np.sum(durations)) if durations else 0.0,
        "mean_travel_time_s": float(np.mean(durations)) if durations else 0.0,
        "total_vehicle_waiting_time_s": float(np.sum(waiting_times)) if waiting_times else 0.0,
        "mean_vehicle_waiting_time_s": float(np.mean(waiting_times)) if waiting_times else 0.0,
        "total_time_loss_s": float(np.sum(time_losses)) if time_losses else 0.0,
        "mean_time_loss_s": float(np.mean(time_losses)) if time_losses else 0.0,
    }


def parse_route_vehicle_demand(path: str, horizon_s: float) -> float:
    """Estimate total vehicle demand from explicit vehicles/trips and flows."""
    if not path or not os.path.exists(path):
        return 0.0
    total = 0.0
    for _event, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag
        if tag in {"vehicle", "trip"}:
            total += 1.0
        elif tag == "flow":
            attrs = elem.attrib
            begin = float(attrs.get("begin", 0.0) or 0.0)
            end = float(attrs.get("end", horizon_s) or horizon_s)
            duration = max(min(end, float(horizon_s)) - max(begin, 0.0), 0.0)
            if attrs.get("number") not in (None, ""):
                try:
                    total += float(attrs["number"])
                except ValueError:
                    pass
            elif attrs.get("vehsPerHour") not in (None, "") or attrs.get("perHour") not in (None, ""):
                rate = attrs.get("vehsPerHour", attrs.get("perHour", "0"))
                try:
                    total += float(rate) * duration / 3600.0
                except ValueError:
                    pass
            elif attrs.get("period") not in (None, ""):
                try:
                    period = max(float(attrs["period"]), 1e-9)
                    total += duration / period
                except ValueError:
                    pass
        elem.clear()
    return float(total)


def make_tripinfo_sumo_cmd(path: str) -> str:
    """SUMO additional command for episode-level tripinfo output."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return f"--tripinfo-output {path}"


def merge_sumo_cmds(*cmds: Optional[str]) -> Optional[str]:
    """Join optional SUMO command fragments while skipping blanks."""
    parts = [str(cmd).strip() for cmd in cmds if cmd and str(cmd).strip()]
    return " ".join(parts) if parts else None


def attach_tripinfo_metrics(result: Dict, path: str, horizon_s: float) -> Dict:
    """Attach tripinfo metrics and promote total travel time to efficiency."""
    trip = parse_tripinfo(path, horizon_s=horizon_s)
    result.update(trip)
    result["tripinfo_xml"] = os.path.abspath(os.path.expanduser(path))
    if trip.get("total_travel_time_s", 0.0) > 0.0:
        result["efficiency"] = -float(trip["total_travel_time_s"])
        result["efficiency_metric"] = "negative_total_travel_time_s"
    return result


def _completion_rates(arrived: float, departed: float, demand: float) -> Dict[str, float]:
    return {
        "completion_rate_departed": float(arrived / departed) if departed > 0.0 else 0.0,
        "completion_rate_demand": float(arrived / demand) if demand > 0.0 else 0.0,
    }


def evaluate_run(deltas_TN: np.ndarray,
                 env_metrics: Dict,
                 ema_state: Optional[np.ndarray] = None,
                 delta_valid: bool = True) -> Dict[str, float]:
    """Unified evaluation layer.

    Args:
        deltas_TN:    np.ndarray [T, N]  sacrifice gaps (per-method δ).
        env_metrics:  dict of raw env metrics; expected keys:
                        - "system_total_waiting_time_series" : list[float]
                        - "agents_total_ped_waiting_time_series" : list[float]
                        - "reward_series" : list[float]  (mean per step)
                      All three are optional; we degrade gracefully.
        ema_state:    prior EMA over per-agent δ̄_i (np.ndarray [N]); if
                      None, ema_theil falls back to single-episode Theil.
        delta_valid:  Forward-compat flag stamped into the returned dict.
                      Under the G-based δ formula every method produces a
                      real number, so all callers pass True. Defaults True.

    Returns:
        {
          "theil_ema": float,
          "theil_raw": float,
          "efficiency": float,
          "ped_wait": float,
          "delta_max": float,
          "delta_mean": float,
          "delta_valid": bool,
        }
        Also stashed under "_ema_next" : np.ndarray [N] for caller to
        feed back as `ema_state` on the next call.
    """
    deltas_TN = np.asarray(deltas_TN, dtype=np.float32)
    if deltas_TN.ndim != 2:
        raise ValueError(f"deltas_TN must be [T,N], got shape {deltas_TN.shape}")

    # Per-agent episode-mean δ̄_i
    delta_agent_mean = deltas_TN.mean(axis=0)   # [N]

    # Single-episode Theil
    theil_raw = theil_t_index(delta_agent_mean, eps=C.THEIL_EPS)

    # EMA Theil — uses same β as Fair-TSC
    ema_next = update_ema(ema_state, delta_agent_mean, beta=C.THEIL_EMA_BETA)
    theil_ema = theil_t_index(ema_next, eps=C.THEIL_EPS)

    # Step-level control metrics. These are useful for training/debugging,
    # but episode-level reporting should prefer tripinfo travel time when
    # available via attach_tripinfo_metrics().
    wait_series = env_metrics.get("system_total_waiting_time_series")
    reward_series = env_metrics.get("reward_series")
    queue_efficiency = -_safe_mean(wait_series) if wait_series else 0.0
    reward_efficiency = _safe_mean(reward_series) if reward_series else 0.0
    if wait_series:
        efficiency = queue_efficiency
        efficiency_metric = "negative_mean_system_waiting_time"
    elif reward_series:
        efficiency = reward_efficiency
        efficiency_metric = "mean_reward"
    else:
        efficiency = 0.0
        efficiency_metric = "none"

    ped_wait = _safe_mean(env_metrics.get("agents_total_ped_waiting_time_series"))
    ped_expected = _safe_mean(env_metrics.get("agents_total_expected_violations_series"))
    num_agents = int(env_metrics.get("num_agents") or deltas_TN.shape[1] or getattr(C, "NUM_AGENTS", 0) or 0)
    ped_risk = env_metrics.get("ped_risk")
    if ped_risk is None:
        ped_risk = normalize_pedestrian_risk(ped_expected, num_agents=num_agents)

    phase_intervals = env_metrics.get("phase_service_intervals")
    if phase_intervals:
        _, theil_intra, max_phase_interval = phase_service_theil_from_intervals(
            phase_intervals, eps=C.THEIL_EPS
        )
    else:
        theil_intra = float(env_metrics.get("theil_intra", 0.0) or 0.0)
        max_phase_interval = float(env_metrics.get("max_phase_interval", 0.0) or 0.0)

    delta_max  = float(delta_agent_mean.max()) if delta_agent_mean.size else 0.0
    delta_mean = float(delta_agent_mean.mean()) if delta_agent_mean.size else 0.0

    return {
        "theil_ema":   float(theil_ema),
        "theil_raw":   float(theil_raw),
        "theil_inter": float(theil_raw),
        "theil_intra": float(theil_intra),
        "max_phase_interval": float(max_phase_interval),
        "efficiency":  float(efficiency),
        "efficiency_metric": efficiency_metric,
        "queue_efficiency": float(queue_efficiency),
        "reward_efficiency": float(reward_efficiency),
        "ped_wait":    float(ped_wait),
        "ped_risk":    float(ped_risk),
        "ped_expected_violations": float(ped_expected),
        "departed_total": float(env_metrics.get("departed_total", 0.0) or 0.0),
        "arrived_total": float(env_metrics.get("arrived_total", 0.0) or 0.0),
        "loaded_total": float(env_metrics.get("loaded_total", 0.0) or 0.0),
        "teleported_total": float(env_metrics.get("teleported_total", 0.0) or 0.0),
        "active_vehicle_count_end": float(env_metrics.get("active_vehicle_count_end", 0.0) or 0.0),
        "pending_vehicle_count_end": float(env_metrics.get("pending_vehicle_count_end", 0.0) or 0.0),
        "min_expected_number_end": float(env_metrics.get("min_expected_number_end", 0.0) or 0.0),
        "total_vehicle_demand": float(env_metrics.get("total_vehicle_demand", 0.0) or 0.0),
        "unfinished_vehicle_demand": float(env_metrics.get("unfinished_vehicle_demand", 0.0) or 0.0),
        "completion_rate_departed": float(env_metrics.get("completion_rate_departed", 0.0) or 0.0),
        "completion_rate_demand": float(env_metrics.get("completion_rate_demand", 0.0) or 0.0),
        "delta_max":   float(delta_max),
        "delta_mean":  float(delta_mean),
        "delta_valid": bool(delta_valid),
        "_ema_next":   ema_next,
    }


# ────────────────────────────────────────────────────────────────────
# Helper: harvest env_metrics from an info dict stream
# ────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """Accumulates per-step env info into the lists evaluate_run() expects.

    Usage:
        coll = MetricsCollector()
        for step in episode:
            info = env.step(...)[-1]
            coll.add(info, mean_reward_this_step)
        env_metrics = coll.finalize()
        result = evaluate_run(deltas_TN, env_metrics)
    """

    def __init__(self):
        self.wait_series = []
        self.ped_wait_series = []
        self.ped_expected_series = []
        self.vehicle_queue_series = []
        self.ped_queue_series = []
        self.reward_series = []
        self.phase_metrics = {}
        self.departed_total = 0.0
        self.arrived_total = 0.0
        self.loaded_total = 0.0
        self.teleported_total = 0.0
        self.last_active_vehicle_count = 0.0
        self.last_pending_vehicle_count = 0.0
        self.last_min_expected_number = 0.0

    def add(self, info: Dict, mean_reward: float = 0.0):
        # `info` is the PettingZoo per-agent info dict (any agent has the
        # system_* + agents_total_* keys mirrored under it).  Or it can
        # be the unwrapped info dict from SumoEnvironment._compute_info.
        if isinstance(info, dict) and info:
            # pick any agent if it's the per-agent wrapper
            probe = info
            if all(isinstance(v, dict) for v in info.values()) and info:
                probe = next(iter(info.values()))
            self.wait_series.append(float(probe.get("system_total_waiting_time", 0.0)))
            self.ped_wait_series.append(float(probe.get("agents_total_ped_waiting_time", 0.0)))
            self.ped_expected_series.append(float(probe.get("agents_total_expected_violations", 0.0)))
            self.vehicle_queue_series.append(
                float(probe.get("agents_total_stopped", probe.get("system_total_stopped", 0.0)))
            )
            self.ped_queue_series.append(float(probe.get("agents_total_ped_queued", 0.0)))
            self.departed_total += float(probe.get("simulation_departed_number", 0.0) or 0.0)
            self.arrived_total += float(probe.get("simulation_arrived_number", 0.0) or 0.0)
            self.loaded_total += float(probe.get("simulation_loaded_number", 0.0) or 0.0)
            self.last_active_vehicle_count = float(probe.get("simulation_active_vehicle_count", 0.0) or 0.0)
            self.last_pending_vehicle_count = float(probe.get("simulation_pending_vehicle_count", 0.0) or 0.0)
            self.last_min_expected_number = float(probe.get("simulation_min_expected_number", 0.0) or 0.0)
            self.teleported_total = max(
                self.teleported_total,
                float(probe.get("simulation_teleported_total_env", 0.0) or 0.0),
            )
            for key in ("theil_intra", "max_phase_interval", "phase_service_mean_interval"):
                if key in probe:
                    self.phase_metrics[key] = float(probe.get(key, 0.0))
        self.reward_series.append(float(mean_reward))

    def finalize(self, env=None) -> Dict:
        out = {
            "system_total_waiting_time_series": self.wait_series,
            "agents_total_ped_waiting_time_series": self.ped_wait_series,
            "agents_total_expected_violations_series": self.ped_expected_series,
            "agents_total_stopped_series": self.vehicle_queue_series,
            "agents_total_ped_queued_series": self.ped_queue_series,
            "reward_series": self.reward_series,
        }
        out.update(self.phase_metrics)
        num_agents = int(getattr(env, "num_agents", 0) or getattr(C, "NUM_AGENTS", 0) or 0)
        out["num_agents"] = num_agents
        out["ped_risk"] = normalize_pedestrian_risk(_safe_mean(self.ped_expected_series), num_agents=num_agents)
        if env is not None:
            try:
                sim_metrics = env.get_simulation_progress_metrics()
                self.last_active_vehicle_count = float(
                    sim_metrics.get("simulation_active_vehicle_count", self.last_active_vehicle_count)
                )
                self.last_pending_vehicle_count = float(
                    sim_metrics.get("simulation_pending_vehicle_count", self.last_pending_vehicle_count)
                )
                self.last_min_expected_number = float(
                    sim_metrics.get("simulation_min_expected_number", self.last_min_expected_number)
                )
                self.departed_total = max(
                    self.departed_total,
                    float(sim_metrics.get("simulation_departed_total_env", self.departed_total)),
                )
                self.arrived_total = max(
                    self.arrived_total,
                    float(sim_metrics.get("simulation_arrived_total_env", self.arrived_total)),
                )
                self.teleported_total = max(
                    self.teleported_total,
                    float(sim_metrics.get("simulation_teleported_total_env", self.teleported_total)),
                )
            except Exception:
                pass
            try:
                out["phase_service_intervals"] = env.get_phase_service_intervals(include_unserved=True)
                out.update(env.get_phase_service_summary())
            except Exception:
                pass
        route_file = getattr(env, "route_file", None) if env is not None else None
        horizon_s = float(getattr(env, "num_seconds", getattr(C, "NUM_SECONDS", 0)) or 0.0)
        demand = parse_route_vehicle_demand(route_file, horizon_s=horizon_s) if route_file else 0.0
        rates = _completion_rates(self.arrived_total, self.departed_total, demand)
        out.update(
            {
                "departed_total": float(self.departed_total),
                "arrived_total": float(self.arrived_total),
                "loaded_total": float(self.loaded_total),
                "teleported_total": float(self.teleported_total),
                "active_vehicle_count_end": float(self.last_active_vehicle_count),
                "pending_vehicle_count_end": float(self.last_pending_vehicle_count),
                "min_expected_number_end": float(self.last_min_expected_number),
                "total_vehicle_demand": float(demand),
                "unfinished_vehicle_demand": float(max(demand - self.arrived_total, 0.0)) if demand else 0.0,
                **rates,
            }
        )
        return out


# ────────────────────────────────────────────────────────────────────
# Shared V^UE loader — used by ALL methods for the unified δ formula.
# ────────────────────────────────────────────────────────────────────
#
# We DEFER `torch` import to the function body so that platforms without
# torch (e.g. lightweight CI lint) can still import `evaluate` for the
# pure-numpy helpers above.


def _default_fair_tsc_ckpt() -> str:
    """Return the default path for the trained Fair-TSC ckpt.

    Resolution order (first hit wins):
      1. env var `FAIR_TSC_CKPT`
      2. newest `final.pt` (else newest `ep_*.pt`) under the current
         demand's `<BASE_DIR>/checkpoints/fair_tsc_4x4_<demand>_*` directory
      3. newest `final.pt` (else newest `ep_*.pt`) under any
         `<BASE_DIR>/checkpoints/fair_tsc_4x4_*` directory
    Raises FileNotFoundError if nothing usable is found.
    """
    import glob

    env_path = os.environ.get("FAIR_TSC_CKPT")
    if env_path:
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                f"FAIR_TSC_CKPT env var points to non-existent path: {env_path}"
            )
        return env_path

    # Prefer the current demand level, then fall back to any Fair-TSC 4x4 run.
    run_dirs = sorted(glob.glob(
        os.path.join(C.BASE_DIR, "checkpoints", f"fair_tsc_4x4_{C.DEMAND_LEVEL}_*")
    ))
    if not run_dirs:
        run_dirs = sorted(glob.glob(
            os.path.join(C.BASE_DIR, "checkpoints", "fair_tsc_4x4_*")
        ))
    for d in reversed(run_dirs):
        f = os.path.join(d, "final.pt")
        if os.path.exists(f):
            return f
        eps = sorted(glob.glob(os.path.join(d, "ep_*.pt")))
        if eps:
            return eps[-1]

    raise FileNotFoundError(
        "No Fair-TSC checkpoint found. Looked for FAIR_TSC_CKPT env var, "
        f"fair_tsc_4x4_{C.DEMAND_LEVEL}_*/final.pt (or ep_*.pt), and any "
        f"fair_tsc_4x4_*/final.pt (or ep_*.pt) under "
        f"{os.path.join(C.BASE_DIR, 'checkpoints')}."
    )


def load_shared_ue_critic(
    ckpt_path: Optional[str] = None,
    env=None,
    device=None,
):
    """Load the SHARED V^UE critic from a Fair-TSC checkpoint.

    The returned module is the actual frozen `SharedCritic`; callers
    invoke `critic(global_obs_batch, agent_idx_batch)` directly.  No
    callable wrapper, no numpy bridge — that's the caller's job (which
    keeps the per-method δ code symmetric: same call shape for V^UE and
    V_self).

    Args:
        ckpt_path: path to a Fair-TSC ckpt dict containing `critic_ue`.
                   None → `_default_fair_tsc_ckpt()`.
        env:       a `FairTSCEnv` (already reset once so `global_obs_dim`
                   / `num_agents` are populated).  Required.
        device:    torch device.  None → cuda if available else cpu.

    Returns:
        A frozen `SharedCritic` in `.eval()` mode with `requires_grad=False`.

    Raises:
        FileNotFoundError if the ckpt cannot be located.
        KeyError if the ckpt dict has no `critic_ue` key (lists keys it does have).
        RuntimeError if shape mismatch on `load_state_dict(strict=True)`.
    """
    if ckpt_path is None:
        ckpt_path = _default_fair_tsc_ckpt()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"V^UE ckpt not found: {ckpt_path}")
    if env is None:
        raise ValueError(
            "load_shared_ue_critic() requires `env` (post-reset) to read "
            "global_obs_dim and num_agents."
        )
    if env.num_agents <= 0 or env.global_obs_dim <= 0:
        raise ValueError(
            "env appears un-reset (num_agents / global_obs_dim are zero). "
            "Call env.reset() before load_shared_ue_critic()."
        )

    import torch  # deferred
    from networks import SharedCritic

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch versions don't support weights_only kwarg
        ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise KeyError(
            f"Ckpt at {ckpt_path} is not a dict (got {type(ckpt).__name__}); "
            "expected a Fair-TSC training save dict containing `critic_ue`."
        )
    if "critic_ue" not in ckpt:
        raise KeyError(
            f"Ckpt at {ckpt_path} is missing the `critic_ue` state_dict key. "
            f"Keys present: {sorted(ckpt.keys())}. "
            "Re-train Fair-TSC or point FAIR_TSC_CKPT at a ckpt that has it."
        )

    critic_ue = SharedCritic(
        global_obs_dim=env.global_obs_dim,
        num_agents=env.num_agents,
        hidden=C.CRITIC_HIDDEN,
    ).to(device)
    try:
        critic_ue.load_state_dict(ckpt["critic_ue"], strict=True)
    except RuntimeError as e:
        # Decorate the error with both the expected (model) and provided
        # (ckpt) shapes so the user can diagnose a wrong-shape ckpt fast.
        model_shapes = {k: tuple(v.shape) for k, v in critic_ue.state_dict().items()}
        ckpt_shapes  = {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                        for k, v in ckpt["critic_ue"].items()}
        raise RuntimeError(
            f"Shape mismatch loading critic_ue from {ckpt_path}.\n"
            f"Model SharedCritic state_dict shapes: {model_shapes}\n"
            f"Ckpt   critic_ue   state_dict shapes: {ckpt_shapes}\n"
            f"Original error: {e}"
        ) from e

    critic_ue.eval()
    for p in critic_ue.parameters():
        p.requires_grad_(False)

    print(f"[evaluate.load_shared_ue_critic] loaded V^UE from {ckpt_path}  "
          f"(global_obs_dim={env.global_obs_dim}, num_agents={env.num_agents})")
    return critic_ue


# Backwards-compatibility alias (old name; new code should use load_shared_ue_critic).
load_v_ue = load_shared_ue_critic


# ────────────────────────────────────────────────────────────────────
# Per-method δ computation
# ────────────────────────────────────────────────────────────────────
#
# δ_i(t) = max( V^UE(s_t, i) − G_t^method(i), 0 )
#
# - V^UE          : the SHARED frozen critic returned by
#                   load_shared_ue_critic.
# - G_t^method(i) : the realized discounted return for agent i under
#                   method M, computed by a backward γ-sweep over the
#                   raw env rewards captured during the eval rollout.
#
# This formula is applied UNIFORMLY to every method (Fixed, MP, IPPO,
# FairLight, Fair-TSC).  No per-method critic is consulted for δ.


def compute_deltas_from_rollout(
    rollout: List[Dict],
    v_ue,            # SharedCritic (frozen)
    num_agents: int,
    gamma: float = None,
) -> np.ndarray:
    """Compute δ_TN = [T, N] = max( V^UE(s_t, i) − G_t(i), 0 ).

    Args:
        rollout:    list of per-step dicts, each with
                      - "global_obs"    : np.ndarray [D_g]  (state BEFORE the step)
                      - "rewards_array" : np.ndarray [N]    (raw env rewards
                                                              in env.agent_ids order)
        v_ue:       the shared frozen SharedCritic from
                    `load_shared_ue_critic`.
        num_agents: N.
        gamma:      discount factor.  None → config.GAMMA (0.99).

    Returns:
        deltas_TN: np.ndarray [T, N], float32, non-negative.
    """
    import torch  # deferred
    if gamma is None:
        gamma = C.GAMMA

    T = len(rollout)
    if T == 0:
        return np.zeros((0, num_agents), dtype=np.float32)

    # Sanity: V^UE agent-id one-hot dim must match `num_agents`.
    if getattr(v_ue, "num_agents", num_agents) != num_agents:
        raise ValueError(
            f"V^UE expects num_agents={v_ue.num_agents} but caller said {num_agents}."
        )

    # ── Stack the rollout into matrices ─────────────────────────────
    g_mat = np.stack(
        [np.asarray(step["global_obs"], dtype=np.float32) for step in rollout],
        axis=0,
    )                                                          # [T, D_g]
    r_mat = np.stack(
        [np.asarray(step["rewards_array"], dtype=np.float32) for step in rollout],
        axis=0,
    )                                                          # [T, N]
    if r_mat.shape != (T, num_agents):
        raise ValueError(
            f"rewards_array stack has shape {r_mat.shape}; expected ({T},{num_agents})."
        )

    # ── Backward γ-sweep to get G[t, i] ─────────────────────────────
    # G[T-1] = R[T-1]; G[t] = R[t] + γ · G[t+1]
    G = np.zeros_like(r_mat)
    G[T - 1] = r_mat[T - 1]
    for t in range(T - 2, -1, -1):
        G[t] = r_mat[t] + gamma * G[t + 1]

    # ── Batched single V^UE forward over [T*N, D_g] ─────────────────
    device = next(v_ue.parameters()).device
    g_tiled = np.repeat(g_mat, num_agents, axis=0)             # [T*N, D_g]
    idx_tiled = np.tile(np.arange(num_agents, dtype=np.int64), T)  # [T*N]

    g_t = torch.from_numpy(g_tiled).to(device)
    i_t = torch.from_numpy(idx_tiled).to(device)
    with torch.no_grad():
        v_ue_flat = v_ue(g_t, i_t).detach().cpu().numpy()      # [T*N]
    V_ue = v_ue_flat.reshape(T, num_agents)                    # [T, N]

    deltas = np.maximum(V_ue - G, 0.0).astype(np.float32)
    return deltas
