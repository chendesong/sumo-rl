"""One-shot algorithm-comparison driver for the Fair-TSC preliminary plot.

Runs all baselines through the SAME `evaluate.evaluate_run` entry point
and produces `comparison_preliminary.csv` with columns:

    method, theil_ema, efficiency, ped_wait, ped_risk, delta_max, delta_valid

Pipeline (all rows use the unified δ formula
        δ_i(t) = max(V^UE(s_t,i) − G_t(i), 0),
where G is the realized discounted return from raw env rewards):

  - fixed_time       : 1 eval episode, no training. δ via shared V^UE
                       + G from this rollout.
  - max_pressure     : 1 eval episode, no training. δ via shared V^UE
                       + G from this rollout.
  - ippo             : 150 train + 1 eval. δ via shared V^UE + G from
                       the eval rollout (IPPO's critic is NOT used).
  - fairsignal       : 150 train + 1 eval using Cai et al.'s
                       intersection-queue Jain-style reward. δ via shared
                       V^UE + G from the eval rollout, with G computed from
                       the RAW env reward (NOT the FairSignal-shaped reward).
  - fair_tsc         : load Fair-TSC ckpt (actor_marl + critic_ue), 1
                       fresh eval. δ via shared V^UE + G from rollout.
                       critic_marl is NOT loaded — not used anymore.
  - fair_tsc_csv_avg : reference row — last-N stage-2 average from the
                       same run's train_log.csv.  Those numbers were
                       computed with V^MARL during training (legacy
                       reference); any discrepancy with `fair_tsc` is
                       expected and intentional.

delta_valid is True for every row.  Efficiency definition: forced through
the "reward_series" path so every method has the same scalar definition.
"""

import csv
import glob
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from evaluate import (
    MetricsCollector,
    compute_deltas_from_rollout,
    evaluate_run,
    load_shared_ue_critic,
)
from safety_eval import normalize_pedestrian_risk


# ────────────────────────────────────────────────────────────────────
# Fair-TSC row: read from existing run's train_log.csv
# ────────────────────────────────────────────────────────────────────

FAIR_TSC_DEFAULT_RUN = os.environ.get(
    "FAIR_TSC_RUN_DIR",
    os.path.join(C.BASE_DIR, "outputs", "fair_tsc_4x4_high_20260517_2359"),
)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def harvest_fair_tsc_metrics(run_dir: str = None, last_n: int = 20) -> Dict:
    """Pull the most recent stage-2 episodes from a Fair-TSC run.

    Returns a dict in the same shape `evaluate_run` produces, plus a
    `_source` field for traceability.  delta_valid=True — these are
    valid Theil/δ measurements from training, but computed with V^MARL
    (legacy reference) rather than the G-based estimator used by the
    other rows.  Any gap to `fair_tsc` is expected, not a bug.
    """
    if run_dir is None:
        run_dir = FAIR_TSC_DEFAULT_RUN
    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        cands = sorted(glob.glob(os.path.join(C.BASE_DIR, "outputs", "fair_tsc_4x4_high_*", "train_log.csv")))
        if not cands:
            raise FileNotFoundError(f"No Fair-TSC train_log.csv found (looked at {log_path})")
        log_path = cands[-1]
        run_dir = os.path.dirname(log_path)

    rows = _read_csv_rows(log_path)
    s2 = [r for r in rows if r.get("stage") == "2"]
    if not s2:
        raise RuntimeError(f"No stage-2 rows in {log_path}; training may still be in stage-1.")
    tail = s2[-last_n:]

    def _f(r, k, default=0.0):
        v = r.get(k)
        try:
            return float(v) if v not in ("", None) else default
        except ValueError:
            return default

    theil_inter = float(np.mean([_f(r, "theil_inter", _f(r, "theil")) for r in tail]))
    theil_ema = float(np.mean([_f(r, "theil_smoothed", theil_inter) for r in tail]))
    theil_raw = theil_inter
    theil_intra = float(np.mean([_f(r, "theil_intra") for r in tail]))
    max_phase_interval = float(np.mean([_f(r, "max_phase_interval") for r in tail]))
    efficiency = float(np.mean([_f(r, "reward_mean")    for r in tail]))
    delta_max  = float(np.mean([_f(r, "delta_max")      for r in tail]))
    delta_mean = float(np.mean([_f(r, "delta_mean")     for r in tail]))

    train_ped_wait = np.asarray([_f(r, "ped_wait", float("nan")) for r in tail], dtype=np.float64)
    train_ped_risk = np.asarray([_f(r, "ped_risk", float("nan")) for r in tail], dtype=np.float64)
    train_ped_expected = np.asarray(
        [_f(r, "ped_expected_violations", float("nan")) for r in tail], dtype=np.float64
    )
    ped_wait = float(np.nanmean(train_ped_wait)) if not np.isnan(train_ped_wait).all() else 0.0
    ped_risk = float(np.nanmean(train_ped_risk)) if not np.isnan(train_ped_risk).all() else 0.0
    ped_expected = (
        float(np.nanmean(train_ped_expected)) if not np.isnan(train_ped_expected).all() else 0.0
    )

    # Older Fair-TSC logs may not contain pedestrian risk directly.
    # Best-effort fallback: read per-episode ep_X.csv files written by
    # upstream sumo_rl under out_csv_name.
    ep_csvs = sorted(glob.glob(os.path.join(run_dir, "ep_*.csv")))
    if ep_csvs and (ped_wait == 0.0 or ped_expected == 0.0):
        try:
            wait_vals = []
            expected_vals = []
            for ep_path in ep_csvs[-last_n:]:
                for r in _read_csv_rows(ep_path):
                    v_wait = r.get("agents_total_ped_waiting_time")
                    v_expected = r.get("agents_total_expected_violations")
                    if v_wait not in (None, ""):
                        try:
                            wait_vals.append(float(v_wait))
                        except ValueError:
                            pass
                    if v_expected not in (None, ""):
                        try:
                            expected_vals.append(float(v_expected))
                        except ValueError:
                            pass
            if wait_vals:
                ped_wait = float(np.mean(wait_vals))
            if expected_vals:
                ped_expected = float(np.mean(expected_vals))
                ped_risk = normalize_pedestrian_risk(ped_expected, num_agents=C.NUM_AGENTS)
        except Exception:
            pass

    return {
        "theil_ema":   theil_ema,
        "theil_raw":   theil_raw,
        "theil_inter": theil_inter,
        "theil_intra": theil_intra,
        "max_phase_interval": max_phase_interval,
        "efficiency":  efficiency,
        "ped_wait":    ped_wait,
        "ped_risk":    ped_risk,
        "ped_expected_violations": ped_expected,
        "delta_max":   delta_max,
        "delta_mean":  delta_mean,
        "delta_valid": True,
        "_source":     log_path,
    }


# ────────────────────────────────────────────────────────────────────
# Fair-TSC fresh-eval path (1 episode with trained actor + critic_marl + V^UE)
# ────────────────────────────────────────────────────────────────────


def fair_tsc_fresh_eval(ckpt_path: Optional[str] = None,
                        seed: Optional[int] = None,
                        additional_sumo_cmd: Optional[str] = None) -> Dict:
    """Run ONE eval episode with the trained Fair-TSC actor_marl.

    δ is computed under the unified G-based formula:
        δ_i(t) = max( V^UE(s_t, i) − G_t(i), 0 )
    where V^UE = `critic_ue` from the ckpt (shared, frozen) and G_t(i)
    is the realized discounted return from the raw env rewards on this
    eval rollout.  critic_marl is NOT loaded — it is no longer used for
    δ under the locked design.
    """
    import torch  # deferred — keeps `run_comparison` importable without torch
    from sumo_env import FairTSCEnv
    from networks import SharedActor
    from evaluate import _default_fair_tsc_ckpt

    if ckpt_path is None:
        ckpt_path = _default_fair_tsc_ckpt()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Fair-TSC ckpt not found at {ckpt_path}; set FAIR_TSC_CKPT or "
            f"train first.")

    if seed is None:
        seed = C.SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    try:
        obs = env.reset(seed=seed)

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        # Only actor_marl + critic_ue are required.  critic_marl is no
        # longer consulted under the G-based δ design.
        for required in ("actor_marl", "critic_ue"):
            if required not in ckpt:
                raise KeyError(
                    f"Fair-TSC ckpt at {ckpt_path} is missing `{required}`. "
                    f"Keys present: {sorted(ckpt.keys())}."
                )

        actor = SharedActor(
            local_obs_dim=env.local_obs_dim,
            num_agents=env.num_agents,
            action_dim=env.action_dim,
            hidden=C.ACTOR_HIDDEN,
        ).to(device)
        actor.load_state_dict(ckpt["actor_marl"])
        actor.eval()

        # Use the same shared loader so V^UE construction stays identical
        # across all methods.
        v_ue = load_shared_ue_critic(ckpt_path=ckpt_path, env=env, device=device)

        coll = MetricsCollector()
        rollout = []
        done = False
        while not done:
            g = env.get_global_obs(obs)
            local_b  = torch.from_numpy(np.stack([obs[a] for a in env.agent_ids])).to(device)
            idx_b    = torch.arange(env.num_agents, device=device)
            with torch.no_grad():
                action, _logp = actor.act(local_b, idx_b, deterministic=True)
            action_dict = {a: int(action[i].item()) for i, a in enumerate(env.agent_ids)}
            next_obs, R, Cp, Cs, done, info = env.step(action_dict)
            r_arr = np.array([R[a] for a in env.agent_ids], dtype=np.float32)
            coll.add(info, mean_reward=float(r_arr.mean()) if r_arr.size else 0.0)
            rollout.append({"global_obs": g, "rewards_array": r_arr})
            obs = next_obs

        env_metrics = coll.finalize(env)

        # δ uses realized G from raw env rewards (NOT critic_marl).
        if len(rollout) == 0:
            deltas_TN = np.zeros((1, env.num_agents), dtype=np.float32)
        else:
            deltas_TN = compute_deltas_from_rollout(
                rollout, v_ue=v_ue, num_agents=env.num_agents, gamma=C.GAMMA,
            )
        result = evaluate_run(deltas_TN, env_metrics, delta_valid=True)
        result["_source"] = ckpt_path
        print(f"[fair_tsc fresh-eval] {result}")
        return result
    finally:
        env.close()


# ────────────────────────────────────────────────────────────────────
# Main driver
# ────────────────────────────────────────────────────────────────────

def _safe_run(label: str, fn):
    """Run `fn`, return its dict result; on any error, log and return a
    sentinel row so the comparison csv still gets written.
    """
    try:
        return fn()
    except Exception as e:
        print(f"[ERROR] {label} failed: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return {"theil_ema": float("nan"), "theil_raw": float("nan"),
                "theil_inter": float("nan"), "theil_intra": float("nan"),
                "max_phase_interval": float("nan"),
                "efficiency": float("nan"), "ped_wait": float("nan"),
                "ped_risk": float("nan"), "ped_expected_violations": float("nan"),
                "delta_max": float("nan"), "delta_mean": float("nan"),
                "delta_valid": False,
                "_error": str(e)}


def _preload_v_ue():
    """Build a throwaway env solely to populate `global_obs_dim` /
    `num_agents`, then load the shared V^UE once and return the module.

    Raises if the ckpt is missing or malformed — NO silent fallback to
    zeros.  The comparison cannot run without V^UE.
    """
    from sumo_env import FairTSCEnv
    env = FairTSCEnv(
        net_file=C.NET_FILE, route_file=C.ROUTE_FILE,
        out_csv_name=None,
        num_seconds=C.NUM_SECONDS, delta_time=C.DELTA_TIME, min_green=C.MIN_GREEN,
    )
    try:
        env.reset(seed=C.SEED)
        v_ue = load_shared_ue_critic(env=env)
    finally:
        env.close()
    return v_ue


def _fmt_num(v, fmt: str) -> str:
    """Format a numeric value or return 'N/A' for non-numerics / NaN."""
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return "N/A"
        return format(v, fmt)
    return str(v)


def main():
    out_dir = os.path.join(C.BASE_DIR, "outputs", "comparison_preliminary")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "comparison_preliminary.csv")
    print(f"[run_comparison] output csv: {csv_path}")

    # ── Pre-load shared V^UE once and reuse across baselines ───────
    print("\n===== loading shared V^UE =====")
    v_ue = _preload_v_ue()

    results: Dict[str, Dict] = {}

    print("\n===== fixed_time =====")
    from baselines.fixed_time import main as run_fixed
    results["fixed_time"] = _safe_run("fixed_time", lambda: run_fixed(v_ue=v_ue))

    print("\n===== max_pressure =====")
    from baselines.max_pressure import main as run_mp
    results["max_pressure"] = _safe_run("max_pressure", lambda: run_mp(v_ue=v_ue))

    print("\n===== ippo (150 episodes) =====")
    from baselines.ippo import train_ippo
    results["ippo"] = _safe_run("ippo", lambda: train_ippo(num_episodes=150, v_ue=v_ue))

    print("\n===== fairsignal (150 episodes) =====")
    from baselines.fairsignal import train_fairsignal
    results["fairsignal"] = _safe_run(
        "fairsignal", lambda: train_fairsignal(num_episodes=150, v_ue=v_ue))

    print("\n===== fair_tsc (fresh 1-ep eval with trained ckpt) =====")
    results["fair_tsc"] = _safe_run("fair_tsc", lambda: fair_tsc_fresh_eval())

    print("\n===== fair_tsc_csv_avg (last-N stage-2 avg from train_log.csv) =====")
    results["fair_tsc_csv_avg"] = _safe_run("fair_tsc_csv_avg",
                                            lambda: harvest_fair_tsc_metrics())

    # ── Dump comparison csv ────────────────────────────────────────
    fields = [
        "method",
        "theil_ema",
        "theil_inter",
        "theil_intra",
        "max_phase_interval",
        "efficiency",
        "ped_wait",
        "ped_risk",
        "ped_expected_violations",
        "delta_max",
        "delta_valid",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, r in results.items():
            w.writerow({
                "method":      method,
                "theil_ema":   r.get("theil_ema"),
                "theil_inter": r.get("theil_inter"),
                "theil_intra": r.get("theil_intra"),
                "max_phase_interval": r.get("max_phase_interval"),
                "efficiency":  r.get("efficiency"),
                "ped_wait":    r.get("ped_wait"),
                "ped_risk":    r.get("ped_risk"),
                "ped_expected_violations": r.get("ped_expected_violations"),
                "delta_max":   r.get("delta_max"),
                "delta_valid": r.get("delta_valid", False),
            })

    print(f"\n[run_comparison] wrote {csv_path}")
    for method, r in results.items():
        dv = bool(r.get("delta_valid", False))
        # All methods now have real δ numbers (G-based δ); only NaNs
        # from a failed _safe_run still render as N/A via _fmt_num.
        te_s = _fmt_num(r.get("theil_inter", r.get("theil_ema")), ".4f")
        ti_s = _fmt_num(r.get("theil_intra"), ".4f")
        mp_s = _fmt_num(r.get("max_phase_interval"), ".1f")
        dm_s = _fmt_num(r.get("delta_max"),  ".4f")
        ef_s = _fmt_num(r.get("efficiency"), ".2f")
        pw_s = _fmt_num(r.get("ped_wait"),   ".1f")
        pr_s = _fmt_num(r.get("ped_risk"),   ".6f")
        print(f"  {method:18s}  Tinter={te_s}  Tintra={ti_s}  max_phase={mp_s}  eff={ef_s}  "
              f"ped={pw_s}  ped_risk={pr_s}  delta_max={dm_s}  delta_valid={dv}")


if __name__ == "__main__":
    main()
