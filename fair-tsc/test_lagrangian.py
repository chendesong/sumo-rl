"""Compatibility smoke tests for the new fairness module."""

import numpy as np

import config as C
from fairness import (
    PIDFairnessController,
    build_per_agent_fair_cost,
    compute_inter_fairness,
    phase_service_theil_from_intervals,
    theil_t_index,
)


def test_theil_unit():
    equal = np.array([1.0, 1.0, 1.0, 1.0])
    unequal = np.array([0.0, 0.0, 0.0, 4.0])
    assert theil_t_index(equal) < 1e-3
    assert theil_t_index(unequal) > theil_t_index(equal)


def test_pid_unit():
    pid = PIDFairnessController(
        target=1.0,
        kp=0.5,
        ki=0.1,
        kd=0.0,
        lambda_max=5.0,
        integral_max=10.0,
        ema_beta=0.0,
    )
    stats = pid.update(1.5)
    assert stats["lambda_fair"] > 0.0
    stats = pid.update(0.5)
    assert stats["lambda_fair"] >= 0.0


def test_dual_level_cost_unit():
    agent_ids = ["a", "b", "c", "d"]
    delta_mean = np.array([1.0, 1.0, 1.0, 4.0], dtype=np.float32)
    t_inter, inter_contrib = compute_inter_fairness(delta_mean, eps=C.THEIL_EPS)
    intervals = {
        "a": {0: [20, 25], 1: [22, 24]},
        "b": {0: [20, 20], 1: [20, 20]},
        "c": {0: [30, 10], 1: [25, 15]},
        "d": {0: [60], 1: [60]},
    }
    intra_by_agent, t_intra, max_interval = phase_service_theil_from_intervals(intervals, agent_ids)
    costs, c_fair = build_per_agent_fair_cost(
        agent_ids,
        inter_contrib,
        intra_by_agent,
        alpha=0.5,
        t_inter_0=1.0,
        t_intra_0=1.0,
        num_agents=len(agent_ids),
    )
    assert t_inter >= 0.0
    assert t_intra >= 0.0
    assert max_interval == 60.0
    assert set(costs) == set(agent_ids)
    assert c_fair >= 0.0
    assert abs(sum(costs.values()) - c_fair) < 1e-5, (
        f"per-agent costs sum {sum(costs.values()):.6f} != C_fair {c_fair:.6f}"
    )


def main():
    test_theil_unit()
    test_pid_unit()
    test_dual_level_cost_unit()
    print("fairness smoke tests passed")


if __name__ == "__main__":
    main()
