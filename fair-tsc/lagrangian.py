"""Compatibility wrapper for the old module name.

The paper/code now use dual-level fairness with a PID adaptive weight.
New code should import from ``fairness`` directly.
"""

from fairness import (  # noqa: F401
    PIDFairnessController,
    apply_fair_advantage,
    build_per_agent_fair_cost,
    compute_inter_fairness,
    compute_sacrifice_gaps,
    phase_service_theil_from_intervals,
    reshape_deltas_to_step_agent,
    theil_t_batch,
    theil_t_contributions,
    theil_t_episode,
    theil_t_index,
)
