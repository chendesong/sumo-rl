"""Baseline controllers for Fair-TSC algorithm comparison.

All baselines reuse `sumo_env.FairTSCEnv` and report metrics via the
single shared `evaluate.evaluate_run` entry point. No baseline computes
Theil internally.
"""
