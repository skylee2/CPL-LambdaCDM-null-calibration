#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Convergence diagnostics for an emcee chain."""
from __future__ import annotations

import numpy as np
import emcee


def split_rhat(chain):
    """Rank-unadjusted split R-hat, treating walkers as independent chains."""
    x = np.asarray(chain, float)
    if x.ndim != 3:
        raise ValueError("chain must have shape (steps, walkers, parameters)")
    n = x.shape[0] // 2
    if n < 2:
        return np.full(x.shape[2], np.inf)
    # Split each walker in time, then arrange (chains, draws, parameters).
    halves = np.concatenate((x[:n].transpose(1, 0, 2), x[-n:].transpose(1, 0, 2)), axis=0)
    chain_means = halves.mean(axis=1)
    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between = n * chain_means.var(axis=0, ddof=1)
    var_hat = (n - 1.0) / n * within + between / n
    return np.sqrt(var_hat / within)


def integrated_time(chain):
    return np.asarray(emcee.autocorr.integrated_time(chain, quiet=True), float)


def diagnose(chain, acceptance_fraction, thresholds):
    x = np.asarray(chain, float)
    tau = integrated_time(x)
    half_tau = integrated_time(x[x.shape[0] // 2 :])
    fractional_tau_change = np.abs(tau - half_tau) / tau
    rhat = split_rhat(x)
    ess = x.shape[0] * x.shape[1] / tau
    length_over_tau = x.shape[0] / tau
    mean_acceptance = float(np.mean(acceptance_fraction))
    finite = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(tau)))
    checks = {
        "finite": finite,
        "length_over_tau": bool(np.all(length_over_tau >= thresholds["minimum_chain_length_in_tau"])),
        "tau_stability": bool(np.all(fractional_tau_change < thresholds["maximum_fractional_tau_change"])),
        "split_rhat": bool(np.all(rhat < thresholds["split_Rhat_max"])),
        "ess": bool(np.all(ess >= thresholds["effective_sample_size_min_per_parameter"])),
        "acceptance": bool(
            thresholds["acceptance_fraction_range"][0]
            <= mean_acceptance
            <= thresholds["acceptance_fraction_range"][1]
        ),
    }
    return {
        "tau": tau.tolist(),
        "half_chain_tau": half_tau.tolist(),
        "fractional_tau_change": fractional_tau_change.tolist(),
        "length_over_tau": length_over_tau.tolist(),
        "split_rhat": rhat.tolist(),
        "ess": ess.tolist(),
        "mean_acceptance_fraction": mean_acceptance,
        "walker_acceptance_fraction": np.asarray(acceptance_fraction).tolist(),
        "checks": checks,
        "converged": bool(all(checks.values())),
    }
