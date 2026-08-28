# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Shared-prior log probability used by the Pedagogic II MCMC pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.likelihoods.joint_likelihood import (
    JointLikelihood,
    JointPrior,
    LCDM_ORDER,
    PARAMETER_ORDER,
)

COMPONENT_BLOB_DTYPE = np.dtype(
    [("chi2_bao", "f8"), ("chi2_cmb", "f8"), ("chi2_sne", "f8"), ("chi2_total", "f8")]
)


@dataclass
class MCMCLogProbability:
    """Bind one realization/model to the same prior used by optimization."""

    likelihood: JointLikelihood
    model: str = "cpl"

    def __post_init__(self):
        self.model = self.model.lower()
        if self.model not in {"lcdm", "cpl"}:
            raise ValueError("model must be 'lcdm' or 'cpl'")
        # This is deliberately object identity, not a separately parsed prior.
        self.prior: JointPrior = self.likelihood.prior
        self.parameter_order = LCDM_ORDER if self.model == "lcdm" else PARAMETER_ORDER

    def theta_dict(self, theta):
        return self.prior.vector_to_theta(theta, self.model)

    def log_prior(self, theta):
        try:
            physical = self.theta_dict(theta)
        except (TypeError, ValueError):
            return -np.inf
        return 0.0 if self.prior.contains(physical) else -np.inf

    def component_chi_square(self, theta, realization=None):
        del realization  # realization is fixed by the JointLikelihood instance
        physical = self.theta_dict(theta)
        return self.likelihood.evaluate(physical, apply_prior=True)

    def log_likelihood(self, theta, realization=None):
        if not np.isfinite(self.log_prior(theta)):
            return -np.inf
        parts = self.component_chi_square(theta, realization)
        return -0.5 * parts["total"]

    def log_probability(self, theta, realization=None):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return (-np.inf, np.inf, np.inf, np.inf, np.inf)
        parts = self.component_chi_square(theta, realization)
        return (
            lp - 0.5 * parts["total"],
            parts["bao"],
            parts["cmb"],
            parts["sne"],
            parts["total"],
        )


def log_prior(theta, probability: MCMCLogProbability):
    return probability.log_prior(theta)


def log_likelihood(theta, realization, probability: MCMCLogProbability):
    return probability.log_likelihood(theta, realization)


def log_probability(theta, realization, probability: MCMCLogProbability):
    return probability.log_probability(theta, realization)
