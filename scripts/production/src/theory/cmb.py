# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Compressed Planck-like CMB distance-prior theory."""
import numpy as np

OBSERVABLE_NAMES = ("R", "l_A", "omega_b")


def covariance_from_config(cfg):
    cov = np.asarray(cfg["cmb"]["covariance"], dtype=float)
    if cov.shape != (3, 3) or not np.allclose(cov, cov.T, rtol=0, atol=0):
        raise ValueError("CMB covariance must be exactly symmetric 3x3")
    np.linalg.cholesky(cov)
    return cov


def theory_vector(theta, early):
    out = np.array([early["R"], early["l_A"], theta["omega_b"]], dtype=float)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError("Invalid compressed-CMB theory")
    return out
