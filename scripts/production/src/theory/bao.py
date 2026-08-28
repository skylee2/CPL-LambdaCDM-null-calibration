# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Reconstructed DESI-DR2-like BAO theory and covariance."""
import numpy as np
from scipy.linalg import block_diag

from .background import D_H, D_M

OBSERVABLE_NAMES = tuple(
    name for _ in range(6) for name in ("DM_over_rd", "DH_over_rd")
)


def covariance_from_config(cfg):
    blocks = [np.asarray(x, dtype=float) for x in cfg["bao"]["covariance_blocks"]]
    if len(blocks) != 6 or any(x.shape != (2, 2) for x in blocks):
        raise ValueError("Expected six 2x2 BAO covariance blocks")
    cov = block_diag(*blocks)
    np.linalg.cholesky(cov)
    return cov


def theory_vector(theta, redshift, r_d):
    z = np.asarray(redshift, dtype=float)
    if z.shape != (6,) or r_d <= 0:
        raise ValueError("Expected six redshifts and positive r_d")
    out = np.empty(12)
    out[0::2] = D_M(z, theta) / r_d
    out[1::2] = D_H(z, theta) / r_d
    if not np.all(np.isfinite(out)):
        raise ValueError("Non-finite BAO theory")
    return out
