# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Official-mask Pantheon+ synthetic-magnitude theory."""
from __future__ import annotations

import numpy as np

from .background import D_M


def read_release(data_path, covariance_path):
    data = np.genfromtxt(data_path, names=True, dtype=None, encoding="utf-8")
    tokens = np.fromfile(covariance_path, sep=" ")
    if tokens.size < 1 or not float(tokens[0]).is_integer():
        raise ValueError("Covariance must start with integer N")
    n = int(tokens[0])
    if tokens.size != 1 + n * n:
        raise ValueError(f"Expected {1+n*n} covariance tokens, found {tokens.size}")
    if len(data) != n:
        raise ValueError(f"N_data={len(data)} does not equal N_cov={n}")
    cov = tokens[1:].reshape(n, n)
    return data, cov


def select_official(data, covariance):
    if len(data) != covariance.shape[0] or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Data/covariance dimension mismatch; truncation is forbidden")
    mask = np.asarray(data["zHD"], dtype=float) > 0.01
    if int(mask.sum()) != 1590:
        raise ValueError(f"Official mask must retain 1590 rows, got {mask.sum()}")
    selected = covariance[np.ix_(mask, mask)]
    np.linalg.cholesky(selected)
    return mask, selected


def luminosity_distance(zHD, zHEL, theta):
    zHD = np.asarray(zHD, dtype=float)
    zHEL = np.asarray(zHEL, dtype=float)
    if zHD.shape != zHEL.shape:
        raise ValueError("zHD and zHEL shapes differ")
    return (1.0 + zHEL) * D_M(zHD, theta)


def theory_vector(zHD, zHEL, theta):
    dl = luminosity_distance(zHD, zHEL, theta)
    out = 5.0 * np.log10(dl) + 25.0 + theta["DeltaM"]
    if not np.all(np.isfinite(out)):
        raise ValueError("Non-finite SNe theory")
    return out
