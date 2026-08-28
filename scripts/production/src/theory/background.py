# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Flat CPL background shared by BAO, CMB, and SNe."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

C_KM_S = 299792.458
OMEGA_GAMMA_H2_27255 = 2.47297533e-5
_GL_X, _GL_W = leggauss(32)


def validate_joint(theta: dict) -> None:
    required = ("omega_m", "omega_b", "H0", "w0", "wa", "DeltaM")
    missing = [k for k in required if k not in theta]
    if missing:
        raise KeyError(f"Missing joint parameters: {missing}")
    h = theta["H0"] / 100.0
    if not (theta["H0"] > 0 and theta["omega_m"] > theta["omega_b"] > 0):
        raise ValueError("Require H0>0 and omega_m>omega_b>0")
    if theta["omega_m"] / h**2 >= 1:
        raise ValueError("Flat model requires Omega_m<1")


def density_fractions(theta: dict, T_CMB=2.7255, N_eff=3.046):
    validate_joint(theta)
    h = theta["H0"] / 100.0
    omega_gamma = OMEGA_GAMMA_H2_27255 * (T_CMB / 2.7255) ** 4
    omega_r = omega_gamma * (1.0 + 0.22710731766 * N_eff)
    Om = theta["omega_m"] / h**2
    Or = omega_r / h**2
    Ode = 1.0 - Om - Or
    if Ode <= 0:
        raise ValueError("Non-positive dark-energy density")
    return Om, Or, Ode


def e2(z, theta: dict, T_CMB=2.7255, N_eff=3.046):
    z = np.asarray(z, dtype=float)
    if np.any(z < 0):
        raise ValueError("Redshift must be non-negative")
    Om, Or, Ode = density_fractions(theta, T_CMB, N_eff)
    zp1 = 1.0 + z
    de = Ode * zp1 ** (3.0 * (1.0 + theta["w0"] + theta["wa"])) * np.exp(
        -3.0 * theta["wa"] * z / zp1
    )
    return Or * zp1**4 + Om * zp1**3 + de


def E(z, theta: dict, **kwargs):
    return np.sqrt(e2(z, theta, **kwargs))


def D_H(z, theta: dict, **kwargs):
    return C_KM_S / theta["H0"] / E(z, theta, **kwargs)


def D_M(z, theta: dict, **kwargs):
    values = np.atleast_1d(np.asarray(z, dtype=float))
    if np.any(values < 0):
        raise ValueError("Redshift must be non-negative")
    if np.ndim(z):
        nodes=0.5*values[:,None]*(_GL_X[None,:]+1.0)
        ans=C_KM_S/theta["H0"]*0.5*values*np.sum(
            _GL_W[None,:]/E(nodes,theta,**kwargs),axis=1
        )
    else:
        # Integrating in x=ln(1+z) keeps the high-redshift scalar integrand
        # smooth.  At z_star this 32-node result agrees with the former
        # adaptive quadrature to better than 3e-11 Mpc.
        xmax=np.log1p(values[0])
        xnodes=0.5*xmax*(_GL_X+1.0)
        znodes=np.expm1(xnodes)
        ans=np.array([C_KM_S/theta["H0"]*0.5*xmax*np.sum(
            _GL_W*np.exp(xnodes)/E(znodes,theta,**kwargs)
        )])
    return ans if np.ndim(z) else float(ans[0])
