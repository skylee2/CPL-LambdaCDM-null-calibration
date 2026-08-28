# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""CLASS calls and transparent two-dimensional early-universe emulator."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from classy import Class
from scipy.interpolate import RectBivariateSpline

from .background import C_KM_S, D_M


OUTPUTS = ("r_d", "r_star", "z_drag", "z_star")


def class_parameters(cfg: dict, theta: dict) -> dict:
    p = dict(cfg["class"]["parameters"])
    p.update(cfg["class"]["precision"])
    p.update(
        H0=float(theta["H0"]),
        omega_b=float(theta["omega_b"]),
        omega_cdm=float(theta["omega_m"] - theta["omega_b"]),
        w0_fld=float(theta["w0"]),
        wa_fld=float(theta["wa"]),
    )
    if p["omega_cdm"] <= 0:
        raise ValueError("omega_cdm must be positive")
    return p


def run_class(cfg: dict, theta: dict) -> dict:
    p = class_parameters(cfg, theta)
    start = time.perf_counter()
    cosmo = Class()
    try:
        cosmo.set(p)
        cosmo.compute()
        d = cosmo.get_current_derived_parameters(
            ["z_d", "rs_d", "z_rec", "rs_rec", "100*theta_s"]
        )
        zstar = float(d["z_rec"])
        dmstar = float(cosmo.comoving_distance(zstar))
        h = theta["H0"] / 100.0
        Om = theta["omega_m"] / h**2
        R = np.sqrt(Om) * theta["H0"] * dmstar / C_KM_S
        lA = np.pi * dmstar / float(d["rs_rec"])
        return {
            "z_drag": float(d["z_d"]),
            "r_d": float(d["rs_d"]),
            "z_star": zstar,
            "r_star": float(d["rs_rec"]),
            "theta_star": float(d["100*theta_s"]) / 100.0,
            "R": float(R),
            "l_A": float(lA),
            "D_M_z_star": dmstar,
            "runtime_seconds": time.perf_counter() - start,
            "class_input": p,
        }
    finally:
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except Exception:
            pass


class EarlyUniverseEmulator:
    version = "pedagogic2-rbs-v1"

    def __init__(self, grid_path):
        raw = np.load(grid_path, allow_pickle=False)
        self.omega_m = raw["omega_m_axis"]
        self.omega_b = raw["omega_b_axis"]
        self.splines = {
            name: RectBivariateSpline(
                self.omega_m, self.omega_b, raw[name], kx=3, ky=3, s=0
            )
            for name in OUTPUTS
        }
        self.grid_path = str(Path(grid_path).resolve())

    def predict(self, omega_m, omega_b) -> dict:
        if not (
            self.omega_m[0] <= omega_m <= self.omega_m[-1]
            and self.omega_b[0] <= omega_b <= self.omega_b[-1]
        ):
            raise ValueError("Early-universe request outside emulator domain")
        return {
            k: float(np.asarray(s(float(omega_m), float(omega_b), grid=False)).item())
            for k, s in self.splines.items()
        }


def cmb_from_emulator(theta: dict, emulator: EarlyUniverseEmulator) -> dict:
    early = emulator.predict(theta["omega_m"], theta["omega_b"])
    dm = D_M(early["z_star"], theta)
    h = theta["H0"] / 100.0
    Om = theta["omega_m"] / h**2
    early["R"] = float(np.sqrt(Om) * theta["H0"] * dm / C_KM_S)
    early["l_A"] = float(np.pi * dm / early["r_star"])
    early["D_M_z_star"] = float(dm)
    return early
