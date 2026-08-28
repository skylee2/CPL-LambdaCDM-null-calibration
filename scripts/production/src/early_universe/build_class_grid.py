#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Build the predeclared regular CLASS grid."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.theory.early_universe import OUTPUTS, run_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "config/fiducial_lcdm.yaml")
    ap.add_argument("--output", default=ROOT / "results/early_universe/class_grid.npz")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ecfg = cfg["emulator"]
    nm, nb = map(int, ecfg["grid_shape"])
    om_axis = np.linspace(*map(float, ecfg["omega_m_range"]), nm)
    ob_axis = np.linspace(*map(float, ecfg["omega_b_range"]), nb)
    base = cfg["joint_parameters"]
    arrays = {k: np.empty((nm, nb)) for k in OUTPUTS}
    runtimes = np.empty((nm, nb))
    start = time.perf_counter()
    for i, om in enumerate(om_axis):
        for j, ob in enumerate(ob_axis):
            theta = dict(omega_m=om, omega_b=ob, H0=float(base["H0_km_s_Mpc"]),
                         w0=-1.0, wa=0.0, DeltaM=float(base["DeltaM_mag"]))
            out = run_class(cfg, theta)
            for k in OUTPUTS:
                arrays[k][i, j] = out[k]
            runtimes[i, j] = out["runtime_seconds"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, omega_m_axis=om_axis, omega_b_axis=ob_axis,
             class_runtime_seconds=runtimes, **arrays)
    meta = dict(shape=[nm, nb], parameter_order=["omega_m","omega_b"],
                outputs=list(OUTPUTS), elapsed_seconds=time.perf_counter()-start,
                mean_class_runtime_seconds=float(runtimes.mean()))
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
