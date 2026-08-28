#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Export compact public tables from the frozen final N100 analysis products.

This is a mechanical export: it does not rerun sampling, optimization, HPD
construction, or any scientific inference.  The input CSV is the production
output ``Main_N100_Final_N0_N99_PerRealization.csv`` and the optional JSON is
its matching ensemble summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PARAMETERS = ("omega_m", "H0", "Omega_m0", "w0", "wa", "wp")
GENERATING = {
    "omega_m": 0.147,
    "H0": 70.0,
    "Omega_m0": 0.3,
    "w0": -1.0,
    "wa": 0.0,
    "wp": -1.0,
}
REPRESENTATIVES = (41, 80, 65, 98)


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["realization_id"]))
    ids = [int(row["realization_id"]) for row in rows]
    if ids != list(range(100)):
        raise RuntimeError(f"expected realization IDs 0..99, found {ids}")
    return rows


def numeric(row: dict, name: str) -> float:
    return float(row[name])


def make_ensemble_summary(rows: list[dict], summary_path: Path | None) -> list[dict]:
    summary = json.loads(summary_path.read_text()) if summary_path else None
    output = []
    for name in PARAMETERS:
        values = np.array([numeric(row, f"{name}_median") for row in rows])
        pulls = np.array([numeric(row, f"{name}_pull_median") for row in rows])
        generated = GENERATING[name]
        computed = {
            "ensemble_mean": float(values.mean()),
            "bias": float(values.mean() - generated),
            "standard_error_of_bias": float(values.std(ddof=1) / math.sqrt(len(values))),
            "rmse": float(np.sqrt(np.mean((values - generated) ** 2))),
            "pull_mean": float(pulls.mean()),
            "pull_width": float(pulls.std(ddof=1)),
        }
        values_source = summary["bias"][name]["median"] if summary else computed
        output.append({
            "parameter": name,
            "generating_value": generated,
            "realization_estimator": "posterior_median",
            "ensemble_mean": values_source["ensemble_mean"],
            "bias": values_source["bias"],
            "standard_error_of_bias": values_source["standard_error_of_bias"],
            "rmse": values_source["rmse"],
            "pull_mean": values_source["pull_mean"],
            "pull_width": values_source["pull_width"],
        })
    return output


def orient_modes(row: dict) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(json.loads(row["w0__wa_covariance_json"]), dtype=float)
    values, vectors = np.linalg.eigh(covariance)
    short = vectors[:, 0]
    if short[0] < 0 or (short[0] == 0 and short[1] < 0):
        short *= -1
    long = vectors[:, 1]
    if long[1] < 0 or (long[1] == 0 and long[0] < 0):
        long *= -1
    delta = np.array([numeric(row, "w0_mean") + 1.0, numeric(row, "wa_mean")])
    displacement = np.array([short @ delta, long @ delta]) / np.sqrt(values)
    return values, displacement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-realization", type=Path, required=True)
    parser.add_argument("--ensemble-summary-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.per_realization)
    out = args.output_dir

    write_csv(out / "ensemble_summary.csv", make_ensemble_summary(rows, args.ensemble_summary_json))

    write_csv(out / "lr_statistics.csv", [{
        "realization": row["realization_label"],
        "Delta_chi2_A": row["delta_chi2_A"],
        "Delta_chi2_LR": row["delta_chi2_LR"],
    } for row in rows])

    write_csv(out / "probe_contributions.csv", [{
        "realization": row["realization_label"],
        "Delta_chi2_BAO_LR": row["delta_chi2_LR_bao"],
        "Delta_chi2_CMB_LR": row["delta_chi2_LR_cmb"],
        "Delta_chi2_SN_LR": row["delta_chi2_LR_sne"],
        "total_Delta_chi2_LR": row["delta_chi2_LR"],
        "classification": row["probe_classification"],
    } for row in rows])

    center_fields = ["realization"]
    center_fields += [f"{name}_{kind}" for name in
                      ("omega_m", "omega_b", "H0", "Omega_m0", "w0", "wa", "DeltaM", "wp")
                      for kind in ("mean", "median")]
    center_rows = []
    for row in rows:
        item = {"realization": row["realization_label"]}
        for field in center_fields[1:]:
            item[field] = row[field]
        center_rows.append(item)
    write_csv(out / "posterior_centers.csv", center_rows, center_fields)

    coverage_fields = [
        "realization",
        "w0_marginal68_contains_generating_value", "w0_marginal95_contains_generating_value",
        "wa_marginal68_contains_generating_value", "wa_marginal95_contains_generating_value",
        "wp_marginal68_contains_generating_value", "wp_marginal95_contains_generating_value",
        "H0_marginal68_contains_generating_value", "H0_marginal95_contains_generating_value",
        "Omega_m0_marginal68_contains_generating_value", "Omega_m0_marginal95_contains_generating_value",
        "HPD68_contains_generating_point", "HPD95_contains_generating_point",
        "ellipse68_contains_generating_point", "ellipse95_contains_generating_point",
    ]
    coverage_rows = []
    for row in rows:
        item = {"realization": row["realization_label"]}
        for name in ("w0", "wa", "wp", "H0", "Omega_m0"):
            item[f"{name}_marginal68_contains_generating_value"] = truth(row[f"{name}_fid_in_68"])
            item[f"{name}_marginal95_contains_generating_value"] = truth(row[f"{name}_fid_in_95"])
        item.update({
            "HPD68_contains_generating_point": truth(row["w0__wa_hpd_68"]),
            "HPD95_contains_generating_point": truth(row["w0__wa_hpd_95"]),
            "ellipse68_contains_generating_point": truth(row["w0__wa_ellipse_68"]),
            "ellipse95_contains_generating_point": truth(row["w0__wa_ellipse_95"]),
        })
        coverage_rows.append(item)
    write_csv(out / "coverage_results.csv", coverage_rows, coverage_fields)

    weak_rows = []
    for row in rows:
        values, displacement = orient_modes(row)
        ratio = values[1] / values[0]
        weak_rows.append({
            "realization": row["realization_label"],
            "lambda_short": values[0],
            "lambda_long": values[1],
            "eigenvalue_ratio": ratio,
            "axis_ratio": math.sqrt(ratio),
            "d_short": displacement[0],
            "d_long": displacement[1],
            "Delta_chi2_LR": row["delta_chi2_LR"],
        })
    write_csv(out / "weak_mode_summary.csv", weak_rows)

    write_csv(out / "pivot_summary.csv", [{
        "realization": row["realization_label"],
        "a_p": row["pivot_a"], "z_p": row["pivot_z"],
        "w_p_mean": row["wp_mean"], "w_p_median": row["wp_median"],
        "w_p_std": row["wp_std"],
        "w_p_q2p5": row["wp_q2p5"], "w_p_q16": row["wp_q16"],
        "w_p_q84": row["wp_q84"], "w_p_q97p5": row["wp_q97p5"],
    } for row in rows])

    write_csv(out / "optimization_diagnostics.csv", [{
        "realization": row["realization_label"],
        "chi2_min_LCDM": row["chi2_min_lcdm"],
        "chi2_min_CPL": row["chi2_min_cpl"],
        "LCDM_random_start_seed": 910000 + int(row["realization_id"]),
        "CPL_random_start_seed": 920000 + int(row["realization_id"]),
        "LCDM_optimizer_success": truth(row["lcdm_optimizer_success"]),
        "CPL_optimizer_success": truth(row["cpl_optimizer_success"]),
        "LCDM_optimizer_method": row["lcdm_optimizer_method"],
        "CPL_optimizer_method": row["cpl_optimizer_method"],
        "LCDM_total_starts": row["lcdm_optimizer_starts"],
        "CPL_total_starts": row["cpl_optimizer_starts"],
        "LCDM_restart_chi2_spread": row["lcdm_restart_spread"],
        "CPL_restart_chi2_spread": row["cpl_restart_spread"],
        "max_restart_chi2_spread": max(numeric(row, "lcdm_restart_spread"), numeric(row, "cpl_restart_spread")),
    } for row in rows])

    representative_rows = []
    wanted = set(REPRESENTATIVES)
    for row in rows:
        if int(row["realization_id"]) not in wanted:
            continue
        item = {"realization_id": row["realization_id"], "realization": row["realization_label"]}
        for name in ("w0", "wa", "H0", "Omega_m0"):
            item[f"{name}_mean"] = row[f"{name}_mean"]
            item[f"{name}_median"] = row[f"{name}_median"]
            item[f"cpl_best_{name}"] = row[f"cpl_best_{name}"]
        representative_rows.append(item)
    write_csv(out / "representative_points.csv", representative_rows)
    print(f"Wrote compact release tables to {out.resolve()}")


if __name__ == "__main__":
    main()
