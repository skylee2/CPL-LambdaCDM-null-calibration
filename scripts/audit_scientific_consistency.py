#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Fail-fast consistency audit for the compact release tables."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read(name: str) -> list[dict]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def yes(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def ids(table: list[dict]) -> list[str]:
    return [row["realization"] for row in table]


def close(actual, expected, tolerance, label) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected} within {tolerance}")


def main() -> None:
    expected_ids = [f"N100-{index}" for index in range(100)]
    table_names = (
        "lr_statistics.csv", "probe_contributions.csv", "posterior_centers.csv",
        "coverage_results.csv", "weak_mode_summary.csv", "pivot_summary.csv",
        "optimization_diagnostics.csv",
    )
    tables = {name: read(name) for name in table_names}
    for name, table in tables.items():
        if ids(table) != expected_ids:
            raise AssertionError(f"{name} does not contain exactly ordered IDs N100-0..N100-99")

    coverage = tables["coverage_results.csv"]
    excluded = [row["realization"] for row in coverage if not yes(row["HPD95_contains_generating_point"])]
    if excluded != ["N100-21", "N100-61", "N100-65", "N100-87"]:
        raise AssertionError(f"primary HPD exclusions differ: {excluded}")
    if sum(yes(row["HPD68_contains_generating_point"]) for row in coverage) != 64:
        raise AssertionError("direct HPD68 inclusion count is not 64/100")
    if sum(yes(row["HPD95_contains_generating_point"]) for row in coverage) != 96:
        raise AssertionError("direct HPD95 inclusion count is not 96/100")
    marginal_expected = {
        "w0": (68, 94), "wa": (63, 94), "wp": (70, 95),
        "H0": (71, 93), "Omega_m0": (65, 94),
    }
    for parameter, expected in marginal_expected.items():
        actual = tuple(sum(yes(row[f"{parameter}_marginal{level}_contains_generating_value"])
                           for row in coverage) for level in (68, 95))
        if actual != expected:
            raise AssertionError(f"{parameter} marginal coverage {actual} != {expected}")

    lr = tables["lr_statistics.csv"]
    values = np.array([float(row["Delta_chi2_LR"]) for row in lr])
    maximum_index = int(np.argmax(values))
    if lr[maximum_index]["realization"] != "N100-65":
        raise AssertionError("LR maximum is not N100-65")
    close(values[maximum_index], 16.411, .0005, "maximum LR")
    close(np.quantile(values, .68, method="linear"), 2.548, .0005, "LR Q68")
    close(np.quantile(values, .90, method="linear"), 4.426, .0005, "LR Q90")
    close(np.quantile(values, .95, method="linear"), 5.529, .0005, "LR Q95")
    close(np.quantile(values, .99, method="linear"), 9.187, .0005, "LR Q99")
    if int(np.count_nonzero(values >= 6.18)) != 4:
        raise AssertionError("LR threshold >=6.18 count is not 4")
    if int(np.count_nonzero(values >= 9.21)) != 1:
        raise AssertionError("LR threshold >=9.21 count is not 1")

    probes = {row["realization"]: row for row in tables["probe_contributions.csv"]}
    n65 = probes["N100-65"]
    components = [float(n65[name]) for name in
                  ("Delta_chi2_BAO_LR", "Delta_chi2_CMB_LR", "Delta_chi2_SN_LR")]
    for actual, expected, label in zip(components, (9.798, 4.857, 1.755), ("BAO", "CMB", "SNe")):
        close(actual, expected, .0005, f"N100-65 {label} component")
    total = float(n65["total_Delta_chi2_LR"])
    close(sum(components), total, 1e-9, "N100-65 component sum")

    weak = tables["weak_mode_summary.csv"]
    ratios = np.array([float(row["eigenvalue_ratio"]) for row in weak])
    close(np.median(ratios), 68.05, .005, "median eigenvalue ratio")
    close(np.sqrt(np.median(ratios)), 8.25, .005, "median principal-axis ratio")
    close(np.quantile(ratios, .16), 58.36, .005, "eigenvalue-ratio Q16")
    close(np.quantile(ratios, .84), 78.80, .005, "eigenvalue-ratio Q84")

    pivot = np.array([float(row["z_p"]) for row in tables["pivot_summary.csv"]])
    close(np.median(pivot), .297, .0005, "median pivot redshift")
    close(np.quantile(pivot, .16), .280, .0005, "pivot-redshift Q16")
    close(np.quantile(pivot, .84), .319, .0005, "pivot-redshift Q84")
    close(pivot.min(), .263, .0005, "pivot-redshift minimum")
    close(pivot.max(), .412, .0005, "pivot-redshift maximum")

    ensemble = {row["parameter"]: row for row in read("ensemble_summary.csv")}
    recovery = {
        "omega_m": (0.146965, -0.000035, 0.000107, 0.001069, -0.015, 1.096),
        "H0": (70.04847, 0.04847, 0.06852, 0.68344, 0.061, 1.017),
        "Omega_m0": (0.299575, -0.000425, 0.000610, 0.006089, -0.093, 1.034),
        "w0": (-0.998820, 0.001180, 0.005316, 0.052910, -0.007, 1.022),
        "wa": (-0.015076, -0.015076, 0.020635, 0.205867, 0.019, 1.079),
        "wp": (-1.001248, -0.001248, 0.002553, 0.025430, -0.039, 1.011),
    }
    fields = ("ensemble_mean", "bias", "standard_error_of_bias", "rmse", "pull_mean", "pull_width")
    tolerances = (5e-6, 5e-6, 5e-6, 5e-6, 5e-4, 5e-4)
    for parameter, expected in recovery.items():
        for field, target, tolerance in zip(fields, expected, tolerances):
            close(ensemble[parameter][field], target, tolerance, f"{parameter} {field}")

    centers = tables["posterior_centers.csv"]
    order = ("Omega_m0", "H0", "w0", "wa", "wp")
    center_matrix = np.array([[float(row[f"{parameter}_median"]) for parameter in order]
                              for row in centers])
    correlation = np.corrcoef(center_matrix, rowvar=False)
    close(correlation[0, 1], -.934, .0005, "r(H0,Omega_m0)")
    close(correlation[2, 3], -.882, .0005, "r(w0,wa)")
    close(correlation[1, 4], -.868, .0005, "r(H0,wp)")
    close(correlation[0, 4], .69, .005, "r(Omega_m0,wp)")
    close(correlation[0, 2], .57, .005, "r(Omega_m0,w0)")

    optimization = tables["optimization_diagnostics.csv"]
    if not all(yes(row["LCDM_optimizer_success"]) and yes(row["CPL_optimizer_success"])
               for row in optimization):
        raise AssertionError("one or more optimizer success flags are false")
    if max(float(row["max_restart_chi2_spread"]) for row in optimization) >= 8e-10:
        raise AssertionError("optimizer restart agreement is not below 8e-10")
    print("SCIENTIFIC CONSISTENCY AUDIT: PASSED")
    print("100 ordered realizations; recovery/pulls/correlations, coverage/exclusions, LR calibration, probe sum, weak modes, pivot, and optimization verified.")


if __name__ == "__main__":
    main()
