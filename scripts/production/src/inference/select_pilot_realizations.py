#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Resolve the frozen six-case pilot selection and optimizer centers."""
from __future__ import annotations

import csv
from pathlib import Path

SELECTION = (
    ("zero_noise", 0, "closure"),
    ("N10", 8, "smallest_delta_chi2"),
    ("N10", 6, "low_delta_chi2"),
    ("N10", 7, "near_typical_delta_chi2"),
    ("N10", 3, "moderately_high_delta_chi2"),
    ("N10", 9, "largest_pilot_delta_chi2"),
)


def select_pilot_rows(csv_path):
    with Path(csv_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r["dataset"], int(r["realization_id"])): r for r in rows}
    selected = []
    for dataset, realization_id, role in SELECTION:
        row = dict(lookup[(dataset, realization_id)])
        row["role"] = role
        selected.append(row)
    return selected


def optimizer_center(row, model):
    names = ("omega_m", "omega_b", "H0", "DeltaM") if model == "lcdm" else (
        "omega_m", "omega_b", "H0", "w0", "wa", "DeltaM"
    )
    return [float(row[f"{model}_{name}"]) for name in names]
