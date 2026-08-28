# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Reproducible independently spawned probe streams."""
from __future__ import annotations

import numpy as np

LABELS = ("bao", "cmb", "sne")


def spawn_probe_streams(master_entropy):
    master = np.random.SeedSequence(master_entropy)
    children = master.spawn(len(LABELS))
    streams = {
        label: np.random.Generator(np.random.PCG64(child))
        for label, child in zip(LABELS, children)
    }
    metadata = {
        "master_entropy": master.entropy,
        "generator_type": "numpy.random.Generator(PCG64)",
        "numpy_version": np.__version__,
        "streams": {
            label: {"spawn_key": list(child.spawn_key)}
            for label, child in zip(LABELS, children)
        },
    }
    return streams, metadata
