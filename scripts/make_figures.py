#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Regenerate the nine artwork PDFs from the compact release products.

Figures 6 and 8 use the released deterministic 40,000-draw subsample for each
of four representative realizations.  All other figures use realization-level
CSV tables only.  Generated PDFs are presentation reproductions; the numerical
CSV products and audit checks are the scientific authority.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cpl-null-calibration-matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import beta, chi2


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "figures" / "generated"
P68, P95 = 0.682689492137, 0.954499736104
FID = {"w0": -1.0, "wa": 0.0, "H0": 70.0, "Omega_m0": 0.3}
ASSETS = {
    1: ("Fig1a.pdf", "Fig1b.pdf"), 2: ("Fig2.pdf",), 3: ("Fig3.pdf",),
    4: ("Fig4.pdf",), 5: ("Fig5.pdf",), 6: ("Fig6.pdf",),
    7: ("Fig7.pdf",), 8: ("Fig8.pdf",),
}


def rows(name: str) -> list[dict]:
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def boolean(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"OUTPUT {path}")


def centers(out: Path) -> None:
    center = rows("posterior_centers.csv")
    coverage = {row["realization"]: row for row in rows("coverage_results.csv")}
    lr = {row["realization"]: float(row["Delta_chi2_LR"]) for row in rows("lr_statistics.csv")}
    labels = {"w0": r"$w_0$", "wa": r"$w_a$",
              "H0": r"$H_0\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$", "Omega_m0": r"$\Omega_{m0}$"}
    for (a, b, filename) in (("w0", "wa", "Fig1a.pdf"), ("H0", "Omega_m0", "Fig1b.pdf")):
        x = np.array([float(row[f"{a}_median"]) for row in center])
        y = np.array([float(row[f"{b}_median"]) for row in center])
        values = np.array([lr[row["realization"]] for row in center])
        excluded = np.array([not boolean(coverage[row["realization"]]["HPD95_contains_generating_point"])
                             for row in center])
        fig, ax = plt.subplots(figsize=(6.4, 5.1))
        scatter = ax.scatter(x, y, c=values, cmap="viridis", s=25)
        ax.scatter(x[excluded], y[excluded], facecolors="none", edgecolors="red", s=58,
                   label="95.45% HPD exclusion")
        ax.plot(FID[a], FID[b], "k*", ms=11, label="generating point")
        ax.set(xlabel=labels[a], ylabel=labels[b]); ax.legend(fontsize=8, frameon=False)
        fig.colorbar(scatter, ax=ax, label=r"$\Delta\chi^2_{\rm LR}$")
        fig.tight_layout(); save(fig, out / filename)


def correlation(out: Path) -> None:
    center = rows("posterior_centers.csv")
    order = ("Omega_m0", "H0", "w0", "wa", "wp")
    matrix = np.array([[float(row[f"{name}_median"]) for name in order] for row in center])
    corr = np.corrcoef(matrix, rowvar=False)
    labels = [r"$\Omega_{m0}$", r"$H_0$", r"$w_0$", r"$w_a$", r"$w_p$"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(5), labels, rotation=30); ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax); fig.tight_layout(); save(fig, out / "Fig2.pdf")


def exact_interval(count: int, total: int, confidence: float = .68) -> tuple[float, float]:
    alpha = 1 - confidence
    low = 0.0 if count == 0 else beta.ppf(alpha / 2, count, total - count + 1)
    high = 1.0 if count == total else beta.ppf(1 - alpha / 2, count + 1, total - count)
    return float(low), float(high)


def coverage(out: Path) -> None:
    data = rows("coverage_results.csv")
    definitions = (
        ("$w_0$ marginal", "w0_marginal"), ("$w_a$ marginal", "wa_marginal"),
        ("$w_p$ marginal", "wp_marginal"), ("$H_0$ marginal", "H0_marginal"),
        (r"$\Omega_{m0}$ marginal", "Omega_m0_marginal"),
        ("$w_0$--$w_a$ direct HPD", "HPD"), ("$w_0$--$w_a$ covariance ellipse", "ellipse"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, level, nominal in zip(axes, (68, 95), (P68, P95)):
        counts = []
        for _, prefix in definitions:
            field = (f"{prefix}{level}_contains_generating_point" if prefix in {"HPD", "ellipse"}
                     else f"{prefix}{level}_contains_generating_value")
            counts.append(sum(boolean(row[field]) for row in data))
        fraction = np.array(counts) / len(data)
        intervals = np.array([exact_interval(count, len(data)) for count in counts])
        y = np.arange(len(definitions))
        ax.errorbar(fraction, y,
                    xerr=np.vstack((fraction - intervals[:, 0], intervals[:, 1] - fraction)),
                    fmt="o", color="#277da1", capsize=3)
        ax.axvline(nominal, color="#d62828", ls="--")
        ax.set(title=f"Nominal {level}.{'27' if level == 68 else '45'}% regions",
               xlabel="coverage fraction", xlim=(.44, 1.08))
        ax.set_yticks(y, [label for label, _ in definitions]); ax.invert_yaxis()
        for yi, count, upper in zip(y, counts, intervals[:, 1]):
            ax.text(upper + .012, yi, f"{count}/100", va="center", fontsize=7)
    fig.tight_layout(pad=1.1); save(fig, out / "Fig3.pdf")


def lr_survival(out: Path) -> None:
    values = np.sort(np.array([float(row["Delta_chi2_LR"]) for row in rows("lr_statistics.csv")]))
    survival = np.arange(len(values), 0, -1) / len(values)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.step(values, survival, where="post", color="#277da1", lw=1.8,
            label=r"empirical survival ($N=100$)")
    x = np.linspace(0, values.max() * 1.04, 500)
    ax.plot(x, chi2.sf(x, 2), "k--", lw=1.2, label=r"asymptotic $\chi^2_2$ reference")
    for threshold, label in ((2.30, "68.27%"), (6.18, "95.45%"), (9.21, "99%")):
        ax.axvline(threshold, color="#f8961e", ls=":", lw=1)
        ax.text(threshold + .10, .55, label, rotation=90, va="center", color="#9c5a00", fontsize=8)
    ax.plot(values[-1], .01, "o", color="#d62828", ms=5)
    ax.annotate(r"N100-65: $\Delta\chi^2_{\rm LR}=16.411$", xy=(values[-1], .01),
                xytext=(10, .022), arrowprops={"arrowstyle": "->", "lw": .8}, fontsize=8)
    ax.axhline(.01, color=".45", ls="-.", lw=.8, label=r"finite-ensemble resolution $1/N=0.01$")
    ax.set_yscale("log"); ax.set_xlim(0, values.max() * 1.08); ax.set_ylim(.0075, 1.08)
    ax.set_xlabel(r"$\Delta\chi^2_{\rm LR}$"); ax.set_ylabel(r"$P(\Delta\chi^2_{\rm LR}\geq x)$")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); save(fig, out / "Fig4.pdf")


def probes(out: Path) -> None:
    data = sorted(rows("probe_contributions.csv"), key=lambda row: float(row["total_Delta_chi2_LR"]), reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(8, 4)); bottom = np.zeros(10)
    for field, label, hatch in (("Delta_chi2_BAO_LR", "BAO", "-"),
                                ("Delta_chi2_CMB_LR", "CMB", "|"),
                                ("Delta_chi2_SN_LR", "SNe", "/")):
        values = np.array([float(row[field]) for row in data])
        ax.bar(range(10), values, bottom=bottom, label=label, hatch=hatch,
               edgecolor="black", linewidth=.35); bottom += values
    ax.set_xticks(range(10), [row["realization"] for row in data], rotation=25, ha="right")
    ax.set_ylabel(r"$\Delta\chi^2_{p,{\rm LR}}$"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, out / "Fig5.pdf")


def hpd_grid(xy: np.ndarray):
    lo, hi = np.quantile(xy, .002, axis=0), np.quantile(xy, .998, axis=0)
    padding = .08 * (hi - lo); lo, hi = lo - padding, hi + padding
    histogram, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=100,
                                       range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    density = gaussian_filter(histogram, 1.15)
    positive = np.sort(density[density > 0])[::-1]
    cumulative = np.cumsum(positive) / positive.sum()
    levels = [positive[min(np.searchsorted(cumulative, probability), len(positive) - 1)]
              for probability in (P95, P68)]
    return density.T, (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, sorted(levels)


def ellipse(ax, xy: np.ndarray, probability: float) -> None:
    center, covariance = xy.mean(axis=0), np.cov(xy, rowvar=False, ddof=1)
    values, vectors = np.linalg.eigh(covariance); angle = np.linspace(0, 2 * np.pi, 300)
    curve = center[:, None] + vectors @ (
        np.sqrt(values * chi2.ppf(probability, 2))[:, None] *
        np.vstack((np.cos(angle), np.sin(angle))))
    ax.plot(curve[0], curve[1], color="#f8961e", ls="--", lw=.9)


def representative(out: Path, plane: str) -> None:
    samples = np.load(DATA / "representative_samples.npz")
    metadata = json.loads((DATA / "representative_samples_metadata.json").read_text())
    points = {int(row["realization_id"]): row for row in rows("representative_points.csv")}
    ids, order = metadata["representative_order"], metadata["parameter_order"]
    roles = {41: "closest to generating cosmology", 80: r"median $w_0$--$w_a$ displacement",
             65: r"largest $\Delta\chi^2_{\rm LR}$", 98: r"largest normalized $H_0$--$\Omega_{m0}$ displacement"}
    if plane == "w0wa":
        a, b, filename = "w0", "wa", "Fig6.pdf"
        xlabel, ylabel = r"$w_0$", r"$w_a$"
    else:
        a, b, filename = "H0", "Omega_m0", "Fig8.pdf"
        xlabel, ylabel = r"$H_0\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$", r"$\Omega_{m0}$"
    ia, ib = order.index(a), order.index(b)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.8)); axes = axes.ravel()
    for ax, realization_id in zip(axes, ids):
        xy = samples[f"N100_{realization_id}_samples"][:, [ia, ib]]
        density, xx, yy, levels = hpd_grid(xy)
        ax.contour(xx, yy, density, levels=levels, colors="#277da1", linewidths=(1, 1.8))
        ellipse(ax, xy, P68); ellipse(ax, xy, P95)
        row = points[realization_id]
        ax.plot(FID[a], FID[b], "*", color="black", ms=8, label="generating point")
        ax.plot(float(row[f"{a}_mean"]), float(row[f"{b}_mean"]), "o", color="#d62828", ms=3.8,
                label="posterior mean")
        ax.plot(float(row[f"{a}_median"]), float(row[f"{b}_median"]), "s", color="#6a4c93", ms=3.5,
                label="posterior median")
        ax.plot(float(row[f"cpl_best_{a}"]), float(row[f"cpl_best_{b}"]), "x", color="#111", ms=5,
                label="CPL optimum")
        if plane == "w0wa":
            line = np.linspace(*ax.get_xlim(), 100); ax.plot(line, -line, ":", color=".35", lw=.7)
        ax.set_title(f"N100-{realization_id}\n{roles[realization_id]}", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5, .99), frameon=False, fontsize=8)
    fig.tight_layout(rect=(.01, .02, 1, .91), h_pad=1.4, w_pad=1.1); save(fig, out / filename)


def weak_modes(out: Path) -> None:
    data = rows("weak_mode_summary.csv")
    x = np.array([float(row["d_short"]) for row in data]); y = np.array([float(row["d_long"]) for row in data])
    lr = np.array([float(row["Delta_chi2_LR"]) for row in data])
    fig, ax = plt.subplots(figsize=(6.7, 5.1))
    scatter = ax.scatter(x, y, c=lr, cmap="viridis", s=27)
    ax.axhline(0, color="k", lw=.55); ax.axvline(0, color="k", lw=.55)
    row65 = next(i for i, row in enumerate(data) if row["realization"] == "N100-65")
    ax.annotate("N100-65", (x[row65], y[row65]), xytext=(-8, -13), textcoords="offset points",
                fontsize=8, ha="right")
    ax.set_xlabel(r"$d_{\rm short}$ (normalized strong displacement)")
    ax.set_ylabel(r"$d_{\rm long}$ (normalized weak displacement)")
    fig.colorbar(scatter, ax=ax, pad=.025, fraction=.05, label=r"$\Delta\chi^2_{\rm LR}$")
    fig.tight_layout(); save(fig, out / "Fig7.pdf")


GENERATORS = {1: centers, 2: correlation, 3: coverage, 4: lr_survival, 5: probes,
              6: lambda out: representative(out, "w0wa"), 7: weak_modes,
              8: lambda out: representative(out, "h0om")}


def check(directory: Path) -> None:
    coverage_data = rows("coverage_results.csv")
    excluded = [row["realization"] for row in coverage_data if not boolean(row["HPD95_contains_generating_point"])]
    assert excluded == ["N100-21", "N100-61", "N100-65", "N100-87"]
    assert sum(boolean(row["HPD68_contains_generating_point"]) for row in coverage_data) == 64
    lr = rows("lr_statistics.csv"); values = np.array([float(row["Delta_chi2_LR"]) for row in lr])
    assert lr[int(np.argmax(values))]["realization"] == "N100-65"
    np.testing.assert_allclose(values.max(), 16.411, atol=.0005)
    np.testing.assert_allclose(np.quantile(values, .95), 5.529, atol=.0005)
    for names in ASSETS.values():
        for name in names:
            path = directory / name
            assert path.is_file() and path.stat().st_size > 1000 and path.read_bytes()[:5] == b"%PDF-"
    print(f"CHECK PASSED: numerical landmarks and nine PDFs in {directory}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--figure", type=int, choices=range(1, 9))
    group.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); output = args.output_dir.resolve()
    if args.check:
        check(output); return
    selected = range(1, 9) if args.all else (args.figure,)
    for number in selected:
        print(f"FIGURE {number}"); GENERATORS[number](output)


if __name__ == "__main__":
    main()
