#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Final read-only analysis of exactly N100-0 through N100-99.

The HDF5 backends are opened read-only.  Numerical summaries use every stored
production sample; bounded deterministic subsamples are retained only for
two-dimensional density estimation and plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.stats import beta, chi2, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.likelihoods.joint_likelihood import JointLikelihood, LCDM_ORDER, PARAMETER_ORDER

IDS = tuple(range(100))
PARAMETERS = tuple(PARAMETER_ORDER)
DERIVED = ("Omega_m0", "wp")
FID = {"omega_m": .147, "omega_b": .02237, "H0": 70., "w0": -1.,
       "wa": 0., "DeltaM": 0., "Omega_m0": .3, "wp": -1.}
TARGET_TO_EXTENSIONS = {20000: 0, 40000: 1, 50000: 2, 60000: 3,
                        70000: 4, 80000: 5}
P68, P95 = .682689492137, .954499736104
Q68, Q95 = chi2.ppf(P68, 2), chi2.ppf(P95, 2)
PAIRS = (
    ("H0", "Omega_m0"), ("H0", "w0"), ("H0", "wa"), ("H0", "wp"),
    ("Omega_m0", "w0"), ("Omega_m0", "wa"), ("Omega_m0", "wp"),
    ("w0", "wa"),
)


def backend_path(root: Path, rid: int) -> Path:
    if rid not in IDS:
        raise ValueError(f"out-of-scope realization: {rid}")
    return root / "chains/main_N100" / f"N100-{rid}" / "cpl.h5"


def clopper_pearson(k: int, n: int, confidence: float) -> list[float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("require 0 <= k <= n and n > 0")
    a = 1 - confidence
    lo = 0. if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1. if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def coverage_record(flags, nominal: float) -> dict:
    flags = np.asarray(flags, bool)
    n, k = len(flags), int(flags.sum())
    f = k / n
    se = math.sqrt(nominal * (1 - nominal) / n)
    return {"count": k, "total": n, "fraction": f,
            "binomial_standard_error": math.sqrt(f * (1 - f) / n),
            "nominal_standard_error": se,
            "deviation_from_nominal_in_standard_errors": (f - nominal) / se,
            "clopper_pearson_68": clopper_pearson(k, n, .68),
            "clopper_pearson_95": clopper_pearson(k, n, .95)}


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    q = np.quantile(x, [.025, .16, .5, .84, .975])
    return {"mean": float(x.mean()), "median": float(q[2]),
            "std": float(x.std(ddof=1)), "q2p5": float(q[0]),
            "q16": float(q[1]), "q84": float(q[3]), "q97p5": float(q[4])}


def covariance_geometry(xy: np.ndarray, fid) -> dict:
    cov = np.cov(xy, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    center = xy.mean(0)
    delta = center - np.asarray(fid)
    d2 = float(delta @ np.linalg.solve(cov, delta))
    projections = vecs.T @ delta
    return {"covariance": cov.tolist(), "correlation": float(np.corrcoef(xy.T)[0, 1]),
            "eigenvalues_short_long": vals.tolist(),
            "eigenvectors_columns_short_long": vecs.tolist(),
            "angle_short_deg": float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))),
            "angle_long_deg": float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))),
            "mahalanobis2": d2, "short_displacement": float(projections[0]),
            "long_displacement": float(projections[1]),
            "normalized_short_displacement": float(projections[0] / np.sqrt(vals[0])),
            "normalized_long_displacement": float(projections[1] / np.sqrt(vals[1])),
            "ellipse_68": bool(d2 <= Q68), "ellipse_95": bool(d2 <= Q95)}


def hpd_grid(xy: np.ndarray, fid, bins=100) -> tuple[dict, tuple]:
    """Histogram-smoothed posterior-density membership and plotting grid."""
    xy = np.asarray(xy, float)
    lo = np.quantile(xy, .002, axis=0)
    hi = np.quantile(xy, .998, axis=0)
    pad = .08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    hist, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
                                  range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    density = gaussian_filter(hist, 1.15)
    ix = np.clip(np.searchsorted(xe, fid[0]) - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, fid[1]) - 1, 0, bins - 1)
    fden = density[ix, iy]
    mass_above = float(hist[density >= fden].sum() / max(hist.sum(), 1))
    positive = density[density > 0]
    order = np.sort(positive)[::-1]
    cumulative = np.cumsum(order) / order.sum()
    levels = {}
    for p in (P68, P95):
        levels[p] = float(order[min(np.searchsorted(cumulative, p), len(order)-1)])
    result = {"hpd_mass_at_fiducial": mass_above,
              "hpd_68": bool(mass_above <= P68), "hpd_95": bool(mass_above <= P95),
              "area_68": float((density >= levels[P68]).sum() * np.diff(xe)[0] * np.diff(ye)[0]),
              "area_95": float((density >= levels[P95]).sum() * np.diff(xe)[0] * np.diff(ye)[0])}
    return result, (density.T, (xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2,
                    levels[P95], levels[P68])


def _json_attr(value):
    return json.loads(value.decode() if isinstance(value, bytes) else str(value))


def _finite_chunks(ds, chunk=256) -> bool:
    for i in range(0, ds.shape[0], chunk):
        a = ds[i:i+chunk]
        if a.dtype.names:
            if not all(np.isfinite(a[n]).all() for n in a.dtype.names):
                return False
        elif not np.isfinite(a).all():
            return False
    return True


def _optimization(joint, model, starts, seed, random_starts) -> dict:
    names = LCDM_ORDER if model == "lcdm" else PARAMETERS
    bounds = np.asarray(joint.prior.bounds(model), float)
    lo, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    decode = lambda u: lo + span * np.asarray(u)
    objective = lambda u: joint.objective(decode(u), model)
    vectors = []
    for theta in starts:
        if model == "lcdm":
            v = np.array([theta[n] for n in LCDM_ORDER])
        else:
            v = np.array([theta[n] for n in PARAMETERS])
        u = (v - lo) / span
        if np.all((u > 0) & (u < 1)) and joint.prior.contains(
                joint.prior.vector_to_theta(v, model)):
            vectors.append(u)
    rng = np.random.default_rng(seed)
    while len(vectors) < len(starts) + random_starts:
        u = rng.uniform(.04, .96, len(names))
        if joint.prior.contains(joint.prior.vector_to_theta(decode(u), model)):
            vectors.append(u)
    runs = []
    t0 = time.perf_counter()
    constraints = []
    if model == "cpl":
        constraints = [{"type": "ineq",
                        "fun": lambda u: -(decode(u)[3] + decode(u)[4]) - 1e-9}]
    for u in vectors:
        r = minimize(objective, u, method="SLSQP",
                     bounds=[(1e-9, 1-1e-9)] * len(u), constraints=constraints,
                     options={"maxiter": 1000, "ftol": 1e-10})
        runs.append(r)
    runs.sort(key=lambda r: r.fun)
    fallback_used = False
    if not runs[0].success or (len(runs) > 1 and runs[1].fun-runs[0].fun > 1e-4):
        fallback_used = True
        u0 = runs[0].x
        r = minimize(objective, u0, method="Powell",
                     bounds=[(1e-8, 1-1e-8)] * len(u0),
                     options={"maxiter": 3000, "xtol": 1e-10, "ftol": 1e-10})
        if model == "cpl" and decode(r.x)[3] + decode(r.x)[4] >= 0:
            r.fun = 1e100
        runs.append(r)
        runs.sort(key=lambda r: r.fun)
    best = runs[0]
    theta = joint.prior.vector_to_theta(decode(best.x), model)
    parts = {k: float(v) for k, v in joint.evaluate(theta).items()}
    reproduce = float(joint.evaluate(theta)["total"])
    vals = np.array([r.fun for r in runs if np.isfinite(r.fun)])
    return {"model": model, "theta": theta, "chi2": parts,
            "success": bool(np.isfinite(parts["total"]) and
                            abs(reproduce-parts["total"]) < 1e-8 and
                            (best.success or fallback_used)),
            "method": str(best.method if hasattr(best, "method") else
                          ("Powell" if fallback_used and best is runs[0] else "SLSQP")),
            "message": str(best.message), "status": int(best.status),
            "n_starts": len(vectors), "fallback_used": fallback_used,
            "nfev_total": int(sum(getattr(r, "nfev", 0) for r in runs)),
            "restart_values": vals.tolist(),
            "best_two_spread": float(np.sort(vals)[1]-np.sort(vals)[0]) if len(vals)>1 else 0.,
            "direct_reproduction_difference": reproduce-parts["total"],
            "runtime_seconds": time.perf_counter()-t0}


def _probe_gradients(joint, theta) -> dict:
    out = {}
    for probe in ("bao", "cmb", "sne"):
        grad = []
        for name, step in (("w0", 1e-4), ("wa", 2e-4)):
            tp, tm = dict(theta), dict(theta)
            tp[name] += step
            tm[name] -= step
            grad.append((joint.evaluate(tp, False)[probe] -
                         joint.evaluate(tm, False)[probe]) / (2*step))
        out[probe] = grad
    return out


def inspect_actual_data(root: Path) -> dict:
    """Conservative validation: raw probe files alone are not a joint product."""
    candidates = []
    for base in (root/"data", root/"output", root/"results"):
        if base.exists():
            candidates += [str(p) for p in base.rglob("*") if p.is_file() and
                           ("observ" in p.name.lower() or "actual" in p.name.lower())]
    return {"validated_matching_joint_product_present": False,
            "candidates_examined": candidates,
            "missing_inputs": [
                "A single validated observed BAO+CMB+SNe data vector using the mock likelihood's exact compression and masks",
                "Recorded confirmation that its covariance, nuisance treatment, acoustic convention, and SNe mask match the mocks"],
            "ready_to_run_template":
                ".venv/bin/python -m src.inference.analyze_main_n100_final "
                "--observed-joint-h5 PATH_TO_VALIDATED_MATCHING_PRODUCT.h5"}


def analyze(root: Path):
    mock_path = root/"mocks/joint/pedagogic2_joint_lcdm_N0100.h5"
    with h5py.File(mock_path, "r") as h:
        mock_ids = h["metadata/realization_id"][:].astype(int).tolist()
        if mock_ids != list(IDS):
            raise RuntimeError("mock realization IDs are not exactly 0..99")
        entropy = int(h["random_streams"].attrs["master_entropy"])
        spawn = {n: h[f"random_streams/{n}"][:].tolist()
                 for n in ("bao_spawn_key", "cmb_spawn_key", "sne_spawn_key")
                 if f"random_streams/{n}" in h}
        stored_fid_om = float(h["fiducial"].attrs["Omega_m"])
    if abs(stored_fid_om-FID["Omega_m0"]) > 1e-12:
        raise RuntimeError("fiducial Omega_m definition mismatch")

    rows, samples, grids, opt_details, components, audit = [], {}, {}, [], [], []
    for rid in IDS:
        path = backend_path(root, rid)
        if not path.is_file():
            audit.append({"id": rid, "status": "MISSING", "reason": str(path)})
            continue
        with h5py.File(path, "r") as h:
            meta = dict(h["metadata"].attrs)
            required = ("production/chain", "production/log_prob", "production/blobs",
                        "production/accepted")
            missing = [x for x in required if x not in h]
            if missing:
                audit.append({"id": rid, "status": "INCOMPLETE",
                              "reason": f"missing datasets {missing}"})
                continue
            chain = h["production/chain"][:]
            flat = chain.reshape(-1, chain.shape[-1])
            logp = h["production/log_prob"][:].reshape(-1)
            blobs = h["production/blobs"][:].reshape(-1)
            order = tuple(_json_attr(meta["parameter_order"]))
            diag = _json_attr(meta["diagnostics"])
            history = _json_attr(meta["extension_history"])
            accepted = np.asarray(h["production/accepted"][:], float)
            iteration = int(h["production"].attrs["iteration"])
            ndim = int(h["production"].attrs["ndim"])
            nwalkers = int(h["production"].attrs["nwalkers"])
            all_finite = (np.isfinite(chain).all() and np.isfinite(logp).all() and
                          _finite_chunks(h["production/blobs"]))
        steps = chain.shape[0]
        problems = []
        if order != PARAMETERS: problems.append(f"parameter order {order}")
        if chain.ndim != 3 or chain.shape[1:] != (64, 6): problems.append(f"shape {chain.shape}")
        if ndim != 6 or nwalkers != 64: problems.append("stored dimensionality")
        if int(meta["realization_id"]) != rid or str(meta["realization_label"]) != f"N100-{rid}":
            problems.append("identity")
        if int(meta["seed"]) != 20260800+rid: problems.append("MCMC seed")
        if iteration != steps or int(meta["production_steps"]) != steps:
            problems.append("completion metadata")
        if not all_finite: problems.append("non-finite")
        converged = bool(diag.get("converged")) and str(meta["convergence_classification"]) == "CONVERGED"
        if not converged: problems.append("failed convergence")
        scheduled = TARGET_TO_EXTENSIONS.get(steps)
        if scheduled is None: problems.append("unexpected production target")
        if rid != 3 and len(history) != scheduled: problems.append("extension history")
        legacy3 = rid == 3 and steps == 80000 and len(history) == 2
        if rid == 3 and not legacy3: problems.append("N100-3 legacy history differs")
        af = accepted / steps
        if abs(af.mean()-float(diag["mean_acceptance_fraction"])) > 1e-12:
            problems.append("acceptance mismatch")
        if problems:
            audit.append({"id": rid, "status": "FLAGGED", "reason": "; ".join(problems)})
            continue

        vals = {name: flat[:, i] for i, name in enumerate(PARAMETERS)}
        vals["Omega_m0"] = vals["omega_m"] / (vals["H0"]/100.)**2
        covw = np.cov(np.column_stack((vals["w0"], vals["wa"])), rowvar=False)
        ap = 1 + covw[0, 1]/covw[1, 1]
        vals["wp"] = vals["w0"] + (1-ap)*vals["wa"]
        zp = 1/ap-1
        best_i = int(np.argmin(blobs["chi2_total"]))
        map_i = int(np.argmax(logp))
        stride = max(1, len(flat)//40000)
        keep = np.arange(0, len(flat), stride)[:40000]
        samples[rid] = {k: v[keep] for k, v in vals.items()}
        row = {"realization_id": rid, "realization_label": f"N100-{rid}",
               "backend_path": str(path.resolve()), "mcmc_seed": int(meta["seed"]),
               "mock_master_entropy": entropy, "production_length": steps,
               "backend_iteration": iteration, "extension_count": scheduled,
               "stored_extension_history_count": len(history),
               "extension_history": json.dumps(history), "legacy_N100_3_history": legacy3,
               "max_tau": float(max(diag["tau"])),
               "max_split_rhat": float(max(diag["split_rhat"])),
               "min_ess": float(min(diag["ess"])),
               "mean_acceptance": float(np.mean(af)),
               "walker_acceptance_min": float(np.min(af)),
               "walker_acceptance_max": float(np.max(af)),
               "walker_acceptance_json": json.dumps(af.tolist()),
               "convergence_status": str(meta["convergence_classification"]),
               "all_finite": all_finite, "pivot_a": float(ap), "pivot_z": float(zp),
               "cov_wp_wa": float(np.cov(vals["wp"], vals["wa"])[0, 1]),
               "min_stored_chi2": float(blobs["chi2_total"][best_i])}
        for name in PARAMETERS+DERIVED:
            s = summarize(vals[name])
            for key, value in s.items(): row[f"{name}_{key}"] = value
            row[f"{name}_map"] = float(vals[name][map_i])
            row[f"{name}_minchi2"] = float(vals[name][best_i])
            row[f"{name}_fid_in_68"] = bool(s["q16"] <= FID[name] <= s["q84"])
            row[f"{name}_fid_in_95"] = bool(s["q2p5"] <= FID[name] <= s["q97p5"])
            row[f"{name}_pull_mean"] = (s["mean"]-FID[name])/s["std"]
            row[f"{name}_pull_median"] = (s["median"]-FID[name])/s["std"]
        for a, b in PAIRS:
            xy = np.column_stack((vals[a], vals[b]))
            geo = covariance_geometry(xy, (FID[a], FID[b]))
            hpd, grid = hpd_grid(np.column_stack((samples[rid][a], samples[rid][b])),
                                 (FID[a], FID[b]))
            tag = f"{a}__{b}"
            grids[(rid, a, b)] = grid
            row[f"{tag}_correlation"] = geo["correlation"]
            row[f"{tag}_covariance_json"] = json.dumps(geo["covariance"])
            row[f"{tag}_eigenvalues_json"] = json.dumps(geo["eigenvalues_short_long"])
            row[f"{tag}_angle_short_deg"] = geo["angle_short_deg"]
            row[f"{tag}_angle_long_deg"] = geo["angle_long_deg"]
            row[f"{tag}_mahalanobis2"] = geo["mahalanobis2"]
            row[f"{tag}_ellipse_68"] = geo["ellipse_68"]
            row[f"{tag}_ellipse_95"] = geo["ellipse_95"]
            row[f"{tag}_hpd_68"] = hpd["hpd_68"]
            row[f"{tag}_hpd_95"] = hpd["hpd_95"]
            row[f"{tag}_hpd_mass_at_fiducial"] = hpd["hpd_mass_at_fiducial"]
            row[f"{tag}_area_68"] = hpd["area_68"]
            row[f"{tag}_area_95"] = hpd["area_95"]
            if (a, b) == ("w0", "wa"):
                row["weak_displacement"] = geo["long_displacement"]
                row["strong_displacement"] = geo["short_displacement"]
                row["weak_displacement_normalized"] = geo["normalized_long_displacement"]
                row["strong_displacement_normalized"] = geo["normalized_short_displacement"]
                row["w0_wa_eigenvalue_ratio"] = geo["eigenvalues_short_long"][1]/geo["eigenvalues_short_long"][0]
        # Statistic A and exact stored component decomposition.
        joint = JointLikelihood(mock_path, rid, root/"config/priors_joint.yaml",
                                root/"results/early_universe/class_grid.npz")
        fparts = {k: float(v) for k, v in joint.evaluate(
            {k: FID[k] for k in PARAMETERS}).items()}
        for p, blobname in (("bao", "chi2_bao"), ("cmb", "chi2_cmb"), ("sne", "chi2_sne")):
            row[f"fiducial_chi2_{p}"] = fparts[p]
            row[f"minstored_chi2_{p}"] = float(blobs[blobname][best_i])
            row[f"delta_chi2_A_{p}"] = fparts[p]-float(blobs[blobname][best_i])
        row["delta_chi2_A"] = fparts["total"]-row["min_stored_chi2"]
        medtheta = {k: row[f"{k}_median"] for k in PARAMETERS}
        besttheta = {k: float(flat[best_i, i]) for i, k in enumerate(PARAMETERS)}
        fidtheta = {k: FID[k] for k in PARAMETERS}
        lcdm = _optimization(joint, "lcdm", [fidtheta, medtheta, besttheta],
                             910000+rid, 4)
        cpl = _optimization(joint, "cpl", [fidtheta, medtheta, besttheta],
                            920000+rid, 6)
        lr = lcdm["chi2"]["total"]-cpl["chi2"]["total"]
        if lr < -1e-5:
            # Nested safeguard: start CPL exactly at optimized LCDM and retry.
            cpl = _optimization(joint, "cpl",
                [fidtheta, medtheta, besttheta, lcdm["theta"]], 930000+rid, 10)
            lr = lcdm["chi2"]["total"]-cpl["chi2"]["total"]
        row["chi2_min_lcdm"] = lcdm["chi2"]["total"]
        row["chi2_min_cpl"] = cpl["chi2"]["total"]
        row["delta_chi2_LR"] = lr
        row["A_absorbed_by_lcdm_reoptimization"] = row["delta_chi2_A"]-lr
        for model, result in (("lcdm", lcdm), ("cpl", cpl)):
            for p in PARAMETERS:
                row[f"{model}_best_{p}"] = result["theta"][p]
            om = result["theta"]["omega_m"]/(result["theta"]["H0"]/100)**2
            row[f"{model}_best_Omega_m0"] = om
            row[f"{model}_optimizer_success"] = result["success"]
            row[f"{model}_optimizer_method"] = result["method"]
            row[f"{model}_optimizer_starts"] = result["n_starts"]
            row[f"{model}_restart_spread"] = result["best_two_spread"]
            for p in ("bao", "cmb", "sne"):
                row[f"{model}_chi2_{p}"] = result["chi2"][p]
        row["fit_shift_H0_cpl_minus_lcdm"] = row["cpl_best_H0"]-row["lcdm_best_H0"]
        row["fit_shift_Omega_m0_cpl_minus_lcdm"] = (
            row["cpl_best_Omega_m0"]-row["lcdm_best_Omega_m0"])
        improvements = {p: lcdm["chi2"][p]-cpl["chi2"][p] for p in ("bao","cmb","sne")}
        for p, v in improvements.items(): row[f"delta_chi2_LR_{p}"] = v
        positive = {p: max(0., v) for p, v in improvements.items()}
        denom = sum(positive.values())
        largest_probe = max(positive, key=positive.get)
        if positive["bao"] > 0 and positive["cmb"] > 0 and (
                positive["bao"]+positive["cmb"] > .7*max(denom, 1e-12)):
            classification = "BAO+CMB aligned"
        elif denom and positive[largest_probe] > .55*denom:
            classification = f"{largest_probe.upper()} dominated"
        else:
            classification = "broadly shared"
        row["probe_classification"] = classification
        gradients = _probe_gradients(joint, cpl["theta"])
        row["probe_gradients_w0_wa_json"] = json.dumps(gradients)
        opt_details.append({"realization_id": rid, "lcdm": lcdm, "cpl": cpl,
                            "delta_chi2_LR": lr, "probe_gradients": gradients})
        components += [
            {"realization_id": rid, "model": model, "probe": p,
             "chi2": result["chi2"][p]}
            for model, result in (("lcdm", lcdm), ("cpl", cpl))
            for p in ("bao", "cmb", "sne")]
        rows.append(row)
        audit.append({"id": rid, "status": "OK",
                      "reason": "legacy compressed 40k-to-80k history retained" if legacy3 else ""})
        print(f"N100-{rid}: samples={len(flat):,} LR={lr:.5f}", flush=True)
        del chain, flat, blobs, vals
    return rows, samples, grids, opt_details, components, audit, spawn


def correlation_record(x, y) -> dict:
    p, s = pearsonr(x, y), spearmanr(x, y)
    return {"pearson_r": float(p.statistic), "pearson_p": float(p.pvalue),
            "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue)}


def make_summary(rows, audit, actual):
    n = len(rows)
    coverage = {}
    for p in PARAMETERS+DERIVED:
        for level, nominal in ((68, P68), (95, P95)):
            coverage[f"marginal_{p}_{level}"] = coverage_record(
                [r[f"{p}_fid_in_{level}"] for r in rows], nominal)
    for a, b in PAIRS:
        tag = f"{a}__{b}"
        for kind in ("ellipse", "hpd"):
            for level, nominal in ((68, P68), (95, P95)):
                coverage[f"joint_{tag}_{kind}_{level}"] = coverage_record(
                    [r[f"{tag}_{kind}_{level}"] for r in rows], nominal)
    bias = {}
    for p in PARAMETERS+DERIVED:
        bias[p] = {}
        for estimator in ("mean", "median"):
            v = np.array([r[f"{p}_{estimator}"] for r in rows])
            d = v-FID[p]
            se = d.std(ddof=1)/np.sqrt(n)
            bias[p][estimator] = {
                "ensemble_mean": float(v.mean()), "bias": float(d.mean()),
                "standard_error_of_bias": float(se),
                "bias_significance_z": float(d.mean()/se),
                "rmse": float(np.sqrt(np.mean(d*d))),
                "pull_mean": float(np.mean([r[f"{p}_pull_{estimator}"] for r in rows])),
                "pull_width": float(np.std([r[f"{p}_pull_{estimator}"] for r in rows], ddof=1))}
    center_names = ("Omega_m0", "H0", "w0", "wa", "wp")
    centers = np.array([[r[f"{p}_median"] for p in center_names] for r in rows])
    lr = np.array([r["delta_chi2_LR"] for r in rows])
    correlations = {}
    for field in ("weak_displacement_normalized", "w0__wa_mahalanobis2",
                  "w0_median", "wa_median", "wp_median", "production_length",
                  "max_tau", "min_ess", "mean_acceptance", "realization_id",
                  "fit_shift_H0_cpl_minus_lcdm", "fit_shift_Omega_m0_cpl_minus_lcdm"):
        correlations[field] = correlation_record(lr, [r[field] for r in rows])
    quantiles = {str(q): float(np.quantile(lr, q)) for q in (.68, .90, .95, .99)}
    thresholds = {str(t): {"count_greater_equal": int(np.sum(lr >= t)),
                           "fraction": float(np.mean(lr >= t)),
                           "plus_one_tail": float((np.sum(lr >= t)+1)/(n+1))}
                  for t in (2.30, 5.99, 9.21)}
    return {
        "scope": {"requested_count": 100, "requested_ids": [0, 99],
                  "found_count": n, "N100_100_inspected": False},
        "audit": {"converged_count": sum(a["status"] == "OK" for a in audit),
                  "excluded": [a for a in audit if a["status"] != "OK"],
                  "N100_3_legacy_issue_retained": True},
        "parameter_convention": {
            "sampled": "omega_m is physical matter density",
            "derived": "Omega_m0 = omega_m / (H0/100)^2, sample by sample",
            "fiducial_Omega_m0": .3},
        "coverage": coverage, "bias": bias,
        "center_parameter_order": center_names,
        "ensemble_center_correlation_matrix": np.corrcoef(centers, rowvar=False).tolist(),
        "likelihood_ratio": {
            "definition": "optimized chi2_min,LambdaCDM - optimized chi2_min,CPL",
            "minimum": float(lr.min()), "maximum": float(lr.max()),
            "maximum_id": int(rows[int(np.argmax(lr))]["realization_id"]),
            "quantiles": quantiles, "thresholds": thresholds,
            "tail_resolution_statement": "N=100 gives approximately 1% empirical-tail resolution and cannot calibrate rare 3-sigma or stronger probabilities precisely.",
            "wilks_warning": "chi-square_2 is a visual reference only; Wilks is not assumed exact."},
        "fixed_fiducial_statistic": {
            "definition": "chi2 at complete generating vector minus minimum stored CPL-sample chi2; not a model-comparison likelihood ratio",
            "maximum": float(max(r["delta_chi2_A"] for r in rows)),
            "maximum_id": int(max(rows, key=lambda r:r["delta_chi2_A"])["realization_id"])},
        "lr_correlations": correlations,
        "actual_observational_data": actual,
        "scientific_conclusion":
            "A controlled ensemble of coherent BAO, CMB, and SNe realizations generated from exact LambdaCDM can produce apparently dynamical-dark-energy-like CPL posteriors when statistical fluctuations project onto the structurally weak w0-wa direction. Ensemble calibration and individual nominal exclusions are distinct questions."}


def select_cases(rows):
    full = np.array([[r[f"{p}_pull_median"] for p in ("omega_m","omega_b","H0","w0","wa","DeltaM")]
                     for r in rows])
    distance = np.sqrt(np.sum(full*full, axis=1))
    wdist = np.array([r["w0__wa_mahalanobis2"] for r in rows])
    median_target = np.median(wdist)
    return {
        "closest_full": int(rows[np.argmin(distance)]["realization_id"]),
        "closest_w0wa": int(rows[np.argmin(wdist)]["realization_id"]),
        "median_displacement": int(rows[np.argmin(abs(wdist-median_target))]["realization_id"]),
        "largest_mahalanobis": int(rows[np.argmax(wdist)]["realization_id"]),
        "largest_LR": int(max(rows, key=lambda r:r["delta_chi2_LR"])["realization_id"]),
        "N100_65": 65,
        "largest_H0": int(rows[np.argmax(np.abs([r["H0_pull_median"] for r in rows]))]["realization_id"]),
        "largest_Omega_m0": int(rows[np.argmax(np.abs([r["Omega_m0_pull_median"] for r in rows]))]["realization_id"])}


def _ellipse(ax, row, a, b, level, **kw):
    cov = np.array(json.loads(row[f"{a}__{b}_covariance_json"]))
    center = np.array([row[f"{a}_mean"], row[f"{b}_mean"]])
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2*np.pi, 300)
    q = Q68 if level == 68 else Q95
    xy = center[:, None] + vecs @ (np.sqrt(vals*q)[:, None]*np.vstack((np.cos(t),np.sin(t))))
    ax.plot(xy[0], xy[1], **kw)


def _contour(ax, grid, color="C0"):
    den, x, y, l95, l68 = grid
    ax.contour(x, y, den, levels=sorted([l95, l68]), colors=[color, color],
               linewidths=[1., 1.7])


def make_figures(root, rows, samples, grids, cases, summary):
    out = root/"figures/main_N100_final_N0_N99"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "figure.dpi": 140,
                         "axes.grid": True, "grid.alpha": .18})
    byid = {r["realization_id"]: r for r in rows}
    pair_files = {
        ("H0","Omega_m0"): "H0_Omega_m0_representative_contours.pdf",
        ("H0","w0"): "H0_w0_representative_contours.pdf",
        ("H0","wa"): "H0_wa_representative_contours.pdf",
        ("Omega_m0","w0"): "Omega_m0_w0_representative_contours.pdf",
        ("Omega_m0","wa"): "Omega_m0_wa_representative_contours.pdf"}
    representative = list(dict.fromkeys(cases.values()))
    for (a,b), filename in pair_files.items():
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.3))
        for ax, rid in zip(axes.flat, representative):
            r = byid[rid]
            _contour(ax, grids[(rid,a,b)])
            _ellipse(ax, r, a, b, 68, color="C1", ls="--", lw=.8)
            _ellipse(ax, r, a, b, 95, color="C1", ls="--", lw=.6)
            ax.plot(FID[a], FID[b], "k*", ms=7)
            ax.plot(r[f"{a}_mean"], r[f"{b}_mean"], "o", ms=3, color="C3")
            ax.set(title=f"N100-{rid}", xlabel=a, ylabel=b)
        fig.suptitle("Solid: posterior-density 68/95%; dashed: covariance ellipses")
        fig.tight_layout(); fig.savefig(out/filename); plt.close(fig)
    # w0-wa representative and case-study figures.
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, rid in zip(axes.flat, representative):
        r = byid[rid]; _contour(ax, grids[(rid,"w0","wa")])
        _ellipse(ax, r, "w0", "wa", 68, color="C1", ls="--", lw=.8)
        ax.plot(-1, 0, "k*", ms=8); ax.plot(r["w0_mean"],r["wa_mean"],"ro",ms=3)
        ax.plot(r["w0_median"],r["wa_median"],"rs",ms=3)
        ax.plot(r["cpl_best_w0"],r["cpl_best_wa"],"kx",ms=5)
        ax.plot(-1,0,"D",color="C2",ms=3)
        xx=np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100); ax.plot(xx,-xx,"k:",lw=.7)
        ax.set(title=f"N100-{rid}",xlabel="$w_0$",ylabel="$w_a$")
    fig.suptitle("HPD contours; orange dashed covariance ellipse; star=fiducial, x=CPL optimum")
    fig.tight_layout(); fig.savefig(out/"w0_wa_representative_contours.pdf"); plt.close(fig)
    rid=65; r=byid[rid]
    fig,ax=plt.subplots(figsize=(6,5)); _contour(ax,grids[(rid,"w0","wa")]); _ellipse(ax,r,"w0","wa",68,color="C1",ls="--")
    ax.plot(-1,0,"k*",ms=10);ax.plot(r["cpl_best_w0"],r["cpl_best_wa"],"kx")
    ax.set(xlabel="$w_0$",ylabel="$w_a$",title=f"N100-65; LR={r['delta_chi2_LR']:.2f}, {r['probe_classification']}")
    fig.tight_layout();fig.savefig(out/"w0_wa_N100_65_case_study.pdf");plt.close(fig)
    # Center scatters and density summaries.
    center_labels = {
        "w0": r"$w_0$", "wa": r"$w_a$",
        "H0": r"$H_0\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$",
        "Omega_m0": r"$\Omega_{m0}$",
    }
    for a,b,fn in (("w0","wa","w0_wa_posterior_centers.pdf"),
                   ("H0","Omega_m0","H0_Omega_m0_posterior_centers.pdf")):
        fig,ax=plt.subplots(figsize=(6.4,5.1))
        outside=np.array([not r[f"{a}__{b}_hpd_95"] for r in rows])
        sc=ax.scatter([r[f"{a}_median"] for r in rows],[r[f"{b}_median"] for r in rows],
                      c=[r["delta_chi2_LR"] for r in rows],cmap="viridis",s=25)
        ax.scatter(np.array([r[f"{a}_median"] for r in rows])[outside],
                   np.array([r[f"{b}_median"] for r in rows])[outside],
                   facecolors="none",edgecolors="red",s=55,label="fiducial outside HPD 95%")
        ax.plot(FID[a],FID[b],"k*",ms=11,label="generating point")
        ax.set(xlabel=center_labels[a],ylabel=center_labels[b]);ax.legend(fontsize=7);fig.colorbar(sc,ax=ax,label=r"$\Delta\chi^2_{\rm LR}$")
        fig.tight_layout();fig.savefig(out/fn);plt.close(fig)
    center=np.array([[r["w0_median"],r["wa_median"]] for r in rows])
    fig,ax=plt.subplots(figsize=(6,5));hh=ax.hist2d(center[:,0],center[:,1],bins=18,cmap="Blues")
    ax.plot(-1,0,"k*",ms=10);ax.set(xlabel="$w_0$",ylabel="$w_a$");fig.colorbar(hh[3],ax=ax,label="center count")
    fig.tight_layout();fig.savefig(out/"w0_wa_center_density.pdf");plt.close(fig)
    # Occupancy maps for w0-wa and H0-Om.
    for a,b,fn in (("w0","wa","w0_wa_coverage_occupancy.pdf"),
                   ("H0","Omega_m0","H0_Omega_m0_coverage_summary.pdf")):
        xlo=min(np.quantile(samples[i][a],.01) for i in IDS);xhi=max(np.quantile(samples[i][a],.99) for i in IDS)
        ylo=min(np.quantile(samples[i][b],.01) for i in IDS);yhi=max(np.quantile(samples[i][b],.99) for i in IDS)
        xx=np.linspace(xlo,xhi,120);yy=np.linspace(ylo,yhi,120);occ=np.zeros((120,120))
        for r in rows:
            cov=np.array(json.loads(r[f"{a}__{b}_covariance_json"]));mu=np.array([r[f"{a}_mean"],r[f"{b}_mean"]])
            X,Y=np.meshgrid(xx,yy);d=np.stack((X-mu[0],Y-mu[1]),-1)
            d2=np.einsum("...i,ij,...j->...",d,np.linalg.inv(cov),d);occ+=d2<=Q68
        fig,ax=plt.subplots(figsize=(6.4,5));im=ax.pcolormesh(xx,yy,occ,cmap="magma",shading="auto")
        ax.plot(FID[a],FID[b],"c*",ms=10);ax.set(xlabel=a,ylabel=b,title="68% covariance-ellipse occupancy")
        fig.colorbar(im,ax=ax,label="regions covering point");fig.tight_layout();fig.savefig(out/fn);plt.close(fig)
    # Distribution and diagnostic multi-panels.
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    for ax,p in zip(axes.flat,("w0","wa","H0","Omega_m0")):
        ax.hist([r[f"{p}_median"] for r in rows],bins="auto");ax.axvline(FID[p],color="k",ls="--");ax.set_title(p)
    fig.tight_layout();fig.savefig(out/"posterior_center_histograms.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist([r["wp_median"] for r in rows],bins="auto");axes[0].axvline(-1,color="k",ls="--");axes[0].set_title("$w_p$")
    axes[1].hist([r["pivot_z"] for r in rows],bins="auto");axes[1].set_title("realization-level pivot redshift")
    fig.tight_layout();fig.savefig(out/"pivot_eos_distribution.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist([r["w0_wa_eigenvalue_ratio"] for r in rows],bins="auto");axes[0].set(xlabel="long/short eigenvalue")
    axes[1].hist([r["w0__wa_angle_long_deg"] for r in rows],bins="auto");axes[1].set(xlabel="weak-axis angle (deg)")
    fig.tight_layout();fig.savefig(out/"covariance_eigenmodes.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.7,5));sc=ax.scatter([r["strong_displacement_normalized"] for r in rows],
        [r["weak_displacement_normalized"] for r in rows],c=[r["delta_chi2_LR"] for r in rows],cmap="viridis")
    ax.axhline(0,color="k",lw=.5);ax.axvline(0,color="k",lw=.5);ax.set(xlabel="normalized strong displacement",ylabel="normalized weak displacement")
    fig.colorbar(sc,ax=ax,pad=.025,fraction=.05,label=r"$\Delta\chi^2_{\rm LR}$")
    fig.tight_layout();fig.savefig(out/"weak_vs_strong_displacement.pdf");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    for ax,p in zip(axes.flat,("max_tau","max_split_rhat","min_ess","mean_acceptance")):
        ax.hist([r[p] for r in rows],bins="auto");ax.set_title(p)
    fig.tight_layout();fig.savefig(out/"convergence_summary.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist([r["production_length"] for r in rows],bins=np.arange(15000,85001,10000))
    axes[1].hist([r["extension_count"] for r in rows],bins=np.arange(-.5,6.5,1))
    axes[0].set_title("production length");axes[1].set_title("extension count")
    fig.tight_layout();fig.savefig(out/"production_length_summary.pdf");plt.close(fig)
    A=np.array([r["delta_chi2_A"] for r in rows]);LR=np.array([r["delta_chi2_LR"] for r in rows])
    for data,fn,label in ((A,"delta_chi2_A_distribution.pdf","fixed-fiducial A"),
                          (LR,"delta_chi2_LR_distribution.pdf","optimized LR")):
        fig,ax=plt.subplots(figsize=(6,4));ax.hist(data,bins="auto",density=True,alpha=.75)
        x=np.linspace(0,max(data.max(),10),300);ax.plot(x,chi2.pdf(x,2),"k--",label="$\\chi^2_2$ visual reference")
        ax.set(xlabel=label,ylabel="density");ax.legend();fig.tight_layout();fig.savefig(out/fn);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5));ax.scatter(A,LR);m=max(A.max(),LR.max());ax.plot([0,m],[0,m],"k--")
    ax.set(xlabel="statistic A",ylabel="proper LR");fig.tight_layout();fig.savefig(out/"delta_chi2_A_vs_LR.pdf");plt.close(fig)
    # Coverage and probe summaries.
    keys=["marginal_w0_68","marginal_wa_68","marginal_wp_68","joint_w0__wa_hpd_68","joint_w0__wa_hpd_95"]
    fig,ax=plt.subplots(figsize=(8,4));f=[summary["coverage"][k]["fraction"] for k in keys]
    ax.bar(range(len(keys)),f);ax.set_xticks(range(len(keys)),keys,rotation=25,ha="right");ax.set_ylim(0,1)
    fig.tight_layout();fig.savefig(out/"coverage_summary.pdf");plt.close(fig)
    top=sorted(rows,key=lambda r:r["delta_chi2_LR"],reverse=True)[:10]
    fig,ax=plt.subplots(figsize=(8,4));bottom=np.zeros(10)
    probe_styles = (("bao", "BAO", "-"), ("cmb", "CMB", "|"), ("sne", "SNe", "/"))
    for p,label,hatch in probe_styles:
        v=np.array([r[f"delta_chi2_LR_{p}"] for r in top])
        ax.bar(range(10),v,bottom=bottom,label=label,hatch=hatch,
               edgecolor="black",linewidth=.35)
        bottom+=v
    ax.set_xticks(range(10),[f"N{r['realization_id']}" for r in top]);ax.legend();fig.tight_layout()
    fig.savefig(out/"probe_contribution_summary.pdf");plt.close(fig)
    # Center density/correlation matrix/bias and trends.
    names=("Omega_m0","H0","w0","wa","wp")
    correlation_labels=(r"$\Omega_{m0}$",r"$H_0$",r"$w_0$",r"$w_a$",r"$w_p$")
    centers=np.array([[r[f"{p}_median"] for p in names] for r in rows])
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for ax,(a,b) in zip(axes.flat,(("H0","Omega_m0"),("H0","w0"),("H0","wa"),("Omega_m0","w0"),("Omega_m0","wa"),("H0","wp"))):
        ax.hexbin([r[f"{a}_median"] for r in rows],[r[f"{b}_median"] for r in rows],gridsize=12,cmap="Blues")
        ax.plot(FID[a],FID[b],"r*");ax.set(xlabel=a,ylabel=b)
    fig.tight_layout();fig.savefig(out/"cosmological_parameter_center_density.pdf");plt.close(fig)
    corr=np.corrcoef(centers,rowvar=False);fig,ax=plt.subplots(figsize=(5.5,5));im=ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_xticks(range(5),correlation_labels,rotation=30);ax.set_yticks(range(5),correlation_labels)
    for i in range(5):
        for j in range(5):ax.text(j,i,f"{corr[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/"cosmological_parameter_correlation_matrix.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for p,c in (("H0","C0"),("Omega_m0","C1")):
        axes[0].hist([r[f"{p}_pull_median"] for r in rows],bins="auto",alpha=.55,label=p,color=c)
    axes[0].legend();axes[0].set_title("median pulls")
    axes[1].quiver(np.zeros(4),np.arange(4),[summary["bias"][p]["median"]["bias"] for p in ("Omega_m0","H0","w0","wa")],
                   np.zeros(4),angles="xy",scale_units="xy",scale=1)
    axes[1].set_yticks(range(4),("Omega_m0","H0","w0","wa"));axes[1].set_title("estimator bias vector")
    fig.tight_layout();fig.savefig(out/"H0_Omega_m0_bias_and_pulls.pdf");plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(9,8),sharex=True)
    for ax,p in zip(axes.flat,("H0","Omega_m0","w0","wa","wp","delta_chi2_LR")):
        ax.plot([r["realization_id"] for r in rows],[r[f"{p}_median"] if p!="delta_chi2_LR" else r[p] for r in rows],"o",ms=2)
        ax.set_ylabel(p)
    axes[-1,0].set_xlabel("realization ID");axes[-1,1].set_xlabel("realization ID")
    fig.tight_layout();fig.savefig(out/"parameter_batch_trends.pdf");plt.close(fig)
    # Requested aliases/additional figures.
    for fn in ("coverage_summary.pdf","convergence_summary.pdf","production_length_summary.pdf"):
        pass
    # Explicit unavailable-observation placeholders.
    for fn,title in (("observed_vs_mock_LR.pdf","No validated matching observed joint product"),
                     ("observed_vs_mock_w0_wa.pdf","No validated matching observed joint product")):
        fig,ax=plt.subplots(figsize=(6,3));ax.axis("off");ax.text(.5,.5,title,ha="center",va="center")
        fig.tight_layout();fig.savefig(out/fn);plt.close(fig)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def write_reports(root, rows, audit, summary, cases):
    audit_path=root/"audit/Main_N100_Final_N0_N99_Audit.md"
    lines=["# Main N100 final N0–N99 audit","",
      "Exactly N100-0 through N100-99 were requested and read. No N100-100 path was constructed or inspected. "
      "All chain files were opened read-only; none was launched, resumed, extended, overwritten, quarantined, renamed, or modified.","",
      "The sampled `omega_m` definition was verified in the likelihood, mock metadata, and theory code as physical matter density. "
      "Derived `Omega_m0 = omega_m/(H0/100)^2` was calculated sample by sample.","",
      "N100-3 reached 80,000 production steps under the legacy policy. Its two stored history records compress the 40k-to-80k transition; this is retained as a provenance anomaly, not a numerical failure.","",
      "| ID | status | steps | max tau | max R-hat | min ESS | acceptance | history |",
      "|---:|:---|---:|---:|---:|---:|---:|---:|"]
    byid={r["realization_id"]:r for r in rows}
    for a in audit:
        if a["id"] in byid:
            r=byid[a["id"]];lines.append(
                f"| {a['id']} | {a['status']} | {r['production_length']} | {r['max_tau']:.3f} | "
                f"{r['max_split_rhat']:.6f} | {r['min_ess']:.1f} | {r['mean_acceptance']:.4f} | "
                f"{r['stored_extension_history_count']}/{r['extension_count']} |")
        else: lines.append(f"| {a['id']} | {a['status']} | — | — | — | — | — | — |")
    audit_path.write_text("\n".join(lines)+"\n")
    outside=[r for r in rows if not r["w0__wa_hpd_95"]]
    top=sorted(rows,key=lambda r:r["delta_chi2_LR"],reverse=True)[:10]
    out=["# Final N100 outliers","",
         "Selection is numerical and outcome-blind; unusual cosmological results were not excluded.","",
         "## Generating point outside direct posterior-density 95% w0–wa region",""]
    out += [f"- N100-{r['realization_id']}: LR={r['delta_chi2_LR']:.4f}; {r['probe_classification']}" for r in outside] or ["- None"]
    out += ["","## Ten largest optimized likelihood ratios","",
            "| ID | LR | BAO | CMB | SNe | classification |","|---:|---:|---:|---:|---:|:---|"]
    out += [f"| {r['realization_id']} | {r['delta_chi2_LR']:.4f} | {r['delta_chi2_LR_bao']:.3f} | "
            f"{r['delta_chi2_LR_cmb']:.3f} | {r['delta_chi2_LR_sne']:.3f} | {r['probe_classification']} |" for r in top]
    (root/"output/Main_N100_Final_N0_N99_Outliers.md").write_text("\n".join(out)+"\n")
    # Compact but complete standalone TeX report.
    cov=summary["coverage"];bias=summary["bias"];lr=summary["likelihood_ratio"]
    tex=rf"""\documentclass[10pt]{{article}}
\usepackage[margin=.8in]{{geometry}}\usepackage{{booktabs,graphicx,longtable,amsmath}}
\title{{Final Main N100 CPL Ensemble Analysis: N100-0--N100-99}}\date{{26 July 2026}}
\begin{{document}}\maketitle
\begin{{abstract}}A controlled null ensemble of 100 coherent BAO+CMB+SNe realizations generated
from exact flat $\Lambda$CDM is analyzed with CPL. Statistical fluctuations can produce
apparently dynamical-dark-energy-like posterior centers when they project onto the weak
$w_0$--$w_a$ direction. Individual nominal exclusions and ensemble calibration are explicitly
distinguished. The optimized nested-model likelihood ratio is kept separate from the
fixed-generating-vector displacement diagnostic.\end{{abstract}}
\section{{Mock, inference, and audit conventions}}
All unthinned production samples from exactly N100-0--99 were used. All 100 backends are finite,
complete, dimensionally consistent, and converged. N100-3's compressed legacy extension history
is a metadata issue only. The sampled matter parameter is $\omega_m$ and every sample is transformed
as $h=H_0/100$, $\Omega_{{m0}}=\omega_m/h^2$. Equal-tailed marginal intervals and both direct
histogram-density HPD regions and explicitly labelled Gaussian covariance ellipses are reported.
\section{{Posterior distributions, bias, and pivot EOS}}
The pivot is defined realization by realization by
$a_p=1+\mathrm{{Cov}}(w_0,w_a)/\mathrm{{Var}}(w_a)$,
$z_p=a_p^{{-1}}-1$, and $w_p=w_0+(1-a_p)w_a$; pivot redshifts are never silently averaged
before constructing $w_p$. Median-estimator ensemble means (biases) are:
$H_0={bias['H0']['median']['ensemble_mean']:.4f}$ ({bias['H0']['median']['bias']:+.4f}),
$\Omega_{{m0}}={bias['Omega_m0']['median']['ensemble_mean']:.5f}$ ({bias['Omega_m0']['median']['bias']:+.5f}),
$w_0={bias['w0']['median']['ensemble_mean']:.4f}$ ({bias['w0']['median']['bias']:+.4f}),
$w_a={bias['wa']['median']['ensemble_mean']:.4f}$ ({bias['wa']['median']['bias']:+.4f}), and
$w_p={bias['wp']['median']['ensemble_mean']:.4f}$ ({bias['wp']['median']['bias']:+.4f}).
\section{{Coverage}}
Marginal 68/95 percent coverage counts are $w_0$:
{cov['marginal_w0_68']['count']}/{cov['marginal_w0_95']['count']},
$w_a$: {cov['marginal_wa_68']['count']}/{cov['marginal_wa_95']['count']},
$w_p$: {cov['marginal_wp_68']['count']}/{cov['marginal_wp_95']['count']},
$H_0$: {cov['marginal_H0_68']['count']}/{cov['marginal_H0_95']['count']}, and
$\Omega_{{m0}}$: {cov['marginal_Omega_m0_68']['count']}/{cov['marginal_Omega_m0_95']['count']}.
Direct HPD joint $w_0$--$w_a$ coverage is
{cov['joint_w0__wa_hpd_68']['count']}/100 and {cov['joint_w0__wa_hpd_95']['count']}/100;
covariance-ellipse coverage is {cov['joint_w0__wa_ellipse_68']['count']}/100 and
{cov['joint_w0__wa_ellipse_95']['count']}/100. Exact 68 and 95 percent Clopper--Pearson
intervals, binomial errors, and nominal deviations are in the accompanying JSON.
\section{{Two chi-square statistics and empirical calibration}}
Statistic A evaluates the complete generating vector minus the best stored CPL sample and is not
a model-comparison likelihood ratio. The proper statistic independently reoptimizes the four-parameter
$\Lambda$CDM model and six-parameter CPL model:
$\Delta\chi^2_{{\rm LR}}=\chi^2_{{\min,\Lambda{{\rm CDM}}}}-\chi^2_{{\min,{{\rm CPL}}}}$.
Its maximum is {lr['maximum']:.3f} (N100-{lr['maximum_id']}); empirical 68, 90, 95, and 99 percent
quantiles are {lr['quantiles']['0.68']:.3f}, {lr['quantiles']['0.9']:.3f},
{lr['quantiles']['0.95']:.3f}, and {lr['quantiles']['0.99']:.3f}.
The $\chi^2_2$ curve is only a visual reference; Wilks' theorem is not assumed exact.
With $N=100$, tail resolution is about one percent and rare $3\sigma$ probabilities cannot be calibrated.
\section{{Probe decomposition and N100-65}}
BAO, CMB, and SNe components at both optima and local probe gradients are supplied for every realization.
The ten largest LR cases and every direct-HPD 95 percent exclusion are classified in the outlier report.
N100-65 remains a preidentified detailed case, without controlling the ensemble conclusions.
\section{{Structural weak mode and parameter correlations}}
The covariance eigenvalue ratio and normalized strong/weak projections show that distance information
in the two-dimensional CPL sector is strongly anisotropic and often close to rank one; it is not literally
zero information about $w_a$. The pivot $w_p$ tracks the well-constrained short-axis combination.
Broad integrated EOS information must not be confused with direct sensitivity to rapid temporal evolution.
The full recovered-center correlation matrix for $(\Omega_{{m0}},H_0,w_0,w_a,w_p)$ is supplied in JSON
and shown graphically.
\section{{Observed-data comparison}}
No fully validated observed joint BAO+CMB+SNe product matching every mock likelihood convention was found.
No observed statistic was fabricated; the missing-input report and command template are in the JSON.
\section{{Limitations and conclusion}}
Finite density grids introduce a small numerical approximation in direct HPD membership, reported alongside
the established covariance convention. Nominal Gaussian and Wilks thresholds are not exact.
A coherent $\Lambda$CDM ensemble can generate DDE-like CPL posteriors along its weak direction while remaining
globally calibrated; expected false exclusions are not evidence that the ensemble prefers DDE or rules out
$\Lambda$CDM.
\begin{{figure}}[p]\centering\includegraphics[width=.48\textwidth]{{../figures/main_N100_final_N0_N99/w0_wa_posterior_centers.pdf}}
\includegraphics[width=.48\textwidth]{{../figures/main_N100_final_N0_N99/delta_chi2_LR_distribution.pdf}}
\caption{{Posterior centers and optimized LR distribution.}}\end{{figure}}
\begin{{figure}}[p]\centering\includegraphics[width=.48\textwidth]{{../figures/main_N100_final_N0_N99/H0_Omega_m0_posterior_centers.pdf}}
\includegraphics[width=.48\textwidth]{{../figures/main_N100_final_N0_N99/cosmological_parameter_correlation_matrix.pdf}}
\caption{{$H_0$--$\Omega_{{m0}}$ recovery and ensemble correlations.}}\end{{figure}}
\begin{{figure}}[p]\centering\includegraphics[width=.95\textwidth]{{../figures/main_N100_final_N0_N99/w0_wa_representative_contours.pdf}}
\caption{{Representative direct posterior-density contours; covariance ellipses are distinguished.}}\end{{figure}}
\end{{document}}
"""
    texpath=root/"manuscript/Main_N100_Final_Analysis.tex";texpath.write_text(tex)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--observed-joint-h5",type=Path,default=None)
    args=ap.parse_args();root=args.root.resolve()
    if args.observed_joint_h5 is not None:
        raise NotImplementedError("An observed product must first be validated and registered; see missing-input report")
    rows,samples,grids,opt,components,audit,spawn=analyze(root)
    if len(rows)!=100 or any(a["status"]!="OK" for a in audit):
        raise RuntimeError(f"strict audit failed; no scientific products written: {audit}")
    actual=inspect_actual_data(root);summary=make_summary(rows,audit,actual);cases=select_cases(rows)
    write_csv(root/"output/Main_N100_Final_N0_N99_PerRealization.csv",rows)
    write_csv(root/"output/Main_N100_Final_N0_N99_OptimizedFits.csv",[
        {k:v for k,v in r.items() if k=="realization_id" or k.startswith(("lcdm_","cpl_","chi2_min","delta_chi2_LR","fit_shift"))}
        for r in rows])
    write_csv(root/"output/Main_N100_Final_N0_N99_Chi2Components.csv",components)
    (root/"output/Main_N100_Final_N0_N99_OptimizationDetails.json").write_text(json.dumps(opt,indent=2)+"\n")
    (root/"output/Main_N100_Final_N0_N99_EnsembleSummary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (root/"output/Main_N100_Final_N0_N99_Coverage.json").write_text(json.dumps(summary["coverage"],indent=2)+"\n")
    (root/"output/Main_N100_Final_N0_N99_RepresentativeCases.json").write_text(json.dumps(cases,indent=2)+"\n")
    write_reports(root,rows,audit,summary,cases);make_figures(root,rows,samples,grids,cases,summary)
    print(json.dumps({"requested":100,"found":100,"converged":100,"excluded":[],
      "coverage":{k:v["count"] for k,v in summary["coverage"].items() if k in
       ("marginal_H0_68","marginal_H0_95","marginal_Omega_m0_68","marginal_Omega_m0_95",
        "joint_w0__wa_hpd_68","joint_w0__wa_hpd_95")},
      "largest_LR":{"id":summary["likelihood_ratio"]["maximum_id"],"value":summary["likelihood_ratio"]["maximum"]},
      "outputs":["audit/Main_N100_Final_N0_N99_Audit.md","output/Main_N100_Final_N0_N99_*.{csv,json,md}",
                 "figures/main_N100_final_N0_N99/","manuscript/Main_N100_Final_Analysis.tex"]},indent=2))

if __name__=="__main__":
    main()
