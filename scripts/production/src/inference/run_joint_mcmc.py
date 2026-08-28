#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Restart-safe Pedagogic II pilot and main-N100 emcee execution interface."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import emcee
import h5py
import numpy as np
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.diagnose_joint_mcmc import diagnose
from src.inference.mcmc_log_probability import MCMCLogProbability
from src.inference.select_pilot_realizations import optimizer_center, select_pilot_rows
from src.likelihoods.joint_likelihood import JointLikelihood

BLOB_DTYPE = [
    ("chi2_bao", float), ("chi2_cmb", float),
    ("chi2_sne", float), ("chi2_total", float),
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_walkers(center, probability, nwalkers, fractional_width, seed):
    center = np.asarray(center, float)
    bounds = np.asarray(probability.prior.bounds(probability.model), float)
    scale = fractional_width * (bounds[:, 1] - bounds[:, 0])
    rng = np.random.default_rng(seed)
    walkers = []
    attempts = 0
    while len(walkers) < nwalkers and attempts < 100000:
        candidate = center + rng.normal(size=center.size) * scale
        attempts += 1
        if not np.isfinite(probability.log_prior(candidate)):
            continue
        if any(np.array_equal(candidate, previous) for previous in walkers):
            continue
        walkers.append(candidate)
    if len(walkers) != nwalkers:
        raise RuntimeError(f"Only initialized {len(walkers)} valid walkers")
    result = np.asarray(walkers)
    if np.unique(result, axis=0).shape[0] != nwalkers:
        raise RuntimeError("Walker initialization is not unique")
    return result


def _thresholds(config):
    ac = config["diagnostics"]["integrated_autocorrelation"]
    return {
        "minimum_chain_length_in_tau": float(ac["minimum_chain_length_in_tau"]),
        "maximum_fractional_tau_change": float(ac["maximum_fractional_tau_change"]),
        "split_Rhat_max": float(config["diagnostics"]["split_Rhat_max"]),
        "effective_sample_size_min_per_parameter": float(
            config["diagnostics"]["effective_sample_size_min_per_parameter"]
        ),
        "acceptance_fraction_range": list(config["diagnostics"]["acceptance_fraction_range"]),
    }

def parse_case(case):
    if case == "zero_noise":
        return {"dataset": "zero_noise", "realization_id": 0, "label": "zero_noise"}
    match = re.fullmatch(r"(N10|N100)-([0-9]+)", str(case))
    if match is None:
        raise ValueError(f"Invalid case {case!r}; expected zero_noise, N10-ID, or N100-ID")
    dataset, text_id = match.groups()
    realization_id = int(text_id)
    upper = 9 if dataset == "N10" else 99
    if not 0 <= realization_id <= upper:
        raise ValueError(f"{dataset} realization ID must be in [0,{upper}]")
    return {"dataset": dataset, "realization_id": realization_id,
            "label": f"{dataset}-{realization_id}"}


def resolve_case(case, optimization_path=None):
    spec = parse_case(case)
    if spec["dataset"] == "N100":
        return spec
    if optimization_path is None:
        raise ValueError("Pilot cases require the optimization-results CSV")
    rows = select_pilot_rows(optimization_path)
    wanted = (spec["dataset"], spec["realization_id"])
    matches = [dict(row) for row in rows
               if (row["dataset"], int(row["realization_id"])) == wanted]
    if len(matches) != 1:
        raise ValueError(f"Unknown or non-unique validated pilot case {case}")
    return {**matches[0], **spec}


def mock_path_for_case(spec, config):
    if spec["dataset"] == "zero_noise":
        return ROOT / "mocks/joint/pedagogic2_joint_lcdm_zero_noise.h5"
    if spec["dataset"] == "N10":
        return ROOT / "mocks/joint/pedagogic2_joint_lcdm_N0010.h5"
    configured = Path(config["mock_file"])
    return configured if configured.is_absolute() else ROOT / configured


def output_path_for_case(spec, model, config):
    root = (ROOT / config.get("output_root", "chains/pilot")
            if not Path(config.get("output_root", "chains/pilot")).is_absolute()
            else Path(config["output_root"]))
    return root / spec["label"] / f"{model}.h5"


def verify_realization_available(mock_path, realization_id):
    with h5py.File(mock_path, "r") as handle:
        ids = np.asarray(handle["metadata/realization_id"][:], int)
    count = int(np.count_nonzero(ids == int(realization_id)))
    if count != 1:
        raise ValueError(f"Realization {realization_id} occurs {count} times in {mock_path}")
    return True


def center_for_case(spec, model, config):
    prefix = f"{model}_"
    names = ("omega_m", "omega_b", "H0", "DeltaM") if model == "lcdm" else (
        "omega_m", "omega_b", "H0", "w0", "wa", "DeltaM"
    )
    if all(prefix + name in spec for name in names):
        return optimizer_center(spec, model)
    if config["initialization"].get("center") != "fiducial":
        raise ValueError(f"No optimizer center is available for {spec['label']}")
    fiducial = config["initialization"]["fiducial"]
    return [float(fiducial[name]) for name in names]


def production_targets(config):
    targets = config.get("production_targets",
                         config.get("rerun_policy", {}).get("total_production_targets"))
    if targets is None:
        initial = int(config["production_steps"])
        factor = int(config["rerun_policy"]["extend_factor"])
        targets = [initial * factor**index
                   for index in range(int(config["rerun_policy"]["maximum_extensions"]) + 1)]
    targets = [int(value) for value in targets]
    expected_count = int(config["rerun_policy"]["maximum_extensions"]) + 1
    if len(targets) != expected_count or targets[0] != int(config["production_steps"]):
        raise ValueError("Production targets do not match initial length/extension count")
    if targets != sorted(set(targets)):
        raise ValueError("Production targets must be strictly increasing")
    return targets


def pending_diagnostic_targets(config, current_iteration):
    """Return indexed targets at or beyond the stored production iteration."""
    current_iteration = int(current_iteration)
    return [
        (index, target) for index, target in enumerate(production_targets(config))
        if target >= current_iteration
    ]


def diagnostic_decision(converged, target, final_target):
    if bool(converged):
        return "stop_converged"
    if int(target) == int(final_target):
        return "quarantine"
    return "continue"


def production_stage_name(target_index):
    names = (
        "initial production", "first extension", "second extension",
        "third extension", "fourth extension", "fifth extension",
    )
    return names[target_index] if target_index < len(names) else f"extension {target_index}"


def inspect_backend(path):
    path = Path(path)
    if not path.exists():
        return {"status": "new", "burnin_iteration": 0, "production_iteration": 0}
    with h5py.File(path, "r") as handle:
        burn = int(handle["burnin"].attrs["iteration"]) if "burnin" in handle else 0
        production = int(handle["production"].attrs["iteration"]) if "production" in handle else 0
        classification = None
        if "metadata" in handle:
            classification = handle["metadata"].attrs.get("convergence_classification")
    if classification == "CONVERGED":
        status = "completed_converged"
    elif classification == "QUARANTINED":
        status = "completed_quarantined"
    else:
        status = "incomplete"
    return {"status": status, "burnin_iteration": burn,
            "production_iteration": production, "classification": classification}

def backend_action(path, overwrite=False):
    state = inspect_backend(path)
    if overwrite and Path(path).exists():
        return "replace", state
    if state["status"] == "new":
        return "start", state
    if state["status"] == "incomplete":
        return "resume", state
    return "protect", state


def _write_run_identity(path, spec, model, mock, config):
    expected = {
        "realization_label": spec["label"], "realization_id": int(spec["realization_id"]),
        "dataset": spec["dataset"], "sampled_model": model,
        "mock_file": str(Path(mock).resolve()),
    }
    with h5py.File(path, "a") as handle:
        group = handle.require_group("run_metadata")
        for key, value in expected.items():
            if key in group.attrs and group.attrs[key] != value:
                raise RuntimeError(
                    f"Backend identity mismatch for {key}: {group.attrs[key]!r} != {value!r}"
                )
            group.attrs[key] = value
        group.attrs["checkpoint_interval_steps"] = int(
            config["storage"]["checkpoint_interval_steps"]
        )
    return expected


def _acceptance_fraction(path, iteration):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["production/accepted"][:], float) / float(iteration)

def select_progress_enabled(cli_value, stream=None):
    """Use the CLI override, otherwise enable bars only on an interactive stdout."""
    if cli_value is not None:
        return bool(cli_value)
    stream = sys.stdout if stream is None else stream
    return bool(getattr(stream, "isatty", lambda: False)())


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if not np.isfinite(seconds):
        return "unknown"
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_checkpoint_status(
    case_id, model, stage, completed, target, elapsed, remaining,
    rate, acceptance, backend_path,
):
    acceptance_text = "unavailable" if acceptance is None else f"{float(acceptance):.6f}"
    rate_text = "unavailable" if rate is None or not np.isfinite(rate) else f"{rate:.3f} step/s"
    return (
        f"[checkpoint] case={case_id} model={model} stage={stage} "
        f"completed={int(completed)} target={int(target)} "
        f"elapsed={format_duration(elapsed)} remaining={format_duration(remaining)} "
        f"rate={rate_text} acceptance={acceptance_text} "
        f"backend={Path(backend_path).resolve()}"
    )


def _mean_acceptance(sampler):
    try:
        values = np.asarray(sampler.acceptance_fraction, float)
        value = float(np.mean(values))
        return value if np.isfinite(value) else None
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None


class RealizationProgress:
    """One persistent, checkpoint-refreshed tqdm line for a realization."""

    def __init__(self, case_id, *, enabled=False, position=0, stream=None, clock=None):
        if position < 0:
            raise ValueError("progress_position must be non-negative")
        self.case_id = str(case_id)
        self.enabled = bool(enabled)
        self.stream = sys.stdout if stream is None else stream
        self.clock = time.perf_counter if clock is None else clock
        self.position = int(position)
        self.bar = None
        self.stage = None
        self.initial_completed = 0
        self.started = None

    def start_stage(self, stage, initial_completed, target_steps):
        self.stage = str(stage)
        self.initial_completed = int(initial_completed)
        self.started = self.clock()
        description = f"case={self.case_id} stage={self.stage}"
        if self.bar is None:
            self.bar = tqdm(
                total=int(target_steps), initial=self.initial_completed,
                desc=description, disable=not self.enabled,
                position=self.position, file=self.stream, dynamic_ncols=True,
                leave=True, unit="step", miniters=1,
                bar_format=(
                    "{desc} | {n_fmt}/{total_fmt} | {percentage:6.2f}% | "
                    "elapsed={elapsed} | ETA={remaining} | rate={rate_fmt}"
                ),
            )
        else:
            now = self.bar._time()
            self.bar.total = int(target_steps)
            self.bar.n = self.initial_completed
            self.bar.last_print_n = self.initial_completed
            self.bar.start_t = now
            self.bar.last_print_t = now
            self.bar._ema_dn = type(self.bar._ema_dn)(self.bar.smoothing)
            self.bar._ema_dt = type(self.bar._ema_dt)(self.bar.smoothing)
            self.bar.set_description_str(description, refresh=False)

    def checkpoint(self, completed):
        completed = int(completed)
        elapsed = max(0.0, self.clock() - self.started)
        newly_sampled = completed - self.initial_completed
        rate = newly_sampled / elapsed if elapsed > 0 else None
        self.bar.update(completed - self.bar.n)
        return elapsed, rate

    def close(self):
        if self.bar is not None:
            self.bar.close()


def advance_sampler(
    sampler, initial_state, steps, checkpoint_interval, *,
    case_id="unknown", model="unknown", stage="sampling",
    initial_completed=0, target_steps=None, backend_path="unknown.h5",
    progress_enabled=False, progress_position=0, stream=None, clock=None,
    verbose_checkpoints=False, progress_display=None,
):
    """Advance in explicit checkpoint-sized calls; HDFBackend persists each step."""
    remaining = int(steps)
    interval = int(checkpoint_interval)
    if remaining < 0 or interval <= 0:
        raise ValueError("Invalid sampling/checkpoint length")
    initial_completed = int(initial_completed)
    target_steps = initial_completed + remaining if target_steps is None else int(target_steps)
    if initial_completed + remaining != target_steps:
        raise ValueError("Initial position plus requested steps must equal target")
    if progress_position < 0:
        raise ValueError("progress_position must be non-negative")
    stream = sys.stdout if stream is None else stream
    clock = time.perf_counter if clock is None else clock
    state = initial_state
    completed = initial_completed
    owns_display = progress_display is None
    display = progress_display or RealizationProgress(
        case_id, enabled=progress_enabled, position=progress_position,
        stream=stream, clock=clock,
    )
    display.start_stage(stage, initial_completed, target_steps)
    try:
        while remaining:
            chunk = min(interval, remaining)
            state = sampler.run_mcmc(state, chunk, progress=False)
            remaining -= chunk
            completed += chunk
            elapsed, rate = display.checkpoint(completed)
            eta = (target_steps - completed) / rate if rate and rate > 0 else np.inf
            if verbose_checkpoints:
                line = format_checkpoint_status(
                    case_id, model, stage, completed, target_steps, elapsed, eta,
                    rate, _mean_acceptance(sampler), backend_path,
                )
                if progress_enabled:
                    tqdm.write(line, file=stream)
                else:
                    print(line, file=stream, flush=True)
                stream.flush()
    finally:
        if owns_display:
            display.close()
    return state


def run_chain(
    row, model, config, seed, overwrite=False,
    progress_enabled=None, progress_position=0, verbose_checkpoints=False,
):
    dataset = row["dataset"]
    realization_id = int(row["realization_id"])
    label = row.get("label") or ("zero_noise" if dataset == "zero_noise"
                                else f"{dataset}-{realization_id}")
    row = {**row, "label": label}
    mock = mock_path_for_case(row, config)
    verify_realization_available(mock, realization_id)
    prior_path = ROOT / "config" / config["priors"]
    grid_path = ROOT / "results/early_universe/class_grid.npz"
    likelihood = JointLikelihood(mock, realization_id, prior_path, grid_path)
    probability = MCMCLogProbability(likelihood, model)
    center = center_for_case(row, model, config)
    walkers = initialize_walkers(
        center, probability, int(config["walkers"]),
        float(config["initialization"]["fractional_prior_width"]), seed
    )
    output = output_path_for_case(row, model, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    action, state = backend_action(output, overwrite)
    if action == "protect":
        raise FileExistsError(
            f"Protected {state['status']} backend {output}; use --overwrite to replace intentionally"
        )
    if action == "replace":
        output.unlink()
        state = inspect_backend(output)
    print(f"backend={output.resolve()}", flush=True)
    _write_run_identity(output, row, model, mock, config)

    start = time.perf_counter()
    progress_display = RealizationProgress(
        label, enabled=bool(progress_enabled), position=progress_position,
    )
    burn_backend = emcee.backends.HDFBackend(output, name="burnin")
    with h5py.File(output, "r") as handle:
        has_burnin = "burnin" in handle
    if not has_burnin:
        burn_backend.reset(len(walkers), walkers.shape[1])
    burn_sampler = emcee.EnsembleSampler(
        len(walkers), walkers.shape[1], probability.log_probability,
        backend=burn_backend, blobs_dtype=BLOB_DTYPE
    )
    burn_remaining = int(config["burn_in_steps"]) - int(burn_backend.iteration)
    if burn_remaining < 0:
        raise RuntimeError("Burn-in backend exceeds configured burn-in length")
    if burn_remaining:
        start_state = walkers if burn_backend.iteration == 0 else None
        burn_initial = int(burn_backend.iteration)
        advance_sampler(
            burn_sampler, start_state, burn_remaining,
            config["storage"]["checkpoint_interval_steps"],
            case_id=label, model=model, stage="burn-in",
            initial_completed=burn_initial,
            target_steps=int(config["burn_in_steps"]),
            backend_path=output, progress_enabled=bool(progress_enabled),
            progress_position=progress_position,
            verbose_checkpoints=verbose_checkpoints,
            progress_display=progress_display,
        )
    burn_state = burn_backend.get_last_sample()

    production_backend = emcee.backends.HDFBackend(output, name="production")
    with h5py.File(output, "r") as handle:
        has_production = "production" in handle
    if not has_production:
        production_backend.reset(len(walkers), walkers.shape[1])
    production_sampler = emcee.EnsembleSampler(
        len(walkers), walkers.shape[1], probability.log_probability,
        backend=production_backend, blobs_dtype=BLOB_DTYPE
    )

    extension_history = []
    thresholds = _thresholds(config)
    diagnostics = None
    targets = production_targets(config)
    starting_iteration = int(production_backend.iteration)
    for target_index, target in pending_diagnostic_targets(config, starting_iteration):
        current = int(production_backend.iteration)
        if current < target:
            sampler_state = burn_state if current == 0 else None
            stage = production_stage_name(target_index)
            advance_sampler(
                production_sampler, sampler_state, target - current,
                config["storage"]["checkpoint_interval_steps"],
                case_id=label, model=model, stage=stage,
                initial_completed=current, target_steps=target,
                backend_path=output, progress_enabled=bool(progress_enabled),
                progress_position=progress_position,
                verbose_checkpoints=verbose_checkpoints,
                progress_display=progress_display,
            )
            if target_index:
                extension_history.append({
                    "extension": target_index, "previous_steps": current,
                    "added_steps": target - current, "target_steps": target,
                })
        iteration = int(production_backend.iteration)
        diagnostics = diagnose(
            production_backend.get_chain(), _acceptance_fraction(output, iteration), thresholds
        )
        with h5py.File(output, "a") as handle:
            run_meta = handle["run_metadata"]
            run_meta.attrs["last_diagnostic_iteration"] = iteration
            run_meta.attrs["last_diagnostics"] = json.dumps(diagnostics)
        decision = diagnostic_decision(diagnostics["converged"], target, targets[-1])
        if decision != "continue":
            break
    if diagnostics is None:
        raise RuntimeError("No production diagnostic was evaluated")

    runtime = time.perf_counter() - start
    manifest = ROOT / "results/early_universe/emulator_manifest.json"
    with h5py.File(output, "a") as handle:
        meta = handle.require_group("metadata")
        meta.attrs.update(
            sampled_model=model,
            realization_label=label,
            realization_id=realization_id,
            dataset=dataset,
            parameter_order=json.dumps(list(probability.parameter_order)),
            prior_config_sha256=sha256(prior_path),
            emulator_manifest_sha256=sha256(manifest),
            emulator_grid_sha256=sha256(grid_path),
            emulator_version=likelihood.emulator.version,
            mock_file_sha256=sha256(mock),
            mock_file=str(mock.resolve()),
            seed=int(seed),
            initialization_center=json.dumps(center),
            initialization_walkers=json.dumps(walkers.tolist()),
            burn_in_steps=int(config["burn_in_steps"]),
            production_steps=int(production_backend.iteration),
            thin=int(config["thin"]),
            checkpoint_interval_steps=int(config["storage"]["checkpoint_interval_steps"]),
            runtime_seconds=float(runtime),
            extension_history=json.dumps(extension_history),
            convergence_classification="CONVERGED" if diagnostics["converged"] else "QUARANTINED",
            diagnostics=json.dumps(diagnostics),
        )
    progress_display.close()
    result = {
        "label": label, "dataset": dataset, "realization_id": realization_id,
        "model": model, "output": str(output.resolve()), "runtime_seconds": runtime,
        "optimizer_center": center, "diagnostics": diagnostics,
        "extension_history": extension_history,
    }
    print(json.dumps(
        {key: value for key, value in result.items() if key != "output"},
        sort_keys=True,
    ), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "config/mcmc_pilot.yaml")
    parser.add_argument("--optimization", default=ROOT / "results/pilot_optimization_results.csv")
    parser.add_argument("--case", help="zero_noise, N10-ID, or N100-ID")
    parser.add_argument("--model", choices=("lcdm", "cpl"))
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress", dest="progress", action="store_true", default=None,
        help="force tqdm progress bars even when stdout is redirected",
    )
    progress_group.add_argument(
        "--no-progress", dest="progress", action="store_false",
        help="disable the animated progress line",
    )
    parser.add_argument(
        "--progress-position", type=int, default=0,
        help="zero-based tqdm terminal row for independent visible processes",
    )
    parser.add_argument(
        "--verbose-checkpoints", action="store_true",
        help="print detailed status after each checkpoint",
    )
    args = parser.parse_args()
    if args.progress_position < 0:
        parser.error("--progress-position must be non-negative")
    show_progress = select_progress_enabled(args.progress, sys.stdout)
    config = yaml.safe_load(Path(args.config).read_text())
    is_main_n100 = config.get("case_prefix") == "N100"
    if is_main_n100 and not args.case:
        parser.error("The N=100 interface requires exactly one --case N100-ID")
    if args.case:
        rows = [resolve_case(args.case, args.optimization)]
        if is_main_n100 and rows[0]["dataset"] != "N100":
            parser.error("mcmc_main_N100.yaml accepts only N100-ID cases")
    else:
        rows = select_pilot_rows(args.optimization)
        rows = [{**row, "label": ("zero_noise" if row["dataset"] == "zero_noise"
                                  else f"N10-{row['realization_id']}")} for row in rows]
    configured_model = str(config.get("model", "CPL")).lower()
    models = [args.model] if args.model else [configured_model]
    tasks = []
    for row_index, row in enumerate(rows):
        for model_index, model in enumerate(models):
            tasks.append((
                row, model, config, args.seed + 100 * row_index + model_index,
                args.overwrite, show_progress,
                args.progress_position + len(tasks) if args.parallel > 1 else args.progress_position,
                args.verbose_checkpoints,
            ))
    if args.parallel > 1:
        results = []
        with ProcessPoolExecutor(max_workers=min(args.parallel, len(tasks))) as executor:
            futures = [executor.submit(run_chain, *task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [run_chain(*task) for task in tasks]
    return results


if __name__ == "__main__":
    main()
