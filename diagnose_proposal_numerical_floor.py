"""Diagnose and remove the absolute-jitter proposal floor for regression-GARCH.

Development-only: no dataset generation, no production chains, and no changes
to the posterior or subsampling estimator.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import feasibility_regression_garch_p50 as p50
import prepare_regression_garch_p25 as preparation25
import profile_regression_garch_p25 as profile25
import regression_garch as rg


OLD_ABSOLUTE_JITTER = 1e-6
P50_CALIBRATION = ((0.65, 20262511), (0.75, 20262512), (0.85, 20262513))
P50_PILOT_ITERATIONS = 500
P50_PILOT_SEED = 20262520
P50_FULL_SEED = 20262521
P50_SUBSAMPLING_SEED = 20262522
P25_OLD_SEED = 20262501
P25_NEW_SEED = 20262502
P25_CHECK_ITERATIONS = 200
P25_CORRECTED_MULTIPLIER = 0.75
ITERATIONS = 2000
BURN_IN = 500

OUTPUT_DIRECTORY = p50.OUTPUT_DIRECTORY / "proposal_floor_diagnostic"
REPORT_PATH = OUTPUT_DIRECTORY / "proposal_floor_report.json"
COMPARISON_CSV = OUTPUT_DIRECTORY / "posterior_comparison_54_parameters.csv"
OVERLAY_PNG = OUTPUT_DIRECTORY / "selected_marginal_overlays.png"
FULL_CHAIN_PATH = OUTPUT_DIRECTORY / "full_data_chain_2000.npz"
SUBSAMPLING_CHAIN_PATH = OUTPUT_DIRECTORY / "subsampling_chain_2000.npz"


def progress(message):
    print(f"[proposal-floor] {message}", file=sys.stderr, flush=True)


def load_model_and_proposal(profile_module, preparation_module, checkpoint):
    dataset = rg.load_regression_GARCH11_dataset(profile_module.DATASET_PATH)
    profile_module.verify_dataset(dataset)
    model = preparation_module.setup_model(dataset)
    with np.load(checkpoint, allow_pickle=False) as stored:
        MAP = np.asarray(stored["phi"], dtype=float).copy()
        Hessian = np.asarray(stored["Hessian_phi"], dtype=float).copy()
        checkpoint_hash = str(stored["dataset_sha256"].item())
    dataset_hash = profile25.sha256_file(profile_module.DATASET_PATH)
    if checkpoint_hash != dataset_hash:
        raise RuntimeError("MAP checkpoint and canonical dataset hashes differ.")
    covariance = np.linalg.inv(-0.5 * (Hessian + Hessian.T))
    covariance = model["ns"]["make_pos_def"](covariance, eps=1e-10)
    d = profile_module.P_REG + 4
    base = (2.38**2 / d) * covariance
    original, transformed = rg.get_param_names_regression_GARCH11(
        profile_module.P_REG
    )
    names = {"params_name": original, "trans_param_names": transformed}
    return dataset, dataset_hash, model, MAP, base, names


def spectrum(covariance):
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    quantiles = np.quantile(eigenvalues, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "eigenvalue_quantiles": {
            "q01": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "q25": float(quantiles[2]),
            "q50": float(quantiles[3]),
            "q75": float(quantiles[4]),
            "q95": float(quantiles[5]),
            "q99": float(quantiles[6]),
        },
        "counts_below": {
            "1e-6": int(np.sum(eigenvalues < 1e-6)),
            "1e-7": int(np.sum(eigenvalues < 1e-7)),
            "1e-8": int(np.sum(eigenvalues < 1e-8)),
            "1e-9": int(np.sum(eigenvalues < 1e-9)),
        },
    }


def jitter_comparison(covariance):
    intended = 0.5 * (covariance + covariance.T)
    old = intended + OLD_ABSOLUTE_JITTER * np.eye(intended.shape[0])
    intended_spectrum = spectrum(intended)
    old_spectrum = spectrum(old)
    repaired, _, new_jitter = rg.numerical_cholesky(intended)
    return {
        "intended": intended_spectrum,
        "old_fixed_1e-6": old_spectrum,
        "old_minimum_eigenvalue_inflation_factor": (
            old_spectrum["minimum_eigenvalue"]
            / intended_spectrum["minimum_eigenvalue"]
        ),
        "old_maximum_eigenvalue_inflation_factor": (
            old_spectrum["maximum_eigenvalue"]
            / intended_spectrum["maximum_eigenvalue"]
        ),
        "old_relative_Frobenius_change": float(
            np.linalg.norm(old - intended, ord="fro")
            / np.linalg.norm(intended, ord="fro")
        ),
        "new_actual_jitter": new_jitter,
        "new_maximum_absolute_covariance_change": float(
            np.max(np.abs(repaired - intended))
        ),
    }


def chain_diagnostics(chain, p_reg):
    phi = np.asarray(chain["samples_phi"])
    theta = np.asarray(chain["samples_theta"])
    jumps = np.linalg.norm(np.diff(phi, axis=0), axis=1)
    moved = jumps > 0.0
    longest = 0
    current = 0
    for did_move in moved:
        if did_move:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return {
        "realised_acceptance_rate": float(np.mean(moved)),
        "mean_acceptance_probability": float(np.mean(chain["alphas_MH"])),
        "accepted_moves": int(np.sum(moved)),
        "unique_states": int(np.unique(phi, axis=0).shape[0]),
        "longest_immobile_run": int(longest),
        "outside_support": int(chain["outside_support"]),
        "all_finite": bool(np.all(np.isfinite(phi)) and np.all(np.isfinite(theta))),
        "seconds": float(chain["run_time"]),
        "seconds_per_iteration": float(chain["run_time"] / len(jumps)),
        "maximum_proposal_jitter": (
            float(chain["maximum_proposal_jitter"])
            if "maximum_proposal_jitter" in chain
            else None
        ),
        "parameter_minimum": np.min(theta, axis=0).tolist(),
        "parameter_maximum": np.max(theta, axis=0).tolist(),
        "maximum_persistence": float(
            np.max(theta[:, p_reg + 1] + theta[:, p_reg + 2])
        ),
    }


def run_corrected(model, MAP, covariance, iterations, seed, names, dataset_id, p_reg):
    np.random.seed(seed)
    random.seed(seed)
    chain = rg.sample_full_data_posterior_numerical_cholesky(
        MAP,
        covariance,
        iterations,
        names,
        dataset_id,
        hinv=model["hinv"],
        log_posterior=model["log_posterior"],
        adapt_cov=False,
        adapt_scale=False,
    )
    return chain, chain_diagnostics(chain, p_reg)


def effective_sample_size(values):
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered) / n)
    if variance <= 0.0:
        return 1.0
    fft_size = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(transform * np.conjugate(transform), n=fft_size)[:n]
    autocovariance /= np.arange(n, 0, -1)
    autocorrelation = autocovariance / autocovariance[0]
    integrated_time = 1.0
    for index in range(1, n - 1, 2):
        pair_sum = autocorrelation[index] + autocorrelation[index + 1]
        if pair_sum <= 0.0:
            break
        integrated_time += 2.0 * pair_sum
    return float(min(n, max(1.0, n / integrated_time)))


def ess_summary(theta, p_reg):
    names = [f"beta_{j}" for j in range(1, p_reg + 1)] + [
        "omega",
        "alpha",
        "beta_garch",
        "nu",
    ]
    values = np.array(
        [effective_sample_size(theta[:, index]) for index in range(theta.shape[1])]
    )
    return {
        "minimum": float(np.min(values)),
        "minimum_parameter": names[int(np.argmin(values))],
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
        "maximum_parameter": names[int(np.argmax(values))],
        "per_parameter": dict(zip(names, values.tolist())),
    }


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    p25_data, p25_hash, model25, MAP25, base25, names25 = load_model_and_proposal(
        profile25, preparation25, preparation25.MAP_CHECKPOINT
    )
    p50_data, p50_hash, model50, MAP50, base50, names50 = load_model_and_proposal(
        p50, p50, p50.MAP_CHECKPOINT
    )
    expected_p50_hash = json.loads(p50.DATASET_CREATION_REPORT.read_text())["sha256"]
    if p50_hash != expected_p50_hash:
        raise RuntimeError("p50 canonical dataset hash changed.")

    progress("computing proposal spectra and fixed-jitter effects")
    eigenvalue_diagnostics = {
        "p25": {
            str(multiplier): jitter_comparison(multiplier * base25)
            for multiplier in (1.0, 0.2, 0.5, 0.75)
        },
        "p50": {
            str(multiplier): jitter_comparison(multiplier * base50)
            for multiplier in (1.0, 0.05, 0.5, 0.75, 1.0)
        },
    }

    progress("running short p25 old/new regression checks")
    np.random.seed(P25_OLD_SEED)
    random.seed(P25_OLD_SEED)
    old25 = model25["ns"]["sample_full_data_posterior_AM"](
        MAP25,
        0.2 * base25,
        P25_CHECK_ITERATIONS,
        names25,
        profile25.DATASET_ID,
        adapt_cov=False,
        adapt_scale=False,
        burn_in=1000,
    )
    old25_diagnostics = chain_diagnostics(old25, profile25.P_REG)
    new25, new25_diagnostics = run_corrected(
        model25,
        MAP25,
        P25_CORRECTED_MULTIPLIER * base25,
        P25_CHECK_ITERATIONS,
        P25_NEW_SEED,
        names25,
        profile25.DATASET_ID,
        profile25.P_REG,
    )

    progress("repeating the focused p50 scalar calibration")
    calibration = []
    for multiplier, seed in P50_CALIBRATION:
        _, diagnostics = run_corrected(
            model50,
            MAP50,
            multiplier * base50,
            120,
            seed,
            names50,
            p50.DATASET_ID,
            p50.P_REG,
        )
        diagnostics.update({"multiplier": multiplier, "seed": seed})
        calibration.append(diagnostics)
        progress(
            f"p50 multiplier={multiplier:.2f}: acceptance="
            f"{diagnostics['realised_acceptance_rate']:.3f}, "
            f"mean alpha={diagnostics['mean_acceptance_probability']:.3f}"
        )
    eligible = [
        result
        for result in calibration
        if 0.20 <= result["realised_acceptance_rate"] <= 0.30
    ]
    if not eligible:
        raise RuntimeError(
            "Corrected p50 proposal did not calibrate into the target interval."
        )
    selected = min(
        eligible,
        key=lambda item: abs(item["realised_acceptance_rate"] - 0.25),
    )
    multiplier = float(selected["multiplier"])

    progress("running improved 500-iteration p50 full-data pilot")
    pilot, pilot_diagnostics = run_corrected(
        model50,
        MAP50,
        multiplier * base50,
        P50_PILOT_ITERATIONS,
        P50_PILOT_SEED,
        names50,
        p50.DATASET_ID,
        p50.P_REG,
    )
    pilot_theta = np.asarray(pilot["samples_theta"])[101:]
    pilot_diagnostics["ESS_after_first_100"] = ess_summary(
        pilot_theta, p50.P_REG
    )
    if not 0.15 <= pilot_diagnostics["realised_acceptance_rate"] <= 0.35:
        raise RuntimeError(
            "Corrected p50 pilot did not maintain sensible acceptance."
        )

    progress("reconstructing existing p50 Taylor quantities")
    densities, gradients, Hessians, q_sum, _, taylor_seconds, _ = (
        p50.construct_taylor_quantities(model50, MAP50)
    )
    existing_report = json.loads(p50.REPORT_PATH.read_text())
    tuning = existing_report["tuning"]
    if (
        tuning["R_max"] != 100
        or tuning["m"] != 2
        or tuning["b"] != 0.0
        or tuning["t_star"] != 1000
    ):
        raise RuntimeError("Existing p50 Rmax=100 tuning differs from requirements.")
    estimator_params = {
        "m_opt": int(tuning["m"]),
        "c_opt": float(tuning["c"]),
        "gamma_opt": float(tuning["gamma"]),
        "t_star": int(tuning["t_star"]),
        "b": float(tuning["b"]),
        "V_tolerance": float(tuning["variance_tolerance"]),
        "R_max": int(tuning["R_max"]),
    }

    progress("running corrected 2,000-iteration p50 full-data chain")
    full_chain, full_diagnostics = run_corrected(
        model50,
        MAP50,
        multiplier * base50,
        ITERATIONS,
        P50_FULL_SEED,
        names50,
        p50.DATASET_ID,
        p50.P_REG,
    )
    progress("running unchanged Rmax=100 subsampling chain with calibrated proposal")
    np.random.seed(P50_SUBSAMPLING_SEED)
    random.seed(P50_SUBSAMPLING_SEED)
    sub_chain = model50["ns"]["sample_subsampling_posterior_random_walk"](
        estimator_params,
        MAP50,
        multiplier * base50,
        ITERATIONS,
        names50,
        p50.DATASET_ID,
        adapt=False,
    )
    sub_diagnostics = chain_diagnostics(sub_chain, p50.P_REG)
    retained_u_max = np.asarray(sub_chain["u_max_prop"])[BURN_IN:]
    sub_diagnostics["u_max_retained"] = p50.distribution_summary(retained_u_max)
    sub_diagnostics["u_max_retained"].update(
        {
            "mean_over_T": float(np.mean(retained_u_max) / p50.T),
            "median_over_T": float(np.median(retained_u_max) / p50.T),
        }
    )

    full_retained = p50.retained_theta(full_chain)
    sub_retained = p50.retained_theta(sub_chain)
    rows = p50.comparison_rows(full_retained, sub_retained)
    comparison_summary = p50.summarize_comparison(rows)
    persistence = p50.derived_persistence_comparison(full_retained, sub_retained)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p50.make_overlay_plot(full_retained, sub_retained, OVERLAY_PNG)
    p50.save_chain(
        FULL_CHAIN_PATH,
        full_chain,
        full_retained,
        P50_FULL_SEED,
        subsampling=False,
    )
    p50.save_chain(
        SUBSAMPLING_CHAIN_PATH,
        sub_chain,
        sub_retained,
        P50_SUBSAMPLING_SEED,
        subsampling=True,
    )

    p25_hash_after = profile25.sha256_file(profile25.DATASET_PATH)
    p50_hash_after = profile25.sha256_file(p50.DATASET_PATH)
    if p25_hash_after != p25_hash or p50_hash_after != p50_hash:
        raise RuntimeError("A canonical dataset changed during diagnostics.")
    report = {
        "development_only": True,
        "canonical_dataset_hashes": {
            "p25_before": p25_hash,
            "p25_after": p25_hash_after,
            "p50_before": p50_hash,
            "p50_after": p50_hash_after,
        },
        "old_absolute_jitter": OLD_ABSOLUTE_JITTER,
        "eigenvalue_diagnostics": eigenvalue_diagnostics,
        "new_stabilisation": {
            "algorithm": (
                "symmetrise; direct Cholesky; only after failure add the smallest "
                "covariance-scale machine-precision jitter that succeeds"
            ),
            "actual_jitter_p25_representative": eigenvalue_diagnostics["p25"]["1.0"]["new_actual_jitter"],
            "actual_jitter_p50_selected": eigenvalue_diagnostics["p50"][str(multiplier)]["new_actual_jitter"],
        },
        "p25_regression_check": {
            "old_sampler": {
                "multiplier": 0.2,
                "iterations": P25_CHECK_ITERATIONS,
                "seed": P25_OLD_SEED,
                **old25_diagnostics,
            },
            "corrected_sampler": {
                "multiplier": P25_CORRECTED_MULTIPLIER,
                "iterations": P25_CHECK_ITERATIONS,
                "seed": P25_NEW_SEED,
                **new25_diagnostics,
            },
        },
        "p50_calibration": {
            "iterations_per_multiplier": 120,
            "results": calibration,
            "selected_multiplier": multiplier,
        },
        "p50_improved_pilot": {
            "iterations": P50_PILOT_ITERATIONS,
            "seed": P50_PILOT_SEED,
            **pilot_diagnostics,
        },
        "reused_tuning": tuning,
        "Taylor_reconstruction_seconds": taylor_seconds,
        "diagnostic_chains": {
            "iterations": ITERATIONS,
            "burn_in": BURN_IN,
            "retained": ITERATIONS - BURN_IN,
            "full_data": {
                "seed": P50_FULL_SEED,
                **full_diagnostics,
                "checkpoint": str(FULL_CHAIN_PATH.resolve()),
            },
            "subsampling": {
                "seed": P50_SUBSAMPLING_SEED,
                **sub_diagnostics,
                "checkpoint": str(SUBSAMPLING_CHAIN_PATH.resolve()),
            },
        },
        "posterior_comparison": {
            "parameters": rows,
            "summary": comparison_summary,
            "persistence": persistence,
            "csv": str(COMPARISON_CSV.resolve()),
            "plot": str(OVERLAY_PNG.resolve()),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress("proposal-floor diagnostic complete")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    progress(f"total elapsed seconds: {time.perf_counter() - start:.1f}")
