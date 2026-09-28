"""Short development posterior comparison for p_reg=25 and Rmax=100.

This is not a manuscript-production run.  It reloads existing checkpoints, runs
2,000 iterations per method, discards 500 iterations, and saves diagnostic-only
outputs under the ignored regression_garch_outputs directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import evaluate_regression_garch_p25_rmax as rmax_diagnostics
import prepare_regression_garch_p25 as preparation
import profile_regression_garch_p25 as profile_p25
import regression_garch as rg


ITERATIONS = 2000
BURN_IN = 500
RETAINED_DRAWS = ITERATIONS - BURN_IN
FULL_DATA_SEED = 20261401
SUBSAMPLING_SEED = 20261402
RMAX = 100

OUTPUT_DIRECTORY = preparation.OUTPUT_DIRECTORY / "posterior_diagnostic_rmax_100"
DEFAULT_REPORT = OUTPUT_DIRECTORY / "posterior_comparison_report.json"
COMPARISON_CSV = OUTPUT_DIRECTORY / "posterior_comparison_29_parameters.csv"
OVERLAY_PNG = OUTPUT_DIRECTORY / "selected_marginal_overlays.png"
FULL_CHAIN_PATH = OUTPUT_DIRECTORY / "full_data_chain_2000.npz"
SUBSAMPLING_CHAIN_PATH = OUTPUT_DIRECTORY / "subsampling_chain_2000.npz"


def progress(message):
    print(f"[posterior-diagnostic] {message}", file=sys.stderr, flush=True)


def simple_parameter_names():
    return [f"beta_{j}" for j in range(1, profile_p25.P_REG + 1)] + [
        "omega",
        "alpha",
        "beta_garch",
        "nu",
    ]


def save_chain(path, chain, retained_theta, seed):
    np.savez_compressed(
        path,
        samples_phi=np.asarray(chain["samples_phi"]),
        samples_theta=np.asarray(chain["samples_theta"]),
        retained_theta=np.asarray(retained_theta),
        alphas_MH=np.asarray(chain["alphas_MH"]),
        seed=np.asarray(seed),
        iterations=np.asarray(ITERATIONS),
        burn_in=np.asarray(BURN_IN),
    )


def save_subsampling_chain(path, chain, retained_theta, seed):
    np.savez_compressed(
        path,
        samples_phi=np.asarray(chain["samples_phi"]),
        samples_theta=np.asarray(chain["samples_theta"]),
        retained_theta=np.asarray(retained_theta),
        alphas_MH=np.asarray(chain["alphas_MH"]),
        log_posthat_samples=np.asarray(chain["log_posthat_samples"]),
        sigma2_lhat_samples=np.asarray(chain["sigma2_lhat_samples"]),
        u_max_prop=np.asarray(chain["u_max_prop"]),
        seed=np.asarray(seed),
        iterations=np.asarray(ITERATIONS),
        burn_in=np.asarray(BURN_IN),
    )


def retained_draws(chain):
    # samples[0] is the initialization; iterations 1,...,500 are burn-in.
    retained = np.asarray(chain["samples_theta"])[BURN_IN + 1 : ITERATIONS + 1]
    if retained.shape != (RETAINED_DRAWS, profile_p25.P_REG + 4):
        raise AssertionError(f"Unexpected retained-draw shape {retained.shape}.")
    return retained


def posterior_comparison(full_draws, subsampling_draws):
    names = simple_parameter_names()
    probabilities = (0.025, 0.50, 0.975)
    rows = []
    for index, name in enumerate(names):
        full = full_draws[:, index]
        subsampling = subsampling_draws[:, index]
        full_mean = float(np.mean(full))
        subsampling_mean = float(np.mean(subsampling))
        full_sd = float(np.std(full, ddof=1))
        subsampling_sd = float(np.std(subsampling, ddof=1))
        full_quantiles = np.quantile(full, probabilities)
        subsampling_quantiles = np.quantile(subsampling, probabilities)
        rows.append(
            {
                "parameter": name,
                "full_mean": full_mean,
                "subsampling_mean": subsampling_mean,
                "full_sd": full_sd,
                "subsampling_sd": subsampling_sd,
                "absolute_standardised_mean_difference": abs(
                    subsampling_mean - full_mean
                )
                / full_sd,
                "sd_ratio_subsampling_over_full": subsampling_sd / full_sd,
                "full_q025": float(full_quantiles[0]),
                "full_q50": float(full_quantiles[1]),
                "full_q975": float(full_quantiles[2]),
                "subsampling_q025": float(subsampling_quantiles[0]),
                "subsampling_q50": float(subsampling_quantiles[1]),
                "subsampling_q975": float(subsampling_quantiles[2]),
            }
        )
    standardized = np.array(
        [row["absolute_standardised_mean_difference"] for row in rows]
    )
    sd_ratios = np.array([row["sd_ratio_subsampling_over_full"] for row in rows])
    summary = {
        "maximum_absolute_standardised_mean_difference": float(np.max(standardized)),
        "maximum_difference_parameter": rows[int(np.argmax(standardized))]["parameter"],
        "median_absolute_standardised_mean_difference": float(
            np.median(standardized)
        ),
        "minimum_sd_ratio": float(np.min(sd_ratios)),
        "minimum_sd_ratio_parameter": rows[int(np.argmin(sd_ratios))]["parameter"],
        "maximum_sd_ratio": float(np.max(sd_ratios)),
        "maximum_sd_ratio_parameter": rows[int(np.argmax(sd_ratios))]["parameter"],
    }
    return rows, summary


def write_comparison_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_overlay_plot(full_draws, subsampling_draws, path):
    selected = [0, 4, 9, 14, 19, 24, 25, 26, 27, 28]
    names = simple_parameter_names()
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.5))
    for axis, index in zip(axes.flat, selected):
        full = full_draws[:, index]
        subsampling = subsampling_draws[:, index]
        pooled = np.concatenate([full, subsampling])
        lower, upper = np.quantile(pooled, [0.0025, 0.9975])
        padding = max(0.08 * (upper - lower), np.finfo(float).eps)
        grid = np.linspace(lower - padding, upper + padding, 400)
        axis.plot(
            grid,
            gaussian_kde(full)(grid),
            color="black",
            linewidth=1.8,
            label="Full data",
        )
        axis.plot(
            grid,
            gaussian_kde(subsampling)(grid),
            color="#3b82f6",
            linewidth=1.8,
            label="Subsampling, Rmax=100",
        )
        axis.set_title(names[index])
        axis.set_yticks([])
        axis.grid(alpha=0.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Development-only posterior marginal overlays: 1,500 retained draws",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(dataset_path, report_path):
    start_total = time.perf_counter()
    checksum_before = profile_p25.sha256_file(dataset_path)
    if checksum_before != preparation.EXPECTED_DATASET_SHA256:
        raise RuntimeError("Canonical dataset checksum does not match.")
    required = (
        preparation.MAP_CHECKPOINT,
        preparation.DEFAULT_REPORT,
        preparation.OUTPUT_DIRECTORY / "rmax_100_smoke_checkpoint.npz",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required checkpoints are unavailable: {missing}")

    progress("loading canonical dataset and validated checkpoints")
    dataset = rg.load_regression_GARCH11_dataset(dataset_path)
    profile_p25.verify_dataset(dataset)
    model = preparation.setup_model(dataset)
    with np.load(preparation.MAP_CHECKPOINT, allow_pickle=False) as checkpoint:
        if str(checkpoint["dataset_sha256"].item()) != checksum_before:
            raise RuntimeError("MAP checkpoint dataset checksum mismatch.")
        MAP = np.asarray(checkpoint["phi"], dtype=float).copy()
        Hessian_phi = np.asarray(checkpoint["Hessian_phi"], dtype=float).copy()
    preparation_report = json.loads(preparation.DEFAULT_REPORT.read_text())
    multiplier = float(
        preparation_report["proposal_calibration"]["selected_multiplier"]
    )
    with np.load(
        preparation.OUTPUT_DIRECTORY / "rmax_100_smoke_checkpoint.npz",
        allow_pickle=False,
    ) as tuning_checkpoint:
        c = float(tuning_checkpoint["c"])
        gamma = float(tuning_checkpoint["gamma"])
        m = int(tuning_checkpoint["m"])
        variance_tolerance = float(tuning_checkpoint["variance_tolerance"])
        checkpoint_probabilities = np.asarray(
            tuning_checkpoint["probabilities"], dtype=float
        ).copy()
    if m != 2 or multiplier != 0.2:
        raise RuntimeError("Validated m or proposal multiplier changed unexpectedly.")
    probabilities, _ = model["ns"]["TPD_probs"](
        profile_p25.T, 1000, gamma, 0.0
    )
    if not np.array_equal(probabilities, checkpoint_probabilities):
        raise RuntimeError("Reconstructed Rmax=100 probabilities differ from checkpoint.")

    negative_Hessian = -0.5 * (Hessian_phi + Hessian_phi.T)
    covariance = np.linalg.inv(negative_Hessian)
    covariance = model["ns"]["make_pos_def"](covariance, eps=1e-10)
    proposal_shape = (2.38**2 / model["d"]) * covariance
    proposal_covariance = multiplier * proposal_shape

    progress("reconstructing unsaved Taylor arrays")
    densities, gradients, Hessians, q_sum, taylor_seconds = (
        preparation.construct_taylor_quantities(model, MAP)
    )
    model["ns"].update({"t_star": 1000, "b": 0.0})
    names_original, names_transformed = rg.get_param_names_regression_GARCH11(
        profile_p25.P_REG
    )
    names = {"params_name": names_original, "trans_param_names": names_transformed}

    progress("running 2,000-iteration full-data development chain")
    np.random.seed(FULL_DATA_SEED)
    random.seed(FULL_DATA_SEED)
    full_chain = model["ns"]["sample_full_data_posterior_AM"](
        MAP,
        proposal_covariance,
        ITERATIONS,
        names,
        profile_p25.DATASET_ID,
        adapt_cov=False,
        adapt_scale=False,
        burn_in=1000,
    )
    full_diagnostics = preparation.chain_diagnostics(full_chain)

    estimator_params = {
        "m_opt": m,
        "c_opt": c,
        "gamma_opt": gamma,
        "t_star": 1000,
        "b": 0.0,
        "V_tolerance": variance_tolerance,
        "R_max": RMAX,
    }
    progress("running 2,000-iteration Rmax=100 subsampling development chain")
    np.random.seed(SUBSAMPLING_SEED)
    random.seed(SUBSAMPLING_SEED)
    subsampling_chain = model["ns"]["sample_subsampling_posterior_random_walk"](
        estimator_params,
        MAP,
        proposal_covariance,
        ITERATIONS,
        names,
        profile_p25.DATASET_ID,
        adapt=False,
    )
    subsampling_diagnostics = preparation.chain_diagnostics(subsampling_chain)

    full_retained = retained_draws(full_chain)
    subsampling_retained = retained_draws(subsampling_chain)
    comparison_rows, comparison_summary = posterior_comparison(
        full_retained, subsampling_retained
    )

    retained_u_max = np.asarray(subsampling_chain["u_max_prop"])[BURN_IN:]
    all_u_max = np.asarray(subsampling_chain["u_max_prop"])
    subsampling_diagnostics["u_max_all_iterations"] = (
        rmax_diagnostics.distribution_summary(all_u_max)
    )
    subsampling_diagnostics["u_max_retained_iterations"] = (
        rmax_diagnostics.distribution_summary(retained_u_max)
    )
    subsampling_diagnostics["u_max_retained_iterations"].update(
        {
            "mean_over_T": float(np.mean(retained_u_max) / profile_p25.T),
            "median_over_T": float(np.median(retained_u_max) / profile_p25.T),
        }
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_chain(FULL_CHAIN_PATH, full_chain, full_retained, FULL_DATA_SEED)
    save_subsampling_chain(
        SUBSAMPLING_CHAIN_PATH,
        subsampling_chain,
        subsampling_retained,
        SUBSAMPLING_SEED,
    )
    write_comparison_csv(comparison_rows, COMPARISON_CSV)
    make_overlay_plot(full_retained, subsampling_retained, OVERLAY_PNG)

    checksum_after = profile_p25.sha256_file(dataset_path)
    if checksum_after != checksum_before:
        raise RuntimeError("Canonical dataset changed during the diagnostic run.")
    report = {
        "development_only": True,
        "iterations": ITERATIONS,
        "burn_in": BURN_IN,
        "retained_draws": RETAINED_DRAWS,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "regenerated": False,
        },
        "fixed_setup": {
            "R_max": RMAX,
            "b": 0.0,
            "t_star": 1000,
            "m": m,
            "c": c,
            "gamma": gamma,
            "proposal_covariance_multiplier": multiplier,
            "Taylor_arrays_reconstructed_because_checkpoint_unavailable": True,
            "Taylor_construction_seconds": taylor_seconds,
            "preparation_recomputed": False,
        },
        "full_data_chain": {
            "seed": FULL_DATA_SEED,
            **full_diagnostics,
            "checkpoint": str(FULL_CHAIN_PATH.resolve()),
        },
        "subsampling_chain": {
            "seed": SUBSAMPLING_SEED,
            **subsampling_diagnostics,
            "checkpoint": str(SUBSAMPLING_CHAIN_PATH.resolve()),
        },
        "posterior_comparison": {
            "parameters": comparison_rows,
            "summary": comparison_summary,
            "csv": str(COMPARISON_CSV.resolve()),
        },
        "marginal_plot": str(OVERLAY_PNG.resolve()),
        "total_seconds": time.perf_counter() - start_total,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress(f"diagnostic comparison complete in {report['total_seconds']:.1f} seconds")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=profile_p25.DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    main(arguments.dataset, arguments.report)
