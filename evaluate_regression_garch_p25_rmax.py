"""Evaluate the discrete Rmax=100 and Rmax=1000 designs for p_reg=25.

The canonical dataset, refined MAP, Laplace proposal shape, calibrated proposal
multiplier, and full-data pilot specification are fixed by the preparation run.
Rmax=10 is read from that report and is never rerun here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import prepare_regression_garch_p25 as preparation
import profile_regression_garch_p25 as profile_p25
import regression_garch as rg


RMAX_VALUES = (100, 1000)
SMOKE_ITERATIONS = 200
COMMON_SMOKE_SEED = 20261301
NEAR_ZERO_THRESHOLD = 1e-10
DEFAULT_REPORT = preparation.OUTPUT_DIRECTORY / "rmax_100_1000_diagnostics.json"


def progress(message):
    print(f"[rmax-diagnostics] {message}", file=sys.stderr, flush=True)


def distribution_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def parameter_envelopes(chain):
    theta = np.asarray(chain["samples_theta"], dtype=float)
    return {
        "minimum": np.min(theta, axis=0).tolist(),
        "maximum": np.max(theta, axis=0).tolist(),
        "beta_reg_overall_minimum": float(np.min(theta[:, : profile_p25.P_REG])),
        "beta_reg_overall_maximum": float(np.max(theta[:, : profile_p25.P_REG])),
        "omega": [float(np.min(theta[:, 25])), float(np.max(theta[:, 25]))],
        "alpha": [float(np.min(theta[:, 26])), float(np.max(theta[:, 26]))],
        "beta_garch": [float(np.min(theta[:, 27])), float(np.max(theta[:, 27]))],
        "nu": [float(np.min(theta[:, 28])), float(np.max(theta[:, 28]))],
    }


def save_smoke_checkpoint(path, chain, tuning):
    np.savez_compressed(
        path,
        samples_phi=np.asarray(chain["samples_phi"]),
        samples_theta=np.asarray(chain["samples_theta"]),
        log_posthat_samples=np.asarray(chain["log_posthat_samples"]),
        sigma2_lhat_samples=np.asarray(chain["sigma2_lhat_samples"]),
        u_max_prop=np.asarray(chain["u_max_prop"]),
        alphas_MH=np.asarray(chain["alphas_MH"]),
        phi_dagger=np.asarray(tuning["phi_cross"]),
        probabilities=np.asarray(tuning["probs_opt"]),
        c=np.asarray(tuning["c_opt"]),
        gamma=np.asarray(tuning["gamma_opt"]),
        m=np.asarray(tuning["m_opt"]),
        variance_tolerance=np.asarray(tuning["V_target"]),
    )


def main(dataset_path, report_path):
    total_start = time.perf_counter()
    checksum_before = profile_p25.sha256_file(dataset_path)
    if checksum_before != preparation.EXPECTED_DATASET_SHA256:
        raise RuntimeError("Canonical dataset checksum is not the required value.")
    if not preparation.MAP_CHECKPOINT.exists():
        raise FileNotFoundError("Required refined-MAP checkpoint is unavailable.")
    if not preparation.DEFAULT_REPORT.exists():
        raise FileNotFoundError("Required preparation report is unavailable.")

    progress("loading canonical dataset and refined MAP checkpoint")
    dataset = rg.load_regression_GARCH11_dataset(dataset_path)
    profile_p25.verify_dataset(dataset)
    model = preparation.setup_model(dataset)
    preparation_report = json.loads(preparation.DEFAULT_REPORT.read_text())
    with np.load(preparation.MAP_CHECKPOINT, allow_pickle=False) as checkpoint:
        checkpoint_checksum = str(checkpoint["dataset_sha256"].item())
        if checkpoint_checksum != checksum_before:
            raise RuntimeError("Refined-MAP checkpoint belongs to a different dataset.")
        MAP = np.asarray(checkpoint["phi"], dtype=float).copy()
        Hessian_phi = np.asarray(checkpoint["Hessian_phi"], dtype=float).copy()

    selected_multiplier = float(
        preparation_report["proposal_calibration"]["selected_multiplier"]
    )
    if selected_multiplier != 0.2:
        raise RuntimeError("Unexpected calibrated proposal multiplier.")
    negative_Hessian = -0.5 * (Hessian_phi + Hessian_phi.T)
    covariance = np.linalg.inv(negative_Hessian)
    covariance = model["ns"]["make_pos_def"](covariance, eps=1e-10)
    proposal_shape = (2.38**2 / model["d"]) * covariance

    names_original, names_transformed = rg.get_param_names_regression_GARCH11(
        profile_p25.P_REG
    )
    names = {"params_name": names_original, "trans_param_names": names_transformed}

    # The preparation run did not store pilot samples, so reproduce only this
    # fixed-seed full-data pilot.  Verify its diagnostics before using it.
    progress("reproducing unavailable full-data pilot from recorded seed")
    pilot, pilot_diagnostics = preparation.run_full_chain(
        model,
        MAP,
        proposal_shape,
        selected_multiplier,
        preparation.TUNING_PILOT_ITERATIONS,
        preparation.TUNING_PILOT_SEED,
        names,
    )
    recorded_pilot = preparation_report["tuning_pilot"]
    if (
        pilot_diagnostics["accepted_moves"] != recorded_pilot["accepted_moves"]
        or pilot_diagnostics["unique_states"] != recorded_pilot["unique_states"]
        or not np.isclose(
            pilot_diagnostics["realised_acceptance_rate"],
            recorded_pilot["realised_acceptance_rate"],
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise RuntimeError("Reproduced full-data pilot does not match preparation diagnostics.")

    progress("reconstructing Taylor quantities at the fixed refined MAP")
    densities, gradients, Hessians, q_sum, taylor_seconds = (
        preparation.construct_taylor_quantities(model, MAP)
    )
    model["ns"].update({"t_star": 1000, "b": 0.0})
    tuning_slice = slice(20, preparation.TUNING_PILOT_ITERATIONS + 1, 10)

    results = {}
    for R_max in RMAX_VALUES:
        progress(f"tuning Rmax={R_max}")
        tuning_start = time.perf_counter()
        tuned = model["ns"]["run_calibrated_tuning_over_Rmax"](
            R_max_list=[R_max],
            MCMC_results={"MCMC_obj": pilot},
            MAP=MAP,
            phiStar=MAP,
            dens_at_phiStar=densities,
            grad_at_phiStar=gradients,
            Hess_at_phiStar=Hessians,
            args=model["args"],
            T=profile_p25.T,
            t_star_frac=0.01,
            b=0.0,
            m_min=2,
            pilot_slice=tuning_slice,
            verbose=False,
        )[R_max]
        tuning_seconds = time.perf_counter() - tuning_start
        analytic_variance = float(tuned["var_opt"])
        variance_tolerance = float(tuned["V_target"])
        probabilities = np.asarray(tuned["probs_opt"], dtype=float)
        expected_u_max = float(
            model["ns"]["expected_umax_safe"](
                probabilities, int(tuned["m_opt"])
            )
        )

        progress(f"running Rmax={R_max} common-seed smoke chain")
        np.random.seed(COMMON_SMOKE_SEED)
        random.seed(COMMON_SMOKE_SEED)
        estimator_params = {
            "m_opt": int(tuned["m_opt"]),
            "c_opt": float(tuned["c_opt"]),
            "gamma_opt": float(tuned["gamma_opt"]),
            "t_star": 1000,
            "b": 0.0,
            "V_tolerance": variance_tolerance,
            "R_max": R_max,
        }
        chain = model["ns"]["sample_subsampling_posterior_random_walk"](
            estimator_params,
            MAP,
            selected_multiplier * proposal_shape,
            SMOKE_ITERATIONS,
            names,
            profile_p25.DATASET_ID,
            adapt=False,
        )
        chain_diagnostics = preparation.chain_diagnostics(chain)
        u_max = np.asarray(chain["u_max_prop"], dtype=float)
        variance_estimates = np.asarray(chain["sigma2_lhat_samples"], dtype=float)
        u_summary = distribution_summary(u_max)
        u_summary.update(
            {
                "mean_over_T": float(np.mean(u_max) / profile_p25.T),
                "median_over_T": float(np.median(u_max) / profile_p25.T),
            }
        )
        variance_summary = distribution_summary(variance_estimates)
        variance_summary.update(
            {
                "fraction_exactly_zero": float(np.mean(variance_estimates == 0.0)),
                "near_zero_threshold": NEAR_ZERO_THRESHOLD,
                "fraction_numerically_near_zero": float(
                    np.mean(np.abs(variance_estimates) <= NEAR_ZERO_THRESHOLD)
                ),
            }
        )
        representative = preparation.representative_variances(
            chain,
            model,
            MAP,
            (densities, gradients, Hessians, q_sum),
            probabilities,
            max(2, int(tuned["m_opt"])),
        )
        checkpoint_path = preparation.OUTPUT_DIRECTORY / (
            f"rmax_{R_max}_smoke_checkpoint.npz"
        )
        save_smoke_checkpoint(checkpoint_path, chain, tuned)
        results[str(R_max)] = {
            "tuning": {
                "c": float(tuned["c_opt"]),
                "gamma": float(tuned["gamma_opt"]),
                "m": int(tuned["m_opt"]),
                "variance_tolerance": variance_tolerance,
                "analytic_WDE_variance_at_phi_dagger": analytic_variance,
                "analytic_variance_over_tolerance": analytic_variance
                / variance_tolerance,
                "constraint_satisfied": bool(
                    analytic_variance <= variance_tolerance * (1.0 + 1e-8)
                ),
                "expected_u_max": expected_u_max,
                "expected_u_max_over_T": expected_u_max / profile_p25.T,
                "seconds": tuning_seconds,
                "phi_dagger": np.asarray(tuned["phi_cross"]).tolist(),
                "theta_dagger": np.asarray(
                    model["hinv"](tuned["phi_cross"]), dtype=float
                ).tolist(),
                "distance_from_refined_MAP": float(
                    np.linalg.norm(np.asarray(tuned["phi_cross"]) - MAP)
                ),
                "nondegenerate_pilot_locations": int(
                    tuned["phi_candidates"].shape[0]
                ),
            },
            "smoke": {
                "iterations": SMOKE_ITERATIONS,
                "seed": COMMON_SMOKE_SEED,
                **chain_diagnostics,
                "parameter_envelopes": parameter_envelopes(chain),
                "u_max": u_summary,
                "stored_variance_estimates": variance_summary,
                "representative_exact_variances": representative,
                "checkpoint": str(checkpoint_path.resolve()),
            },
        }

    reference = preparation_report
    reference_tuning = reference["tuning"]
    reference_smoke = reference["subsampling_smoke"]
    comparison = [
        {
            "R_max": 10,
            "primary_candidate": False,
            "source": "existing preparation report; not rerun",
            "c": reference_tuning["c"],
            "gamma": reference_tuning["gamma"],
            "m": reference_tuning["m"],
            "variance_tolerance": reference_tuning["variance_tolerance"],
            "analytic_variance_at_phi_dagger": reference_tuning[
                "analytic_WDE_variance_at_phi_dagger"
            ],
            "analytic_variance_over_tolerance": reference_tuning["binding_ratio"],
            "expected_u_max": reference_tuning["expected_u_max"],
            "expected_u_max_over_T": reference_tuning["expected_u_max"]
            / profile_p25.T,
            "empirical_mean_u_max_over_T": reference_smoke["u_max_mean"]
            / profile_p25.T,
            "empirical_median_u_max_over_T": reference_smoke["u_max_quantiles"][
                "0.5"
            ]
            / profile_p25.T,
            "seconds_per_iteration": reference_smoke["seconds_per_iteration"],
            "realised_acceptance_rate": reference_smoke["realised_acceptance_rate"],
            "mean_acceptance_probability": reference_smoke[
                "mean_acceptance_probability"
            ],
            "variance_estimator_median": reference_smoke[
                "estimated_WDE_variance_quantiles"
            ]["0.5"],
            "variance_estimator_q95": reference_smoke[
                "estimated_WDE_variance_quantiles"
            ]["0.95"],
            "variance_estimator_maximum": reference_smoke[
                "estimated_WDE_variance_quantiles"
            ]["1.0"],
            "fraction_exactly_zero": None,
            "fraction_numerically_near_zero": None,
            "fraction_note": (
                "not stored in the existing report; recorded quantiles imply "
                "at least 25% exactly zero and at least 50% <=1e-10"
            ),
        }
    ]
    for R_max in RMAX_VALUES:
        result = results[str(R_max)]
        tuning = result["tuning"]
        smoke = result["smoke"]
        comparison.append(
            {
                "R_max": R_max,
                "primary_candidate": R_max == 100,
                "source": "new diagnostic run",
                "c": tuning["c"],
                "gamma": tuning["gamma"],
                "m": tuning["m"],
                "variance_tolerance": tuning["variance_tolerance"],
                "analytic_variance_at_phi_dagger": tuning[
                    "analytic_WDE_variance_at_phi_dagger"
                ],
                "analytic_variance_over_tolerance": tuning[
                    "analytic_variance_over_tolerance"
                ],
                "expected_u_max": tuning["expected_u_max"],
                "expected_u_max_over_T": tuning["expected_u_max_over_T"],
                "empirical_mean_u_max_over_T": smoke["u_max"]["mean_over_T"],
                "empirical_median_u_max_over_T": smoke["u_max"]["median_over_T"],
                "seconds_per_iteration": smoke["seconds_per_iteration"],
                "realised_acceptance_rate": smoke["realised_acceptance_rate"],
                "mean_acceptance_probability": smoke[
                    "mean_acceptance_probability"
                ],
                "variance_estimator_median": smoke["stored_variance_estimates"][
                    "median"
                ],
                "variance_estimator_q95": smoke["stored_variance_estimates"]["q95"],
                "variance_estimator_maximum": smoke["stored_variance_estimates"][
                    "maximum"
                ],
                "fraction_exactly_zero": smoke["stored_variance_estimates"][
                    "fraction_exactly_zero"
                ],
                "fraction_numerically_near_zero": smoke[
                    "stored_variance_estimates"
                ]["fraction_numerically_near_zero"],
                "fraction_note": None,
            }
        )

    checksum_after = profile_p25.sha256_file(dataset_path)
    if checksum_after != checksum_before:
        raise RuntimeError("Canonical dataset changed during diagnostics.")
    report = {
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "regenerated": False,
        },
        "fixed_setup": {
            "refined_MAP": MAP.tolist(),
            "proposal_covariance_multiplier": selected_multiplier,
            "b": 0.0,
            "t_star": 1000,
            "pilot_reproduced_because_checkpoint_unavailable": True,
            "pilot_diagnostics": pilot_diagnostics,
            "Taylor_construction_seconds": taylor_seconds,
            "Rmax_10_rerun": False,
        },
        "near_zero_definition": f"absolute estimate <= {NEAR_ZERO_THRESHOLD}",
        "results": results,
        "comparison": comparison,
        "total_seconds": time.perf_counter() - total_start,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress(f"diagnostics complete in {report['total_seconds']:.1f} seconds")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=profile_p25.DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    main(arguments.dataset, arguments.report)
