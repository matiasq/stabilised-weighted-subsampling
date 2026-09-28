"""Refine and calibrate the canonical p_reg=25 regression-GARCH experiment.

This driver never generates data.  It verifies and reloads the canonical archive,
refines the profiling MAP, calibrates a scalar random-walk covariance multiplier,
reruns TPD tuning, and performs short post-calibration smoke tests.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from autograd import grad, hessian
from autograd import jacobian as autograd_jacobian

import profile_regression_garch_p25 as profile_p25
import regression_garch as rg
import validate_regression_garch as validation


EXPECTED_DATASET_SHA256 = (
    "07a016de0093074e65d29f3a2a87fb5e9f4af0f38fe0f50ce10fa3e9fad295cd"
)
OUTPUT_DIRECTORY = Path(__file__).with_name("regression_garch_outputs") / (
    "p25_dataset_001"
)
MAP_CHECKPOINT = OUTPUT_DIRECTORY / "refined_map_checkpoint.npz"
DEFAULT_REPORT = OUTPUT_DIRECTORY / "preparation_report.json"

# Exact mode reported by the production profiling run.
PROFILE_MAP_THETA = np.array(
    [
        0.11913735188360544,
        0.11350975510240595,
        0.12133172264497469,
        0.11757707578474785,
        0.11740263594778792,
        0.11684879900489777,
        0.12023182092863496,
        0.12153202416491898,
        0.11815935606095918,
        0.1182935896957146,
        0.11591908671438772,
        0.12269009450155832,
        0.11916804938129127,
        0.11746733059797529,
        0.11641273513966564,
        0.11867854368244024,
        0.12068984026761272,
        0.11662911997047715,
        0.11769595790271575,
        0.11601442323844088,
        0.12446113390626129,
        0.12247074103980139,
        0.11542042286115844,
        0.11407261568361862,
        0.12250799384798801,
        0.07466260304631304,
        0.15654139694590144,
        0.7862549285270874,
        8.267939781044724,
    ]
)

CALIBRATION_MULTIPLIERS = (0.05, 0.10, 0.15, 0.20, 0.30)
CALIBRATION_ITERATIONS = 80
CALIBRATION_SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
TUNING_PILOT_ITERATIONS = 300
TUNING_PILOT_SEED = 20261201
FULL_SMOKE_ITERATIONS = 200
FULL_SMOKE_SEED = 20261202
SUBSAMPLING_SMOKE_ITERATIONS = 200
SUBSAMPLING_SMOKE_SEED = 20261203


def progress(message):
    print(f"[prepare-p25] {message}", file=sys.stderr, flush=True)


def setup_model(dataset):
    X = dataset["X"]
    y = dataset["y"]
    X_start = dataset["X_start"]
    y_start = np.array([dataset["y_start"]])
    sigma2_start = np.array([dataset["sigma2_start"]])
    ns = validation.load_existing_notebook_functions()
    h, hinv, hinv_p, hinv_pp, jacdiag, logdet, log_prior = (
        validation.configure_namespace_for_regression_model(
            ns, X, X_start, beta_reg_sd=1.0
        )
    )
    d = profile_p25.P_REG + 4
    ns.update(
        {
            "T": profile_p25.T,
            "p": 1,
            "q": 1,
            "y": y,
            "y_start": y_start,
            "sigma2_start": sigma2_start,
            "n_params": d,
            "GARCH_type": "Regression-GARCH",
            "error_type": "Student-t",
            "phiStar": None,
        }
    )

    def log_prior_phi(phi):
        return log_prior(hinv(phi), 1, 1) + logdet(phi)

    grad_log_prior_phi = grad(log_prior_phi)
    Hess_log_prior_phi = hessian(log_prior_phi)

    def log_posterior(phi):
        theta = hinv(phi)
        prior = log_prior(theta, 1, 1)
        if np.isinf(float(prior)):
            return -np.inf, True
        value = (
            ns["log_likelihood"](
                theta,
                profile_p25.T,
                1,
                1,
                y,
                y_start,
                sigma2_start,
            )
            + prior
            + logdet(phi)
        )
        return float(value), False

    ns["log_posterior"] = lambda phi, *unused: log_posterior(phi)
    return {
        "ns": ns,
        "X": X,
        "y": y,
        "X_start": X_start,
        "y_start": y_start,
        "sigma2_start": sigma2_start,
        "args": (
            profile_p25.T,
            1,
            1,
            y,
            y_start,
            sigma2_start,
        ),
        "h": h,
        "hinv": hinv,
        "hinv_p": hinv_p,
        "hinv_pp": hinv_pp,
        "jacdiag": jacdiag,
        "logdet": logdet,
        "log_prior": log_prior,
        "grad_log_prior_phi": grad_log_prior_phi,
        "Hess_log_prior_phi": Hess_log_prior_phi,
        "log_posterior": log_posterior,
        "d": d,
    }


def make_stationarity_map(model):
    h = model["h"]
    hinv = model["hinv"]
    jacdiag = model["jacdiag"]
    theta_to_psi, psi_to_theta, _ = (
        rg.initiate_regression_GARCH11_persistence_reparam_functions(
            p_reg=profile_p25.P_REG,
            h=h,
            hinv=hinv,
            jacdiag=jacdiag,
            np_backend=validation.np_autograd,
            delta=0.02,
        )
    )

    def phi_from_psi(psi):
        return h(psi_to_theta(psi))

    jacobian_phi_from_psi = autograd_jacobian(phi_from_psi)
    Hessian_phi_from_psi = autograd_jacobian(jacobian_phi_from_psi)
    return theta_to_psi, psi_to_theta, phi_from_psi, jacobian_phi_from_psi, Hessian_phi_from_psi


def posterior_phi_derivatives(phi, model):
    theta = np.asarray(model["hinv"](phi), dtype=float)
    log_densities = rg.log_densities_regression_GARCH11(
        theta,
        model["y"],
        model["X"],
        model["y_start"],
        model["X_start"],
        model["sigma2_start"],
    )
    grad_obs, Hess_obs = rg.grad_Hess_log_densities_regression_GARCH11(
        theta,
        model["y"],
        model["X"],
        model["y_start"],
        model["X_start"],
        model["sigma2_start"],
        h=model["h"],
        hinv_p=model["hinv_p"],
        hinv_pp=model["hinv_pp"],
        partials_1_deriv_t_l_t=model["ns"]["partials_1_deriv_t_l_t"],
        partials_2_deriv_t_l_t=model["ns"]["partials_2_deriv_t_l_t"],
    )
    objective = (
        float(np.sum(log_densities))
        + float(model["log_prior"](theta, 1, 1))
        + float(model["logdet"](phi))
    )
    gradient = np.sum(grad_obs, axis=0) + np.asarray(
        model["grad_log_prior_phi"](phi), dtype=float
    )
    Hessian = np.sum(Hess_obs, axis=0) + np.asarray(
        model["Hess_log_prior_phi"](phi), dtype=float
    )
    Hessian = 0.5 * (Hessian + Hessian.T)
    del grad_obs, Hess_obs
    gc.collect()
    return objective, gradient, Hessian


def refine_map(model, maximum_steps=6, psi_tolerance=1e-4, phi_tolerance=1e-3):
    (
        theta_to_psi,
        psi_to_theta,
        phi_from_psi,
        jacobian_phi_from_psi,
        Hessian_phi_from_psi,
    ) = make_stationarity_map(model)
    psi = np.asarray(theta_to_psi(PROFILE_MAP_THETA), dtype=float)
    history = []
    start = time.perf_counter()

    for iteration in range(maximum_steps + 1):
        phi = np.asarray(phi_from_psi(psi), dtype=float)
        objective, gradient_phi, Hessian_phi = posterior_phi_derivatives(phi, model)
        J = np.asarray(jacobian_phi_from_psi(psi), dtype=float)
        transform_Hessians = np.asarray(Hessian_phi_from_psi(psi), dtype=float)
        gradient_psi = J.T @ gradient_phi
        Hessian_psi = J.T @ Hessian_phi @ J + np.tensordot(
            gradient_phi, transform_Hessians, axes=(0, 0)
        )
        Hessian_psi = 0.5 * (Hessian_psi + Hessian_psi.T)
        record = {
            "iteration": iteration,
            "objective": objective,
            "gradient_infinity_norm_phi": float(np.max(np.abs(gradient_phi))),
            "gradient_infinity_norm_psi": float(np.max(np.abs(gradient_psi))),
            "persistence": float(
                model["hinv"](phi)[profile_p25.P_REG + 1]
                + model["hinv"](phi)[profile_p25.P_REG + 2]
            ),
        }
        history.append(record)
        progress(
            "MAP Newton iteration "
            f"{iteration}: objective={objective:.6f}, "
            f"|g_psi|_inf={record['gradient_infinity_norm_psi']:.3e}, "
            f"|g_phi|_inf={record['gradient_infinity_norm_phi']:.3e}"
        )
        if (
            record["gradient_infinity_norm_psi"] <= psi_tolerance
            and record["gradient_infinity_norm_phi"] <= phi_tolerance
        ):
            termination = "gradient tolerances satisfied"
            break
        if iteration == maximum_steps:
            termination = "maximum Newton steps reached"
            break

        negative_Hessian = -Hessian_psi
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(negative_Hessian)))
        if minimum_eigenvalue <= 1e-10:
            negative_Hessian = model["ns"]["make_pos_def"](
                negative_Hessian, eps=1e-8
            )
            record["negative_Hessian_repaired"] = True
        else:
            record["negative_Hessian_repaired"] = False
        record["negative_Hessian_minimum_eigenvalue"] = minimum_eigenvalue
        direction = np.linalg.solve(negative_Hessian, gradient_psi)
        directional_derivative = float(gradient_psi @ direction)
        if directional_derivative <= 0.0:
            raise RuntimeError("Newton direction is not an ascent direction.")

        step_size = 1.0
        accepted = False
        for backtrack in range(25):
            psi_candidate = psi + step_size * direction
            phi_candidate = np.asarray(phi_from_psi(psi_candidate), dtype=float)
            candidate_objective, outside = model["log_posterior"](phi_candidate)
            if (
                not outside
                and np.isfinite(candidate_objective)
                and candidate_objective
                >= objective + 1e-4 * step_size * directional_derivative
            ):
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            raise RuntimeError("Analytic Newton refinement could not find an ascent step.")
        record["step_size"] = step_size
        record["backtracks"] = backtrack
        record["predicted_directional_derivative"] = directional_derivative
        psi = psi_candidate
    else:
        raise AssertionError("Unreachable refinement loop state.")

    seconds = time.perf_counter() - start
    final_phi = np.asarray(phi_from_psi(psi), dtype=float)
    final_theta = np.asarray(psi_to_theta(psi), dtype=float)
    improvement = history[-1]["objective"] - history[0]["objective"]
    gradient_reduction = (
        history[0]["gradient_infinity_norm_phi"]
        / history[-1]["gradient_infinity_norm_phi"]
    )
    if gradient_reduction < 10.0:
        raise RuntimeError(
            "MAP refinement did not materially reduce the phi-gradient norm; "
            f"reduction factor was only {gradient_reduction:.3f}."
        )
    return {
        "phi": final_phi,
        "theta": final_theta,
        "psi": psi,
        "objective": history[-1]["objective"],
        "gradient_phi": gradient_phi,
        "gradient_psi": gradient_psi,
        "Hessian_phi": Hessian_phi,
        "Hessian_psi": Hessian_psi,
        "history": history,
        "termination": termination,
        "seconds": seconds,
        "objective_improvement": improvement,
        "phi_gradient_reduction_factor": gradient_reduction,
    }


def save_map_checkpoint(result, dataset_sha256):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MAP_CHECKPOINT,
        phi=result["phi"],
        theta=result["theta"],
        psi=result["psi"],
        Hessian_phi=result["Hessian_phi"],
        Hessian_psi=result["Hessian_psi"],
        gradient_phi=result["gradient_phi"],
        gradient_psi=result["gradient_psi"],
        objective=np.asarray(result["objective"]),
        dataset_sha256=np.asarray(dataset_sha256),
    )


def chain_diagnostics(chain):
    samples_phi = np.asarray(chain["samples_phi"])
    samples_theta = np.asarray(chain["samples_theta"])
    jumps = np.linalg.norm(np.diff(samples_phi, axis=0), axis=1)
    moved = jumps > 0.0
    longest_immobile = 0
    current_immobile = 0
    for did_move in moved:
        if did_move:
            current_immobile = 0
        else:
            current_immobile += 1
            longest_immobile = max(longest_immobile, current_immobile)
    return {
        "mean_acceptance_probability": float(np.mean(chain["alphas_MH"])),
        "realised_acceptance_rate": float(np.mean(moved)),
        "accepted_moves": int(np.sum(moved)),
        "unique_states": int(np.unique(samples_phi, axis=0).shape[0]),
        "longest_immobile_run": int(longest_immobile),
        "jump_norm_mean": float(np.mean(jumps)),
        "jump_norm_median_when_moved": float(np.median(jumps[moved]))
        if np.any(moved)
        else 0.0,
        "parameter_min": np.min(samples_theta, axis=0).tolist(),
        "parameter_max": np.max(samples_theta, axis=0).tolist(),
        "all_finite": bool(
            np.all(np.isfinite(samples_phi)) and np.all(np.isfinite(samples_theta))
        ),
        "maximum_persistence": float(
            np.max(
                samples_theta[:, profile_p25.P_REG + 1]
                + samples_theta[:, profile_p25.P_REG + 2]
            )
        ),
        "outside_support": int(chain["outside_support"]),
        "seconds": float(chain["run_time"]),
        "seconds_per_iteration": float(chain["run_time"] / len(jumps)),
    }


def run_full_chain(model, MAP, covariance, multiplier, iterations, seed, names):
    np.random.seed(seed)
    random.seed(seed)
    chain = model["ns"]["sample_full_data_posterior_AM"](
        MAP,
        multiplier * covariance,
        iterations,
        names,
        profile_p25.DATASET_ID,
        adapt_cov=False,
        adapt_scale=False,
        burn_in=1000,
    )
    return chain, chain_diagnostics(chain)


def construct_taylor_quantities(model, MAP):
    theta = np.asarray(model["hinv"](MAP), dtype=float)
    start = time.perf_counter()
    densities = rg.log_densities_regression_GARCH11(
        theta,
        model["y"],
        model["X"],
        model["y_start"],
        model["X_start"],
        model["sigma2_start"],
    )
    gradients, Hessians = rg.grad_Hess_log_densities_regression_GARCH11(
        theta,
        model["y"],
        model["X"],
        model["y_start"],
        model["X_start"],
        model["sigma2_start"],
        h=model["h"],
        hinv_p=model["hinv_p"],
        hinv_pp=model["hinv_pp"],
        partials_1_deriv_t_l_t=model["ns"]["partials_1_deriv_t_l_t"],
        partials_2_deriv_t_l_t=model["ns"]["partials_2_deriv_t_l_t"],
    )
    seconds = time.perf_counter() - start
    A = np.sum(densities)
    B = np.sum(gradients, axis=0)
    C = np.sum(Hessians, axis=0)

    def q_sum(phi):
        delta = phi - MAP
        return A + B @ delta + 0.5 * delta @ C @ delta

    model["ns"].update(
        {
            "phiStar": MAP,
            "dens_at_phiStar": densities,
            "grad_at_phiStar": gradients,
            "Hess_at_phiStar": Hessians,
            "q_sum": q_sum,
        }
    )
    return densities, gradients, Hessians, q_sum, seconds


def representative_variances(chain, model, MAP, Taylor, probabilities, m):
    densities, gradients, Hessians, _ = Taylor
    samples_phi = np.asarray(chain["samples_phi"])
    distances = np.linalg.norm(samples_phi - MAP, axis=1)
    target_distances = np.quantile(distances, [0.0, 0.25, 0.5, 0.75, 1.0])
    indices = []
    for target in target_distances:
        index = int(np.argmin(np.abs(distances - target)))
        if index not in indices:
            indices.append(index)
    results = []
    for index in indices:
        phi = samples_phi[index]
        theta = model["hinv"](phi)
        q_values = model["ns"]["eval_q_k"](
            phi, MAP, densities, gradients, Hessians, order=2
        )
        residuals = model["ns"]["log_densities"](theta, *model["args"]) - q_values
        analytic_variance = model["ns"]["variance_WDE"](
            residuals, probabilities, m
        )
        results.append(
            {
                "chain_index": index,
                "distance_from_MAP": float(distances[index]),
                "analytic_WDE_variance": float(analytic_variance),
                "stored_estimated_WDE_variance": float(
                    chain["sigma2_lhat_samples"][index]
                ),
            }
        )
    return results


def main(dataset_path, report_path):
    start_total = time.perf_counter()
    if not dataset_path.exists():
        raise FileNotFoundError(
            "The canonical dataset is missing; this driver will not regenerate it."
        )
    dataset_sha256_before = profile_p25.sha256_file(dataset_path)
    if dataset_sha256_before != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Canonical dataset SHA-256 does not match the required value.")
    progress("canonical dataset checksum verified; loading without regeneration")
    dataset = rg.load_regression_GARCH11_dataset(dataset_path)
    profile_p25.verify_dataset(dataset)
    model = setup_model(dataset)
    names_original, names_transformed = rg.get_param_names_regression_GARCH11(
        profile_p25.P_REG
    )
    names = {"params_name": names_original, "trans_param_names": names_transformed}

    progress("refining the profiling MAP with analytic Newton steps")
    map_result = refine_map(model)
    save_map_checkpoint(map_result, dataset_sha256_before)
    MAP = map_result["phi"]
    negative_Hessian_phi = -0.5 * (
        map_result["Hessian_phi"] + map_result["Hessian_phi"].T
    )
    covariance = np.linalg.inv(negative_Hessian_phi)
    covariance = model["ns"]["make_pos_def"](covariance, eps=1e-10)
    laplace_random_walk_shape = (2.38**2 / model["d"]) * covariance

    progress("calibrating scalar proposal covariance multipliers")
    calibration = []
    for multiplier, seed in zip(CALIBRATION_MULTIPLIERS, CALIBRATION_SEEDS):
        chain, diagnostics = run_full_chain(
            model,
            MAP,
            laplace_random_walk_shape,
            multiplier,
            CALIBRATION_ITERATIONS,
            seed,
            names,
        )
        diagnostics.update({"multiplier": multiplier, "seed": seed})
        calibration.append(diagnostics)
        progress(
            f"multiplier={multiplier:.3f}: realised acceptance="
            f"{diagnostics['realised_acceptance_rate']:.3f}, "
            f"mean alpha={diagnostics['mean_acceptance_probability']:.3f}"
        )
    eligible = [
        result
        for result in calibration
        if 0.20 <= result["realised_acceptance_rate"] <= 0.30
    ]
    selection_pool = eligible if eligible else calibration
    selected_calibration = min(
        selection_pool,
        key=lambda result: abs(result["realised_acceptance_rate"] - 0.25),
    )
    selected_multiplier = selected_calibration["multiplier"]
    progress(f"selected covariance multiplier {selected_multiplier:.3f}")

    progress("running improved full-data tuning pilot")
    pilot, pilot_diagnostics = run_full_chain(
        model,
        MAP,
        laplace_random_walk_shape,
        selected_multiplier,
        TUNING_PILOT_ITERATIONS,
        TUNING_PILOT_SEED,
        names,
    )

    progress("constructing Taylor quantities at the refined MAP")
    densities, gradients, Hessians, q_sum, taylor_seconds = (
        construct_taylor_quantities(model, MAP)
    )
    model["ns"].update({"t_star": 1000, "b": 0.0})
    tuning_slice = slice(20, TUNING_PILOT_ITERATIONS + 1, 10)
    selected_pilot_states = pilot["samples_phi"][tuning_slice]
    progress("rerunning TPD tuning with b=0, t_star=1000, Rmax=10")
    tuning_start = time.perf_counter()
    tuning_all = model["ns"]["run_calibrated_tuning_over_Rmax"](
        R_max_list=[10],
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
    )
    tuning_seconds = time.perf_counter() - tuning_start
    tuned = tuning_all[10]
    analytic_variance_phi_dagger = float(tuned["var_opt"])
    variance_constraint_satisfied = bool(
        analytic_variance_phi_dagger <= tuned["V_target"] * (1.0 + 1e-8)
    )
    if not variance_constraint_satisfied:
        raise RuntimeError("Tuned analytic WDE variance violates its variance tolerance.")

    estimator_params = {
        "m_opt": int(tuned["m_opt"]),
        "c_opt": float(tuned["c_opt"]),
        "gamma_opt": float(tuned["gamma_opt"]),
        "t_star": 1000,
        "b": 0.0,
        "V_tolerance": float(tuned["V_target"]),
        "R_max": 10,
    }
    progress("running post-calibration full-data smoke test")
    full_smoke, full_smoke_diagnostics = run_full_chain(
        model,
        MAP,
        laplace_random_walk_shape,
        selected_multiplier,
        FULL_SMOKE_ITERATIONS,
        FULL_SMOKE_SEED,
        names,
    )
    progress("running post-calibration subsampling smoke test")
    np.random.seed(SUBSAMPLING_SMOKE_SEED)
    random.seed(SUBSAMPLING_SMOKE_SEED)
    subsampling_smoke = model["ns"]["sample_subsampling_posterior_random_walk"](
        estimator_params,
        MAP,
        selected_multiplier * laplace_random_walk_shape,
        SUBSAMPLING_SMOKE_ITERATIONS,
        names,
        profile_p25.DATASET_ID,
        adapt=False,
    )
    subsampling_diagnostics = chain_diagnostics(subsampling_smoke)
    probabilities = np.asarray(tuned["probs_opt"], dtype=float)
    representative = representative_variances(
        subsampling_smoke,
        model,
        MAP,
        (densities, gradients, Hessians, q_sum),
        probabilities,
        max(2, int(tuned["m_opt"])),
    )
    sigma2_estimates = np.asarray(subsampling_smoke["sigma2_lhat_samples"])
    u_max = np.asarray(subsampling_smoke["u_max_prop"])

    dataset_sha256_after = profile_p25.sha256_file(dataset_path)
    if dataset_sha256_after != dataset_sha256_before:
        raise RuntimeError("Canonical dataset changed during preparation.")
    total_seconds = time.perf_counter() - start_total
    report = {
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256_before": dataset_sha256_before,
            "sha256_after": dataset_sha256_after,
            "regenerated": False,
        },
        "refined_MAP": {
            "starting_theta": PROFILE_MAP_THETA.tolist(),
            "theta": map_result["theta"].tolist(),
            "phi": map_result["phi"].tolist(),
            "objective": float(map_result["objective"]),
            "objective_improvement": float(map_result["objective_improvement"]),
            "gradient_infinity_norm_phi": float(
                np.max(np.abs(map_result["gradient_phi"]))
            ),
            "gradient_infinity_norm_psi": float(
                np.max(np.abs(map_result["gradient_psi"]))
            ),
            "phi_gradient_reduction_factor": float(
                map_result["phi_gradient_reduction_factor"]
            ),
            "persistence": float(
                map_result["theta"][profile_p25.P_REG + 1]
                + map_result["theta"][profile_p25.P_REG + 2]
            ),
            "seconds": float(map_result["seconds"]),
            "termination": map_result["termination"],
            "history": map_result["history"],
            "checkpoint": str(MAP_CHECKPOINT.resolve()),
        },
        "proposal_calibration": {
            "definition": (
                "proposal covariance = multiplier * (2.38^2/d) * "
                "Laplace covariance"
            ),
            "iterations_per_multiplier": CALIBRATION_ITERATIONS,
            "results": calibration,
            "selected_multiplier": float(selected_multiplier),
        },
        "tuning_pilot": {
            "iterations": TUNING_PILOT_ITERATIONS,
            "seed": TUNING_PILOT_SEED,
            **pilot_diagnostics,
            "selected_locations_requested": int(selected_pilot_states.shape[0]),
            "selected_locations_distinct": int(
                np.unique(selected_pilot_states, axis=0).shape[0]
            ),
            "selected_locations_nondegenerate": int(tuned["phi_candidates"].shape[0]),
        },
        "Taylor_control_variates": {
            "seconds": taylor_seconds,
            "density_shape": list(densities.shape),
            "gradient_shape": list(gradients.shape),
            "Hessian_shape": list(Hessians.shape),
        },
        "tuning": {
            "seconds": tuning_seconds,
            "R_max": 10,
            "b": 0.0,
            "t_star": 1000,
            "phi_dagger": np.asarray(tuned["phi_cross"]).tolist(),
            "theta_dagger": np.asarray(model["hinv"](tuned["phi_cross"])).tolist(),
            "c": float(tuned["c_opt"]),
            "gamma": float(tuned["gamma_opt"]),
            "m": int(tuned["m_opt"]),
            "variance_tolerance": float(tuned["V_target"]),
            "analytic_WDE_variance_at_phi_dagger": analytic_variance_phi_dagger,
            "binding_ratio": float(tuned["binding_ratio_final"]),
            "variance_constraint_satisfied": variance_constraint_satisfied,
            "expected_u_max": float(
                model["ns"]["expected_umax_safe"](
                    probabilities, int(tuned["m_opt"])
                )
            ),
        },
        "full_data_smoke": {
            "iterations": FULL_SMOKE_ITERATIONS,
            "seed": FULL_SMOKE_SEED,
            **full_smoke_diagnostics,
        },
        "subsampling_smoke": {
            "iterations": SUBSAMPLING_SMOKE_ITERATIONS,
            "seed": SUBSAMPLING_SMOKE_SEED,
            **subsampling_diagnostics,
            "u_max_quantiles": {
                str(q): float(np.quantile(u_max, q))
                for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
            },
            "u_max_mean": float(np.mean(u_max)),
            "estimated_WDE_variance_quantiles": {
                str(q): float(np.quantile(sigma2_estimates, q))
                for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
            },
            "estimated_WDE_variance_mean": float(np.mean(sigma2_estimates)),
            "representative_analytic_WDE_variances": representative,
        },
        "warnings": [],
        "total_seconds": total_seconds,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress(f"preparation complete in {total_seconds:.1f} seconds")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=profile_p25.DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        final_report = main(arguments.dataset, arguments.report)
        final_report["warnings"] = [str(item.message) for item in caught_warnings]
        arguments.report.write_text(
            json.dumps(final_report, indent=2, sort_keys=True) + "\n"
        )
