"""Development-only feasibility diagnostic for p_reg=50 regression-GARCH.

This driver creates at most one canonical dataset and refuses to overwrite it.
It uses the validated p_reg=25 model, optimisation, Taylor, tuning, and sampler
conventions, but it does not run manuscript-production chains or DC-BATS.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from autograd import grad, hessian
from autograd import jacobian as autograd_jacobian
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

import profile_regression_garch_p25 as profile_tools
import regression_garch as rg
import validate_regression_garch as validation


T = 100000
P_REG = 50
D = P_REG + 4
BURN_IN_SIMULATION = 1000
RHO = 0.5
DATASET_SEED = 20260922
DATASET_ID = "regression_garch_p50_dataset_001"
DATASET_PATH = Path(__file__).with_name("data") / f"{DATASET_ID}.npz"

OMEGA = 0.07134079
ALPHA = 0.16151262
BETA_GARCH = 0.78591371
NU = 8.0

CALIBRATION_MULTIPLIERS = (0.05, 0.10, 0.15, 0.20, 0.30)
CALIBRATION_ITERATIONS = 80
CALIBRATION_SEEDS = (20262101, 20262102, 20262103, 20262104, 20262105)
TUNING_PILOT_ITERATIONS = 300
TUNING_PILOT_SEED = 20262201
FULL_CHAIN_SEED = 20262301
SUBSAMPLING_CHAIN_SEED = 20262302
ITERATIONS = 2000
MCMC_BURN_IN = 500
RETAINED_DRAWS = ITERATIONS - MCMC_BURN_IN
RMAX = 100
T_STAR = 1000
B_OFFSET = 0.0

OUTPUT_DIRECTORY = Path(__file__).with_name("regression_garch_outputs") / (
    "p50_dataset_001"
)
DATASET_CREATION_REPORT = OUTPUT_DIRECTORY / "dataset_creation_report.json"
MAP_CHECKPOINT = OUTPUT_DIRECTORY / "refined_map_checkpoint.npz"
MAP_STAGE_REPORT = OUTPUT_DIRECTORY / "map_stage_report.json"
TAYLOR_STAGE_REPORT = OUTPUT_DIRECTORY / "taylor_stage_report.json"
REPORT_PATH = OUTPUT_DIRECTORY / "feasibility_report.json"
COMPARISON_CSV = OUTPUT_DIRECTORY / "posterior_comparison_54_parameters.csv"
OVERLAY_PNG = OUTPUT_DIRECTORY / "selected_marginal_overlays.png"
FULL_CHAIN_PATH = OUTPUT_DIRECTORY / "full_data_chain_2000.npz"
SUBSAMPLING_CHAIN_PATH = OUTPUT_DIRECTORY / "subsampling_chain_2000.npz"


def progress(message):
    print(f"[p50-feasibility] {message}", file=sys.stderr, flush=True)


def design_quantities():
    indices = np.arange(P_REG)
    Sigma = RHO ** np.abs(indices[:, None] - indices[None, :])
    beta_raw = np.ones(P_REG)
    beta_reg = beta_raw / np.sqrt(beta_raw @ Sigma @ beta_raw)
    return Sigma, beta_reg


def generate_canonical_dataset_once(path):
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing canonical dataset: {path}"
        )
    Sigma, beta_reg = design_quantities()
    rng = np.random.default_rng(DATASET_SEED)
    X_with_initial = rng.multivariate_normal(
        np.zeros(P_REG),
        Sigma,
        size=T + BURN_IN_SIMULATION + 1,
        method="cholesky",
    )
    t_draws_with_initial = rng.standard_t(
        NU, size=T + BURN_IN_SIMULATION + 1
    )
    sigma2_initial = OMEGA / (1.0 - ALPHA - BETA_GARCH)
    standardised_initial = t_draws_with_initial[0] * np.sqrt((NU - 2.0) / NU)
    z_initial = np.sqrt(sigma2_initial) * standardised_initial
    y_initial = float(X_with_initial[0] @ beta_reg + z_initial)

    full = rg.simulate_regression_GARCH11(
        X_with_initial[1:],
        beta_reg,
        OMEGA,
        ALPHA,
        BETA_GARCH,
        NU,
        y_start=y_initial,
        X_start=X_with_initial[0],
        sigma2_start=sigma2_initial,
        student_t_draws=t_draws_with_initial[1:],
    )
    start_index = BURN_IN_SIMULATION - 1
    retained = slice(BURN_IN_SIMULATION, BURN_IN_SIMULATION + T)
    simulation = {
        "X": full["X"][retained].copy(),
        "y": full["y"][retained].copy(),
        "z": full["z"][retained].copy(),
        "sigma2": full["sigma2"][retained].copy(),
        "student_t_draws": full["student_t_draws"][retained].copy(),
        "standardised_innovations": full["standardised_innovations"][
            retained
        ].copy(),
        "X_start": full["X"][start_index].copy(),
        "y_start": float(full["y"][start_index]),
        "z_start": float(full["z"][start_index]),
        "sigma2_start": float(full["sigma2"][start_index]),
        "student_t_draw_start": float(full["student_t_draws"][start_index]),
        "standardised_innovation_start": float(
            full["standardised_innovations"][start_index]
        ),
        "seed": DATASET_SEED,
        "dataset_id": DATASET_ID,
        "burn_in": BURN_IN_SIMULATION,
        "rho": RHO,
        "signal_variance": float(beta_reg @ Sigma @ beta_reg),
        "Sigma": Sigma,
        "params": {
            "p_reg": P_REG,
            "beta_reg": beta_reg,
            "omega": OMEGA,
            "alpha": ALPHA,
            "beta_garch": BETA_GARCH,
            "nu": NU,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    rg.save_regression_GARCH11_dataset(path, simulation)
    loaded = rg.load_regression_GARCH11_dataset(path)
    array_names = (
        "X",
        "y",
        "z",
        "sigma2",
        "student_t_draws",
        "standardised_innovations",
        "X_start",
        "Sigma",
    )
    arrays_exact = all(
        np.array_equal(simulation[name], loaded[name]) for name in array_names
    )
    scalars_exact = all(
        simulation[name] == loaded[name]
        for name in (
            "y_start",
            "z_start",
            "sigma2_start",
            "student_t_draw_start",
            "standardised_innovation_start",
            "seed",
            "dataset_id",
            "burn_in",
            "rho",
            "signal_variance",
        )
    )
    params_exact = np.array_equal(
        simulation["params"]["beta_reg"], loaded["params"]["beta_reg"]
    ) and all(
        simulation["params"][name] == loaded["params"][name]
        for name in ("p_reg", "omega", "alpha", "beta_garch", "nu")
    )
    if not (arrays_exact and scalars_exact and params_exact):
        raise AssertionError("Canonical dataset failed exact save/load round trip.")
    checks = verify_dataset(loaded)
    checksum = profile_tools.sha256_file(path)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    creation_report = {
        "dataset_id": DATASET_ID,
        "path": str(path.resolve()),
        "sha256": checksum,
        "seed": DATASET_SEED,
        "exact_save_load_round_trip": True,
        "checks": checks,
        "beta_reg": beta_reg.tolist(),
    }
    DATASET_CREATION_REPORT.write_text(
        json.dumps(creation_report, indent=2, sort_keys=True) + "\n"
    )
    return loaded, creation_report


def verify_dataset(dataset):
    Sigma, beta_expected = design_quantities()
    params = dataset["params"]
    if dataset["X"].shape != (T, P_REG) or dataset["y"].shape != (T,):
        raise AssertionError("Canonical dataset has unexpected dimensions.")
    if dataset["dataset_id"] != DATASET_ID or dataset["seed"] != DATASET_SEED:
        raise AssertionError("Canonical dataset identity or seed is incorrect.")
    if dataset["burn_in"] != BURN_IN_SIMULATION or dataset["rho"] != RHO:
        raise AssertionError("Canonical dataset design metadata is incorrect.")
    if not np.array_equal(dataset["Sigma"], Sigma):
        raise AssertionError("Stored covariance matrix differs from the design.")
    if not np.allclose(params["beta_reg"], beta_expected, rtol=0.0, atol=1e-15):
        raise AssertionError("Stored regression coefficients differ from the design.")
    for name, expected in {
        "p_reg": P_REG,
        "omega": OMEGA,
        "alpha": ALPHA,
        "beta_garch": BETA_GARCH,
        "nu": NU,
    }.items():
        if params[name] != expected:
            raise AssertionError(f"Stored parameter {name} differs from the design.")
    signal = dataset["X"] @ params["beta_reg"]
    identity_error = np.max(np.abs(dataset["y"] - signal - dataset["z"]))
    start_identity_error = abs(
        dataset["y_start"]
        - dataset["X_start"] @ params["beta_reg"]
        - dataset["z_start"]
    )
    start_innovation_error = abs(
        dataset["z_start"] / np.sqrt(dataset["sigma2_start"])
        - dataset["standardised_innovation_start"]
    )
    if max(identity_error, start_identity_error, start_innovation_error) > 1e-12:
        raise AssertionError("Response/residual/innovation identity check failed.")
    if not np.all(dataset["sigma2"] > 0.0):
        raise AssertionError("Conditional variance path is not strictly positive.")
    return {
        "designed_signal_variance": float(
            params["beta_reg"] @ Sigma @ params["beta_reg"]
        ),
        "realised_signal_variance": float(np.var(signal, ddof=1)),
        "standardised_innovation_variance": float(
            np.var(dataset["standardised_innovations"], ddof=1)
        ),
        "response_residual_identity_max_absolute": float(identity_error),
        "presample_response_residual_identity_absolute": float(
            start_identity_error
        ),
        "presample_innovation_identity_absolute": float(start_innovation_error),
        "sigma2_minimum": float(np.min(dataset["sigma2"])),
        "sigma2_maximum": float(np.max(dataset["sigma2"])),
        "all_sigma2_positive": True,
    }


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
    ns.update(
        {
            "T": T,
            "p": 1,
            "q": 1,
            "y": y,
            "y_start": y_start,
            "sigma2_start": sigma2_start,
            "n_params": D,
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
                theta, T, 1, 1, y, y_start, sigma2_start
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
        "args": (T, 1, 1, y, y_start, sigma2_start),
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
    }


def persistence_maps(model, autograd_backend):
    theta_to_psi, psi_to_theta, _ = (
        rg.initiate_regression_GARCH11_persistence_reparam_functions(
            p_reg=P_REG,
            h=model["h"],
            hinv=model["hinv"],
            jacdiag=model["jacdiag"],
            np_backend=autograd_backend,
            delta=0.02,
        )
    )
    return theta_to_psi, psi_to_theta


def gradient_log_posterior_phi(phi, model):
    return model["ns"]["grad_log_likelihood"](
        model["hinv"](phi), *model["args"]
    ) + np.asarray(model["grad_log_prior_phi"](phi), dtype=float)


def initial_map_optimisation(model, theta_start):
    theta_to_psi, psi_to_theta = persistence_maps(model, np)
    psi_start = np.asarray(theta_to_psi(theta_start), dtype=float)
    idx_alpha = P_REG + 1
    idx_beta = P_REG + 2

    def phi_from_psi(psi):
        return np.asarray(model["h"](psi_to_theta(psi)), dtype=float)

    def objective(psi):
        value, outside = model["log_posterior"](phi_from_psi(psi))
        return np.inf if outside else -value

    def objective_gradient(psi):
        theta = np.asarray(psi_to_theta(psi), dtype=float)
        phi = np.asarray(model["h"](theta), dtype=float)
        gradient_phi = gradient_log_posterior_phi(phi, model)
        sigmoid_eta = 1.0 / (1.0 + np.exp(-psi[idx_alpha]))
        persistence = theta[idx_alpha] + theta[idx_beta]
        gradient_psi = gradient_phi.copy()
        gradient_psi[idx_alpha] = (
            gradient_phi[idx_alpha] + gradient_phi[idx_beta]
        ) * (1.0 - sigmoid_eta)
        gradient_psi[idx_beta] = (
            gradient_phi[idx_alpha] * theta[idx_beta] / persistence
            - gradient_phi[idx_beta] * theta[idx_alpha] / persistence
        )
        return -gradient_psi

    bounds = (
        [(-20.0, 20.0)] * P_REG
        + [(-20.0, 5.0), (-20.0, 20.0), (-20.0, 20.0), (-10.0, 10.0)]
    )
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = minimize(
            objective,
            psi_start,
            jac=objective_gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150, "gtol": 1e-5, "ftol": 1e-10, "maxls": 50},
        )
    seconds = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f"Initial MAP optimisation failed: {result.message}")
    theta = np.asarray(psi_to_theta(result.x), dtype=float)
    phi = np.asarray(model["h"](theta), dtype=float)
    return {
        "theta": theta,
        "phi": phi,
        "psi": np.asarray(result.x, dtype=float),
        "objective": float(-result.fun),
        "gradient_phi": gradient_log_posterior_phi(phi, model),
        "gradient_psi": -np.asarray(result.jac, dtype=float),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "termination": str(result.message),
        "seconds": seconds,
        "warnings": [str(item.message) for item in caught],
    }


def posterior_phi_derivatives(phi, model):
    theta = np.asarray(model["hinv"](phi), dtype=float)
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
    objective = (
        float(np.sum(densities))
        + float(model["log_prior"](theta, 1, 1))
        + float(model["logdet"](phi))
    )
    gradient = np.sum(gradients, axis=0) + np.asarray(
        model["grad_log_prior_phi"](phi), dtype=float
    )
    Hessian = np.sum(Hessians, axis=0) + np.asarray(
        model["Hess_log_prior_phi"](phi), dtype=float
    )
    Hessian = 0.5 * (Hessian + Hessian.T)
    del densities, gradients, Hessians
    gc.collect()
    return objective, gradient, Hessian


def refine_map(model, theta_start, maximum_steps=6):
    theta_to_psi, psi_to_theta = persistence_maps(model, validation.np_autograd)

    def phi_from_psi(psi):
        return model["h"](psi_to_theta(psi))

    jacobian_phi_from_psi = autograd_jacobian(phi_from_psi)
    Hessian_phi_from_psi = autograd_jacobian(jacobian_phi_from_psi)
    psi = np.asarray(theta_to_psi(theta_start), dtype=float)
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
                model["hinv"](phi)[P_REG + 1]
                + model["hinv"](phi)[P_REG + 2]
            ),
        }
        history.append(record)
        progress(
            f"MAP Newton iteration {iteration}: objective={objective:.6f}, "
            f"|g_psi|_inf={record['gradient_infinity_norm_psi']:.3e}, "
            f"|g_phi|_inf={record['gradient_infinity_norm_phi']:.3e}"
        )
        if (
            record["gradient_infinity_norm_psi"] <= 1e-4
            and record["gradient_infinity_norm_phi"] <= 1e-3
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
        for backtrack in range(25):
            candidate_psi = psi + step_size * direction
            candidate_phi = np.asarray(phi_from_psi(candidate_psi), dtype=float)
            candidate_objective, outside = model["log_posterior"](candidate_phi)
            if (
                not outside
                and np.isfinite(candidate_objective)
                and candidate_objective
                >= objective + 1e-4 * step_size * directional_derivative
            ):
                break
            step_size *= 0.5
        else:
            raise RuntimeError("Newton refinement could not find an ascent step.")
        record["step_size"] = step_size
        record["backtracks"] = backtrack
        psi = candidate_psi

    if termination != "gradient tolerances satisfied":
        raise RuntimeError(
            "MAP refinement did not reach the validated gradient tolerances."
        )
    return {
        "theta": np.asarray(psi_to_theta(psi), dtype=float),
        "phi": np.asarray(phi_from_psi(psi), dtype=float),
        "psi": psi,
        "objective": objective,
        "gradient_phi": gradient_phi,
        "gradient_psi": gradient_psi,
        "Hessian_phi": Hessian_phi,
        "Hessian_psi": Hessian_psi,
        "termination": termination,
        "history": history,
        "seconds": time.perf_counter() - start,
    }


def save_map_checkpoint(result, dataset_sha256):
    np.savez_compressed(
        MAP_CHECKPOINT,
        theta=result["theta"],
        phi=result["phi"],
        psi=result["psi"],
        gradient_phi=result["gradient_phi"],
        gradient_psi=result["gradient_psi"],
        Hessian_phi=result["Hessian_phi"],
        Hessian_psi=result["Hessian_psi"],
        objective=np.asarray(result["objective"]),
        dataset_sha256=np.asarray(dataset_sha256),
    )


def construct_taylor_quantities(model, MAP):
    theta = np.asarray(model["hinv"](MAP), dtype=float)
    gc.collect()
    start = time.perf_counter()
    with profile_tools.MemorySampler() as memory:
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
    stored_bytes = densities.nbytes + gradients.nbytes + Hessians.nbytes
    memory_report = memory.report()
    memory_report.update(
        {
            "stored_arrays_MiB": stored_bytes / 1024**2,
            "largest_temporary_increment_MiB": max(
                0.0,
                memory.peak_bytes - memory.baseline_bytes - stored_bytes,
            )
            / 1024**2,
            "temporary_increment_over_Hessian_size": max(
                0.0,
                memory.peak_bytes - memory.baseline_bytes - stored_bytes,
            )
            / Hessians.nbytes,
        }
    )
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
            "t_star": T_STAR,
            "b": B_OFFSET,
        }
    )
    return densities, gradients, Hessians, q_sum, C, seconds, memory_report


def chain_diagnostics(chain):
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
        "mean_acceptance_probability": float(np.mean(chain["alphas_MH"])),
        "realised_acceptance_rate": float(np.mean(moved)),
        "accepted_moves": int(np.sum(moved)),
        "unique_states": int(np.unique(phi, axis=0).shape[0]),
        "longest_immobile_run": int(longest),
        "outside_support": int(chain["outside_support"]),
        "all_finite": bool(np.all(np.isfinite(phi)) and np.all(np.isfinite(theta))),
        "seconds": float(chain["run_time"]),
        "seconds_per_iteration": float(chain["run_time"] / len(jumps)),
        "maximum_persistence": float(
            np.max(theta[:, P_REG + 1] + theta[:, P_REG + 2])
        ),
    }


def run_full_chain(model, MAP, proposal_shape, multiplier, iterations, seed, names):
    np.random.seed(seed)
    random.seed(seed)
    chain = model["ns"]["sample_full_data_posterior_AM"](
        MAP,
        multiplier * proposal_shape,
        iterations,
        names,
        DATASET_ID,
        adapt_cov=False,
        adapt_scale=False,
        burn_in=1000,
    )
    return chain, chain_diagnostics(chain)


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


def retained_theta(chain):
    draws = np.asarray(chain["samples_theta"])[MCMC_BURN_IN + 1 : ITERATIONS + 1]
    if draws.shape != (RETAINED_DRAWS, D):
        raise AssertionError(f"Unexpected retained draw shape {draws.shape}.")
    return draws


def comparison_rows(full, subsampling):
    names = [f"beta_{j}" for j in range(1, P_REG + 1)] + [
        "omega",
        "alpha",
        "beta_garch",
        "nu",
    ]
    rows = []
    for index, name in enumerate(names):
        full_values = full[:, index]
        sub_values = subsampling[:, index]
        full_sd = float(np.std(full_values, ddof=1))
        sub_sd = float(np.std(sub_values, ddof=1))
        full_q = np.quantile(full_values, [0.025, 0.5, 0.975])
        sub_q = np.quantile(sub_values, [0.025, 0.5, 0.975])
        rows.append(
            {
                "parameter": name,
                "full_mean": float(np.mean(full_values)),
                "subsampling_mean": float(np.mean(sub_values)),
                "full_sd": full_sd,
                "subsampling_sd": sub_sd,
                "absolute_standardised_mean_difference": float(
                    abs(np.mean(sub_values) - np.mean(full_values)) / full_sd
                ),
                "sd_ratio_subsampling_over_full": sub_sd / full_sd,
                "full_q025": float(full_q[0]),
                "full_q50": float(full_q[1]),
                "full_q975": float(full_q[2]),
                "subsampling_q025": float(sub_q[0]),
                "subsampling_q50": float(sub_q[1]),
                "subsampling_q975": float(sub_q[2]),
            }
        )
    return rows


def summarize_comparison(rows):
    discrepancies = np.array(
        [row["absolute_standardised_mean_difference"] for row in rows]
    )
    ratios = np.array([row["sd_ratio_subsampling_over_full"] for row in rows])
    overlap = [
        max(row["full_q025"], row["subsampling_q025"])
        <= min(row["full_q975"], row["subsampling_q975"])
        for row in rows
    ]
    return {
        "maximum_absolute_standardised_mean_difference": float(
            np.max(discrepancies)
        ),
        "maximum_discrepancy_parameter": rows[int(np.argmax(discrepancies))][
            "parameter"
        ],
        "median_absolute_standardised_mean_difference": float(
            np.median(discrepancies)
        ),
        "minimum_sd_ratio": float(np.min(ratios)),
        "minimum_sd_ratio_parameter": rows[int(np.argmin(ratios))]["parameter"],
        "maximum_sd_ratio": float(np.max(ratios)),
        "maximum_sd_ratio_parameter": rows[int(np.argmax(ratios))]["parameter"],
        "all_marginal_95_intervals_overlap": bool(all(overlap)),
        "nonoverlapping_parameters": [
            row["parameter"] for row, does_overlap in zip(rows, overlap) if not does_overlap
        ],
    }


def derived_persistence_comparison(full, subsampling):
    full_values = full[:, P_REG + 1] + full[:, P_REG + 2]
    sub_values = subsampling[:, P_REG + 1] + subsampling[:, P_REG + 2]
    full_sd = float(np.std(full_values, ddof=1))
    sub_sd = float(np.std(sub_values, ddof=1))
    full_q = np.quantile(full_values, [0.025, 0.5, 0.975])
    sub_q = np.quantile(sub_values, [0.025, 0.5, 0.975])
    return {
        "full_mean": float(np.mean(full_values)),
        "subsampling_mean": float(np.mean(sub_values)),
        "full_sd": full_sd,
        "subsampling_sd": sub_sd,
        "absolute_standardised_mean_difference": float(
            abs(np.mean(sub_values) - np.mean(full_values)) / full_sd
        ),
        "sd_ratio_subsampling_over_full": sub_sd / full_sd,
        "full_q025": float(full_q[0]),
        "full_q50": float(full_q[1]),
        "full_q975": float(full_q[2]),
        "subsampling_q025": float(sub_q[0]),
        "subsampling_q50": float(sub_q[1]),
        "subsampling_q975": float(sub_q[2]),
    }


def make_overlay_plot(full, subsampling, path):
    selected = [0, 9, 19, 29, 39, 49, 50, 51, 52, 53, "persistence"]
    names = [f"beta_{j}" for j in range(1, P_REG + 1)] + [
        "omega",
        "alpha",
        "beta_garch",
        "nu",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for axis, item in zip(axes.flat, selected):
        if item == "persistence":
            full_values = full[:, P_REG + 1] + full[:, P_REG + 2]
            sub_values = subsampling[:, P_REG + 1] + subsampling[:, P_REG + 2]
            title = "persistence"
        else:
            full_values = full[:, item]
            sub_values = subsampling[:, item]
            title = names[item]
        pooled = np.concatenate([full_values, sub_values])
        lower, upper = np.quantile(pooled, [0.0025, 0.9975])
        padding = max(0.08 * (upper - lower), np.finfo(float).eps)
        grid = np.linspace(lower - padding, upper + padding, 400)
        axis.plot(grid, gaussian_kde(full_values)(grid), color="black", label="Full data")
        axis.plot(
            grid,
            gaussian_kde(sub_values)(grid),
            color="#3b82f6",
            label="Subsampling, Rmax=100",
        )
        axis.set_title(title)
        axis.set_yticks([])
        axis.grid(alpha=0.15)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Development-only p_reg=50 posterior overlays: 1,500 retained draws",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_chain(path, chain, retained, seed, subsampling=False):
    payload = {
        "samples_phi": np.asarray(chain["samples_phi"]),
        "samples_theta": np.asarray(chain["samples_theta"]),
        "retained_theta": retained,
        "alphas_MH": np.asarray(chain["alphas_MH"]),
        "seed": np.asarray(seed),
        "iterations": np.asarray(ITERATIONS),
        "burn_in": np.asarray(MCMC_BURN_IN),
    }
    if subsampling:
        payload.update(
            {
                "log_posthat_samples": np.asarray(chain["log_posthat_samples"]),
                "sigma2_lhat_samples": np.asarray(chain["sigma2_lhat_samples"]),
                "u_max_prop": np.asarray(chain["u_max_prop"]),
            }
        )
    np.savez_compressed(path, **payload)


def run_feasibility(dataset_path):
    total_start = time.perf_counter()
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Canonical p_reg=50 archive is absent; run with --dataset-only first."
        )
    creation = json.loads(DATASET_CREATION_REPORT.read_text())
    checksum_before = profile_tools.sha256_file(dataset_path)
    if checksum_before != creation["sha256"]:
        raise RuntimeError("Canonical dataset checksum differs from its creation record.")
    dataset = rg.load_regression_GARCH11_dataset(dataset_path)
    checks = verify_dataset(dataset)
    model = setup_model(dataset)
    theta_true = np.concatenate(
        [dataset["params"]["beta_reg"], [OMEGA, ALPHA, BETA_GARCH, NU]]
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    progress("running initial and refined MAP optimisation")
    with profile_tools.MemorySampler() as map_memory:
        initial = initial_map_optimisation(model, theta_true)
        refined = refine_map(model, initial["theta"])
    save_map_checkpoint(refined, checksum_before)
    map_memory_report = map_memory.report()
    map_stage_report = {
        "dataset_sha256": checksum_before,
        "initial_objective": initial["objective"],
        "initial_gradient_infinity_norm_phi": float(
            np.max(np.abs(initial["gradient_phi"]))
        ),
        "initial_gradient_infinity_norm_psi": float(
            np.max(np.abs(initial["gradient_psi"]))
        ),
        "initial_iterations": initial["iterations"],
        "initial_function_evaluations": initial["function_evaluations"],
        "initial_termination": initial["termination"],
        "initial_seconds": initial["seconds"],
        "initial_warnings": initial["warnings"],
        "refined_objective": refined["objective"],
        "refined_gradient_infinity_norm_phi": float(
            np.max(np.abs(refined["gradient_phi"]))
        ),
        "refined_gradient_infinity_norm_psi": float(
            np.max(np.abs(refined["gradient_psi"]))
        ),
        "refined_theta": refined["theta"].tolist(),
        "refined_phi": refined["phi"].tolist(),
        "refined_persistence": float(
            refined["theta"][P_REG + 1] + refined["theta"][P_REG + 2]
        ),
        "refined_termination": refined["termination"],
        "refined_history": refined["history"],
        "refinement_seconds": refined["seconds"],
        "total_seconds": initial["seconds"] + refined["seconds"],
        "memory": map_memory_report,
    }
    MAP_STAGE_REPORT.write_text(
        json.dumps(map_stage_report, indent=2, sort_keys=True) + "\n"
    )

    progress("constructing complete second-order Taylor arrays")
    densities, gradients, Hessians, q_sum, C, taylor_seconds, taylor_memory = (
        construct_taylor_quantities(model, refined["phi"])
    )
    array_details = {
        "density": {"shape": list(densities.shape), "bytes": int(densities.nbytes)},
        "gradient": {"shape": list(gradients.shape), "bytes": int(gradients.nbytes)},
        "Hessian": {"shape": list(Hessians.shape), "bytes": int(Hessians.nbytes)},
    }
    TAYLOR_STAGE_REPORT.write_text(
        json.dumps(
            {
                "dataset_sha256": checksum_before,
                "arrays": array_details,
                "total_stored_bytes": int(
                    densities.nbytes + gradients.nbytes + Hessians.nbytes
                ),
                "total_stored_MiB": (
                    densities.nbytes + gradients.nbytes + Hessians.nbytes
                )
                / 1024**2,
                "seconds": taylor_seconds,
                "memory": taylor_memory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    Hessian_log_posterior = C + np.asarray(
        model["Hess_log_prior_phi"](refined["phi"]), dtype=float
    )
    Hessian_log_posterior = 0.5 * (
        Hessian_log_posterior + Hessian_log_posterior.T
    )
    covariance = np.linalg.inv(-Hessian_log_posterior)
    covariance = model["ns"]["make_pos_def"](covariance, eps=1e-10)
    proposal_shape = (2.38**2 / D) * covariance
    names_original, names_transformed = rg.get_param_names_regression_GARCH11(P_REG)
    names = {"params_name": names_original, "trans_param_names": names_transformed}

    progress("calibrating scalar proposal multiplier")
    calibration = []
    for multiplier, seed in zip(CALIBRATION_MULTIPLIERS, CALIBRATION_SEEDS):
        chain, diagnostics = run_full_chain(
            model,
            refined["phi"],
            proposal_shape,
            multiplier,
            CALIBRATION_ITERATIONS,
            seed,
            names,
        )
        diagnostics.update({"multiplier": multiplier, "seed": seed})
        calibration.append(diagnostics)
        progress(
            f"multiplier={multiplier:.2f}: acceptance="
            f"{diagnostics['realised_acceptance_rate']:.3f}"
        )
    eligible = [
        result
        for result in calibration
        if 0.20 <= result["realised_acceptance_rate"] <= 0.30
    ]
    calibration_target_met = bool(eligible)
    selection_pool = eligible if eligible else calibration
    selected = min(
        selection_pool,
        key=lambda result: abs(result["realised_acceptance_rate"] - 0.25),
    )
    multiplier = float(selected["multiplier"])
    if not calibration_target_met:
        progress(
            "no prespecified multiplier reached the 0.20-0.30 acceptance target; "
            "using the existing closest-to-0.25 fallback rule"
        )

    progress("running full-data tuning pilot")
    pilot, pilot_diagnostics = run_full_chain(
        model,
        refined["phi"],
        proposal_shape,
        multiplier,
        TUNING_PILOT_ITERATIONS,
        TUNING_PILOT_SEED,
        names,
    )
    tuning_slice = slice(20, TUNING_PILOT_ITERATIONS + 1, 10)
    selected_pilot_states = pilot["samples_phi"][tuning_slice]

    progress("tuning Rmax=100 with b=0 and t_star=1000")
    tuning_start = time.perf_counter()
    tuned = model["ns"]["run_calibrated_tuning_over_Rmax"](
        R_max_list=[RMAX],
        MCMC_results={"MCMC_obj": pilot},
        MAP=refined["phi"],
        phiStar=refined["phi"],
        dens_at_phiStar=densities,
        grad_at_phiStar=gradients,
        Hess_at_phiStar=Hessians,
        args=model["args"],
        T=T,
        t_star_frac=0.01,
        b=B_OFFSET,
        m_min=2,
        pilot_slice=tuning_slice,
        verbose=False,
    )[RMAX]
    tuning_seconds = time.perf_counter() - tuning_start
    probabilities = np.asarray(tuned["probs_opt"], dtype=float)
    expected_u_max = float(
        model["ns"]["expected_umax_safe"](
            probabilities, int(tuned["m_opt"])
        )
    )
    analytic_variance = float(tuned["var_opt"])
    variance_tolerance = float(tuned["V_target"])

    estimator_params = {
        "m_opt": int(tuned["m_opt"]),
        "c_opt": float(tuned["c_opt"]),
        "gamma_opt": float(tuned["gamma_opt"]),
        "t_star": T_STAR,
        "b": B_OFFSET,
        "V_tolerance": variance_tolerance,
        "R_max": RMAX,
    }

    progress("running 2,000-iteration full-data development chain")
    full_chain, full_diagnostics = run_full_chain(
        model,
        refined["phi"],
        proposal_shape,
        multiplier,
        ITERATIONS,
        FULL_CHAIN_SEED,
        names,
    )
    progress("running 2,000-iteration subsampling development chain")
    np.random.seed(SUBSAMPLING_CHAIN_SEED)
    random.seed(SUBSAMPLING_CHAIN_SEED)
    sub_chain = model["ns"]["sample_subsampling_posterior_random_walk"](
        estimator_params,
        refined["phi"],
        multiplier * proposal_shape,
        ITERATIONS,
        names,
        DATASET_ID,
        adapt=False,
    )
    sub_diagnostics = chain_diagnostics(sub_chain)
    retained_u_max = np.asarray(sub_chain["u_max_prop"])[MCMC_BURN_IN:]
    u_max_summary = distribution_summary(retained_u_max)
    u_max_summary.update(
        {
            "mean_over_T": float(np.mean(retained_u_max) / T),
            "median_over_T": float(np.median(retained_u_max) / T),
        }
    )

    full_retained = retained_theta(full_chain)
    sub_retained = retained_theta(sub_chain)
    rows = comparison_rows(full_retained, sub_retained)
    comparison_summary = summarize_comparison(rows)
    persistence = derived_persistence_comparison(full_retained, sub_retained)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_overlay_plot(full_retained, sub_retained, OVERLAY_PNG)
    save_chain(
        FULL_CHAIN_PATH,
        full_chain,
        full_retained,
        FULL_CHAIN_SEED,
        subsampling=False,
    )
    save_chain(
        SUBSAMPLING_CHAIN_PATH,
        sub_chain,
        sub_retained,
        SUBSAMPLING_CHAIN_SEED,
        subsampling=True,
    )

    checksum_after = profile_tools.sha256_file(dataset_path)
    if checksum_after != checksum_before:
        raise RuntimeError("Canonical dataset changed during feasibility run.")
    report = {
        "development_only": True,
        "dataset": {
            **creation,
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "checks_reverified": checks,
            "regenerated_during_feasibility": False,
        },
        "MAP": {
            "initial_optimisation": {
                "objective": initial["objective"],
                "gradient_infinity_norm_phi": float(
                    np.max(np.abs(initial["gradient_phi"]))
                ),
                "gradient_infinity_norm_psi": float(
                    np.max(np.abs(initial["gradient_psi"]))
                ),
                "iterations": initial["iterations"],
                "function_evaluations": initial["function_evaluations"],
                "termination": initial["termination"],
                "seconds": initial["seconds"],
                "warnings": initial["warnings"],
            },
            "refined": {
                "objective": refined["objective"],
                "gradient_infinity_norm_phi": float(
                    np.max(np.abs(refined["gradient_phi"]))
                ),
                "gradient_infinity_norm_psi": float(
                    np.max(np.abs(refined["gradient_psi"]))
                ),
                "theta": refined["theta"].tolist(),
                "phi": refined["phi"].tolist(),
                "persistence": float(
                    refined["theta"][P_REG + 1]
                    + refined["theta"][P_REG + 2]
                ),
                "termination": refined["termination"],
                "history": refined["history"],
                "seconds": refined["seconds"],
            },
            "total_seconds": initial["seconds"] + refined["seconds"],
            "memory": map_memory_report,
            "checkpoint": str(MAP_CHECKPOINT.resolve()),
        },
        "Taylor": {
            "arrays": array_details,
            "total_stored_bytes": int(
                densities.nbytes + gradients.nbytes + Hessians.nbytes
            ),
            "total_stored_MiB": (
                densities.nbytes + gradients.nbytes + Hessians.nbytes
            )
            / 1024**2,
            "seconds": taylor_seconds,
            "memory": taylor_memory,
        },
        "proposal_calibration": {
            "definition": (
                "proposal covariance = multiplier * (2.38^2/d) * "
                "Laplace covariance"
            ),
            "iterations_per_multiplier": CALIBRATION_ITERATIONS,
            "results": calibration,
            "selected_multiplier": multiplier,
            "target_acceptance_interval": [0.20, 0.30],
            "target_interval_met": calibration_target_met,
            "selection_used_existing_closest_fallback": not calibration_target_met,
        },
        "pilot": {
            "iterations": TUNING_PILOT_ITERATIONS,
            "seed": TUNING_PILOT_SEED,
            **pilot_diagnostics,
            "selected_locations_requested": int(selected_pilot_states.shape[0]),
            "selected_locations_distinct": int(
                np.unique(selected_pilot_states, axis=0).shape[0]
            ),
            "selected_locations_nondegenerate": int(
                tuned["phi_candidates"].shape[0]
            ),
        },
        "tuning": {
            "R_max": RMAX,
            "b": B_OFFSET,
            "t_star": T_STAR,
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
            "expected_u_max_over_T": expected_u_max / T,
            "phi_dagger": np.asarray(tuned["phi_cross"]).tolist(),
            "distance_phi_dagger_from_MAP": float(
                np.linalg.norm(np.asarray(tuned["phi_cross"]) - refined["phi"])
            ),
            "seconds": tuning_seconds,
        },
        "chains": {
            "iterations": ITERATIONS,
            "burn_in": MCMC_BURN_IN,
            "retained_draws": RETAINED_DRAWS,
            "full_data": {
                "seed": FULL_CHAIN_SEED,
                **full_diagnostics,
                "checkpoint": str(FULL_CHAIN_PATH.resolve()),
            },
            "subsampling": {
                "seed": SUBSAMPLING_CHAIN_SEED,
                **sub_diagnostics,
                "u_max_retained": u_max_summary,
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
        "total_seconds": time.perf_counter() - total_start,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress(f"feasibility run complete in {report['total_seconds']:.1f} seconds")
    print(json.dumps(report, indent=2, sort_keys=True))


def main(dataset_path, dataset_only):
    if dataset_only:
        progress("creating canonical p_reg=50 dataset once")
        _, report = generate_canonical_dataset_once(dataset_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    run_feasibility(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--dataset-only", action="store_true")
    arguments = parser.parse_args()
    main(arguments.dataset, arguments.dataset_only)
