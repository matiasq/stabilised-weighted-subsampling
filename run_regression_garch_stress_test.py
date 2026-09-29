"""Restart-safe Version-3 regression-GARCH dimensional stress test.

This driver orchestrates the validated implementation in ``regression_garch.py``
and the existing notebook functions loaded by ``validate_regression_garch.py``.
It intentionally does not execute on import.  Production work is performed only
when explicitly invoked from the command line or the Version-3 notebook.

The ten datasets are processed one at a time.  Every expensive stage writes a
dataset-hash-bound checkpoint before the next stage begins.  Dataset 001 is
never generated here unless it is absent, in which case execution fails: those
two archives are immutable canonical inputs with pre-specified SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib

# Preserve Jupyter's interactive backend when this API is imported by the notebook.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from autograd import grad, hessian
from autograd import jacobian as autograd_jacobian
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

import regression_garch as rg
import validate_regression_garch as validation


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "regression_garch_experiment_config.json"
DATA_DIRECTORY = ROOT / "data"
OUTPUT_ROOT = ROOT / "regression_garch_outputs"


def require_writable_output_root():
    if OUTPUT_ROOT.exists() and not os.access(OUTPUT_ROOT, os.W_OK | os.X_OK):
        raise PermissionError(
            f"{OUTPUT_ROOT} is not writable by the current process. Preserve its "
            "contents and make the directory writable by the notebook user before "
            "starting dataset preparation or production inference."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_config(path=CONFIG_PATH):
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported regression-GARCH experiment configuration.")
    mcmc = config["mcmc"]
    if mcmc["b"] != 0.0 or mcmc["R_max"] != 100 or mcmc["t_star"] != 1000:
        raise ValueError("Production configuration must use b=0, R_max=100, t_star=1000.")
    if mcmc["iterations"] - mcmc["burn_in"] != mcmc["retained_draws"]:
        raise ValueError("MCMC iteration, burn-in, and retained-draw settings disagree.")
    for p_text, case in config["cases"].items():
        p_reg = int(p_text)
        if case["total_dimension"] != p_reg + 4:
            raise ValueError(f"Incorrect total dimension for p={p_reg}.")
        if sorted(case["dataset_seeds"]) != ["001", "002", "003", "004", "005"]:
            raise ValueError(f"Exactly five fixed datasets are required for p={p_reg}.")
    return config


def progress(message):
    print(f"[regression-garch-production] {message}", file=sys.stderr, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array):
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def atomic_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **payload)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.save(output, array, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def dataset_name(p_reg, dataset_id):
    return f"regression_garch_p{p_reg}_dataset_{dataset_id}"


def dataset_path(p_reg, dataset_id):
    return DATA_DIRECTORY / f"{dataset_name(p_reg, dataset_id)}.npz"


def output_directory(p_reg, dataset_id):
    return OUTPUT_ROOT / f"p{p_reg}" / f"dataset_{dataset_id}"


def design_quantities(p_reg, rho):
    indices = np.arange(p_reg)
    Sigma = rho ** np.abs(indices[:, None] - indices[None, :])
    beta_raw = np.ones(p_reg)
    beta_reg = beta_raw / np.sqrt(beta_raw @ Sigma @ beta_raw)
    return Sigma, beta_reg


def simulate_dataset_once(path, p_reg, dataset_id, seed, config):
    """Generate only noncanonical datasets, refusing every overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
    if dataset_id == "001":
        raise FileNotFoundError(
            f"Immutable canonical dataset 001 is missing and will not be regenerated: {path}"
        )
    dgp = config["data_generating_process"]
    T = int(dgp["T"])
    burn_in = int(dgp["simulation_burn_in"])
    Sigma, beta_reg = design_quantities(p_reg, float(dgp["rho"]))
    rng = np.random.default_rng(seed)
    X_all = rng.multivariate_normal(
        np.zeros(p_reg), Sigma, size=T + burn_in + 1, method="cholesky"
    )
    t_all = rng.standard_t(float(dgp["nu"]), size=T + burn_in + 1)
    omega = float(dgp["omega"])
    alpha = float(dgp["alpha"])
    beta_garch = float(dgp["beta_garch"])
    nu = float(dgp["nu"])
    sigma2_initial = omega / (1.0 - alpha - beta_garch)
    innovation_initial = t_all[0] * np.sqrt((nu - 2.0) / nu)
    z_initial = np.sqrt(sigma2_initial) * innovation_initial
    y_initial = float(X_all[0] @ beta_reg + z_initial)
    full = rg.simulate_regression_GARCH11(
        X_all[1:], beta_reg, omega, alpha, beta_garch, nu,
        y_start=y_initial, X_start=X_all[0], sigma2_start=sigma2_initial,
        student_t_draws=t_all[1:],
    )
    start_index = burn_in - 1
    retained = slice(burn_in, burn_in + T)
    simulation = {
        "X": full["X"][retained].copy(),
        "y": full["y"][retained].copy(),
        "z": full["z"][retained].copy(),
        "sigma2": full["sigma2"][retained].copy(),
        "student_t_draws": full["student_t_draws"][retained].copy(),
        "standardised_innovations": full["standardised_innovations"][retained].copy(),
        "X_start": full["X"][start_index].copy(),
        "y_start": float(full["y"][start_index]),
        "z_start": float(full["z"][start_index]),
        "sigma2_start": float(full["sigma2"][start_index]),
        "student_t_draw_start": float(full["student_t_draws"][start_index]),
        "standardised_innovation_start": float(full["standardised_innovations"][start_index]),
        "seed": int(seed),
        "dataset_id": dataset_name(p_reg, dataset_id),
        "burn_in": burn_in,
        "rho": float(dgp["rho"]),
        "signal_variance": float(beta_reg @ Sigma @ beta_reg),
        "Sigma": Sigma,
        "params": {
            "p_reg": p_reg, "beta_reg": beta_reg, "omega": omega,
            "alpha": alpha, "beta_garch": beta_garch, "nu": nu,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    rg.save_regression_GARCH11_dataset(temporary, simulation)
    # A complete load must succeed before the immutable final filename appears.
    rg.load_regression_GARCH11_dataset(temporary)
    temporary.replace(path)


def verify_dataset(dataset, p_reg, dataset_id, seed, config):
    dgp = config["data_generating_process"]
    T = int(dgp["T"])
    Sigma, beta_reg = design_quantities(p_reg, float(dgp["rho"]))
    if dataset["X"].shape != (T, p_reg) or dataset["y"].shape != (T,):
        raise AssertionError("Dataset dimensions differ from the pre-specified design.")
    if dataset["dataset_id"] != dataset_name(p_reg, dataset_id):
        raise AssertionError("Dataset identifier differs from its filename/configuration.")
    if dataset["seed"] != seed or dataset["burn_in"] != dgp["simulation_burn_in"]:
        raise AssertionError("Dataset seed or burn-in metadata differs from configuration.")
    if dataset["rho"] != dgp["rho"] or not np.array_equal(dataset["Sigma"], Sigma):
        raise AssertionError("Dataset covariance design differs from configuration.")
    if not np.allclose(dataset["params"]["beta_reg"], beta_reg, rtol=0, atol=1e-15):
        raise AssertionError("Dense equal regression coefficients differ from the design.")
    for name in ("omega", "alpha", "beta_garch", "nu"):
        if dataset["params"][name] != dgp[name]:
            raise AssertionError(f"Dataset true {name} differs from configuration.")
    residual_error = np.max(
        np.abs(dataset["y"] - dataset["X"] @ beta_reg - dataset["z"])
    )
    start_error = abs(
        dataset["y_start"] - dataset["X_start"] @ beta_reg - dataset["z_start"]
    )
    if max(residual_error, start_error) > 1e-12 or not np.all(dataset["sigma2"] > 0):
        raise AssertionError("Dataset residual identity or variance checks failed.")
    return {
        "response_residual_identity_max_absolute": float(residual_error),
        "presample_residual_identity_absolute": float(start_error),
        "designed_signal_variance": float(beta_reg @ Sigma @ beta_reg),
        "realised_signal_variance": float(np.var(dataset["X"] @ beta_reg, ddof=1)),
        "all_sigma2_positive": True,
    }


def ensure_dataset(p_reg, dataset_id, config):
    case = config["cases"][str(p_reg)]
    seed = int(case["dataset_seeds"][dataset_id])
    path = dataset_path(p_reg, dataset_id)
    if not path.exists():
        progress(
            f"p={p_reg} (d={p_reg + 4}) dataset {dataset_id}: "
            f"creating missing dataset with fixed seed {seed}"
        )
        simulate_dataset_once(path, p_reg, dataset_id, seed, config)
    else:
        progress(
            f"p={p_reg} (d={p_reg + 4}) dataset {dataset_id}: verifying existing dataset"
        )
    checksum = sha256_file(path)
    if dataset_id == "001" and checksum != case["canonical_dataset_001_sha256"]:
        raise RuntimeError(f"Canonical p={p_reg} dataset 001 SHA-256 mismatch.")
    dataset = rg.load_regression_GARCH11_dataset(path)
    checks = verify_dataset(dataset, p_reg, dataset_id, seed, config)
    manifest = {
        "filename": str(path.relative_to(ROOT)), "p": p_reg,
        "total_parameter_dimension": p_reg + 4, "dataset_id": dataset_id,
        "seed": seed, "sha256": checksum, "canonical": dataset_id == "001",
        "checks": checks,
    }
    manifest_path = output_directory(p_reg, dataset_id) / "dataset_manifest.json"
    if not manifest_path.exists() or json.loads(
        manifest_path.read_text(encoding="utf-8")
    ).get("sha256") != checksum:
        atomic_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in ("filename", "dataset_id", "seed", "sha256")}))
    return dataset, checksum, manifest


def setup_model(dataset, p_reg, config):
    X, y, X_start = dataset["X"], dataset["y"], dataset["X_start"]
    y_start = np.asarray([dataset["y_start"]])
    sigma2_start = np.asarray([dataset["sigma2_start"]])
    ns = validation.load_existing_notebook_functions()
    h, hinv, hinv_p, hinv_pp, jacdiag, logdet, log_prior = (
        validation.configure_namespace_for_regression_model(ns, X, X_start, beta_reg_sd=1.0)
    )
    T = int(config["data_generating_process"]["T"])
    d = p_reg + 4
    ns.update({
        "T": T, "p": 1, "q": 1, "y": y, "y_start": y_start,
        "sigma2_start": sigma2_start, "n_params": d,
        "GARCH_type": "Regression-GARCH", "error_type": "Student-t", "phiStar": None,
    })

    def log_prior_phi(phi):
        return log_prior(hinv(phi), 1, 1) + logdet(phi)

    grad_prior = grad(log_prior_phi)
    Hess_prior = hessian(log_prior_phi)

    def log_posterior(phi):
        theta = hinv(phi)
        prior = log_prior(theta, 1, 1)
        if np.isinf(float(prior)):
            return -np.inf, True
        value = ns["log_likelihood"](theta, T, 1, 1, y, y_start, sigma2_start)
        return float(value + prior + logdet(phi)), False

    ns["log_posterior"] = lambda phi, *unused: log_posterior(phi)
    return {
        "ns": ns, "X": X, "y": y, "X_start": X_start,
        "y_start": y_start, "sigma2_start": sigma2_start,
        "args": (T, 1, 1, y, y_start, sigma2_start), "p_reg": p_reg, "d": d,
        "h": h, "hinv": hinv, "hinv_p": hinv_p, "hinv_pp": hinv_pp,
        "jacdiag": jacdiag, "logdet": logdet, "log_prior": log_prior,
        "grad_log_prior_phi": grad_prior, "Hess_log_prior_phi": Hess_prior,
        "log_posterior": log_posterior,
    }


def persistence_maps(model, backend):
    theta_to_psi, psi_to_theta, _ = rg.initiate_regression_GARCH11_persistence_reparam_functions(
        p_reg=model["p_reg"], h=model["h"], hinv=model["hinv"],
        jacdiag=model["jacdiag"], np_backend=backend, delta=0.02,
    )
    return theta_to_psi, psi_to_theta


def gradient_log_posterior_phi(phi, model):
    return model["ns"]["grad_log_likelihood"](
        model["hinv"](phi), *model["args"]
    ) + np.asarray(model["grad_log_prior_phi"](phi), dtype=float)


def map_psi_bounds(p_reg):
    """Numerical regularity bounds used by both MAP optimisers."""
    return [(-20.0, 20.0)] * p_reg + [
        (-20.0, 5.0), (-20.0, 20.0), (-20.0, 20.0), (-10.0, 10.0)
    ]


def initial_map_optimisation(model, theta_start):
    theta_to_psi, psi_to_theta = persistence_maps(model, np)
    psi_start = np.asarray(theta_to_psi(theta_start), dtype=float)
    p_reg = model["p_reg"]
    idx_alpha, idx_beta = p_reg + 1, p_reg + 2

    def phi_from_psi(psi):
        return np.asarray(model["h"](psi_to_theta(psi)), dtype=float)

    def objective(psi):
        value, outside = model["log_posterior"](phi_from_psi(psi))
        return np.inf if outside else -value

    def objective_gradient(psi):
        theta = np.asarray(psi_to_theta(psi), dtype=float)
        gradient_phi = gradient_log_posterior_phi(model["h"](theta), model)
        sigmoid_eta = 1.0 / (1.0 + np.exp(-psi[idx_alpha]))
        persistence = theta[idx_alpha] + theta[idx_beta]
        gradient_psi = gradient_phi.copy()
        gradient_psi[idx_alpha] = (gradient_phi[idx_alpha] + gradient_phi[idx_beta]) * (1.0 - sigmoid_eta)
        gradient_psi[idx_beta] = (
            gradient_phi[idx_alpha] * theta[idx_beta] / persistence
            - gradient_phi[idx_beta] * theta[idx_alpha] / persistence
        )
        return -gradient_psi

    bounds = map_psi_bounds(p_reg)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = minimize(
            objective, psi_start, jac=objective_gradient, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 150, "gtol": 1e-5, "ftol": 1e-10, "maxls": 50},
        )
    if not result.success:
        raise RuntimeError(f"Initial MAP optimisation failed: {result.message}")
    theta = np.asarray(psi_to_theta(result.x), dtype=float)
    phi = np.asarray(model["h"](theta), dtype=float)
    return {
        "theta": theta, "phi": phi, "psi": np.asarray(result.x),
        "objective": float(-result.fun), "gradient_phi": gradient_log_posterior_phi(phi, model),
        "gradient_psi": -np.asarray(result.jac), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev), "termination": str(result.message),
        "seconds": time.perf_counter() - started,
        "warnings": [str(item.message) for item in caught],
    }


def posterior_phi_derivatives(phi, model):
    theta = np.asarray(model["hinv"](phi), dtype=float)
    densities = rg.log_densities_regression_GARCH11(
        theta, model["y"], model["X"], model["y_start"], model["X_start"], model["sigma2_start"]
    )
    gradients, Hessians = rg.grad_Hess_log_densities_regression_GARCH11(
        theta, model["y"], model["X"], model["y_start"], model["X_start"], model["sigma2_start"],
        h=model["h"], hinv_p=model["hinv_p"], hinv_pp=model["hinv_pp"],
        partials_1_deriv_t_l_t=model["ns"]["partials_1_deriv_t_l_t"],
        partials_2_deriv_t_l_t=model["ns"]["partials_2_deriv_t_l_t"],
    )
    objective = float(np.sum(densities)) + float(model["log_prior"](theta, 1, 1)) + float(model["logdet"](phi))
    gradient = np.sum(gradients, axis=0) + np.asarray(model["grad_log_prior_phi"](phi), dtype=float)
    Hessian = np.sum(Hessians, axis=0) + np.asarray(model["Hess_log_prior_phi"](phi), dtype=float)
    del densities, gradients, Hessians
    gc.collect()
    return objective, gradient, 0.5 * (Hessian + Hessian.T)


def refine_map(model, theta_start, maximum_steps=6):
    theta_to_psi, psi_to_theta = persistence_maps(model, validation.np_autograd)

    def phi_from_psi(psi):
        return model["h"](psi_to_theta(psi))

    jacobian_phi = autograd_jacobian(phi_from_psi)
    Hessian_phi_transform = autograd_jacobian(jacobian_phi)

    def evaluate_derivatives(candidate_psi):
        candidate_psi = np.asarray(candidate_psi, dtype=float)
        phi = np.asarray(phi_from_psi(candidate_psi), dtype=float)
        objective, gradient_phi, Hessian_phi = posterior_phi_derivatives(phi, model)
        J = np.asarray(jacobian_phi(candidate_psi), dtype=float)
        transform_Hessians = np.asarray(
            Hessian_phi_transform(candidate_psi), dtype=float
        )
        gradient_psi = J.T @ gradient_phi
        Hessian_psi = J.T @ Hessian_phi @ J + np.tensordot(
            gradient_phi, transform_Hessians, axes=(0, 0)
        )
        Hessian_psi = 0.5 * (Hessian_psi + Hessian_psi.T)
        return {
            "psi": candidate_psi,
            "theta": np.asarray(psi_to_theta(candidate_psi), dtype=float),
            "phi": phi,
            "objective": float(objective),
            "gradient_phi": np.asarray(gradient_phi, dtype=float),
            "gradient_psi": np.asarray(gradient_psi, dtype=float),
            "Hessian_phi": np.asarray(Hessian_phi, dtype=float),
            "Hessian_psi": np.asarray(Hessian_psi, dtype=float),
        }

    def negative_objective(candidate_psi):
        candidate_psi = np.asarray(candidate_psi, dtype=float)
        if not np.all(np.isfinite(candidate_psi)):
            return np.inf
        candidate_phi = np.asarray(phi_from_psi(candidate_psi), dtype=float)
        if not np.all(np.isfinite(candidate_phi)):
            return np.inf
        value, outside = model["log_posterior"](candidate_phi)
        return np.inf if outside or not np.isfinite(value) else -float(value)

    def negative_objective_gradient(candidate_psi):
        candidate_psi = np.asarray(candidate_psi, dtype=float)
        candidate_phi = np.asarray(phi_from_psi(candidate_psi), dtype=float)
        gradient_phi = gradient_log_posterior_phi(candidate_phi, model)
        J = np.asarray(jacobian_phi(candidate_psi), dtype=float)
        return -(J.T @ gradient_phi)

    def gradient_tolerances_satisfied(state):
        return bool(
            np.max(np.abs(state["gradient_psi"])) <= 1e-4
            and np.max(np.abs(state["gradient_phi"])) <= 1e-3
        )

    def precision_limited_stationarity(state):
        """Conservative quantitative test for a machine-precision plateau."""
        negative_Hessian = -state["Hessian_psi"]
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(negative_Hessian)))
        predicted_gain = np.inf
        if minimum_eigenvalue > 0.0:
            try:
                newton_direction = np.linalg.solve(
                    negative_Hessian, state["gradient_psi"]
                )
                predicted_gain = float(
                    0.5 * state["gradient_psi"] @ newton_direction
                )
            except np.linalg.LinAlgError:
                predicted_gain = np.inf
        objective_resolution = float(
            1024.0
            * np.finfo(float).eps
            * max(1.0, abs(state["objective"]))
        )
        close_gradient = (
            np.max(np.abs(state["gradient_psi"])) <= 1e-3
            and np.max(np.abs(state["gradient_phi"])) <= 1e-2
        )
        return {
            "passed": bool(
                close_gradient
                and np.isfinite(predicted_gain)
                and predicted_gain >= 0.0
                and predicted_gain <= objective_resolution
            ),
            "negative_Hessian_minimum_eigenvalue_psi": minimum_eigenvalue,
            "predicted_objective_gain": predicted_gain,
            "objective_resolution": objective_resolution,
        }

    psi = np.asarray(theta_to_psi(theta_start), dtype=float)
    history = []
    started = time.perf_counter()
    state = evaluate_derivatives(psi)
    refinement_starting_objective = state["objective"]
    newton_stop_reason = None
    ordinary_newton_succeeded = False
    newton_iterations = 0
    newton_steps_accepted = 0

    for iteration in range(maximum_steps + 1):
        record = {
            "iteration": iteration, "objective": state["objective"],
            "gradient_infinity_norm_phi": float(
                np.max(np.abs(state["gradient_phi"]))
            ),
            "gradient_infinity_norm_psi": float(
                np.max(np.abs(state["gradient_psi"]))
            ),
        }
        history.append(record)
        progress(
            f"MAP Newton iteration {iteration}: "
            f"objective={state['objective']:.12g}, "
            f"|g_psi|_inf={record['gradient_infinity_norm_psi']:.3e}, "
            f"|g_phi|_inf={record['gradient_infinity_norm_phi']:.3e}"
        )
        if gradient_tolerances_satisfied(state):
            termination = "gradient tolerances satisfied"
            newton_stop_reason = termination
            ordinary_newton_succeeded = True
            break
        if iteration == maximum_steps:
            newton_stop_reason = "maximum Newton steps reached"
            break

        newton_iterations += 1
        negative_Hessian = -state["Hessian_psi"]
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(negative_Hessian)))
        if minimum_eigenvalue <= 1e-10:
            negative_Hessian = model["ns"]["make_pos_def"](negative_Hessian, eps=1e-8)
            record["negative_Hessian_repaired"] = True
        else:
            record["negative_Hessian_repaired"] = False
        record["negative_Hessian_minimum_eigenvalue"] = minimum_eigenvalue
        direction = np.linalg.solve(negative_Hessian, state["gradient_psi"])
        directional_derivative = float(state["gradient_psi"] @ direction)
        record["directional_derivative"] = directional_derivative
        if directional_derivative <= 0:
            newton_stop_reason = "Newton direction was not an ascent direction"
            break
        line_search_objective, outside = model["log_posterior"](state["phi"])
        if outside or not np.isfinite(line_search_objective):
            newton_stop_reason = (
                "aggregate log posterior was invalid at the Newton point"
            )
            break
        line_search_objective = float(line_search_objective)
        record["line_search_objective"] = line_search_objective
        record["objective_evaluation_difference"] = float(
            line_search_objective - state["objective"]
        )
        step_size = 1.0
        for backtrack in range(25):
            candidate = psi + step_size * direction
            candidate_phi = np.asarray(phi_from_psi(candidate), dtype=float)
            candidate_value, outside = model["log_posterior"](candidate_phi)
            sufficient_ascent = (
                line_search_objective
                + 1e-4 * step_size * directional_derivative
            )
            if (
                not outside
                and np.isfinite(candidate_value)
                and candidate_value >= sufficient_ascent
            ):
                break
            step_size *= 0.5
        else:
            record.update({"step_size": step_size, "backtracks": 25})
            newton_stop_reason = (
                "Newton Armijo line search exhausted 25 backtracking steps"
            )
            break
        record.update({"step_size": step_size, "backtracks": backtrack})
        psi = candidate
        newton_steps_accepted += 1
        state = evaluate_derivatives(psi)

    fallback_used = not ordinary_newton_succeeded
    fallback_optimizer = None
    fallback_status = None
    fallback_message = None
    fallback_success = None
    fallback_iterations = None
    fallback_function_evaluations = None
    fallback_starting_objective = None
    fallback_objective_improvement = None

    if fallback_used:
        fallback_optimizer = "L-BFGS-B"
        fallback_starting_objective = -negative_objective(psi)
        if not np.isfinite(fallback_starting_objective):
            raise RuntimeError(
                "MAP fallback starting point has a non-finite objective after "
                f"Newton stopped: {newton_stop_reason}."
            )
        progress(
            "MAP Newton refinement stopped ("
            f"{newton_stop_reason}); starting L-BFGS-B fallback from the current "
            "best Newton point"
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fallback = minimize(
                    negative_objective,
                    psi,
                    jac=negative_objective_gradient,
                    method=fallback_optimizer,
                    bounds=map_psi_bounds(model["p_reg"]),
                    options={
                        "maxiter": 300,
                        "gtol": 1e-7,
                        "ftol": 1e-15,
                        "maxls": 50,
                    },
                )
        except Exception as exc:
            raise RuntimeError(
                "MAP L-BFGS-B fallback failed after Newton stopped: "
                f"{newton_stop_reason}."
            ) from exc
        fallback_status = int(fallback.status)
        fallback_message = str(fallback.message)
        fallback_success = bool(fallback.success)
        fallback_iterations = int(fallback.nit)
        fallback_function_evaluations = int(fallback.nfev)
        if caught:
            fallback_message += "; warnings: " + "; ".join(
                str(item.message) for item in caught
            )
        psi = np.asarray(fallback.x, dtype=float)
        if not np.all(np.isfinite(psi)):
            raise RuntimeError(
                "MAP fallback returned non-finite optimisation parameters: "
                f"status={fallback_status}, message={fallback_message}"
            )
        try:
            state = evaluate_derivatives(psi)
        except Exception as exc:
            raise RuntimeError(
                "MAP fallback final gradient/Hessian evaluation failed: "
                f"status={fallback_status}, message={fallback_message}"
            ) from exc
        fallback_final_objective = -negative_objective(psi)
        comparison_tolerance = float(
            128.0
            * np.finfo(float).eps
            * max(1.0, abs(fallback_starting_objective))
        )
        fallback_objective_improvement = float(
            fallback_final_objective - fallback_starting_objective
        )
        if (
            not np.isfinite(fallback_final_objective)
            or fallback_objective_improvement < -comparison_tolerance
        ):
            raise RuntimeError(
                "MAP fallback worsened the exact log posterior beyond numerical "
                f"tolerance ({fallback_objective_improvement:.3e} < "
                f"{-comparison_tolerance:.3e}); status={fallback_status}, "
                f"message={fallback_message}"
            )

    finite_state = all(
        np.all(np.isfinite(state[name]))
        for name in (
            "psi", "theta", "phi", "gradient_phi", "gradient_psi",
            "Hessian_phi", "Hessian_psi",
        )
    ) and np.isfinite(state["objective"])
    strict_stationarity = gradient_tolerances_satisfied(state)
    precision_diagnostics = precision_limited_stationarity(state)
    final_map_validation_passed = bool(
        finite_state
        and (strict_stationarity or precision_diagnostics["passed"])
    )
    if not final_map_validation_passed:
        raise RuntimeError(
            "MAP refinement and fallback did not produce a validated stationary "
            "point: "
            f"finite_state={finite_state}, "
            f"|g_psi|_inf={np.max(np.abs(state['gradient_psi'])):.3e}, "
            f"|g_phi|_inf={np.max(np.abs(state['gradient_phi'])):.3e}, "
            f"predicted_gain={precision_diagnostics['predicted_objective_gain']:.3e}, "
            f"fallback_status={fallback_status}, message={fallback_message}"
        )

    if fallback_used:
        if strict_stationarity:
            termination = "fallback gradient tolerances satisfied"
        else:
            termination = "fallback precision-limited stationarity satisfied"

    report = {
        "ordinary_newton_refinement_succeeded": ordinary_newton_succeeded,
        "fallback_optimisation_used": fallback_used,
        "newton_stop_reason": newton_stop_reason,
        "refinement_starting_objective": refinement_starting_objective,
        "final_objective": state["objective"],
        "final_gradient_norm": float(np.linalg.norm(state["gradient_phi"])),
        "final_gradient_infinity_norm_phi": float(
            np.max(np.abs(state["gradient_phi"]))
        ),
        "final_gradient_infinity_norm_psi": float(
            np.max(np.abs(state["gradient_psi"]))
        ),
        "newton_iterations": newton_iterations,
        "newton_steps_accepted": newton_steps_accepted,
        "fallback_optimizer": fallback_optimizer,
        "fallback_termination_status": fallback_status,
        "fallback_termination_message": fallback_message,
        "fallback_optimizer_reported_success": fallback_success,
        "fallback_iterations": fallback_iterations,
        "fallback_function_evaluations": fallback_function_evaluations,
        "fallback_starting_objective": fallback_starting_objective,
        "fallback_objective_improvement": fallback_objective_improvement,
        "strict_gradient_tolerances_satisfied": strict_stationarity,
        "precision_limited_stationarity": precision_diagnostics,
        "final_map_validation_passed": final_map_validation_passed,
    }
    return {
        **state,
        "termination": termination,
        "history": history,
        "seconds": time.perf_counter() - started,
        "report": report,
    }


def load_npz_if_matching(path, dataset_sha256, required):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if "dataset_sha256" not in stored or str(stored["dataset_sha256"].item()) != dataset_sha256:
            return None
        if any(name not in stored for name in required):
            return None
        return {name: np.asarray(stored[name]).copy() for name in stored.files}


def obtain_map(model, dataset, checksum, p_reg, dataset_id, out):
    checkpoint = out / "map_hessian.npz"
    cached = load_npz_if_matching(checkpoint, checksum, ("theta", "phi", "Hessian_phi"))
    if cached is not None:
        progress(f"p={p_reg} dataset {dataset_id}: loading cached MAP/Hessian")
        return cached
    legacy = ROOT / "regression_garch_outputs" / f"p{p_reg}_dataset_001" / "refined_map_checkpoint.npz"
    if dataset_id == "001":
        reused = load_npz_if_matching(legacy, checksum, ("theta", "phi", "Hessian_phi"))
        if reused is not None:
            progress(f"p={p_reg} dataset 001: reusing validated legacy MAP/Hessian")
            reused["source"] = np.asarray(str(legacy.relative_to(ROOT)))
            atomic_npz(checkpoint, **reused)
            return reused
        progress(
            f"p={p_reg} dataset 001: legacy MAP checkpoint unavailable; "
            "recomputing from the immutable canonical data"
        )
    theta_true = np.concatenate([
        dataset["params"]["beta_reg"],
        [dataset["params"][key] for key in ("omega", "alpha", "beta_garch", "nu")],
    ])
    progress(f"p={p_reg} dataset {dataset_id}: initial and refined MAP optimisation")
    initial = initial_map_optimisation(model, theta_true)
    refined = refine_map(model, initial["theta"])
    atomic_npz(
        checkpoint, theta=refined["theta"], phi=refined["phi"], psi=refined["psi"],
        gradient_phi=refined["gradient_phi"], gradient_psi=refined["gradient_psi"],
        Hessian_phi=refined["Hessian_phi"], Hessian_psi=refined["Hessian_psi"],
        objective=np.asarray(refined["objective"]), dataset_sha256=np.asarray(checksum),
    )
    atomic_json(out / "map_report.json", {
        "dataset_sha256": checksum, "initial_seconds": initial["seconds"],
        "initial_iterations": initial["iterations"], "refinement_seconds": refined["seconds"],
        "termination": refined["termination"], "history": refined["history"],
        **refined["report"],
    })
    return load_npz_if_matching(checkpoint, checksum, ("theta", "phi", "Hessian_phi"))


def proposal_covariance(map_stage, p_reg, multiplier):
    negative_Hessian = -0.5 * (map_stage["Hessian_phi"] + map_stage["Hessian_phi"].T)
    covariance = np.linalg.inv(negative_Hessian)
    covariance = 0.5 * (covariance + covariance.T)
    intended = multiplier * (2.38 ** 2 / (p_reg + 4)) * covariance
    repaired, _, jitter = rg.numerical_cholesky(intended)
    return repaired, float(jitter)


def chain_diagnostics(chain, p_reg):
    phi = np.asarray(chain["samples_phi"])
    moved = np.linalg.norm(np.diff(phi, axis=0), axis=1) > 0
    return {
        "mean_acceptance_probability": float(np.mean(chain["alphas_MH"])),
        "realised_acceptance_rate": float(np.mean(moved)),
        "outside_support": int(chain["outside_support"]),
        "maximum_persistence": float(np.max(
            np.asarray(chain["samples_theta"])[:, p_reg + 1]
            + np.asarray(chain["samples_theta"])[:, p_reg + 2]
        )),
        "maximum_proposal_jitter": float(chain.get("maximum_proposal_jitter", chain.get("proposal_jitter_initial", 0.0))),
    }


def save_chain(
    path, chain, checksum, seed, iterations, burn_in, cpu_seconds,
    proposal_sha256, subsampling, tuning_signature="",
):
    retained = np.asarray(chain["samples_theta"])[burn_in + 1:iterations + 1]
    if retained.shape[0] != iterations - burn_in:
        raise AssertionError("Retained-chain length differs from production convention.")
    payload = {
        "samples_phi": np.asarray(chain["samples_phi"]),
        "samples_theta": np.asarray(chain["samples_theta"]),
        "retained_theta": retained, "alphas_MH": np.asarray(chain["alphas_MH"]),
        "dataset_sha256": np.asarray(checksum), "seed": np.asarray(seed),
        "iterations": np.asarray(iterations), "burn_in": np.asarray(burn_in),
        "wall_seconds": np.asarray(chain["run_time"]), "cpu_seconds": np.asarray(cpu_seconds),
        "proposal_sha256": np.asarray(proposal_sha256),
        "tuning_signature": np.asarray(tuning_signature),
    }
    if subsampling:
        payload.update({
            "log_posthat_samples": np.asarray(chain["log_posthat_samples"]),
            "sigma2_lhat_samples": np.asarray(chain["sigma2_lhat_samples"]),
            "u_max_prop": np.asarray(chain["u_max_prop"]),
        })
    atomic_npz(path, **payload)
    return retained


def run_or_load_full(path, model, MAP, covariance, iterations, burn_in, seed, checksum, dataset_label):
    proposal_hash = sha256_array(covariance)
    cached = load_npz_if_matching(
        path, checksum,
        ("retained_theta", "alphas_MH", "cpu_seconds", "proposal_sha256", "seed", "iterations", "burn_in"),
    )
    if (
        cached is not None
        and cached["retained_theta"].shape == (iterations - burn_in, model["d"])
        and str(cached["proposal_sha256"].item()) == proposal_hash
        and int(cached["seed"]) == seed
        and int(cached["iterations"]) == iterations
        and int(cached["burn_in"]) == burn_in
    ):
        progress(f"p={model['p_reg']} {dataset_label}: loading cached {Path(path).stem}")
        return cached
    progress(
        f"p={model['p_reg']} {dataset_label}: running {Path(path).stem} "
        f"({iterations} iterations, burn-in {burn_in})"
    )
    np.random.seed(seed)
    random.seed(seed)
    cpu_start = time.process_time()
    chain = rg.sample_full_data_posterior_numerical_cholesky(
        MAP, covariance, iterations, rg.get_param_names_regression_GARCH11(model["p_reg"]),
        dataset_label, hinv=model["hinv"], log_posterior=model["log_posterior"],
        adapt_cov=False, adapt_scale=False, burn_in=burn_in,
    )
    cpu_seconds = time.process_time() - cpu_start
    save_chain(
        path, chain, checksum, seed, iterations, burn_in, cpu_seconds,
        proposal_hash, False,
    )
    return load_npz_if_matching(path, checksum, ("retained_theta", "alphas_MH", "cpu_seconds"))


def taylor_paths(out):
    directory = out / "taylor_control_variates"
    return directory, directory / "densities.npy", directory / "gradients.npy", directory / "Hessians.npy", directory / "metadata.json"


def construct_or_load_taylor(model, MAP, checksum, out, config):
    directory, density_path, gradient_path, Hessian_path, metadata_path = taylor_paths(out)
    expected_shapes = ((model["args"][0],), (model["args"][0], model["d"]), (model["args"][0], model["d"], model["d"]))
    if metadata_path.exists() and all(path.exists() for path in (density_path, gradient_path, Hessian_path)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in (density_path, gradient_path, Hessian_path)]
        if (
            metadata.get("dataset_sha256") == checksum
            and np.array_equal(np.asarray(metadata.get("MAP")), np.asarray(MAP))
            and tuple(array.shape for array in arrays) == expected_shapes
        ):
            progress(f"p={model['p_reg']}: loading cached Taylor control variates")
            return install_taylor(model, MAP, arrays, config), metadata
        del arrays
    progress(f"constructing Taylor control variates at p={model['p_reg']}")
    started = time.perf_counter()
    theta = np.asarray(model["hinv"](MAP), dtype=float)
    densities = rg.log_densities_regression_GARCH11(
        theta, model["y"], model["X"], model["y_start"], model["X_start"], model["sigma2_start"]
    )
    gradients, Hessians = rg.grad_Hess_log_densities_regression_GARCH11(
        theta, model["y"], model["X"], model["y_start"], model["X_start"], model["sigma2_start"],
        h=model["h"], hinv_p=model["hinv_p"], hinv_pp=model["hinv_pp"],
        partials_1_deriv_t_l_t=model["ns"]["partials_1_deriv_t_l_t"],
        partials_2_deriv_t_l_t=model["ns"]["partials_2_deriv_t_l_t"],
    )
    directory.mkdir(parents=True, exist_ok=True)
    atomic_npy(density_path, densities)
    atomic_npy(gradient_path, gradients)
    atomic_npy(Hessian_path, Hessians)
    metadata = {
        "dataset_sha256": checksum, "MAP": np.asarray(MAP).tolist(),
        "seconds": time.perf_counter() - started,
        "arrays": {
            "densities": {"shape": list(densities.shape), "bytes": int(densities.nbytes)},
            "gradients": {"shape": list(gradients.shape), "bytes": int(gradients.nbytes)},
            "Hessians": {"shape": list(Hessians.shape), "bytes": int(Hessians.nbytes)},
        },
    }
    atomic_json(metadata_path, metadata)
    del densities, gradients, Hessians
    gc.collect()
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in (density_path, gradient_path, Hessian_path)]
    return install_taylor(model, MAP, arrays, config), metadata


def install_taylor(model, MAP, arrays, config):
    densities, gradients, Hessians = arrays
    A, B, C = float(np.sum(densities)), np.sum(gradients, axis=0), np.sum(Hessians, axis=0)

    def q_sum(phi):
        delta = phi - MAP
        return A + B @ delta + 0.5 * delta @ C @ delta

    model["ns"].update({
        "phiStar": MAP, "dens_at_phiStar": densities, "grad_at_phiStar": gradients,
        "Hess_at_phiStar": Hessians, "q_sum": q_sum,
        "t_star": config["mcmc"]["t_star"], "b": 0.0,
    })
    return densities, gradients, Hessians, q_sum


def run_or_load_tuning(model, MAP, Taylor, pilot, checksum, out, config):
    path = out / "tuning_rmax_100.npz"
    cached = load_npz_if_matching(
        path, checksum,
        ("probabilities", "m", "c", "gamma", "expected_u_max", "R_max", "t_star", "b"),
    )
    if (
        cached is not None
        and cached["probabilities"].shape == (model["args"][0],)
        and int(cached["R_max"]) == 100
        and int(cached["t_star"]) == 1000
        and float(cached["b"]) == 0.0
    ):
        progress(f"p={model['p_reg']}: loading cached b=0, R_max=100 tuning")
        return cached
    progress(f"p={model['p_reg']}: running calibrated b=0, R_max=100 tuning")
    mcmc = config["mcmc"]
    densities, gradients, Hessians, _ = Taylor
    pilot_slice = slice(mcmc["pilot_slice_start"], mcmc["pilot_iterations"] + 1, mcmc["pilot_slice_step"])
    tuning_started = time.perf_counter()
    tuned = model["ns"]["run_calibrated_tuning_over_Rmax"](
        R_max_list=[100], MCMC_results={"MCMC_obj": {"samples_phi": pilot["samples_phi"]}},
        MAP=MAP, phiStar=MAP, dens_at_phiStar=densities, grad_at_phiStar=gradients,
        Hess_at_phiStar=Hessians, args=model["args"], T=model["args"][0],
        t_star_frac=0.01, b=0.0, m_min=mcmc["m_min"], pilot_slice=pilot_slice, verbose=False,
    )[100]
    probabilities = np.asarray(tuned["probs_opt"], dtype=float)
    expected_u_max = float(model["ns"]["expected_umax_safe"](probabilities, int(tuned["m_opt"])))
    atomic_npz(
        path, probabilities=probabilities, m=np.asarray(int(tuned["m_opt"])),
        c=np.asarray(float(tuned["c_opt"])), gamma=np.asarray(float(tuned["gamma_opt"])),
        expected_u_max=np.asarray(expected_u_max), variance_tolerance=np.asarray(float(tuned["V_target"])),
        analytic_variance=np.asarray(float(tuned["var_opt"])), seconds=np.asarray(time.perf_counter() - tuning_started),
        dataset_sha256=np.asarray(checksum), R_max=np.asarray(100), t_star=np.asarray(1000), b=np.asarray(0.0),
    )
    return load_npz_if_matching(path, checksum, ("probabilities", "m", "c", "gamma", "expected_u_max"))


def estimator_params(tuning):
    return {
        "m_opt": int(tuning["m"]), "c_opt": float(tuning["c"]),
        "gamma_opt": float(tuning["gamma"]), "t_star": 1000, "b": 0.0,
        "V_tolerance": float(tuning["variance_tolerance"]), "R_max": 100,
    }


def run_or_load_subsampling(path, model, MAP, covariance, tuning, iterations, burn_in, seed, checksum, dataset_label):
    proposal_hash = sha256_array(covariance)
    tuning_text = json.dumps(estimator_params(tuning), sort_keys=True, separators=(",", ":"))
    tuning_hash = hashlib.sha256(tuning_text.encode("utf-8")).hexdigest()
    cached = load_npz_if_matching(
        path, checksum,
        ("retained_theta", "alphas_MH", "cpu_seconds", "u_max_prop", "proposal_sha256", "tuning_signature", "seed", "iterations", "burn_in"),
    )
    if (
        cached is not None
        and cached["retained_theta"].shape == (iterations - burn_in, model["d"])
        and str(cached["proposal_sha256"].item()) == proposal_hash
        and str(cached["tuning_signature"].item()) == tuning_hash
        and int(cached["seed"]) == seed
        and int(cached["iterations"]) == iterations
        and int(cached["burn_in"]) == burn_in
    ):
        progress(f"p={model['p_reg']} {dataset_label}: loading cached b=0 subsampling chain")
        return cached
    progress(
        f"p={model['p_reg']} {dataset_label}: running b=0 subsampling chain "
        f"({iterations} iterations, burn-in {burn_in})"
    )
    np.random.seed(seed)
    random.seed(seed)
    cpu_start = time.process_time()
    chain = model["ns"]["sample_subsampling_posterior_random_walk"](
        estimator_params(tuning), MAP, covariance, iterations,
        rg.get_param_names_regression_GARCH11(model["p_reg"]), dataset_label, adapt=False,
    )
    cpu_seconds = time.process_time() - cpu_start
    save_chain(
        path, chain, checksum, seed, iterations, burn_in, cpu_seconds,
        proposal_hash, True, tuning_signature=tuning_hash,
    )
    return load_npz_if_matching(path, checksum, ("retained_theta", "alphas_MH", "cpu_seconds", "u_max_prop"))


def parameter_names(p_reg):
    return [f"beta_reg_{j}" for j in range(1, p_reg + 1)] + ["omega", "alpha", "beta_garch", "nu"]


def comparison_rows(full, subsampling, p_reg):
    rows = []
    for index, name in enumerate(parameter_names(p_reg)):
        x, y = full[:, index], subsampling[:, index]
        xq, yq = np.quantile(x, (0.025, 0.5, 0.975)), np.quantile(y, (0.025, 0.5, 0.975))
        xsd, ysd = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
        rows.append({
            "parameter": name, "full_mean": float(np.mean(x)), "subsampling_mean": float(np.mean(y)),
            "full_sd": xsd, "subsampling_sd": ysd,
            "absolute_standardised_mean_difference": float(abs(np.mean(y) - np.mean(x)) / xsd),
            "sd_ratio_subsampling_over_full": ysd / xsd,
            "full_q025": float(xq[0]), "full_q50": float(xq[1]), "full_q975": float(xq[2]),
            "subsampling_q025": float(yq[0]), "subsampling_q50": float(yq[1]), "subsampling_q975": float(yq[2]),
            "intervals_overlap": bool(max(xq[0], yq[0]) <= min(xq[2], yq[2])),
        })
    return rows


def save_comparison(rows, path):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_overlay(full, subsampling, p_reg, path):
    beta_indices = np.unique(np.linspace(0, p_reg - 1, min(6, p_reg), dtype=int)).tolist()
    selected = beta_indices + list(range(p_reg, p_reg + 4))
    names = parameter_names(p_reg)
    ncols, nrows = 5, int(np.ceil(len(selected) / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.4 * nrows), squeeze=False)
    for axis, index in zip(axes.flat, selected):
        pooled = np.concatenate([full[:, index], subsampling[:, index]])
        lower, upper = np.quantile(pooled, (0.0025, 0.9975))
        grid = np.linspace(lower, upper, 400)
        axis.plot(grid, gaussian_kde(full[:, index])(grid), color="black", label="Full-data MCMC")
        axis.plot(grid, gaussian_kde(subsampling[:, index])(grid), color="#0072B2", label="b=0 subsampling MCMC")
        axis.set_title(names[index])
        axis.set_yticks([])
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(path)


def summarise_dataset(p_reg, dataset_id, seed, checksum, full, subsampling, tuning, proposal_jitter, out, config):
    summary_path = out / "summary.json"
    full_proposal_hash = str(full["proposal_sha256"].item())
    subsampling_proposal_hash = str(subsampling["proposal_sha256"].item())
    tuning_signature = str(subsampling["tuning_signature"].item())
    required_products = [out / "posterior_comparison.csv"]
    if dataset_id == "001":
        required_products.append(out / "selected_marginal_overlays.png")
    if summary_path.exists() and all(path.exists() for path in required_products):
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            cached_summary.get("dataset_sha256") == checksum
            and cached_summary.get("full_chain_proposal_sha256") == full_proposal_hash
            and cached_summary.get("subsampling_chain_proposal_sha256") == subsampling_proposal_hash
            and cached_summary.get("subsampling_tuning_signature") == tuning_signature
        ):
            progress(f"p={p_reg} dataset {dataset_id}: loading cached posterior summary")
            return cached_summary
    progress(f"p={p_reg} dataset {dataset_id}: creating posterior summary and diagnostics")
    rows = comparison_rows(full["retained_theta"], subsampling["retained_theta"], p_reg)
    save_comparison(rows, out / "posterior_comparison.csv")
    if dataset_id == "001":
        make_overlay(full["retained_theta"], subsampling["retained_theta"], p_reg, out / "selected_marginal_overlays.png")
    discrepancies = np.asarray([row["absolute_standardised_mean_difference"] for row in rows])
    ratios = np.asarray([row["sd_ratio_subsampling_over_full"] for row in rows])
    overlaps = np.asarray([row["intervals_overlap"] for row in rows])
    full_cpu, sub_cpu = float(full["cpu_seconds"]), float(subsampling["cpu_seconds"])
    full_moved = np.linalg.norm(np.diff(full["samples_phi"], axis=0), axis=1) > 0
    subsampling_moved = np.linalg.norm(
        np.diff(subsampling["samples_phi"], axis=0), axis=1
    ) > 0
    retained_u = np.asarray(subsampling["u_max_prop"])[config["mcmc"]["burn_in"]:]
    summary = {
        "p": p_reg, "total_parameter_dimension": p_reg + 4, "dataset_id": dataset_id,
        "seed": seed, "dataset_sha256": checksum,
        "full_data_acceptance_rate": float(np.mean(full_moved)),
        "subsampling_acceptance_rate": float(np.mean(subsampling_moved)),
        "full_data_mean_acceptance_probability": float(np.mean(full["alphas_MH"])),
        "subsampling_mean_acceptance_probability": float(np.mean(subsampling["alphas_MH"])),
        "full_data_CPU_seconds": full_cpu, "subsampling_CPU_seconds": sub_cpu,
        "R_CPU": full_cpu / sub_cpu, "tuned_c": float(tuning["c"]),
        "tuned_gamma": float(tuning["gamma"]), "tuned_m": int(tuning["m"]),
        "expected_u_max": float(tuning["expected_u_max"]),
        "expected_u_max_over_T": float(tuning["expected_u_max"] / config["data_generating_process"]["T"]),
        "realised_average_u_max": float(np.mean(retained_u)),
        "realised_average_u_max_over_T": float(np.mean(retained_u) / config["data_generating_process"]["T"]),
        "marginal_95_intervals_overlapping": int(np.sum(overlaps)),
        "marginal_95_intervals_total": len(rows), "marginal_95_intervals_overlap_proportion": float(np.mean(overlaps)),
        "maximum_standardised_posterior_mean_difference": float(np.max(discrepancies)),
        "median_standardised_posterior_mean_difference": float(np.median(discrepancies)),
        "minimum_posterior_SD_ratio": float(np.min(ratios)),
        "maximum_posterior_SD_ratio": float(np.max(ratios)),
        "proposal_cholesky_jitter": proposal_jitter, "R_max": 100, "t_star": 1000, "b": 0.0,
        "full_chain_proposal_sha256": full_proposal_hash,
        "subsampling_chain_proposal_sha256": subsampling_proposal_hash,
        "subsampling_tuning_signature": tuning_signature,
        "iterations": config["mcmc"]["iterations"], "burn_in": config["mcmc"]["burn_in"],
        "retained_draws": config["mcmc"]["retained_draws"],
        "interpretation": "Computational robustness/timing replicate; not a repeated-sampling coverage experiment.",
    }
    atomic_json(summary_path, summary)
    return summary


def run_dataset(p_reg, dataset_id, config):
    progress(f"starting p={p_reg} (d={p_reg + 4}), dataset {dataset_id}")
    dataset, checksum, manifest = ensure_dataset(p_reg, dataset_id, config)
    out = output_directory(p_reg, dataset_id)
    out.mkdir(parents=True, exist_ok=True)
    model = setup_model(dataset, p_reg, config)
    progress(f"p={p_reg} dataset {dataset_id}: model configured")
    map_stage = obtain_map(model, dataset, checksum, p_reg, dataset_id, out)
    multiplier = float(config["cases"][str(p_reg)]["proposal_covariance_multiplier"])
    covariance, proposal_jitter = proposal_covariance(map_stage, p_reg, multiplier)
    progress(
        f"p={p_reg} dataset {dataset_id}: proposal constructed "
        f"(multiplier={multiplier:.2f}, Cholesky jitter={proposal_jitter:.3e})"
    )
    if proposal_jitter != 0.0:
        warnings.warn(f"Proposal required numerical jitter {proposal_jitter:.3e}.")
    mcmc = config["mcmc"]
    seed_offset = p_reg * 10 + int(dataset_id)
    pilot_path = out / "pilot_chain.npz"
    pilot = run_or_load_full(
        pilot_path, model, map_stage["phi"], covariance, mcmc["pilot_iterations"], 0,
        config["seed_bases"]["pilot"] + seed_offset, checksum, manifest["filename"],
    )
    Taylor, _ = construct_or_load_taylor(model, map_stage["phi"], checksum, out, config)
    tuning = run_or_load_tuning(model, map_stage["phi"], Taylor, pilot, checksum, out, config)
    full = run_or_load_full(
        out / "full_data_chain.npz", model, map_stage["phi"], covariance,
        mcmc["iterations"], mcmc["burn_in"], config["seed_bases"]["full_data"] + seed_offset,
        checksum, manifest["filename"],
    )
    subsampling = run_or_load_subsampling(
        out / "subsampling_chain.npz", model, map_stage["phi"], covariance, tuning,
        mcmc["iterations"], mcmc["burn_in"], config["seed_bases"]["subsampling"] + seed_offset,
        checksum, manifest["filename"],
    )
    summary = summarise_dataset(
        p_reg, dataset_id, manifest["seed"], checksum, full, subsampling,
        tuning, proposal_jitter, out, config,
    )
    del Taylor, pilot, full, subsampling, model, dataset
    gc.collect()
    progress(f"completed p={p_reg} (d={p_reg + 4}), dataset {dataset_id}")
    return summary


def aggregate_dimension(p_reg, config):
    summaries = []
    for dataset_id in sorted(config["cases"][str(p_reg)]["dataset_seeds"]):
        path = output_directory(p_reg, dataset_id) / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    speeds = np.asarray([summary["R_CPU"] for summary in summaries])
    aggregate = {
        "p": p_reg, "total_parameter_dimension": p_reg + 4,
        "R_CPU_values": speeds.tolist(), "median_R_CPU": float(np.median(speeds)),
        "range_R_CPU": [float(np.min(speeds)), float(np.max(speeds))],
        "dataset_ids": [summary["dataset_id"] for summary in summaries],
        "interpretation": "Five computational robustness/timing datasets; these are not coverage replicates.",
    }
    atomic_json(OUTPUT_ROOT / f"p{p_reg}" / "five_dataset_summary.json", aggregate)
    progress(
        f"p={p_reg} (d={p_reg + 4}): saved five-dataset CPU summary; "
        f"median R_CPU={aggregate['median_R_CPU']:.6g}, range={aggregate['range_R_CPU']}"
    )
    return aggregate


def prepare_datasets(config, p_values=(25, 50)):
    for p_reg in p_values:
        for dataset_id in sorted(config["cases"][str(p_reg)]["dataset_seeds"]):
            progress(f"dataset preparation: p={p_reg} (d={p_reg + 4}), dataset {dataset_id}")
            dataset, _, _ = ensure_dataset(p_reg, dataset_id, config)
            del dataset
            gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-datasets", action="store_true", help="Only create/verify datasets; do not run inference.")
    parser.add_argument("--p", type=int, choices=(25, 50), action="append", help="Dimension case; repeat to select both.")
    parser.add_argument("--dataset", choices=("001", "002", "003", "004", "005"), action="append", help="Dataset id; repeat as needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    require_writable_output_root()
    p_values = args.p or [25, 50]
    if args.prepare_datasets:
        prepare_datasets(config, p_values)
        return
    selected_ids = args.dataset
    for p_reg in p_values:
        ids = selected_ids or sorted(config["cases"][str(p_reg)]["dataset_seeds"])
        for dataset_id in ids:
            progress(f"starting p={p_reg} (d={p_reg + 4}), dataset {dataset_id}")
            run_dataset(p_reg, dataset_id, config)
        expected_all = set(config["cases"][str(p_reg)]["dataset_seeds"])
        completed = {
            dataset_id
            for dataset_id in expected_all
            if (output_directory(p_reg, dataset_id) / "summary.json").exists()
        }
        if completed == expected_all:
            print(json.dumps(aggregate_dimension(p_reg, config), indent=2))


if __name__ == "__main__":
    main()
