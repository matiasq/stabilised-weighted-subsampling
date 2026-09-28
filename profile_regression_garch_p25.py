"""Production-scale profiling run for one Student-t regression-GARCH dataset.

This is deliberately a profiling/smoke-test driver, not the final experiment.  It
creates (once) and thereafter reloads one canonical T=100000, p_reg=25 dataset.
It does not run DC-BATS or the final long MCMC chains.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import resource
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
from autograd import grad, hessian
from scipy.optimize import minimize

import regression_garch as rg
import validate_regression_garch as validation


T = 100000
P_REG = 25
BURN_IN = 1000
RHO = 0.5
DATASET_SEED = 20260921
DATASET_ID = "regression_garch_p25_dataset_001"
DATASET_PATH = Path(__file__).with_name("data") / f"{DATASET_ID}.npz"

OMEGA = 0.07134079
ALPHA = 0.16151262
BETA_GARCH = 0.78591371
NU = 8.0

PILOT_MCMC_SEED = 20261001
FULL_SMOKE_SEED = 20261002
SUBSAMPLING_SMOKE_SEED = 20261003
ESTIMATOR_TIMING_SEED = 20261004


def current_rss_bytes():
    with Path("/proc/self/status").open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS was not present in /proc/self/status.")


def progress(message):
    print(f"[profile] {message}", file=sys.stderr, flush=True)


class MemorySampler:
    """Sample process RSS during one profiling phase."""

    def __init__(self, interval_seconds=0.005):
        self.interval_seconds = interval_seconds
        self.baseline_bytes = None
        self.peak_bytes = None
        self.end_bytes = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, current_rss_bytes())
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        gc.collect()
        self.baseline_bytes = current_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self.end_bytes = current_rss_bytes()

    def report(self):
        mib = 1024**2
        return {
            "baseline_MiB": self.baseline_bytes / mib,
            "peak_MiB": self.peak_bytes / mib,
            "end_MiB": self.end_bytes / mib,
            "peak_increment_MiB": (self.peak_bytes - self.baseline_bytes) / mib,
            "peak_above_end_MiB": max(0, self.peak_bytes - self.end_bytes) / mib,
        }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def design_quantities():
    indices = np.arange(P_REG)
    Sigma = RHO ** np.abs(indices[:, None] - indices[None, :])
    beta_raw = np.ones(P_REG)
    beta_reg = beta_raw / np.sqrt(beta_raw @ Sigma @ beta_raw)
    return Sigma, beta_reg


def generate_canonical_dataset(path):
    Sigma, beta_reg = design_quantities()
    signal_variance = float(beta_reg @ Sigma @ beta_reg)
    rng = np.random.default_rng(DATASET_SEED)

    # The first row/draw defines the presample state before the burn-in begins.
    X_with_initial = rng.multivariate_normal(
        np.zeros(P_REG), Sigma, size=T + BURN_IN + 1, method="cholesky"
    )
    t_draws_with_initial = rng.standard_t(NU, size=T + BURN_IN + 1)
    sigma2_initial = OMEGA / (1.0 - ALPHA - BETA_GARCH)
    standardised_initial = t_draws_with_initial[0] * np.sqrt((NU - 2.0) / NU)
    z_initial = np.sqrt(sigma2_initial) * standardised_initial
    y_initial = float(X_with_initial[0] @ beta_reg + z_initial)

    full_simulation = rg.simulate_regression_GARCH11(
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

    start_index = BURN_IN - 1
    retained = slice(BURN_IN, BURN_IN + T)
    simulation = {
        "X": full_simulation["X"][retained].copy(),
        "y": full_simulation["y"][retained].copy(),
        "z": full_simulation["z"][retained].copy(),
        "sigma2": full_simulation["sigma2"][retained].copy(),
        "student_t_draws": full_simulation["student_t_draws"][retained].copy(),
        "standardised_innovations": full_simulation[
            "standardised_innovations"
        ][retained].copy(),
        "X_start": full_simulation["X"][start_index].copy(),
        "y_start": float(full_simulation["y"][start_index]),
        "z_start": float(full_simulation["z"][start_index]),
        "sigma2_start": float(full_simulation["sigma2"][start_index]),
        "student_t_draw_start": float(
            full_simulation["student_t_draws"][start_index]
        ),
        "standardised_innovation_start": float(
            full_simulation["standardised_innovations"][start_index]
        ),
        "seed": DATASET_SEED,
        "dataset_id": DATASET_ID,
        "burn_in": BURN_IN,
        "rho": RHO,
        "signal_variance": signal_variance,
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
    return simulation


def verify_dataset(dataset):
    Sigma, beta_expected = design_quantities()
    params = dataset["params"]
    expected_scalars = {
        "p_reg": P_REG,
        "omega": OMEGA,
        "alpha": ALPHA,
        "beta_garch": BETA_GARCH,
        "nu": NU,
    }
    if dataset["X"].shape != (T, P_REG) or dataset["y"].shape != (T,):
        raise AssertionError("Canonical dataset has unexpected retained dimensions.")
    if any(params[name] != value for name, value in expected_scalars.items()):
        raise AssertionError("Canonical dataset parameter metadata does not match the design.")
    if dataset["seed"] != DATASET_SEED or dataset["burn_in"] != BURN_IN:
        raise AssertionError("Canonical dataset seed or burn-in metadata is incorrect.")
    if dataset["dataset_id"] != DATASET_ID or dataset["rho"] != RHO:
        raise AssertionError("Canonical dataset identity/design metadata is incorrect.")
    if not np.array_equal(dataset["Sigma"], Sigma):
        raise AssertionError("Stored design covariance is not the requested Toeplitz matrix.")
    if not np.allclose(params["beta_reg"], beta_expected, rtol=0.0, atol=1e-15):
        raise AssertionError("Stored beta_reg does not have the requested scaling.")

    signal_variance = float(params["beta_reg"] @ Sigma @ params["beta_reg"])
    mean_residual_error = float(
        np.max(np.abs(dataset["y"] - dataset["X"] @ params["beta_reg"] - dataset["z"]))
    )
    start_residual_error = abs(
        dataset["y_start"]
        - dataset["X_start"] @ params["beta_reg"]
        - dataset["z_start"]
    )
    start_innovation_error = abs(
        dataset["z_start"] / np.sqrt(dataset["sigma2_start"])
        - dataset["standardised_innovation_start"]
    )
    if max(mean_residual_error, start_residual_error, start_innovation_error) > 1e-12:
        raise AssertionError("Canonical response/residual/presample identities failed.")
    if not np.all(dataset["sigma2"] > 0.0):
        raise AssertionError("Canonical conditional variances are not all positive.")
    return {
        "signal_variance_exact": signal_variance,
        "sample_signal_variance": float(np.var(dataset["X"] @ params["beta_reg"], ddof=1)),
        "mean_residual_identity_max_absolute": mean_residual_error,
        "start_residual_identity_absolute": float(start_residual_error),
        "start_innovation_identity_absolute": float(start_innovation_error),
        "standardised_innovation_sample_variance": float(
            np.var(dataset["standardised_innovations"], ddof=1)
        ),
        "sigma2_min": float(np.min(dataset["sigma2"])),
        "sigma2_max": float(np.max(dataset["sigma2"])),
    }


def timed_repetitions(function, repetitions):
    durations = np.empty(repetitions)
    values = []
    for i in range(repetitions):
        start = time.perf_counter()
        values.append(function())
        durations[i] = time.perf_counter() - start
    return values, {
        "repetitions": repetitions,
        "mean_seconds": float(np.mean(durations)),
        "median_seconds": float(np.median(durations)),
        "min_seconds": float(np.min(durations)),
        "max_seconds": float(np.max(durations)),
    }


def profile(path):
    total_start = time.perf_counter()
    progress("generating or loading the canonical dataset")
    generated_now = not path.exists()
    generation_start = time.perf_counter()
    generated = generate_canonical_dataset(path) if generated_now else None
    generation_seconds = time.perf_counter() - generation_start

    load_start = time.perf_counter()
    dataset = rg.load_regression_GARCH11_dataset(path)
    load_seconds = time.perf_counter() - load_start
    if generated is not None:
        exact_array_names = (
            "X",
            "y",
            "z",
            "sigma2",
            "student_t_draws",
            "standardised_innovations",
            "X_start",
        )
        if not all(
            np.array_equal(generated[name], dataset[name]) for name in exact_array_names
        ):
            raise AssertionError("Canonical dataset did not round-trip exactly.")
    data_checks = verify_dataset(dataset)
    progress("canonical dataset verified; configuring the model")

    X = dataset["X"]
    y = dataset["y"]
    X_start = dataset["X_start"]
    y_start_arr = np.array([dataset["y_start"]])
    sigma2_start_arr = np.array([dataset["sigma2_start"]])
    theta_true = np.concatenate(
        [
            dataset["params"]["beta_reg"],
            [OMEGA, ALPHA, BETA_GARCH, NU],
        ]
    )

    ns = validation.load_existing_notebook_functions()
    h, hinv, hinv_p, hinv_pp, jacdiag, logdet, log_prior = (
        validation.configure_namespace_for_regression_model(
            ns, X, X_start, beta_reg_sd=1.0
        )
    )
    d = P_REG + 4
    ns.update(
        {
            "T": T,
            "p": 1,
            "q": 1,
            "y": y,
            "y_start": y_start_arr,
            "sigma2_start": sigma2_start_arr,
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

    def log_posterior_local(phi, T_arg, p, q, y_arg, y_start_arg, sigma2_start_arg):
        theta = hinv(phi)
        prior = log_prior(theta, p, q)
        if np.isinf(float(prior)):
            return -np.inf, True
        value = (
            ns["log_likelihood"](
                theta, T_arg, p, q, y_arg, y_start_arg, sigma2_start_arg
            )
            + prior
            + logdet(phi)
        )
        return float(value), False

    ns["log_posterior"] = log_posterior_local

    def log_post_value(phi):
        return log_posterior_local(
            phi, T, 1, 1, y, y_start_arr, sigma2_start_arr
        )[0]

    def grad_log_post(phi):
        return ns["grad_log_likelihood"](
            hinv(phi), T, 1, 1, y, y_start_arr, sigma2_start_arr
        ) + np.asarray(grad_log_prior_phi(phi), dtype=float)

    theta_to_psi, psi_to_theta, _ = (
        rg.initiate_regression_GARCH11_persistence_reparam_functions(
            p_reg=P_REG,
            h=h,
            hinv=hinv,
            jacdiag=jacdiag,
            np_backend=np,
            delta=0.02,
        )
    )
    psi_start = np.asarray(theta_to_psi(theta_true), dtype=float)
    idx_omega = P_REG
    idx_alpha = P_REG + 1
    idx_beta_garch = P_REG + 2
    idx_nu = P_REG + 3

    def phi_from_psi(psi):
        return np.asarray(h(psi_to_theta(psi)), dtype=float)

    def log_post_value_psi(psi):
        return log_post_value(phi_from_psi(psi))

    def grad_log_post_psi(psi):
        theta = np.asarray(psi_to_theta(psi), dtype=float)
        phi = np.asarray(h(theta), dtype=float)
        grad_phi = grad_log_post(phi)
        sigmoid_eta = 1.0 / (1.0 + np.exp(-psi[idx_alpha]))
        persistence = theta[idx_alpha] + theta[idx_beta_garch]
        grad_psi = grad_phi.copy()
        grad_psi[idx_alpha] = (
            grad_phi[idx_alpha] + grad_phi[idx_beta_garch]
        ) * (1.0 - sigmoid_eta)
        grad_psi[idx_beta_garch] = (
            grad_phi[idx_alpha] * theta[idx_beta_garch] / persistence
            - grad_phi[idx_beta_garch] * theta[idx_alpha] / persistence
        )
        return grad_psi

    # Wide numerical guards only; none is expected to be scientifically active.
    map_bounds = (
        [(-20.0, 20.0)] * P_REG
        + [(-20.0, 5.0), (-20.0, 20.0), (-20.0, 20.0), (-10.0, 10.0)]
    )
    captured_warnings = []
    progress("starting MAP optimisation")
    map_start = time.perf_counter()
    with MemorySampler() as map_memory, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        map_result = minimize(
            lambda psi: -log_post_value_psi(psi),
            psi_start,
            jac=lambda psi: -grad_log_post_psi(psi),
            method="L-BFGS-B",
            bounds=map_bounds,
            options={"maxiter": 150, "gtol": 1e-5, "ftol": 1e-10, "maxls": 50},
        )
        captured_warnings.extend(str(item.message) for item in caught)
        if map_result.success:
            MAP = phi_from_psi(map_result.x)
            map_gradient_phi = grad_log_post(MAP)
    map_seconds = time.perf_counter() - map_start
    if not map_result.success:
        raise RuntimeError(f"MAP optimisation failed: {map_result.message}")
    progress(f"MAP optimisation complete in {map_seconds:.1f} seconds")

    args = (T, 1, 1, y, y_start_arr, sigma2_start_arr)
    full_values, full_likelihood_timing = timed_repetitions(
        lambda: ns["log_likelihood"](hinv(MAP), *args), repetitions=5
    )
    if not np.all(np.isfinite(full_values)):
        raise AssertionError("Non-finite full likelihood during timing.")

    gc.collect()
    progress("constructing Taylor control-variate arrays")
    taylor_start = time.perf_counter()
    with MemorySampler() as taylor_memory:
        dens_at_phiStar, grad_at_phiStar, Hess_at_phiStar = ns[
            "initiate_control_variate_quantities"
        ](
            ns["log_densities"],
            ns["grad_log_densities"],
            ns["Hess_log_densities"],
            MAP,
            args,
        )
    taylor_seconds = time.perf_counter() - taylor_start
    if not all(
        np.all(np.isfinite(value))
        for value in (dens_at_phiStar, grad_at_phiStar, Hess_at_phiStar)
    ):
        raise AssertionError("Taylor construction produced non-finite values.")
    progress(f"Taylor construction complete in {taylor_seconds:.1f} seconds")

    A = np.sum(dens_at_phiStar)
    B = np.sum(grad_at_phiStar, axis=0)
    C = np.sum(Hess_at_phiStar, axis=0)

    def q_sum(phi):
        delta = phi - MAP
        return A + B @ delta + 0.5 * delta @ C @ delta

    ns.update(
        {
            "phiStar": MAP,
            "dens_at_phiStar": dens_at_phiStar,
            "grad_at_phiStar": grad_at_phiStar,
            "Hess_at_phiStar": Hess_at_phiStar,
            "q_sum": q_sum,
        }
    )

    Hess_log_post = C + np.asarray(Hess_log_prior_phi(MAP), dtype=float)
    Hess_log_post = 0.5 * (Hess_log_post + Hess_log_post.T)
    covariance = np.linalg.inv(-Hess_log_post)
    covariance = ns["make_pos_def"](covariance, eps=1e-10)
    proposal_cov = (2.38**2 / d) * covariance

    names_original, names_transformed = rg.get_param_names_regression_GARCH11(P_REG)
    names = {"params_name": names_original, "trans_param_names": names_transformed}

    np.random.seed(PILOT_MCMC_SEED)
    random.seed(PILOT_MCMC_SEED)
    progress("running the short full-data pilot required for tuning")
    pilot = ns["sample_full_data_posterior_AM"](
        MAP,
        proposal_cov,
        60,
        names,
        DATASET_ID,
        adapt_cov=False,
        adapt_scale=False,
        burn_in=1000,
    )
    pilot_results = {"MCMC_obj": pilot}

    ns.update({"t_star": int(0.01 * T), "b": 0.0})
    progress("running the existing tuning procedure with b=0")
    tuning_start = time.perf_counter()
    tuning = ns["run_calibrated_tuning_over_Rmax"](
        R_max_list=[10],
        MCMC_results=pilot_results,
        MAP=MAP,
        phiStar=MAP,
        dens_at_phiStar=dens_at_phiStar,
        grad_at_phiStar=grad_at_phiStar,
        Hess_at_phiStar=Hess_at_phiStar,
        args=args,
        T=T,
        t_star_frac=0.01,
        b=0.0,
        m_min=2,
        pilot_slice=slice(1, 61, 5),
        verbose=False,
    )
    tuning_seconds = time.perf_counter() - tuning_start
    tuned = tuning[10]
    progress(
        f"tuning complete: c={tuned['c_opt']:.6g}, "
        f"gamma={tuned['gamma_opt']:.6g}, m={tuned['m_opt']}"
    )

    probabilities = np.asarray(tuned["probs_opt"], dtype=float)
    m_opt = max(2, int(tuned["m_opt"]))
    timing_rng = random.Random(ESTIMATOR_TIMING_SEED)
    theta_MAP = hinv(MAP)

    def one_subsampling_estimate():
        u = timing_rng.choices(range(T), weights=probabilities, k=m_opt)
        u_max = max(u)
        l_prefix = ns["log_densities"](
            theta_MAP, u_max, 1, 1, y[: u_max + 1], y_start_arr, sigma2_start_arr
        )
        q_values = ns["eval_q_k"](
            MAP,
            MAP,
            dens_at_phiStar[u],
            grad_at_phiStar[u],
            Hess_at_phiStar[u],
            order=2,
        )
        estimate = q_sum(MAP) + ns["log_est_WDE"](
            l_prefix[u] - q_values, probabilities[u], m_opt
        )
        return float(estimate), int(u_max)

    estimator_values, subsampling_estimator_timing = timed_repetitions(
        one_subsampling_estimate, repetitions=100
    )
    estimator_log_values = [value[0] for value in estimator_values]
    estimator_u_max = [value[1] for value in estimator_values]
    if not np.all(np.isfinite(estimator_log_values)):
        raise AssertionError("Non-finite subsampling likelihood estimate during timing.")

    np.random.seed(FULL_SMOKE_SEED)
    random.seed(FULL_SMOKE_SEED)
    progress("running the short full-data MCMC smoke test")
    full_smoke = ns["sample_full_data_posterior_AM"](
        MAP,
        proposal_cov,
        50,
        names,
        DATASET_ID,
        adapt_cov=False,
        adapt_scale=False,
        burn_in=1000,
    )

    estimator_params = {
        "m_opt": int(tuned["m_opt"]),
        "c_opt": float(tuned["c_opt"]),
        "gamma_opt": float(tuned["gamma_opt"]),
        "t_star": int(0.01 * T),
        "b": 0.0,
        "V_tolerance": float(tuned["V_target"]),
        "R_max": 10,
    }
    np.random.seed(SUBSAMPLING_SMOKE_SEED)
    random.seed(SUBSAMPLING_SMOKE_SEED)
    progress("running the short subsampling-MCMC smoke test")
    subsampling_smoke = ns["sample_subsampling_posterior_random_walk"](
        estimator_params,
        MAP,
        proposal_cov,
        50,
        names,
        DATASET_ID,
        adapt=False,
    )

    required_finite = (
        full_smoke["samples_theta"],
        subsampling_smoke["samples_theta"],
        subsampling_smoke["log_posthat_samples"],
        subsampling_smoke["sigma2_lhat_samples"],
        subsampling_smoke["u_max_prop"],
    )
    if not all(np.all(np.isfinite(value)) for value in required_finite):
        raise AssertionError("Smoke tests produced non-finite stored values.")

    array_bytes = {
        "density": int(dens_at_phiStar.nbytes),
        "gradient": int(grad_at_phiStar.nbytes),
        "Hessian": int(Hess_at_phiStar.nbytes),
    }
    stored_taylor_bytes = sum(array_bytes.values())
    taylor_memory_report = taylor_memory.report()
    taylor_memory_report["stored_arrays_MiB"] = stored_taylor_bytes / 1024**2
    taylor_memory_report["peak_increment_minus_stored_arrays_MiB"] = (
        (taylor_memory.peak_bytes - taylor_memory.baseline_bytes - stored_taylor_bytes)
        / 1024**2
    )
    taylor_memory_report["extra_peak_to_Hessian_ratio"] = max(
        0.0,
        taylor_memory.peak_bytes
        - taylor_memory.baseline_bytes
        - stored_taylor_bytes,
    ) / Hess_at_phiStar.nbytes

    total_seconds = time.perf_counter() - total_start
    peak_rss_MiB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    progress(f"profiling complete in {total_seconds:.1f} seconds")
    return {
        "dataset": {
            "id": DATASET_ID,
            "path": str(path.resolve()),
            "generated_now": generated_now,
            "seed": DATASET_SEED,
            "burn_in": BURN_IN,
            "T": T,
            "p_reg": P_REG,
            "dimension": d,
            "rho": RHO,
            "archive_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "generation_seconds": generation_seconds,
            "load_seconds": load_seconds,
            "beta_reg": dataset["params"]["beta_reg"].tolist(),
            "checks": data_checks,
            "presample": {
                "X_start": dataset["X_start"].tolist(),
                "y_start": dataset["y_start"],
                "z_start": dataset["z_start"],
                "sigma2_start": dataset["sigma2_start"],
                "student_t_draw_start": dataset["student_t_draw_start"],
                "standardised_innovation_start": dataset[
                    "standardised_innovation_start"
                ],
            },
        },
        "prior": {
            "beta_reg": "independent N(0, 1)",
            "other_components": "unchanged existing GARCH/Student-t priors",
        },
        "MAP": {
            "success": bool(map_result.success),
            "message": str(map_result.message),
            "iterations": int(map_result.nit),
            "function_evaluations": int(map_result.nfev),
            "gradient_evaluations": int(map_result.njev),
            "gradient_infinity_norm_phi": float(np.max(np.abs(map_gradient_phi))),
            "gradient_infinity_norm_psi": float(np.max(np.abs(map_result.jac))),
            "optimisation_coordinates": "stationarity-preserving Psi (delta=0.02)",
            "any_numerical_bound_active": bool(
                any(
                    np.isclose(value, lower, atol=1e-6)
                    or np.isclose(value, upper, atol=1e-6)
                    for value, (lower, upper) in zip(map_result.x, map_bounds)
                )
            ),
            "seconds": map_seconds,
            "memory": map_memory.report(),
            "theta": np.asarray(hinv(MAP), dtype=float).tolist(),
        },
        "Taylor_control_variates": {
            "seconds": taylor_seconds,
            "density_shape": list(dens_at_phiStar.shape),
            "gradient_shape": list(grad_at_phiStar.shape),
            "Hessian_shape": list(Hess_at_phiStar.shape),
            "array_bytes": array_bytes,
            "memory": taylor_memory_report,
        },
        "timing": {
            "full_data_likelihood": full_likelihood_timing,
            "subsampling_likelihood_estimate": subsampling_estimator_timing,
            "subsampling_estimate_u_max_mean": float(np.mean(estimator_u_max)),
            "subsampling_estimate_u_max_min": int(np.min(estimator_u_max)),
            "subsampling_estimate_u_max_max": int(np.max(estimator_u_max)),
        },
        "pilot_full_MCMC": {
            "iterations": 60,
            "seed": PILOT_MCMC_SEED,
            "seconds": float(pilot["run_time"]),
            "seconds_per_iteration": float(pilot["run_time"] / 60),
            "mean_acceptance_probability": float(np.mean(pilot["alphas_MH"])),
            "outside_support": int(pilot["outside_support"]),
        },
        "tuning": {
            "seconds": tuning_seconds,
            "R_max": 10,
            "b": 0.0,
            "t_star": int(0.01 * T),
            "c_opt": float(tuned["c_opt"]),
            "gamma_opt": float(tuned["gamma_opt"]),
            "m_opt": int(tuned["m_opt"]),
            "V_target": float(tuned["V_target"]),
            "expected_u_max": float(
                ns["expected_umax_safe"](probabilities, int(tuned["m_opt"]))
            ),
        },
        "full_MCMC_smoke": {
            "iterations": 50,
            "seed": FULL_SMOKE_SEED,
            "seconds": float(full_smoke["run_time"]),
            "seconds_per_iteration": float(full_smoke["run_time"] / 50),
            "mean_acceptance_probability": float(np.mean(full_smoke["alphas_MH"])),
            "outside_support": int(full_smoke["outside_support"]),
            "stored_shape": list(full_smoke["samples_theta"].shape),
        },
        "subsampling_MCMC_smoke": {
            "iterations": 50,
            "seed": SUBSAMPLING_SMOKE_SEED,
            "seconds": float(subsampling_smoke["run_time"]),
            "seconds_per_iteration": float(subsampling_smoke["run_time"] / 50),
            "mean_acceptance_probability": float(
                np.mean(subsampling_smoke["alphas_MH"])
            ),
            "outside_support": int(subsampling_smoke["outside_support"]),
            "stored_shape": list(subsampling_smoke["samples_theta"].shape),
            "variance_estimate_min": float(
                np.min(subsampling_smoke["sigma2_lhat_samples"])
            ),
            "variance_estimate_max": float(
                np.max(subsampling_smoke["sigma2_lhat_samples"])
            ),
            "u_max_min": float(np.min(subsampling_smoke["u_max_prop"])),
            "u_max_mean": float(np.mean(subsampling_smoke["u_max_prop"])),
            "u_max_max": float(np.max(subsampling_smoke["u_max_prop"])),
        },
        "warnings": captured_warnings,
        "process_peak_rss_MiB": peak_rss_MiB,
        "total_seconds": total_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = profile(args.dataset)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
