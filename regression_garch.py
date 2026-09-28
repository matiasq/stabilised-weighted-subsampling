"""Student-t linear-regression GARCH(1,1) extension.

This module is intentionally separate from the existing notebook implementation.  It
adds only model-specific code and exposes adapters with the call signature expected by
the notebook's existing likelihood, Taylor-control-variate, and MCMC functions.

The parameter ordering is

    theta = (beta_reg[0:p_reg], omega, alpha, beta_garch, nu).

The covariate matrix is always supplied explicitly; this module never generates an
experimental design matrix.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.special import gammaln


def numerical_cholesky(covariance, *, max_tries=12):
    """Factor a covariance using no jitter unless floating-point repair is needed.

    The matrix is symmetrised first.  Positive-definite inputs are factorised
    exactly as supplied.  If Cholesky fails, the initial repair is tied to the
    matrix scale and machine precision, and is increased only until the
    factorisation succeeds.  The returned jitter is the actual diagonal value
    added to the symmetrised covariance.
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain only finite values.")
    symmetric = 0.5 * (covariance + covariance.T)
    try:
        return symmetric, np.linalg.cholesky(symmetric), 0.0
    except np.linalg.LinAlgError:
        pass

    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(np.diag(symmetric)))), np.finfo(float).tiny)
    roundoff = np.finfo(float).eps * scale
    jitter = max(roundoff, -float(eigenvalues[0]) + roundoff)
    identity = np.eye(symmetric.shape[0])
    for _ in range(max_tries):
        repaired = symmetric + jitter * identity
        try:
            return repaired, np.linalg.cholesky(repaired), float(jitter)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(
        "Cholesky failed after adaptive covariance-scale jitter."
    )


def sample_full_data_posterior_numerical_cholesky(
    phi_init,
    prop_cov_init,
    N,
    names,
    dataset_name,
    *,
    hinv,
    log_posterior,
    adapt_cov=False,
    adapt_scale=False,
    burn_in=2000,
    target_acceptance=0.23,
    progress_every=100,
):
    """Full-data random-walk Metropolis with numerical-only Cholesky repair.

    This preserves the proposal shape and scalar multiplier supplied by the
    caller.  Unlike the notebook sampler, it does not add a fixed ``1e-6 I``
    term.  The interface dependencies are explicit so the existing notebook
    need not be modified.
    """
    phi_init = np.asarray(phi_init, dtype=float).reshape(-1)
    d = phi_init.size
    covariance = np.asarray(prop_cov_init, dtype=float)
    if covariance.shape != (d, d):
        raise ValueError("prop_cov_init shape does not match phi_init.")
    samples_phi = np.zeros((N + 1, d))
    samples_theta = np.zeros((N + 1, d))
    log_p_samples = np.zeros(N + 1)
    alphas_MH = np.zeros(N)
    proposal_jitters = np.zeros(N)
    samples_phi[0] = phi_init
    samples_theta[0] = hinv(phi_init)

    phi_current = phi_init.copy()
    log_current, outside = log_posterior(phi_current)
    if outside or not np.isfinite(log_current):
        raise ValueError("Initial state is outside posterior support.")
    log_p_samples[0] = log_current
    outside_support = 0
    log_scale = np.log(0.1) if adapt_scale else 0.0
    running_covariance = covariance.copy()
    running_mean = phi_current.copy()
    start = time.time()

    fixed_factor = None
    fixed_jitter = None
    if not adapt_cov and not adapt_scale:
        _, fixed_factor, fixed_jitter = numerical_cholesky(running_covariance)

    for iteration in range(1, N + 1):
        if progress_every and iteration % progress_every == 0:
            print(
                f"Iteration {iteration}. Mean acc="
                f"{np.mean(alphas_MH[:iteration]):.2f}. "
                f"Time={time.time() - start:.2f}. "
                f"Outside support={outside_support}."
            )
        if fixed_factor is None:
            scale = np.exp(log_scale)
            effective = scale**2 * running_covariance
            _, factor, jitter = numerical_cholesky(effective)
        else:
            factor = fixed_factor
            jitter = fixed_jitter
        proposal_jitters[iteration - 1] = jitter
        phi_proposed = phi_current + factor @ np.random.randn(d)
        log_proposed, violates = log_posterior(phi_proposed)
        if violates or not np.isfinite(log_proposed):
            outside_support += 1
            alpha_MH = 0.0
        else:
            log_ratio = log_proposed - log_current
            alpha_MH = 1.0 if log_ratio >= 0.0 else float(np.exp(log_ratio))
        alphas_MH[iteration - 1] = alpha_MH
        if np.random.rand() < alpha_MH:
            phi_current = phi_proposed
            log_current = log_proposed
            accepted = 1.0
        else:
            accepted = 0.0
        samples_phi[iteration] = phi_current
        samples_theta[iteration] = hinv(phi_current)
        log_p_samples[iteration] = log_current

        if adapt_scale:
            log_scale += (accepted - target_acceptance) / iteration
        if adapt_cov and iteration > burn_in:
            weight = (iteration - burn_in) ** (-0.6)
            updated_mean = (
                (1.0 - weight) * running_mean
                + weight * samples_phi[iteration]
            )
            difference = samples_phi[iteration] - running_mean
            running_covariance = (
                (1.0 - weight) * running_covariance
                + weight * np.outer(difference, difference)
            )
            running_mean = updated_mean

    return {
        "samples_theta": samples_theta,
        "samples_phi": samples_phi,
        "log_p_samples": log_p_samples,
        "alphas_MH": alphas_MH,
        "outside_support": outside_support,
        "N": N,
        "run_time": time.time() - start,
        "names": names,
        "dataset_name": dataset_name,
        "proposal_jitters": proposal_jitters,
        "maximum_proposal_jitter": float(np.max(proposal_jitters)),
    }


def get_param_names_regression_GARCH11(p_reg: int):
    """Return original- and transformed-parameter display names."""
    if not isinstance(p_reg, int) or p_reg < 1:
        raise ValueError("p_reg must be a positive integer.")

    beta_names = [rf"$\beta_{{\mathrm{{reg}},{j}}}$" for j in range(1, p_reg + 1)]
    beta_phi_names = [rf"$\widetilde{{\beta}}_{{\mathrm{{reg}},{j}}}$" for j in range(1, p_reg + 1)]
    param_names = beta_names + [
        r"$\omega$",
        r"$\alpha$",
        r"$\beta_{\mathrm{garch}}$",
        r"$\nu$",
    ]
    trans_param_names = beta_phi_names + [
        r"$\widetilde{\omega}$",
        r"$\widetilde{\alpha}$",
        r"$\widetilde{\beta}_{\mathrm{garch}}$",
        r"$\widetilde{\nu}$",
    ]
    return param_names, trans_param_names


def regression_GARCH11_transform_names(p_reg: int):
    """Transformation specification for the existing transformation factory."""
    if not isinstance(p_reg, int) or p_reg < 1:
        raise ValueError("p_reg must be a positive integer.")
    return p_reg * ["identity"] + ["log", "log", "log", "log_shift2"]


def _as_scalar_start(value, name: str) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"{name} must contain exactly one value for GARCH(1,1).")
    return float(arr[0])


def _validate_regression_data(y, X, y_start, X_start, sigma2_start):
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    X_start = np.asarray(X_start, dtype=float).reshape(-1)
    y_start = _as_scalar_start(y_start, "y_start")
    sigma2_start = _as_scalar_start(sigma2_start, "sigma2_start")

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if X.shape[0] != y.size:
        raise ValueError("X and y must have the same number of observations.")
    if X.shape[1] < 1:
        raise ValueError("X must have at least one column.")
    if X_start.shape != (X.shape[1],):
        raise ValueError("X_start must have one value per column of X.")
    if sigma2_start <= 0:
        raise ValueError("sigma2_start must be positive.")
    if not all(np.all(np.isfinite(x)) for x in (y, X, X_start)):
        raise ValueError("y, X, and X_start must be finite.")
    if not np.isfinite(y_start) or not np.isfinite(sigma2_start):
        raise ValueError("Presample values must be finite.")
    return y, X, y_start, X_start, sigma2_start


def _unpack_theta(theta, p_reg: int):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != p_reg + 4:
        raise ValueError(
            f"theta has length {theta.size}; expected p_reg + 4 = {p_reg + 4}."
        )
    beta_reg = theta[:p_reg]
    omega, alpha, beta_garch, nu = theta[p_reg:]
    return theta, beta_reg, omega, alpha, beta_garch, nu


def simulate_regression_GARCH11(
    X,
    beta_reg,
    omega,
    alpha,
    beta_garch,
    nu,
    y_start,
    X_start,
    sigma2_start,
    *,
    seed=None,
    rng=None,
    student_t_draws=None,
):
    """Simulate a Student-t regression-GARCH(1,1) series for a supplied X.

    ``student_t_draws`` are unstandardised t_nu draws.  Supplying them provides
    an exact reproducibility hook independent of an RNG implementation.
    """
    X = np.asarray(X, dtype=float)
    beta_reg = np.asarray(beta_reg, dtype=float).reshape(-1)
    X_start = np.asarray(X_start, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[1] != beta_reg.size:
        raise ValueError("X must have shape (T, len(beta_reg)).")
    if X_start.shape != beta_reg.shape:
        raise ValueError("X_start must have shape (len(beta_reg),).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(beta_reg)):
        raise ValueError("X and beta_reg must be finite.")
    if not np.all(np.isfinite(X_start)):
        raise ValueError("X_start must be finite.")
    if omega <= 0 or alpha < 0 or beta_garch < 0:
        raise ValueError("Require omega > 0 and alpha, beta_garch >= 0.")
    if alpha + beta_garch >= 1:
        raise ValueError("Require alpha + beta_garch < 1 for simulation.")
    if nu <= 2:
        raise ValueError("nu must be greater than 2.")
    if sigma2_start <= 0:
        raise ValueError("sigma2_start must be positive.")
    if rng is not None and seed is not None:
        raise ValueError("Supply at most one of rng and seed.")

    T, p_reg = X.shape
    if student_t_draws is None:
        rng = np.random.default_rng(seed) if rng is None else rng
        student_t_draws = rng.standard_t(df=nu, size=T)
    else:
        student_t_draws = np.asarray(student_t_draws, dtype=float).reshape(-1)
        if student_t_draws.size != T:
            raise ValueError("student_t_draws must have length X.shape[0].")

    standardised_innovations = student_t_draws * np.sqrt((nu - 2.0) / nu)
    y = np.empty(T, dtype=float)
    z = np.empty(T, dtype=float)
    sigma2 = np.empty(T, dtype=float)

    z_start = float(y_start) - float(X_start @ beta_reg)
    z_prev = z_start
    sigma2_prev = float(sigma2_start)
    for t in range(T):
        sigma2_t = omega + alpha * z_prev**2 + beta_garch * sigma2_prev
        z_t = np.sqrt(sigma2_t) * standardised_innovations[t]
        y_t = float(X[t] @ beta_reg) + z_t

        sigma2[t] = sigma2_t
        z[t] = z_t
        y[t] = y_t
        z_prev = z_t
        sigma2_prev = sigma2_t

    return {
        "X": X.copy(),
        "y": y,
        "z": z,
        "sigma2": sigma2,
        "student_t_draws": np.asarray(student_t_draws),
        "standardised_innovations": standardised_innovations,
        "y_start": float(y_start),
        "X_start": X_start.copy(),
        "z_start": z_start,
        "student_t_draw_start": z_start / (
            np.sqrt(float(sigma2_start)) * np.sqrt((nu - 2.0) / nu)
        ),
        "standardised_innovation_start": z_start / np.sqrt(float(sigma2_start)),
        "sigma2_start": float(sigma2_start),
        "seed": seed,
        "params": {
            "p_reg": p_reg,
            "beta_reg": beta_reg.copy(),
            "omega": float(omega),
            "alpha": float(alpha),
            "beta_garch": float(beta_garch),
            "nu": float(nu),
        },
    }


def save_regression_GARCH11_dataset(path, simulation):
    """Save one realised dataset for reuse by every downstream method.

    The archive contains the supplied design, response, latent simulation values,
    presample values, parameters, and seed.  Downstream full-data, subsampling, and
    DC-BATS drivers should load this archive instead of simulating independently.
    """
    required = {
        "X",
        "y",
        "z",
        "sigma2",
        "student_t_draws",
        "standardised_innovations",
        "y_start",
        "X_start",
        "z_start",
        "student_t_draw_start",
        "standardised_innovation_start",
        "sigma2_start",
        "seed",
        "params",
    }
    missing = required.difference(simulation)
    if missing:
        raise ValueError(f"simulation is missing required entries: {sorted(missing)}")

    params = simulation["params"]
    metadata = {
        "format": "regression_GARCH11_dataset_v2",
        "seed": None if simulation["seed"] is None else int(simulation["seed"]),
        "dataset_id": simulation.get("dataset_id"),
        "burn_in": int(simulation.get("burn_in", 0)),
        "rho": simulation.get("rho"),
        "signal_variance": simulation.get("signal_variance"),
        "p_reg": int(params["p_reg"]),
        "omega": float(params["omega"]),
        "alpha": float(params["alpha"]),
        "beta_garch": float(params["beta_garch"]),
        "nu": float(params["nu"]),
    }
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    payload = {
        "X": np.asarray(simulation["X"], dtype=float),
        "y": np.asarray(simulation["y"], dtype=float),
        "z": np.asarray(simulation["z"], dtype=float),
        "sigma2": np.asarray(simulation["sigma2"], dtype=float),
        "student_t_draws": np.asarray(simulation["student_t_draws"], dtype=float),
        "standardised_innovations": np.asarray(
            simulation["standardised_innovations"], dtype=float
        ),
        "y_start": np.asarray(float(simulation["y_start"])),
        "X_start": np.asarray(simulation["X_start"], dtype=float),
        "z_start": np.asarray(float(simulation["z_start"])),
        "student_t_draw_start": np.asarray(float(simulation["student_t_draw_start"])),
        "standardised_innovation_start": np.asarray(
            float(simulation["standardised_innovation_start"])
        ),
        "sigma2_start": np.asarray(float(simulation["sigma2_start"])),
        "beta_reg": np.asarray(params["beta_reg"], dtype=float),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if "Sigma" in simulation:
        payload["Sigma"] = np.asarray(simulation["Sigma"], dtype=float)
    np.savez_compressed(path, **payload)
    return path


def load_regression_GARCH11_dataset(path):
    """Load an archive written by :func:`save_regression_GARCH11_dataset`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format") != "regression_GARCH11_dataset_v2":
            raise ValueError("Unsupported regression-GARCH dataset archive format.")
        result = {
            name: np.asarray(archive[name], dtype=float).copy()
            for name in (
                "X",
                "y",
                "z",
                "sigma2",
                "student_t_draws",
                "standardised_innovations",
                "X_start",
            )
        }
        result.update(
            {
                "y_start": float(archive["y_start"]),
                "sigma2_start": float(archive["sigma2_start"]),
                "z_start": float(archive["z_start"]),
                "student_t_draw_start": float(archive["student_t_draw_start"]),
                "standardised_innovation_start": float(
                    archive["standardised_innovation_start"]
                ),
                "seed": metadata["seed"],
                "dataset_id": metadata["dataset_id"],
                "burn_in": int(metadata["burn_in"]),
                "rho": metadata["rho"],
                "signal_variance": metadata["signal_variance"],
                "params": {
                    "p_reg": int(metadata["p_reg"]),
                    "beta_reg": np.asarray(archive["beta_reg"], dtype=float).copy(),
                    "omega": float(metadata["omega"]),
                    "alpha": float(metadata["alpha"]),
                    "beta_garch": float(metadata["beta_garch"]),
                    "nu": float(metadata["nu"]),
                },
            }
        )
        if "Sigma" in archive:
            result["Sigma"] = np.asarray(archive["Sigma"], dtype=float).copy()
    if (
        result["X"].shape[1] != result["params"]["p_reg"]
        or result["params"]["beta_reg"].shape != (result["params"]["p_reg"],)
    ):
        raise ValueError("Dataset archive has inconsistent p_reg metadata.")
    return result


def log_densities_regression_GARCH11(
    theta, y, X, y_start, X_start, sigma2_start
):
    """Per-observation standardised Student-t log densities in theta space."""
    y, X, y_start, X_start, sigma2_start = _validate_regression_data(
        y, X, y_start, X_start, sigma2_start
    )
    T, p_reg = X.shape
    bad = -1e12 * np.ones(T)
    try:
        theta, beta_reg, omega, alpha, beta_garch, nu = _unpack_theta(theta, p_reg)
    except ValueError:
        return bad

    if not np.all(np.isfinite(theta)):
        return bad
    if omega <= 0 or alpha < 0 or beta_garch < 0 or nu <= 2:
        return bad

    out = np.empty(T, dtype=float)
    z_prev = y_start - float(X_start @ beta_reg)
    sigma2_prev = sigma2_start
    for t in range(T):
        sigma2_t = omega + alpha * z_prev**2 + beta_garch * sigma2_prev
        if not np.isfinite(sigma2_t) or sigma2_t <= 0:
            return bad

        mean_t = float(X[t] @ beta_reg)
        resid_t = y[t] - mean_t
        val = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * (nu - 2.0))
            - 0.5 * np.log(sigma2_t)
            - ((nu + 1.0) / 2.0)
            * np.log(1.0 + resid_t**2 / (sigma2_t * (nu - 2.0)))
        )
        if not np.isfinite(val):
            return bad
        out[t] = val
        z_prev = resid_t
        sigma2_prev = sigma2_t
    return out


def _regression_garch_derivatives_theta(
    theta,
    y,
    X,
    y_start,
    X_start,
    sigma2_start,
    partials_1_deriv_t_l_t: Callable,
    partials_2_deriv_t_l_t: Callable,
):
    """Return contribution gradients/Hessians with respect to original theta.

    With ``c[t-1] = d z[t-1] / d theta_v``, the recursive variance terms are

    ``g[t] = e_omega + z[t-1]^2 e_alpha + h[t-1] e_beta``
    ``       + 2 alpha z[t-1] c[t-1] + beta_garch g[t-1]``

    and the Hessian is obtained by differentiating this expression directly.
    Here ``theta_v`` excludes nu, which affects the measurement density but not
    the conditional-variance path.  The scalar Student-t partial derivatives are
    supplied by the existing notebook implementation.
    """
    y, X, y_start, X_start, sigma2_start = _validate_regression_data(
        y, X, y_start, X_start, sigma2_start
    )
    T, p_reg = X.shape
    theta, beta_reg, omega, alpha, beta_garch, nu = _unpack_theta(theta, p_reg)
    if not np.all(np.isfinite(theta)) or omega <= 0 or alpha < 0 or beta_garch < 0 or nu <= 2:
        raise ValueError("theta is outside the valid regression-GARCH parameter space.")

    d_v = p_reg + 3
    d = d_v + 1
    idx_omega = p_reg
    idx_alpha = p_reg + 1
    idx_beta_garch = p_reg + 2

    e_omega = np.zeros(d_v)
    e_alpha = np.zeros(d_v)
    e_beta = np.zeros(d_v)
    e_omega[idx_omega] = 1.0
    e_alpha[idx_alpha] = 1.0
    e_beta[idx_beta_garch] = 1.0

    grad_out = np.empty((T, d), dtype=float)
    Hess_out = np.empty((T, d, d), dtype=float)

    z_prev = y_start - float(X_start @ beta_reg)
    c_prev = np.zeros(d_v)
    c_prev[:p_reg] = -X_start
    sigma2_prev = sigma2_start
    grad_sigma2_prev = np.zeros(d_v)
    Hess_sigma2_prev = np.zeros((d_v, d_v))

    for t in range(T):
        sigma2_t = omega + alpha * z_prev**2 + beta_garch * sigma2_prev
        if not np.isfinite(sigma2_t) or sigma2_t <= 0:
            raise ValueError("The conditional variance recursion became invalid.")

        grad_sigma2 = (
            e_omega
            + z_prev**2 * e_alpha
            + sigma2_prev * e_beta
            + 2.0 * alpha * z_prev * c_prev
            + beta_garch * grad_sigma2_prev
        )

        Hess_sigma2 = (
            2.0 * alpha * np.outer(c_prev, c_prev)
            + 2.0
            * z_prev
            * (np.outer(e_alpha, c_prev) + np.outer(c_prev, e_alpha))
            + np.outer(e_beta, grad_sigma2_prev)
            + np.outer(grad_sigma2_prev, e_beta)
            + beta_garch * Hess_sigma2_prev
        )

        mean_t = float(X[t] @ beta_reg)
        dmean = np.zeros(d_v)
        dmean[:p_reg] = X[t]

        l_m = partials_1_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dmu")
        l_h = partials_1_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dsigma2")
        l_nu = partials_1_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dv")

        l_mm = partials_2_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dmudmu")
        l_hh = partials_2_deriv_t_l_t(
            mean_t, nu, sigma2_t, y[t], "dsigma2dsigma2"
        )
        l_hm = partials_2_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dsigma2dmu")
        l_mnu = partials_2_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dmudv")
        l_hnu = partials_2_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dsigma2dv")
        l_nunu = partials_2_deriv_t_l_t(mean_t, nu, sigma2_t, y[t], "dvdv")

        grad_v = l_m * dmean + l_h * grad_sigma2
        Hess_v = (
            l_mm * np.outer(dmean, dmean)
            + l_hm
            * (np.outer(dmean, grad_sigma2) + np.outer(grad_sigma2, dmean))
            + l_hh * np.outer(grad_sigma2, grad_sigma2)
            + l_h * Hess_sigma2
        )
        cross_nu = l_mnu * dmean + l_hnu * grad_sigma2

        grad_out[t, :d_v] = grad_v
        grad_out[t, -1] = l_nu
        Hess_out[t, :d_v, :d_v] = Hess_v
        Hess_out[t, :d_v, -1] = cross_nu
        Hess_out[t, -1, :d_v] = cross_nu
        Hess_out[t, -1, -1] = l_nunu

        z_prev = y[t] - mean_t
        c_prev = np.zeros(d_v)
        c_prev[:p_reg] = -X[t]
        sigma2_prev = sigma2_t
        grad_sigma2_prev = grad_sigma2
        Hess_sigma2_prev = Hess_sigma2

    return grad_out, Hess_out


def grad_log_densities_regression_GARCH11(
    theta,
    y,
    X,
    y_start,
    X_start,
    sigma2_start,
    *,
    h,
    hinv_p,
    partials_1_deriv_t_l_t,
    partials_2_deriv_t_l_t,
):
    """Per-observation gradient with respect to transformed phi."""
    grad_theta, _ = _regression_garch_derivatives_theta(
        theta,
        y,
        X,
        y_start,
        X_start,
        sigma2_start,
        partials_1_deriv_t_l_t,
        partials_2_deriv_t_l_t,
    )
    phi = h(theta)
    return grad_theta * np.asarray(hinv_p(phi), dtype=float)


def grad_Hess_log_densities_regression_GARCH11(
    theta,
    y,
    X,
    y_start,
    X_start,
    sigma2_start,
    *,
    h,
    hinv_p,
    hinv_pp,
    partials_1_deriv_t_l_t,
    partials_2_deriv_t_l_t,
):
    """Return transformed gradients and Hessians from one recursive pass."""
    grad_theta, Hess_theta = _regression_garch_derivatives_theta(
        theta,
        y,
        X,
        y_start,
        X_start,
        sigma2_start,
        partials_1_deriv_t_l_t,
        partials_2_deriv_t_l_t,
    )
    phi = h(theta)
    first = np.asarray(hinv_p(phi), dtype=float)
    second = np.asarray(hinv_pp(phi), dtype=float)
    grad_phi = grad_theta * first
    Hess_phi = Hess_theta * np.outer(first, first)[None, :, :]
    diag_idx = np.arange(len(theta))
    Hess_phi[:, diag_idx, diag_idx] += grad_theta * second
    return grad_phi, Hess_phi


def Hess_log_densities_regression_GARCH11(
    theta,
    y,
    X,
    y_start,
    X_start,
    sigma2_start,
    *,
    h,
    hinv_p,
    hinv_pp,
    partials_1_deriv_t_l_t,
    partials_2_deriv_t_l_t,
):
    """Per-observation Hessian with respect to transformed phi."""
    _, Hess_phi = grad_Hess_log_densities_regression_GARCH11(
        theta,
        y,
        X,
        y_start,
        X_start,
        sigma2_start,
        h=h,
        hinv_p=hinv_p,
        hinv_pp=hinv_pp,
        partials_1_deriv_t_l_t=partials_1_deriv_t_l_t,
        partials_2_deriv_t_l_t=partials_2_deriv_t_l_t,
    )
    return Hess_phi


def initiate_regression_GARCH11_model_functions(
    X,
    X_start,
    *,
    h,
    hinv_p,
    hinv_pp,
    partials_1_deriv_t_l_t,
    partials_2_deriv_t_l_t,
):
    """Return closures matching the existing notebook GARCH call signature."""
    X = np.asarray(X, dtype=float)
    X_start = np.asarray(X_start, dtype=float).reshape(-1)
    if X.ndim != 2 or X_start.shape != (X.shape[1],):
        raise ValueError("X/X_start shapes are inconsistent.")

    def _data_for_call(T, p, q, y):
        if p != 1 or q != 1:
            raise ValueError("The regression extension currently supports GARCH(1,1) only.")
        y_arr = np.asarray(y).reshape(-1)
        if y_arr.size > X.shape[0]:
            raise ValueError("The likelihood call requests more rows than X contains.")
        if T not in (y_arr.size, y_arr.size - 1):
            # Existing recursive calls occasionally pass a zero-based maximum index.
            # len(y), rather than T, is authoritative in the original implementation.
            pass
        return X[: y_arr.size]

    def log_densities(theta, T, p, q, y, y_start, sigma2_start):
        X_call = _data_for_call(T, p, q, y)
        return log_densities_regression_GARCH11(
            theta, y, X_call, y_start, X_start, sigma2_start
        )

    def grad_log_densities(theta, T, p, q, y, y_start, sigma2_start):
        X_call = _data_for_call(T, p, q, y)
        return grad_log_densities_regression_GARCH11(
            theta,
            y,
            X_call,
            y_start,
            X_start,
            sigma2_start,
            h=h,
            hinv_p=hinv_p,
            partials_1_deriv_t_l_t=partials_1_deriv_t_l_t,
            partials_2_deriv_t_l_t=partials_2_deriv_t_l_t,
        )

    def Hess_log_densities(theta, T, p, q, y, y_start, sigma2_start):
        X_call = _data_for_call(T, p, q, y)
        return Hess_log_densities_regression_GARCH11(
            theta,
            y,
            X_call,
            y_start,
            X_start,
            sigma2_start,
            h=h,
            hinv_p=hinv_p,
            hinv_pp=hinv_pp,
            partials_1_deriv_t_l_t=partials_1_deriv_t_l_t,
            partials_2_deriv_t_l_t=partials_2_deriv_t_l_t,
        )

    return log_densities, grad_log_densities, Hess_log_densities


def initiate_regression_GARCH11_prior(
    *,
    normal_logpdf_var,
    halfnormal_logpdf,
    shifted_gamma_logpdf,
    np_backend,
    beta_reg_mean=0.0,
    beta_reg_sd=10.0,
    omega_scale=1.0,
    alpha_scale=0.2,
    beta_garch_scale=0.8,
    nu_shape=2.0,
    nu_rate=1.0,
):
    """Create a prior with the existing GARCH and nu prior components.

    ``beta_reg_mean`` and ``beta_reg_sd`` are validation defaults only.  They
    are deliberately explicit because the scientific prior for the eventual
    p_reg=25/50 experiment has not been selected.
    """
    if beta_reg_sd <= 0:
        raise ValueError("beta_reg_sd must be positive.")

    def log_prior(theta, p, q):
        if p != 1 or q != 1:
            raise ValueError("The regression extension currently supports GARCH(1,1) only.")
        p_reg = len(theta) - 4
        if p_reg < 1:
            raise ValueError("theta must contain at least one regression coefficient.")
        beta_reg = theta[:p_reg]
        omega, alpha, beta_garch, nu = theta[p_reg:]

        lp = np_backend.sum(
            normal_logpdf_var(beta_reg, beta_reg_mean, beta_reg_sd**2)
        )
        lp = lp + halfnormal_logpdf(omega, omega_scale)
        lp = lp + halfnormal_logpdf(alpha, alpha_scale)
        lp = lp + halfnormal_logpdf(beta_garch, beta_garch_scale)
        lp = lp + shifted_gamma_logpdf(nu, nu_shape, nu_rate, full=True)
        support = (omega > 0.0) & (alpha >= 0.0) & (beta_garch >= 0.0)
        support = support & ((alpha + beta_garch) < 1.0) & (nu > 2.0)
        return np_backend.where(support, lp, -np_backend.inf)

    return log_prior


def log_penalty_regression_GARCH11(
    theta,
    p,
    q,
    *,
    np_backend=np,
    lambda_pos=10.0,
    lambda_stat=100.0,
    delta_stat=0.02,
    lambda_nu_low=0.0,
    nu_min=2.0,
):
    """Constraint-only optimisation penalty for regression-GARCH(1,1)."""
    if p != 1 or q != 1:
        raise ValueError("The regression extension currently supports GARCH(1,1) only.")
    p_reg = len(theta) - 4
    omega, alpha, beta_garch, nu = theta[p_reg:]
    neg = (
        np_backend.maximum(0.0, -omega) ** 2
        + np_backend.maximum(0.0, -alpha) ** 2
        + np_backend.maximum(0.0, -beta_garch) ** 2
    )
    stat = np_backend.maximum(0.0, alpha + beta_garch - (1.0 - delta_stat)) ** 2
    nu_shortfall = np_backend.maximum(0.0, nu_min - nu) ** 2
    return (
        lambda_pos * neg
        + lambda_stat * stat
        + lambda_nu_low * nu_shortfall
    )


def initiate_regression_GARCH11_persistence_reparam_functions(
    *, p_reg, h, hinv, jacdiag, np_backend, delta=0.02, eps=1e-12
):
    """Stationarity-preserving proposal map for the new parameter ordering."""
    if p_reg < 1:
        raise ValueError("p_reg must be positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0,1).")
    d = p_reg + 4
    idx_alpha = p_reg + 1
    idx_beta = p_reg + 2
    persistence_idx = {idx_alpha, idx_beta}
    untouched_idx = [j for j in range(d) if j not in persistence_idx]

    def _sigmoid(x):
        return 1.0 / (1.0 + np_backend.exp(-x))

    def theta_to_psi(theta):
        if len(theta) != d:
            raise ValueError(f"Expected theta length {d}.")
        phi = h(theta)
        alpha = theta[idx_alpha]
        beta = theta[idx_beta]
        persistence = alpha + beta
        scaled = np_backend.clip(persistence / (1.0 - delta), eps, 1.0 - eps)
        eta = np_backend.log(scaled) - np_backend.log(1.0 - scaled)
        log_ratio = np_backend.log(np_backend.maximum(alpha, eps)) - np_backend.log(
            np_backend.maximum(beta, eps)
        )
        out = [phi[j] for j in range(d)]
        out[idx_alpha] = eta
        out[idx_beta] = log_ratio
        return np_backend.stack(out)

    def psi_to_theta(psi):
        if len(psi) != d:
            raise ValueError(f"Expected psi length {d}.")
        theta_base = hinv(psi)
        persistence = (1.0 - delta) * _sigmoid(psi[idx_alpha])
        ratio = np_backend.exp(psi[idx_beta])
        alpha = persistence * ratio / (1.0 + ratio)
        beta = persistence / (1.0 + ratio)
        out = [theta_base[j] for j in range(d)]
        out[idx_alpha] = alpha
        out[idx_beta] = beta
        return np_backend.stack(out)

    def logabsdet_dpsi_dtheta(theta):
        phi = h(theta)
        dtheta_dphi = jacdiag(phi)
        untouched = -np_backend.sum(
            np_backend.log(np_backend.maximum(np_backend.abs(dtheta_dphi[untouched_idx]), eps))
        )
        alpha = theta[idx_alpha]
        beta = theta[idx_beta]
        remain = 1.0 - (alpha + beta) / (1.0 - delta)
        block = (
            -np_backend.log(np_backend.maximum(alpha, eps))
            - np_backend.log(np_backend.maximum(beta, eps))
            - np_backend.log(np_backend.maximum(remain, eps))
        )
        return untouched + block

    return theta_to_psi, psi_to_theta, logabsdet_dpsi_dtheta
