"""Validation runner for the Student-t regression-GARCH(1,1) extension.

The runner deliberately loads only the function-definition cells from the existing
notebook.  It does not execute or modify any existing experiment.  By exercising the
real notebook functions, it also guards against accidental changes to the submitted
GARCH implementation.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import inspect
import json
import random
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numdifftools as nd
import numpy as np
import pandas as pd
import scipy.stats as sps
from autograd import grad, hessian
from autograd import jacobian as Jacobian_autograd
from autograd import numpy as np_autograd
from autograd.scipy.special import gammaln as autograd_gammaln
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import basinhopping, brentq, minimize, minimize_scalar
from scipy.special import digamma, gammaln, polygamma

import regression_garch as rg


NOTEBOOK_PATH = Path(__file__).with_name("stabilised_weighted_data_subsampling.ipynb")


# These hashes were recorded from the unmodified Student-t GARCH(1,1) functions
# before the regression-GARCH implementation was added.  Arrays are rounded to
# 12 decimal places and serialised as little-endian float64 before hashing.
EXISTING_GARCH_FIXTURE_HASHES = {
    "log_densities": "990bfc315bfefabd9ef63b4d99506247863398ffd787854e73c547ffc5a7cb35",
    "log_likelihood": "4674ac8961c1066b40d2fb44b901eb4048fb73500db02768641b60762ac25f48",
    "grad_densities": "534cc80131def2ed79bb36e254a0a3a2574aba6b8cc6407a204167059e2eb63b",
    "grad_log_likelihood": "247c1141d6e5c7c1b85d27c2c6d24f95bf2ce1c6d9a65c561b279b453379d3cb",
    "Hess_densities": "a38c5c0aae78d3a7299400d108fcc5f71e13895f1d28dfd37c460f0ef1f9319e",
    "Hess_log_likelihood": "495af63b215221269de6e4b4f42b979e70dfb07904ef64fe45f01a522f921c70",
}


def load_existing_notebook_functions():
    """Load definition cells without running any application cells."""
    with NOTEBOOK_PATH.open(encoding="utf-8") as f:
        notebook = json.load(f)

    namespace = globals().copy()
    for cell_index in (7, 9, 11, 13):
        source = "".join(notebook["cells"][cell_index]["source"])
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{cell_index}", "exec"), namespace)
    return namespace


def array_hash(x):
    rounded = np.ascontiguousarray(np.round(np.asarray(x), 12).astype("<f8"))
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def discrepancy(reference, candidate, relative_floor=1e-12):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    absolute = np.abs(candidate - reference)
    denominator = np.maximum(np.maximum(np.abs(reference), np.abs(candidate)), relative_floor)
    return {
        "max_absolute": float(np.max(absolute)),
        "max_relative": float(np.max(absolute / denominator)),
    }


def existing_garch_fixture(ns):
    theta = np.array([0.05, 0.08, 0.12, 0.78, 7.5])
    y = np.array([0.10, -0.20, 0.35, -0.15, 0.05, 0.42, -0.31])
    y_start = np.array([0.15])
    sigma2_start = np.array([0.8])
    T = len(y)
    transforms = ["identity", "log", "log", "log", "log_shift2"]
    h, hinv, hinv_p, hinv_pp, _, _ = ns["initiate_transformation_functions"](transforms)

    log_densities = ns["log_densities_GARCH"](
        theta, T, 1, 1, y, y_start, sigma2_start, "Student-t"
    )
    grad_densities = ns["grad_log_densities_GARCH"](
        theta, T, 1, 1, y, y_start, sigma2_start, "Student-t", h, hinv_p
    )
    Hess_densities = ns["Hess_log_densities_GARCH"](
        theta, T, 1, 1, y, y_start, sigma2_start, "Student-t", h, hinv_p, hinv_pp
    )
    values = {
        "log_densities": log_densities,
        "log_likelihood": np.array([np.sum(log_densities)]),
        "grad_densities": grad_densities,
        "grad_log_likelihood": np.sum(grad_densities, axis=0),
        "Hess_densities": Hess_densities,
        "Hess_log_likelihood": np.sum(Hess_densities, axis=0),
    }
    hashes = {name: array_hash(value) for name, value in values.items()}
    if hashes != EXISTING_GARCH_FIXTURE_HASHES:
        mismatch = {
            name: {"expected": EXISTING_GARCH_FIXTURE_HASHES[name], "actual": hashes[name]}
            for name in hashes
            if hashes[name] != EXISTING_GARCH_FIXTURE_HASHES[name]
        }
        raise AssertionError(f"Existing GARCH regression fixture changed: {mismatch}")
    return {
        "passed": True,
        "hashes": hashes,
        "log_densities": log_densities.tolist(),
        "total_log_likelihood": float(np.sum(log_densities)),
        "total_gradient": np.sum(grad_densities, axis=0).tolist(),
        "total_Hessian": np.sum(Hess_densities, axis=0).tolist(),
    }


def make_transforms(ns, p_reg):
    return ns["initiate_transformation_functions"](
        rg.regression_GARCH11_transform_names(p_reg)
    )


def derivative_keywords(ns, h, hinv_p, hinv_pp=None):
    out = {
        "h": h,
        "hinv_p": hinv_p,
        "partials_1_deriv_t_l_t": ns["partials_1_deriv_t_l_t"],
        "partials_2_deriv_t_l_t": ns["partials_2_deriv_t_l_t"],
    }
    if hinv_pp is not None:
        out["hinv_pp"] = hinv_pp
    return out


def constant_mean_reduction(ns):
    T = 60
    X = np.ones((T, 1))
    X_start = np.ones(1)
    theta = np.array([0.17, 0.075, 0.13, 0.76, 8.5])
    y_start_scalar = 0.11
    sigma2_start_scalar = 0.72
    draws = np.random.default_rng(4101).standard_t(theta[-1], size=T)
    sim = rg.simulate_regression_GARCH11(
        X,
        theta[:1],
        *theta[1:],
        y_start=y_start_scalar,
        X_start=X_start,
        sigma2_start=sigma2_start_scalar,
        student_t_draws=draws,
    )
    y = sim["y"]
    y_start = np.array([y_start_scalar])
    sigma2_start = np.array([sigma2_start_scalar])
    h, hinv, hinv_p, hinv_pp, _, _ = make_transforms(ns, 1)

    existing_log = ns["log_densities_GARCH"](
        theta, T, 1, 1, y, y_start, sigma2_start, "Student-t"
    )
    new_log = rg.log_densities_regression_GARCH11(
        theta, y, X, y_start_scalar, X_start, sigma2_start_scalar
    )
    existing_grad = ns["grad_log_densities_GARCH"](
        theta, T, 1, 1, y, y_start, sigma2_start, "Student-t", h, hinv_p
    )
    new_grad = rg.grad_log_densities_regression_GARCH11(
        theta,
        y,
        X,
        y_start_scalar,
        X_start,
        sigma2_start_scalar,
        **derivative_keywords(ns, h, hinv_p),
    )
    existing_Hess = ns["Hess_log_densities_GARCH"](
        theta, T, 1, 1, y, y_start, sigma2_start, "Student-t", h, hinv_p, hinv_pp
    )
    new_Hess = rg.Hess_log_densities_regression_GARCH11(
        theta,
        y,
        X,
        y_start_scalar,
        X_start,
        sigma2_start_scalar,
        **derivative_keywords(ns, h, hinv_p, hinv_pp),
    )

    comparisons = {
        "per_observation_log_densities": discrepancy(existing_log, new_log),
        "total_log_likelihood": discrepancy(np.sum(existing_log), np.sum(new_log)),
        "per_observation_gradients": discrepancy(existing_grad, new_grad),
        "total_gradient": discrepancy(np.sum(existing_grad, axis=0), np.sum(new_grad, axis=0)),
        "per_observation_Hessians": discrepancy(existing_Hess, new_Hess),
        "total_Hessian": discrepancy(np.sum(existing_Hess, axis=0), np.sum(new_Hess, axis=0)),
    }
    regression_prior = rg.initiate_regression_GARCH11_prior(
        normal_logpdf_var=ns["normal_logpdf_var"],
        halfnormal_logpdf=ns["halfnormal_logpdf"],
        shifted_gamma_logpdf=ns["shifted_gamma_logpdf"],
        np_backend=np_autograd,
    )
    prior_comparison = discrepancy(
        ns["log_prior_GARCH"](theta, 1, 1, "Student-t"),
        regression_prior(theta, 1, 1),
    )
    comparisons["log_prior"] = prior_comparison
    worst_absolute = max(x["max_absolute"] for x in comparisons.values())
    worst_relative = max(x["max_relative"] for x in comparisons.values())
    if worst_absolute > 1e-10 or worst_relative > 1e-9:
        raise AssertionError(f"Constant-mean reduction failed: {comparisons}")
    return {
        "passed": True,
        "T": T,
        "comparisons": comparisons,
        "worst_absolute": worst_absolute,
        "worst_relative": worst_relative,
    }


def numerical_derivative_validation(ns):
    rng = np.random.default_rng(5201)
    T = 50
    p_reg = 3
    X = rng.normal(size=(T, p_reg))
    X_start = rng.normal(size=p_reg)
    y_start = 0.08
    sigma2_start = 0.75
    theta_data = np.array([0.20, -0.12, 0.08, 0.08, 0.12, 0.76, 8.0])
    sim = rg.simulate_regression_GARCH11(
        X,
        theta_data[:p_reg],
        *theta_data[p_reg:],
        y_start=y_start,
        X_start=X_start,
        sigma2_start=sigma2_start,
        seed=5202,
    )
    y = sim["y"]

    parameter_vectors = [
        theta_data,
        np.array([0.15, -0.05, 0.12, 0.10, 0.09, 0.80, 6.5]),
        np.array([0.25, -0.18, 0.03, 0.06, 0.16, 0.70, 11.0]),
    ]
    h, hinv, hinv_p, hinv_pp, _, _ = make_transforms(ns, p_reg)
    results = []

    for point_index, theta in enumerate(parameter_vectors, start=1):
        phi = np.asarray(h(theta), dtype=float)

        def total_loglik(phi_arg):
            return np.sum(
                rg.log_densities_regression_GARCH11(
                    hinv(phi_arg), y, X, y_start, X_start, sigma2_start
                )
            )

        analytic_grad = np.sum(
            rg.grad_log_densities_regression_GARCH11(
                theta,
                y,
                X,
                y_start,
                X_start,
                sigma2_start,
                **derivative_keywords(ns, h, hinv_p),
            ),
            axis=0,
        )
        analytic_Hess = np.sum(
            rg.Hess_log_densities_regression_GARCH11(
                theta,
                y,
                X,
                y_start,
                X_start,
                sigma2_start,
                **derivative_keywords(ns, h, hinv_p, hinv_pp),
            ),
            axis=0,
        )
        numeric_grad = nd.Gradient(total_loglik, method="central", step=1e-5)(phi)
        numeric_Hess = nd.Hessian(total_loglik, method="central", step=2e-4)(phi)

        idx_omega = p_reg
        idx_alpha = p_reg + 1
        idx_beta_garch = p_reg + 2
        idx_nu = p_reg + 3
        blocks = {
            "full_gradient": discrepancy(numeric_grad, analytic_grad),
            "beta_reg_gradient": discrepancy(numeric_grad[:p_reg], analytic_grad[:p_reg]),
            "full_Hessian": discrepancy(numeric_Hess, analytic_Hess),
            "beta_beta": discrepancy(
                numeric_Hess[:p_reg, :p_reg], analytic_Hess[:p_reg, :p_reg]
            ),
            "beta_omega": discrepancy(
                numeric_Hess[:p_reg, idx_omega], analytic_Hess[:p_reg, idx_omega]
            ),
            "beta_alpha": discrepancy(
                numeric_Hess[:p_reg, idx_alpha], analytic_Hess[:p_reg, idx_alpha]
            ),
            "beta_beta_garch": discrepancy(
                numeric_Hess[:p_reg, idx_beta_garch],
                analytic_Hess[:p_reg, idx_beta_garch],
            ),
            "beta_nu": discrepancy(
                numeric_Hess[:p_reg, idx_nu], analytic_Hess[:p_reg, idx_nu]
            ),
        }
        symmetry = float(np.max(np.abs(analytic_Hess - analytic_Hess.T)))
        results.append(
            {
                "point": point_index,
                "blocks": blocks,
                "Hessian_symmetry_max_absolute": symmetry,
            }
        )

    theta = parameter_vectors[0]
    phi = np.asarray(h(theta), dtype=float)

    def late_log_density(phi_arg):
        return rg.log_densities_regression_GARCH11(
            hinv(phi_arg), y, X, y_start, X_start, sigma2_start
        )[-1]

    late_grad = rg.grad_log_densities_regression_GARCH11(
        theta,
        y,
        X,
        y_start,
        X_start,
        sigma2_start,
        **derivative_keywords(ns, h, hinv_p),
    )[-1]
    late_Hess = rg.Hess_log_densities_regression_GARCH11(
        theta,
        y,
        X,
        y_start,
        X_start,
        sigma2_start,
        **derivative_keywords(ns, h, hinv_p, hinv_pp),
    )[-1]
    late_numeric_grad = nd.Gradient(late_log_density, method="central", step=1e-5)(phi)
    late_numeric_Hess = nd.Hessian(late_log_density, method="central", step=2e-4)(phi)
    late = {
        "time_index": T,
        "gradient": discrepancy(late_numeric_grad, late_grad),
        "Hessian": discrepancy(late_numeric_Hess, late_Hess),
        "Hessian_symmetry_max_absolute": float(np.max(np.abs(late_Hess - late_Hess.T))),
    }

    worst_grad_abs = max(r["blocks"]["full_gradient"]["max_absolute"] for r in results)
    worst_Hess_abs = max(r["blocks"]["full_Hessian"]["max_absolute"] for r in results)
    worst_symmetry = max(r["Hessian_symmetry_max_absolute"] for r in results)
    if worst_grad_abs > 2e-5 or worst_Hess_abs > 1e-3 or worst_symmetry > 1e-12:
        raise AssertionError("Numerical derivative validation exceeded tolerances.")
    if late["gradient"]["max_absolute"] > 2e-5 or late["Hessian"]["max_absolute"] > 1e-3:
        raise AssertionError("Late-contribution derivative validation exceeded tolerances.")
    return {
        "passed": True,
        "T": T,
        "p_reg": p_reg,
        "points": results,
        "late_contribution": late,
        "worst_full_gradient_max_absolute": worst_grad_abs,
        "worst_full_Hessian_max_absolute": worst_Hess_abs,
        "worst_Hessian_symmetry_max_absolute": worst_symmetry,
    }


def configure_namespace_for_regression_model(ns, X, X_start, beta_reg_sd=10.0):
    p_reg = X.shape[1]
    h, hinv, hinv_p, hinv_pp, jacdiag, logdet = make_transforms(ns, p_reg)
    model_functions = rg.initiate_regression_GARCH11_model_functions(
        X,
        X_start,
        h=h,
        hinv_p=hinv_p,
        hinv_pp=hinv_pp,
        partials_1_deriv_t_l_t=ns["partials_1_deriv_t_l_t"],
        partials_2_deriv_t_l_t=ns["partials_2_deriv_t_l_t"],
    )
    log_prior = rg.initiate_regression_GARCH11_prior(
        normal_logpdf_var=ns["normal_logpdf_var"],
        halfnormal_logpdf=ns["halfnormal_logpdf"],
        shifted_gamma_logpdf=ns["shifted_gamma_logpdf"],
        np_backend=np_autograd,
        beta_reg_mean=0.0,
        beta_reg_sd=beta_reg_sd,
    )
    ns.update(
        {
            "h": h,
            "hinv": hinv,
            "hinv_p": hinv_p,
            "hinv_pp": hinv_pp,
            "jacdiag": jacdiag,
            "logdet": logdet,
            "log_densities": model_functions[0],
            "grad_log_densities": model_functions[1],
            "Hess_log_densities": model_functions[2],
            "log_prior": log_prior,
        }
    )
    return h, hinv, hinv_p, hinv_pp, jacdiag, logdet, log_prior


def end_to_end_validation(ns, T=600, p_reg=3):
    start_total = time.perf_counter()
    rng_X = np.random.default_rng(6301)
    X = rng_X.normal(size=(T, p_reg))
    X_start = rng_X.normal(size=p_reg)
    theta_true = np.array([0.18, -0.12, 0.07, 0.08, 0.12, 0.76, 8.0])
    y_start = float(X_start @ theta_true[:p_reg] + 0.10)
    sigma2_start = theta_true[p_reg] / (
        1.0 - theta_true[p_reg + 1] - theta_true[p_reg + 2]
    )
    sim = rg.simulate_regression_GARCH11(
        X,
        theta_true[:p_reg],
        *theta_true[p_reg:],
        y_start=y_start,
        X_start=X_start,
        sigma2_start=sigma2_start,
        seed=6302,
    )
    with tempfile.TemporaryDirectory(prefix="regression-garch-validation-") as temp_dir:
        dataset_path = Path(temp_dir) / "dataset.npz"
        saved_path = rg.save_regression_GARCH11_dataset(dataset_path, sim)
        loaded_sim = rg.load_regression_GARCH11_dataset(saved_path)
    arrays_to_compare = (
        "X",
        "y",
        "z",
        "sigma2",
        "student_t_draws",
        "standardised_innovations",
        "X_start",
    )
    if not all(np.array_equal(sim[name], loaded_sim[name]) for name in arrays_to_compare):
        raise AssertionError("Saved regression-GARCH dataset did not round-trip exactly.")
    beta_equal = np.array_equal(
        sim["params"]["beta_reg"], loaded_sim["params"]["beta_reg"]
    )
    scalar_params = set(sim["params"]).difference({"beta_reg"})
    scalar_equal = all(
        sim["params"][name] == loaded_sim["params"][name]
        for name in scalar_params
    )
    if not (beta_equal and scalar_equal):
        raise AssertionError("Saved dataset parameter metadata changed on round trip.")
    sim = loaded_sim
    X = sim["X"]
    X_start = sim["X_start"]
    y = sim["y"]
    y_start_arr = np.array([y_start])
    sigma2_start_arr = np.array([sigma2_start])

    h, hinv, hinv_p, hinv_pp, jacdiag, logdet, log_prior = (
        configure_namespace_for_regression_model(ns, X, X_start)
    )
    d = p_reg + 4
    transform_roundtrip = discrepancy(theta_true, hinv(h(theta_true)))
    theta_to_psi, psi_to_theta, logabsdet_dpsi_dtheta = (
        rg.initiate_regression_GARCH11_persistence_reparam_functions(
            p_reg=p_reg,
            h=h,
            hinv=hinv,
            jacdiag=jacdiag,
            np_backend=np,
        )
    )
    psi_true = theta_to_psi(theta_true)
    proposal_roundtrip = discrepancy(theta_true, psi_to_theta(psi_true))
    proposal_log_jacobian = float(logabsdet_dpsi_dtheta(theta_true))
    if transform_roundtrip["max_absolute"] > 1e-12:
        raise AssertionError("Regression-GARCH parameter transformation failed to invert.")
    if proposal_roundtrip["max_absolute"] > 1e-12:
        raise AssertionError("Persistence proposal map failed to invert.")
    if not np.isfinite(proposal_log_jacobian):
        raise AssertionError("Persistence proposal-map Jacobian is not finite.")
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

    phi_start = np.asarray(h(theta_true), dtype=float)
    map_start = time.perf_counter()
    map_result = minimize(
        lambda phi: -log_post_value(phi),
        phi_start,
        jac=lambda phi: -grad_log_post(phi),
        method="L-BFGS-B",
        options={"maxiter": 300, "gtol": 1e-7, "ftol": 1e-11, "maxls": 50},
    )
    map_seconds = time.perf_counter() - map_start
    if not map_result.success:
        raise AssertionError(f"MAP optimisation failed: {map_result.message}")
    MAP = np.asarray(map_result.x, dtype=float)

    Hess_log_post = ns["Hess_log_likelihood"](
        hinv(MAP), T, 1, 1, y, y_start_arr, sigma2_start_arr
    ) + np.asarray(Hess_log_prior_phi(MAP), dtype=float)
    covariance = np.linalg.inv(-0.5 * (Hess_log_post + Hess_log_post.T))
    covariance = ns["make_pos_def"](covariance, eps=1e-8)

    args = (T, 1, 1, y, y_start_arr, sigma2_start_arr)
    cv_start = time.perf_counter()
    dens_at_phiStar, grad_at_phiStar, Hess_at_phiStar = ns[
        "initiate_control_variate_quantities"
    ](ns["log_densities"], ns["grad_log_densities"], ns["Hess_log_densities"], MAP, args)
    cv_seconds = time.perf_counter() - cv_start
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

    # Pilot chain used by the existing tuning machinery.
    names_original, names_transformed = rg.get_param_names_regression_GARCH11(p_reg)
    names = {"params_name": names_original, "trans_param_names": names_transformed}
    proposal_cov = 0.15 * (2.38**2 / d) * covariance
    np.random.seed(6303)
    random.seed(6303)
    pilot = ns["sample_full_data_posterior_AM"](
        MAP,
        proposal_cov,
        80,
        names,
        "validation_regression_garch",
        adapt_cov=False,
        adapt_scale=False,
        burn_in=1000,
    )
    pilot_results = {"MCMC_obj": pilot}

    ns.update({"t_star": int(0.01 * T), "b": 0.0})
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
        pilot_slice=slice(1, 80, 4),
        verbose=False,
    )
    tuning_seconds = time.perf_counter() - tuning_start
    tuned = tuning[10]

    # Weighted difference-estimator validation away from the exact Taylor centre.
    direction = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    phi_eval = MAP + 0.20 * direction
    full_log = ns["log_likelihood"](
        hinv(phi_eval), T, 1, 1, y, y_start_arr, sigma2_start_arr
    )
    q_values = ns["eval_q_k"](
        phi_eval, MAP, dens_at_phiStar, grad_at_phiStar, Hess_at_phiStar, order=2
    )
    residuals = ns["log_densities"](
        hinv(phi_eval), *args
    ) - q_values
    probabilities = np.asarray(tuned["probs_opt"], dtype=float)
    m_wde = max(int(tuned["m_opt"]), 2)
    analytic_variance = ns["variance_WDE"](residuals, probabilities, m_wde)
    rng_wde = np.random.default_rng(6304)
    n_wde_rep = 100000
    sampled_indices = rng_wde.choice(T, size=(n_wde_rep, m_wde), p=probabilities)
    estimates = q_sum(phi_eval) + np.mean(
        residuals[sampled_indices] / probabilities[sampled_indices], axis=1
    )
    empirical_mean = float(np.mean(estimates))
    empirical_variance = float(np.var(estimates, ddof=1))
    mean_error = empirical_mean - full_log
    mean_monte_carlo_se = float(np.sqrt(empirical_variance / n_wde_rep))
    mean_standard_errors = abs(mean_error) / mean_monte_carlo_se
    variance_relative_error = abs(empirical_variance - analytic_variance) / analytic_variance
    centred_estimates = estimates - empirical_mean
    empirical_fourth_central_moment = float(np.mean(centred_estimates**4))
    variance_estimator_variance = (
        empirical_fourth_central_moment
        - ((n_wde_rep - 3.0) / (n_wde_rep - 1.0)) * empirical_variance**2
    ) / n_wde_rep
    variance_monte_carlo_se = float(np.sqrt(max(variance_estimator_variance, 0.0)))
    variance_confidence_interval_95 = [
        max(0.0, empirical_variance - 1.96 * variance_monte_carlo_se),
        empirical_variance + 1.96 * variance_monte_carlo_se,
    ]
    analytic_variance_compatible = bool(
        variance_confidence_interval_95[0]
        <= analytic_variance
        <= variance_confidence_interval_95[1]
    )
    if mean_standard_errors > 4.0 or not analytic_variance_compatible:
        raise AssertionError("Weighted difference-estimator validation failed.")

    # Short validation chains only; these are not inferential results.
    full_proposal_cov = 0.08 * (2.38**2 / d) * covariance
    np.random.seed(6305)
    random.seed(6305)
    full_chain = ns["sample_full_data_posterior_AM"](
        MAP,
        full_proposal_cov,
        250,
        names,
        "validation_regression_garch",
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
    np.random.seed(6306)
    random.seed(6306)
    subsampling_chain = ns["sample_subsampling_posterior_random_walk"](
        estimator_params,
        MAP,
        full_proposal_cov,
        250,
        names,
        "validation_regression_garch",
        adapt=False,
    )

    full_acceptance = float(np.mean(full_chain["alphas_MH"]))
    subsampling_acceptance = float(np.mean(subsampling_chain["alphas_MH"]))
    required_finite = [
        dens_at_phiStar,
        grad_at_phiStar,
        Hess_at_phiStar,
        full_chain["samples_phi"],
        full_chain["samples_theta"],
        subsampling_chain["samples_phi"],
        subsampling_chain["samples_theta"],
        subsampling_chain["log_posthat_samples"],
        subsampling_chain["sigma2_lhat_samples"],
        subsampling_chain["u_max_prop"],
    ]
    if not all(np.all(np.isfinite(x)) for x in required_finite):
        raise AssertionError("The end-to-end run produced non-finite stored values.")
    if full_chain["samples_theta"].shape != (251, d):
        raise AssertionError("Unexpected full-chain storage shape.")
    if subsampling_chain["samples_theta"].shape != (251, d):
        raise AssertionError("Unexpected subsampling-chain storage shape.")
    if not (0.0 < full_acceptance < 1.0 and 0.0 < subsampling_acceptance < 1.0):
        raise AssertionError("Validation chains have degenerate acceptance behaviour.")

    total_seconds = time.perf_counter() - start_total
    return {
        "passed": True,
        "T": T,
        "p_reg": p_reg,
        "dimension": d,
        "simulation_seed": 6302,
        "saved_dataset_roundtrip": {
            "passed": True,
            "arrays_exact": True,
            "format": "regression_GARCH11_dataset_v2",
        },
        "parameter_maps": {
            "transform_roundtrip": transform_roundtrip,
            "proposal_roundtrip": proposal_roundtrip,
            "proposal_logabsdet_dpsi_dtheta": proposal_log_jacobian,
        },
        "MAP": {
            "success": bool(map_result.success),
            "iterations": int(map_result.nit),
            "gradient_infinity_norm": float(np.max(np.abs(map_result.jac))),
            "seconds": map_seconds,
        },
        "Taylor_control_variates": {
            "density_shape": list(dens_at_phiStar.shape),
            "gradient_shape": list(grad_at_phiStar.shape),
            "Hessian_shape": list(Hess_at_phiStar.shape),
            "Hessian_megabytes": float(Hess_at_phiStar.nbytes / 1024**2),
            "seconds": cv_seconds,
        },
        "WDE": {
            "replications": n_wde_rep,
            "m": m_wde,
            "true_log_likelihood": float(full_log),
            "empirical_mean": empirical_mean,
            "empirical_mean_minus_exact": float(mean_error),
            "empirical_mean_monte_carlo_se": mean_monte_carlo_se,
            "mean_error_in_standard_errors": float(mean_standard_errors),
            "analytic_variance": float(analytic_variance),
            "empirical_variance": empirical_variance,
            "variance_relative_error": float(variance_relative_error),
            "empirical_fourth_central_moment": empirical_fourth_central_moment,
            "empirical_variance_monte_carlo_se": variance_monte_carlo_se,
            "empirical_variance_confidence_interval_95": variance_confidence_interval_95,
            "analytic_variance_compatible": analytic_variance_compatible,
        },
        "tuning": {
            "R_max": 10,
            "b": 0.0,
            "t_star": int(0.01 * T),
            "c_opt": float(tuned["c_opt"]),
            "gamma_opt": float(tuned["gamma_opt"]),
            "m_opt": int(tuned["m_opt"]),
            "V_target": float(tuned["V_target"]),
            "expected_u_max": float(
                ns["expected_umax_safe"](tuned["probs_opt"], int(tuned["m_opt"]))
            ),
            "seconds": tuning_seconds,
        },
        "full_MCMC": {
            "stored_shape": list(full_chain["samples_theta"].shape),
            "mean_acceptance_probability": full_acceptance,
            "outside_support": int(full_chain["outside_support"]),
            "seconds": float(full_chain["run_time"]),
        },
        "subsampling_MCMC": {
            "stored_shape": list(subsampling_chain["samples_theta"].shape),
            "mean_acceptance_probability": subsampling_acceptance,
            "outside_support": int(subsampling_chain["outside_support"]),
            "variance_estimate_min": float(np.min(subsampling_chain["sigma2_lhat_samples"])),
            "variance_estimate_max": float(np.max(subsampling_chain["sigma2_lhat_samples"])),
            "u_max_min": float(np.min(subsampling_chain["u_max_prop"])),
            "u_max_mean": float(np.mean(subsampling_chain["u_max_prop"])),
            "u_max_max": float(np.max(subsampling_chain["u_max_prop"])),
            "seconds": float(subsampling_chain["run_time"]),
        },
        "total_seconds": total_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path; stdout is always populated.",
    )
    args = parser.parse_args()

    ns = load_existing_notebook_functions()
    report = {
        "existing_GARCH_fixture": existing_garch_fixture(ns),
        "constant_mean_reduction": constant_mean_reduction(ns),
        "numerical_derivatives": numerical_derivative_validation(ns),
        "small_end_to_end": end_to_end_validation(ns),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
