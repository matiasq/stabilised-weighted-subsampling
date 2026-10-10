#!/usr/bin/env python3
"""Generate the two canonical regression-GARCH dataset-001 archives.

The design and seeds are read from ``regression_garch_experiment_config.json``.
Existing files are verified and retained unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import regression_garch as rg


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "regression_garch_experiment_config.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "data"
SUPPORTED_P = (25, 50)
DATASET_ID = "001"

# These hashes were obtained in the final reproducibility environment. A
# different NumPy/platform combination may produce a bytewise-different archive
# even when all numerical checks below pass.
REFERENCE_SHA256 = {
    25: "07a016de0093074e65d29f3a2a87fb5e9f4af0f38fe0f50ce10fa3e9fad295cd",
    50: "fa821d80909ca019ad41fdb3938f240df1b8e91e965bac1a05525c4186180b65",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    for p_reg in SUPPORTED_P:
        case = config.get("cases", {}).get(str(p_reg), {})
        if DATASET_ID not in case.get("dataset_seeds", {}):
            raise ValueError(f"Configuration has no dataset-001 seed for p={p_reg}.")
        configured_hash = case.get("canonical_dataset_001_sha256")
        if configured_hash != REFERENCE_SHA256[p_reg]:
            raise ValueError(f"Configuration reference hash differs for p={p_reg}.")
    return config


def design_quantities(p_reg: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(p_reg)
    Sigma = rho ** np.abs(indices[:, None] - indices[None, :])
    beta_raw = np.ones(p_reg)
    beta_reg = beta_raw / np.sqrt(beta_raw @ Sigma @ beta_raw)
    return Sigma, beta_reg


def simulate_canonical_dataset(p_reg: int, config: dict) -> dict:
    """Simulate dataset 001 using the final pre-specified design."""
    dgp = config["data_generating_process"]
    case = config["cases"][str(p_reg)]
    T = int(dgp["T"])
    burn_in = int(dgp["simulation_burn_in"])
    rho = float(dgp["rho"])
    seed = int(case["dataset_seeds"][DATASET_ID])
    omega = float(dgp["omega"])
    alpha = float(dgp["alpha"])
    beta_garch = float(dgp["beta_garch"])
    nu = float(dgp["nu"])

    Sigma, beta_reg = design_quantities(p_reg, rho)
    rng = np.random.default_rng(seed)

    # The first draw supplies the state preceding the simulation burn-in.
    X_all = rng.multivariate_normal(
        np.zeros(p_reg), Sigma, size=T + burn_in + 1, method="cholesky"
    )
    t_all = rng.standard_t(nu, size=T + burn_in + 1)
    sigma2_initial = omega / (1.0 - alpha - beta_garch)
    innovation_initial = t_all[0] * np.sqrt((nu - 2.0) / nu)
    z_initial = np.sqrt(sigma2_initial) * innovation_initial
    y_initial = float(X_all[0] @ beta_reg + z_initial)

    full = rg.simulate_regression_GARCH11(
        X_all[1:],
        beta_reg,
        omega,
        alpha,
        beta_garch,
        nu,
        y_start=y_initial,
        X_start=X_all[0],
        sigma2_start=sigma2_initial,
        student_t_draws=t_all[1:],
    )

    start_index = burn_in - 1
    retained = slice(burn_in, burn_in + T)
    return {
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
        "seed": seed,
        "dataset_id": f"regression_garch_p{p_reg}_dataset_{DATASET_ID}",
        "burn_in": burn_in,
        "rho": rho,
        "signal_variance": float(beta_reg @ Sigma @ beta_reg),
        "Sigma": Sigma,
        "params": {
            "p_reg": p_reg,
            "beta_reg": beta_reg,
            "omega": omega,
            "alpha": alpha,
            "beta_garch": beta_garch,
            "nu": nu,
        },
    }


def validate_dataset(dataset: dict, p_reg: int, config: dict) -> dict:
    """Check dimensions, metadata, simulation identities, and GARCH recursion."""
    dgp = config["data_generating_process"]
    case = config["cases"][str(p_reg)]
    T = int(dgp["T"])
    burn_in = int(dgp["simulation_burn_in"])
    rho = float(dgp["rho"])
    seed = int(case["dataset_seeds"][DATASET_ID])
    Sigma, beta_reg = design_quantities(p_reg, rho)
    params = dataset["params"]

    expected_id = f"regression_garch_p{p_reg}_dataset_{DATASET_ID}"
    if dataset["X"].shape != (T, p_reg) or dataset["y"].shape != (T,):
        raise ValueError(f"p={p_reg}: unexpected dataset dimensions.")
    if dataset["dataset_id"] != expected_id or dataset["seed"] != seed:
        raise ValueError(f"p={p_reg}: dataset identity or seed is incorrect.")
    if dataset["burn_in"] != burn_in or dataset["rho"] != rho:
        raise ValueError(f"p={p_reg}: burn-in or covariance metadata is incorrect.")
    if not np.array_equal(dataset["Sigma"], Sigma):
        raise ValueError(f"p={p_reg}: stored Toeplitz covariance is incorrect.")
    if not np.allclose(params["beta_reg"], beta_reg, rtol=0.0, atol=1e-15):
        raise ValueError(f"p={p_reg}: stored regression coefficients are incorrect.")
    for name in ("omega", "alpha", "beta_garch", "nu"):
        if params[name] != float(dgp[name]):
            raise ValueError(f"p={p_reg}: stored {name} differs from the design.")
    if params["p_reg"] != p_reg:
        raise ValueError(f"p={p_reg}: stored regression dimension is incorrect.")

    arrays = (
        "X",
        "y",
        "z",
        "sigma2",
        "student_t_draws",
        "standardised_innovations",
        "X_start",
        "Sigma",
    )
    if not all(np.all(np.isfinite(dataset[name])) for name in arrays):
        raise ValueError(f"p={p_reg}: dataset contains non-finite values.")
    if not np.all(dataset["sigma2"] > 0.0):
        raise ValueError(f"p={p_reg}: conditional variances must be positive.")

    residual_error = float(
        np.max(np.abs(dataset["y"] - dataset["X"] @ beta_reg - dataset["z"]))
    )
    start_residual_error = float(
        abs(dataset["y_start"] - dataset["X_start"] @ beta_reg - dataset["z_start"])
    )
    innovation_error = float(
        np.max(
            np.abs(
                dataset["z"] / np.sqrt(dataset["sigma2"])
                - dataset["standardised_innovations"]
            )
        )
    )
    omega = float(dgp["omega"])
    alpha = float(dgp["alpha"])
    beta_garch = float(dgp["beta_garch"])
    expected_sigma2 = np.empty(T)
    expected_sigma2[0] = (
        omega
        + alpha * dataset["z_start"] ** 2
        + beta_garch * dataset["sigma2_start"]
    )
    expected_sigma2[1:] = (
        omega
        + alpha * dataset["z"][:-1] ** 2
        + beta_garch * dataset["sigma2"][:-1]
    )
    garch_error = float(np.max(np.abs(dataset["sigma2"] - expected_sigma2)))
    if max(residual_error, start_residual_error, innovation_error, garch_error) > 1e-12:
        raise ValueError(f"p={p_reg}: simulation identities failed numerical checks.")

    return {
        "response_residual_identity_max_absolute": residual_error,
        "presample_residual_identity_absolute": start_residual_error,
        "standardised_innovation_identity_max_absolute": innovation_error,
        "garch_recursion_max_absolute": garch_error,
        "designed_signal_variance": float(beta_reg @ Sigma @ beta_reg),
    }


def save_safely(path: Path, simulation: dict) -> None:
    """Validate a complete temporary archive before atomically replacing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        rg.save_regression_GARCH11_dataset(temporary, simulation)
        # Loading here catches incomplete or malformed archives before publication.
        rg.load_regression_GARCH11_dataset(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def report_hash(path: Path, p_reg: int, strict_reference_hash: bool) -> str:
    checksum = sha256_file(path)
    expected = REFERENCE_SHA256[p_reg]
    if checksum == expected:
        print(f"  SHA-256: {checksum} (matches reference)")
    else:
        message = (
            f"p={p_reg}: SHA-256 {checksum} differs from reference {expected}. "
            "The archive passed the numerical and metadata checks; bytewise output "
            "can depend on the NumPy/platform combination."
        )
        if strict_reference_hash:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr)
        print(f"  SHA-256: {checksum} (reference differs)")
    return checksum


def prepare_one(
    p_reg: int,
    output_directory: Path,
    config: dict,
    overwrite: bool,
    strict_reference_hash: bool,
) -> None:
    path = output_directory / f"regression_garch_p{p_reg}_dataset_001.npz"
    seed = int(config["cases"][str(p_reg)]["dataset_seeds"][DATASET_ID])
    if path.exists() and not overwrite:
        print(f"p={p_reg}, seed={seed}: verifying existing {path}")
        dataset = rg.load_regression_GARCH11_dataset(path)
        validate_dataset(dataset, p_reg, config)
        report_hash(path, p_reg, strict_reference_hash)
        return

    action = "replacing" if path.exists() else "generating"
    print(f"p={p_reg}, seed={seed}: {action} {path}")
    simulation = simulate_canonical_dataset(p_reg, config)
    validate_dataset(simulation, p_reg, config)
    save_safely(path, simulation)
    validate_dataset(rg.load_regression_GARCH11_dataset(path), p_reg, config)
    report_hash(path, p_reg, strict_reference_hash)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p",
        type=int,
        nargs="+",
        choices=SUPPORTED_P,
        default=list(SUPPORTED_P),
        help="Regression dimensions to generate (default: 25 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Output directory (default: repository data/ directory).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Experiment configuration JSON file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing dataset after validating its replacement.",
    )
    parser.add_argument(
        "--strict-reference-hash",
        action="store_true",
        help="Treat a reference SHA-256 mismatch as an error.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    output_directory = args.output_dir.resolve()
    for p_reg in dict.fromkeys(args.p):
        prepare_one(
            p_reg,
            output_directory,
            config,
            args.overwrite,
            args.strict_reference_hash,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
