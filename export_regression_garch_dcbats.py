"""Deterministically export canonical regression-GARCH data for R/DC-BATS.

The exporter never simulates data.  It verifies the immutable dataset-001 NPZ
hash before writing R-friendly files.  With ``--include-python-results`` it also
exports the completed Python full-data/subsampling draws and creates the
pre-specified median-R_CPU-to-K resource configuration required by the R runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

import regression_garch as rg
from run_regression_garch_stress_test import (
    CONFIG_PATH,
    OUTPUT_ROOT,
    ROOT,
    atomic_json,
    dataset_path,
    load_config,
    output_directory,
    require_writable_output_root,
    sha256_file,
    verify_dataset,
)


EXPORT_ROOT = OUTPUT_ROOT / "dcbats_exports"
RESOURCE_CSV = OUTPUT_ROOT / "dcbats_resource_config.csv"
RESOURCE_JSON = OUTPUT_ROOT / "dcbats_resource_config.json"


def atomic_savetxt(path, values, header):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(path.name + ".tmp")
    np.savetxt(temporary, values, delimiter=",", header=header, comments="", fmt="%.17g")
    temporary.replace(path)


def export_directory(p_reg):
    return EXPORT_ROOT / f"p{p_reg}" / "dataset_001"


def validate_existing_export(out, checksum):
    metadata_path = out / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_npz_sha256") != checksum:
        raise RuntimeError(f"Existing R export has the wrong source hash: {out}")
    required = ("X.csv", "series.csv", "presample.csv", "true_parameters.csv")
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"Existing R export is incomplete ({missing}); refusing silent overwrite.")
    return True


def export_data(p_reg, config):
    case = config["cases"][str(p_reg)]
    source_path = dataset_path(p_reg, "001")
    checksum = sha256_file(source_path)
    if checksum != case["canonical_dataset_001_sha256"]:
        raise RuntimeError(f"Canonical p={p_reg} dataset SHA-256 mismatch.")
    out = export_directory(p_reg)
    if validate_existing_export(out, checksum):
        return out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"Partial R data export has no valid metadata; inspect before retrying: {out}"
        )
    dataset = rg.load_regression_GARCH11_dataset(source_path)
    verify_dataset(dataset, p_reg, "001", case["dataset_seeds"]["001"], config)
    out.mkdir(parents=True, exist_ok=True)
    atomic_savetxt(
        out / "X.csv", dataset["X"],
        ",".join(f"x{index}" for index in range(1, p_reg + 1)),
    )
    atomic_savetxt(
        out / "series.csv", np.column_stack([dataset["y"], dataset["sigma2"]]),
        "y,sigma2_true",
    )
    atomic_savetxt(
        out / "presample.csv",
        np.asarray([[dataset["y_start"], dataset["sigma2_start"], *dataset["X_start"]]]),
        ",".join(["y_start", "sigma2_start"] + [f"x_start_{index}" for index in range(1, p_reg + 1)]),
    )
    theta_true = np.concatenate([
        dataset["params"]["beta_reg"],
        [dataset["params"][key] for key in ("omega", "alpha", "beta_garch", "nu")],
    ])
    names = [f"beta_reg_{index}" for index in range(1, p_reg + 1)] + ["omega", "alpha", "beta_garch", "nu"]
    atomic_savetxt(out / "true_parameters.csv", theta_true[None, :], ",".join(names))
    atomic_json(out / "metadata.json", {
        "schema_version": 1, "p": p_reg, "total_parameter_dimension": p_reg + 4,
        "dataset_id": "001", "seed": case["dataset_seeds"]["001"],
        "T": config["data_generating_process"]["T"],
        "source_npz": str(source_path.relative_to(ROOT)),
        "source_npz_sha256": checksum,
        "model": "no-intercept regression-GARCH(1,1) with standardized Student-t errors",
        "oracle_shard_variance_initialisation_available": True,
    })
    return out


def export_draws(p_reg, out):
    python_out = output_directory(p_reg, "001")
    required = (
        python_out / "full_data_chain.npz",
        python_out / "subsampling_chain.npz",
        python_out / "map_hessian.npz",
        python_out / "summary.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Complete the Python dataset-001 production run before exporting draws: "
            + ", ".join(missing)
        )
    names = [f"beta_reg_{index}" for index in range(1, p_reg + 1)] + ["omega", "alpha", "beta_garch", "nu"]
    results_metadata = out / "python_results_metadata.json"
    result_exports = (
        out / "full_data_draws.csv",
        out / "subsampling_draws.csv",
        out / "map_initial_values.csv",
    )
    if results_metadata.exists():
        if not all(path.exists() for path in result_exports):
            raise RuntimeError(f"Python-result export is incomplete: {out}")
        return
    if any(path.exists() for path in result_exports):
        raise RuntimeError(
            f"Partial Python-result export has no valid metadata; inspect before retrying: {out}"
        )
    with np.load(required[0], allow_pickle=False) as full:
        atomic_savetxt(out / "full_data_draws.csv", full["retained_theta"], ",".join(names))
    with np.load(required[1], allow_pickle=False) as subsampling:
        atomic_savetxt(out / "subsampling_draws.csv", subsampling["retained_theta"], ",".join(names))
    with np.load(required[2], allow_pickle=False) as map_stage:
        atomic_savetxt(out / "map_initial_values.csv", map_stage["theta"][None, :], ",".join(names))
    summary = json.loads(required[3].read_text(encoding="utf-8"))
    atomic_json(results_metadata, {
        "dataset_sha256": summary["dataset_sha256"], "p": p_reg,
        "total_parameter_dimension": p_reg + 4, "full_data_CPU_seconds": summary["full_data_CPU_seconds"],
        "subsampling_CPU_seconds": summary["subsampling_CPU_seconds"], "R_CPU": summary["R_CPU"],
        "full_data_draws": 10000, "subsampling_draws": 10000,
    })


def select_k_stress(median_R_CPU):
    """Ordinary nearest-integer rounding for the positive speed-up ratio."""
    median_R_CPU = float(median_R_CPU)
    if not np.isfinite(median_R_CPU) or median_R_CPU <= 0:
        raise ValueError("median_R_CPU must be a positive finite value.")
    K_stress = int(math.floor(median_R_CPU + 0.5))
    if K_stress < 1:
        raise ValueError("Rounded K_stress is zero, so no valid shard configuration exists.")
    return K_stress


def balanced_shard_configuration(T, K):
    """Return the all-observation contiguous-shard sizes and likelihood powers."""
    T = int(T)
    K = int(K)
    if T < 1 or K < 1 or K > T:
        raise ValueError("Balanced shards require 1 <= K <= T.")
    q, r = divmod(T, K)
    shard_sizes = [q + 1] * r + [q] * (K - r)
    likelihood_powers = [T / size for size in shard_sizes]
    if sum(shard_sizes) != T or max(shard_sizes) - min(shard_sizes) > 1:
        raise AssertionError("Balanced shard construction failed.")
    return q, r, shard_sizes, likelihood_powers


def write_resource_configuration(config):
    rows = []
    T = int(config["data_generating_process"]["T"])
    standard_K = [int(value) for value in config["dc_bats"]["standard_K"]]
    if standard_K != [10, 20]:
        raise ValueError("The pre-specified standard DC-BATS settings must be K=10 and K=20.")
    for p_reg in (25, 50):
        summary_path = OUTPUT_ROOT / f"p{p_reg}" / "five_dataset_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Five-dataset median R_CPU is unavailable; DC-BATS must not run yet: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        speedups = [float(value) for value in summary["R_CPU_values"]]
        if len(speedups) != 5:
            raise ValueError(f"Expected five R_CPU values for p={p_reg}.")
        median = float(np.median(speedups))
        if not np.isclose(median, float(summary["median_R_CPU"]), rtol=0.0, atol=1e-12):
            raise ValueError(f"Stored median_R_CPU is inconsistent for p={p_reg}.")
        K_stress = select_k_stress(median)
        roles_by_K = {10: ["standard_10"], 20: ["standard_20"]}
        roles_by_K.setdefault(K_stress, []).append("cpu_matched")
        ordered_K = list(dict.fromkeys([*standard_K, K_stress]))
        for K in ordered_K:
            roles = roles_by_K[K]
            q, r, shard_sizes, likelihood_powers = balanced_shard_configuration(T, K)
            row = {
                "p": p_reg,
                "total_parameter_dimension": p_reg + 4,
                "dataset_id": "001",
                "median_R_CPU": median,
                "K_stress": K_stress,
                "K": K,
                "K_roles": "|".join(roles),
                "is_standard_10": "standard_10" in roles,
                "is_standard_20": "standard_20" in roles,
                "is_cpu_matched": "cpu_matched" in roles,
                "R_CPU_values_json": json.dumps(speedups, separators=(",", ":")),
                "R_CPU_1": speedups[0],
                "R_CPU_2": speedups[1],
                "R_CPU_3": speedups[2],
                "R_CPU_4": speedups[3],
                "R_CPU_5": speedups[4],
                "T_total": T,
                "T_used": T,
                "dropped_observations": 0,
                "base_shard_size_q": q,
                "larger_shard_count_r": r,
                "minimum_shard_size": min(shard_sizes),
                "maximum_shard_size": max(shard_sizes),
                "shard_sizes_json": json.dumps(shard_sizes, separators=(",", ":")),
                "shard_likelihood_powers_json": json.dumps(likelihood_powers, separators=(",", ":")),
                "rule": config["dc_bats"]["k_rule"],
            }
            rows.append(row)
    RESOURCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESOURCE_CSV.with_name(RESOURCE_CSV.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESOURCE_CSV)
    atomic_json(RESOURCE_JSON, {
        "schema_version": 2, "selection_uses_inferential_results": False,
        "timing_interpretation": "K perfectly parallel shard workers; K_stress is chosen before examining any DC-BATS timing or agreement.",
        "partition_definition": "All T observations are assigned exactly once to K contiguous shards; r=T mod K shards have size floor(T/K)+1 and the remaining shards have size floor(T/K).",
        "likelihood_power_definition": "Shard k uses likelihood power T/n_k; the prior remains unpowered and is included once.",
        "standard_settings": standard_K,
        "cpu_matched_definition": "ordinary nearest-integer rounding of the five-dataset median R_CPU",
        "rows": rows,
    })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-python-results", action="store_true")
    parser.add_argument("--p", type=int, choices=(25, 50), action="append")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)
    require_writable_output_root()
    for p_reg in args.p or (25, 50):
        out = export_data(p_reg, config)
        if args.include_python_results:
            export_draws(p_reg, out)
    if args.include_python_results and not args.p:
        print(json.dumps(write_resource_configuration(config), indent=2))


if __name__ == "__main__":
    main()
