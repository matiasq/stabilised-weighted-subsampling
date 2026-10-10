#!/usr/bin/env python3
"""Plot cached regression-GARCH posterior draws without running experiments.

The script reads completed Python full-data/subsampling chains and completed
DC-BATS barycentre draws.  It deliberately does not import the production
driver or any model, optimisation, simulation, or MCMC implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent
PYTHON_OUTPUT_ROOT = ROOT / "regression_garch_outputs"
DCBATS_OUTPUT_ROOT = ROOT / "results" / "DC_BATS_regression_garch"


class PlotInputError(RuntimeError):
    """Raised when a required cached plotting input is absent or invalid."""


@dataclass(frozen=True)
class PosteriorDraws:
    label: str
    path: Path
    values: np.ndarray
    color: str
    linestyle: str
    linewidth: float


def parameter_names(p_reg: int) -> list[str]:
    return [f"beta_reg_{j}" for j in range(1, p_reg + 1)] + [
        "omega",
        "alpha",
        "beta_garch",
        "nu",
    ]


def default_parameters(p_reg: int) -> list[str]:
    if p_reg == 25:
        beta_indices = [1, 5, 10, 15, 20, 25]
    else:
        beta_indices = np.unique(
            np.linspace(1, p_reg, min(6, p_reg), dtype=int)
        ).tolist()
    return [f"beta_reg_{j}" for j in beta_indices] + [
        "omega",
        "alpha",
        "beta_garch",
        "nu",
    ]


def normalise_dataset_id(value: str) -> str:
    dataset_id = str(value).strip().lower()
    if dataset_id.startswith("dataset_"):
        dataset_id = dataset_id.removeprefix("dataset_")
    if not dataset_id.isdigit():
        raise argparse.ArgumentTypeError(
            "dataset must be a numeric id such as 001 or dataset_001"
        )
    return dataset_id.zfill(3)


def normalise_requested_parameters(
    requested: Sequence[str] | None, p_reg: int
) -> list[str]:
    available = parameter_names(p_reg)
    if not requested:
        return default_parameters(p_reg)

    expanded: list[str] = []
    for item in requested:
        expanded.extend(part.strip() for part in item.split(",") if part.strip())
    aliases = {
        "beta_garch11": "beta_garch",
        "beta_garch_11": "beta_garch",
    }
    normalised: list[str] = []
    for name in expanded:
        name = aliases.get(name, name)
        if name.startswith("beta_reg_"):
            suffix = name.removeprefix("beta_reg_")
            if suffix.isdigit():
                name = f"beta_reg_{int(suffix)}"
        if name not in available:
            raise PlotInputError(
                f"Unknown parameter {name!r} for p={p_reg}. Available names are: "
                + ", ".join(available)
            )
        if name not in normalised:
            normalised.append(name)
    if not normalised:
        raise PlotInputError("At least one parameter must be requested.")
    return normalised


def flatten_K_groups(groups: Sequence[Sequence[int]] | None) -> list[int]:
    values = [10, 20] if groups is None else [item for group in groups for item in group]
    result: list[int] = []
    for value in values:
        if value < 1:
            raise PlotInputError(f"DC-BATS K must be positive, received {value}.")
        if value not in result:
            result.append(value)
    return result


def read_json(path: Path, role: str) -> dict:
    if not path.is_file():
        raise PlotInputError(f"Missing {role}: {path}")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise PlotInputError(f"Could not read valid JSON from {role} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlotInputError(f"Expected a JSON object in {role}: {path}")
    return value


def validate_draw_matrix(values: np.ndarray, source: Path, expected_columns: int) -> None:
    if values.ndim != 2 or values.shape[1] != expected_columns:
        raise PlotInputError(
            f"Unexpected draw shape in {source}: {values.shape}; expected (*, {expected_columns})."
        )
    if values.shape[0] < 2:
        raise PlotInputError(f"Fewer than two posterior draws were found in {source}.")
    if not np.all(np.isfinite(values)):
        raise PlotInputError(f"Non-finite posterior draws were found in {source}.")


def load_python_draws(
    p_reg: int, dataset_id: str, selected_names: Sequence[str]
) -> tuple[list[PosteriorDraws], str, list[tuple[str, Path]]]:
    directory = PYTHON_OUTPUT_ROOT / f"p{p_reg}" / f"dataset_{dataset_id}"
    summary_path = directory / "summary.json"
    summary = read_json(summary_path, "completed Python summary")
    expected_dimension = p_reg + 4
    expected = {
        "p": p_reg,
        "dataset_id": dataset_id,
        "total_parameter_dimension": expected_dimension,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise PlotInputError(
                f"Python summary metadata mismatch for {key!r} in {summary_path}: "
                f"expected {expected_value!r}, found {summary.get(key)!r}."
            )
    dataset_sha256 = summary.get("dataset_sha256")
    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
        raise PlotInputError(f"Missing or invalid dataset SHA-256 in {summary_path}.")

    all_names = parameter_names(p_reg)
    selected_indices = [all_names.index(name) for name in selected_names]
    specifications = [
        ("Full-data MCMC", directory / "full_data_chain.npz", "black", "-", 2.6),
        (
            "Subsampling MCMC",
            directory / "subsampling_chain.npz",
            "#0072B2",
            "--",
            2.2,
        ),
    ]
    loaded: list[PosteriorDraws] = []
    loaded_files: list[tuple[str, Path]] = [("Python completion summary", summary_path)]
    for label, path, color, linestyle, linewidth in specifications:
        if not path.is_file():
            raise PlotInputError(
                f"Required cached {label} chain is missing: {path}. "
                "This plotting script will not generate it."
            )
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "retained_theta" not in archive:
                    raise PlotInputError(f"Missing 'retained_theta' in {path}.")
                values = np.asarray(archive["retained_theta"], dtype=float)
                stored_hash = (
                    str(archive["dataset_sha256"].item())
                    if "dataset_sha256" in archive
                    else None
                )
        except (OSError, ValueError) as exc:
            raise PlotInputError(f"Could not read cached chain {path}: {exc}") from exc
        validate_draw_matrix(values, path, expected_dimension)
        if stored_hash != dataset_sha256:
            raise PlotInputError(
                f"Dataset SHA-256 mismatch between {path} and {summary_path}."
            )
        loaded.append(
            PosteriorDraws(
                label=label,
                path=path,
                values=values[:, selected_indices].copy(),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        )
        loaded_files.append((label, path))
    return loaded, dataset_sha256, loaded_files


def find_barycentre_csv(directory: Path) -> Path:
    candidates = sorted(directory.glob("bary_draws*.csv"))
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise PlotInputError(
            f"Expected exactly one DC-BATS barycentre draw CSV in {directory}; found {names}."
        )
    return candidates[0]


def load_selected_csv_columns(
    path: Path, selected_names: Sequence[str]
) -> np.ndarray:
    columns = {name: [] for name in selected_names}
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames or []
            missing = [name for name in selected_names if name not in fieldnames]
            if missing:
                raise PlotInputError(
                    f"Missing requested columns in {path}: {', '.join(missing)}. "
                    f"Available columns are: {', '.join(fieldnames)}"
                )
            for row_number, row in enumerate(reader, start=2):
                for name in selected_names:
                    try:
                        columns[name].append(float(row[name]))
                    except (TypeError, ValueError) as exc:
                        raise PlotInputError(
                            f"Invalid numeric value for {name!r} at {path}:{row_number}."
                        ) from exc
    except OSError as exc:
        raise PlotInputError(f"Could not read DC-BATS draws {path}: {exc}") from exc
    values = np.column_stack([columns[name] for name in selected_names])
    validate_draw_matrix(values, path, len(selected_names))
    return values


def load_dcbats_draws(
    p_reg: int,
    dataset_id: str,
    K_values: Sequence[int],
    selected_names: Sequence[str],
    dataset_sha256: str,
) -> tuple[list[PosteriorDraws], list[tuple[str, Path]], list[str]]:
    base = DCBATS_OUTPUT_ROOT / f"p{p_reg}" / f"dataset_{dataset_id}"
    dc_color = "#009E73"
    linestyles = ["-", "--", "-.", ":"]
    loaded: list[PosteriorDraws] = []
    loaded_files: list[tuple[str, Path]] = []
    skipped: list[str] = []

    for position, K in enumerate(K_values):
        directory = base / f"K_{K}"
        completion_path = directory / "COMPLETED.json"
        metadata_path = directory / "metadata.json"
        if not completion_path.is_file():
            skipped.append(
                f"DC-BATS K={K} is not complete: completion marker is missing at "
                f"{completion_path}"
            )
            continue
        try:
            completion = read_json(completion_path, f"DC-BATS K={K} completion marker")
            metadata = read_json(metadata_path, f"DC-BATS K={K} metadata")
            expected_metadata = {
                "p": p_reg,
                "K": K,
                "canonical_dataset_id": dataset_id,
                "source_npz_sha256": dataset_sha256,
            }
            for key, expected_value in expected_metadata.items():
                if metadata.get(key) != expected_value:
                    raise PlotInputError(
                        f"DC-BATS K={K} metadata mismatch for {key!r} in "
                        f"{metadata_path}: expected {expected_value!r}, "
                        f"found {metadata.get(key)!r}."
                    )
            if completion.get("K") != K:
                raise PlotInputError(
                    f"DC-BATS completion marker K mismatch in {completion_path}."
                )
            if completion.get("source_npz_sha256") != dataset_sha256:
                raise PlotInputError(
                    f"DC-BATS completion marker dataset hash mismatch in {completion_path}."
                )
            barycentre_path = find_barycentre_csv(directory)
            values = load_selected_csv_columns(barycentre_path, selected_names)
        except PlotInputError as exc:
            skipped.append(f"DC-BATS K={K} was skipped: {exc}")
            continue

        loaded.append(
            PosteriorDraws(
                label=f"DC-BATS K={K}",
                path=barycentre_path,
                values=values,
                color=dc_color,
                linestyle=linestyles[position % len(linestyles)],
                linewidth=2.0,
            )
        )
        loaded_files.extend(
            [
                (f"DC-BATS K={K} completion marker", completion_path),
                (f"DC-BATS K={K} metadata", metadata_path),
                (f"DC-BATS K={K} barycentre draws", barycentre_path),
            ]
        )
    return loaded, loaded_files, skipped


def display_parameter_name(name: str) -> str:
    if name.startswith("beta_reg_"):
        index = int(name.removeprefix("beta_reg_"))
        return rf"$\beta_{{\mathrm{{reg}},{index}}}$"
    return {
        "omega": r"$\omega$",
        "alpha": r"$\alpha$",
        "beta_garch": r"$\beta_{\mathrm{GARCH}}$",
        "nu": r"$\nu$",
    }.get(name, name)


def density_grid(methods: Sequence[PosteriorDraws], column: int) -> np.ndarray:
    pooled = np.concatenate([method.values[:, column] for method in methods])
    lower, upper = np.quantile(pooled, (0.001, 0.999))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = float(np.min(pooled)), float(np.max(pooled))
    if upper <= lower:
        raise PlotInputError(
            "Cannot construct a KDE grid because all loaded draws are identical."
        )
    padding = 0.08 * (upper - lower)
    return np.linspace(lower - padding, upper + padding, 500)


def make_figure(
    methods: Sequence[PosteriorDraws],
    selected_names: Sequence[str],
    p_reg: int,
    dataset_id: str,
) -> plt.Figure:
    n_panels = len(selected_names)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.5, 3.1 * nrows + 1.0),
        squeeze=False,
    )

    for column, (axis, name) in enumerate(zip(axes.flat, selected_names)):
        grid = density_grid(methods, column)
        for method in methods:
            values = method.values[:, column]
            if np.unique(values).size < 2 or np.std(values, ddof=1) <= 0.0:
                raise PlotInputError(
                    f"Cannot estimate a KDE for {name} from {method.path}: "
                    "the retained draws have zero variation."
                )
            try:
                density = gaussian_kde(values)(grid)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise PlotInputError(
                    f"KDE estimation failed for {name} using {method.path}: {exc}"
                ) from exc
            axis.plot(
                grid,
                density,
                label=method.label,
                color=method.color,
                linestyle=method.linestyle,
                linewidth=method.linewidth,
            )
        axis.set_xlabel(display_parameter_name(name), fontsize=17)
        axis.set_yticks([])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(axis="x", labelsize=13, length=6, width=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in axes.flat[n_panels:]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = ["Subsampling MCMC"] + [
        method.label for method in methods if method.label.startswith("DC-BATS")
    ] + ["Full-data MCMC"]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(
        f"Regression-GARCH posterior marginals: p={p_reg}, d={p_reg + 4}, "
        f"dataset {dataset_id}",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.94))
    return fig


def save_figure(fig: plt.Figure, output_directory: Path, stem: str) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, options in (
        (".pdf", {}),
        (".png", {"dpi": 240}),
    ):
        destination = output_directory / f"{stem}{suffix}"
        temporary = destination.with_name(f".{destination.stem}.tmp{suffix}")
        fig.savefig(temporary, bbox_inches="tight", **options)
        temporary.replace(destination)
        outputs.append(destination)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot posterior KDEs from completed regression-GARCH Python and "
            "DC-BATS caches. This command never runs an experiment."
        )
    )
    parser.add_argument("--p", type=int, default=25, help="Number of regressors (default: 25).")
    parser.add_argument(
        "--dataset",
        type=normalise_dataset_id,
        default="001",
        help="Dataset id, e.g. 001 or dataset_001 (default: 001).",
    )
    parser.add_argument(
        "--K",
        "--k",
        dest="K_groups",
        action="append",
        nargs="+",
        type=int,
        metavar="K",
        help=(
            "DC-BATS shard counts. Accepts '--K 10 20' and repeated uses such "
            "as '--K 10 --K 20'. Default: 10 20."
        ),
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        help=(
            "Optional parameter names (space- or comma-separated), e.g. "
            "--parameters omega alpha beta_garch nu beta_reg_1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory. Default: "
            "results/DC_BATS_regression_garch/p<P>/dataset_<ID>/plots/."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    if args.p < 1:
        raise PlotInputError(f"p must be positive, received {args.p}.")
    dataset_id = normalise_dataset_id(args.dataset)
    K_values = flatten_K_groups(args.K_groups)
    selected_names = normalise_requested_parameters(args.parameters, args.p)

    python_methods, dataset_sha256, python_files = load_python_draws(
        args.p, dataset_id, selected_names
    )
    dcbats_methods, dcbats_files, skipped = load_dcbats_draws(
        args.p,
        dataset_id,
        K_values,
        selected_names,
        dataset_sha256,
    )
    methods = python_methods + dcbats_methods

    print("Loaded completed regression-GARCH posterior inputs:")
    for role, path in python_files + dcbats_files:
        print(f"  - {role}: {path}")
    print("Posterior draw counts:")
    for method in methods:
        print(f"  - {method.label}: {method.values.shape[0]:,}")
    for message in skipped:
        print(f"WARNING: {message}", file=sys.stderr)
    if not dcbats_methods:
        print(
            "WARNING: no requested completed DC-BATS result was available; "
            "the figure will contain only the two Python methods.",
            file=sys.stderr,
        )

    output_directory = args.output
    if output_directory is None:
        output_directory = (
            DCBATS_OUTPUT_ROOT
            / f"p{args.p}"
            / f"dataset_{dataset_id}"
            / "plots"
        )
    if output_directory.exists() and not output_directory.is_dir():
        raise PlotInputError(
            f"--output must name a directory, but this is not a directory: {output_directory}"
        )

    loaded_K = [
        int(method.label.removeprefix("DC-BATS K=")) for method in dcbats_methods
    ]
    K_suffix = "_".join(f"K{K}" for K in loaded_K) if loaded_K else "python_only"
    stem = f"posterior_kde_p{args.p}_dataset_{dataset_id}_{K_suffix}"
    fig = make_figure(methods, selected_names, args.p, dataset_id)
    try:
        outputs = save_figure(fig, output_directory, stem)
    finally:
        plt.close(fig)
    for path in outputs:
        print(f"Wrote: {path}")
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except PlotInputError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
