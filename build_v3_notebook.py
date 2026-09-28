"""Build the Version-3 reproduction notebook without executing any cell."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "stabilised_weighted_data_subsampling.ipynb"
TARGET = ROOT / "stabilised_weighted_data_subsampling_v3.ipynb"
EXPECTED_SOURCE_SHA256 = "bfb61acc292bc395bac0a72e5c3849a15186d4483d30d0e2fc0e83abc3c9814d"


def source_lines(text):
    return text.splitlines(keepends=True) or [""]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source_lines(text),
    }


SECTION_7_CELLS = [
    markdown(r"""## 7. Regression-GARCH dimensional stress test and extended DC-BATS comparison

This Version-3 experiment studies a no-intercept regression-GARCH(1,1) model with standardized Student-$t$ errors as the number of regressors increases. The cases are $p=25$ regressors (total parameter dimension $d=29$) and $p=50$ regressors (total parameter dimension $d=54$). In each case, five independently simulated datasets are used as computational robustness and timing replicates. They are **not** a repeated-sampling coverage experiment. Dataset 001 is predesignated for detailed posterior comparisons.

For each case, $T=100{,}000$ observations follow $y_t=x_t^\top\beta+z_t$, with iid $x_t$ having Toeplitz covariance $\Sigma_{jk}=0.5^{|j-k|}$. The regression coefficients are dense and equal, scaled so that $\beta^\top\Sigma\beta=1$. The true volatility parameters are $\omega=0.07134079$, $\alpha=0.16151262$, and $\beta_{\mathrm{GARCH}}=0.78591371$. Innovations have $\nu=8$ degrees of freedom, with $\nu$ estimated during inference, and simulation burn-in is 1,000 observations. The Bayesian model uses independent $N(0,1)$ priors for the regression coefficients and the already validated priors and transformations for the remaining parameters.

The full-data and stabilised weighted subsampling analyses use the same dataset, priors, transformations, MAP/Hessian calculations, second-order Taylor control variates, and random-walk proposal. The revised TPD construction uses $b=0$, $R_{\max}=100$, and $t^\star=1000$. Proposal covariance matrices are symmetrised and factorised with zero jitter first; numerical jitter is added only if Cholesky genuinely fails. No KSD is computed in this experiment.
"""),
    markdown("""### 7.1 Pre-specified design, seeds, and canonical dataset provenance

The configuration below was fixed before production results were observed. The two dataset-001 archives are immutable canonical inputs and are verified by SHA-256 before use. Datasets 002--005 are generated only when absent, are never silently overwritten, and use the same validated simulation/serialization implementation as dataset 001.
"""),
    code("""import gc
import hashlib
import json
import subprocess
from IPython.display import Image

REGRESSION_GARCH_CONFIG_PATH = Path("regression_garch_experiment_config.json")
with REGRESSION_GARCH_CONFIG_PATH.open(encoding="utf-8") as source:
    regression_garch_config = json.load(source)

# Fixed, pre-specified dataset seeds (visible here by design).
REGRESSION_GARCH_DATASET_SEEDS = {
    25: {"001": 20260921, "002": 20260923, "003": 20260924, "004": 20260925, "005": 20260926},
    50: {"001": 20260922, "002": 20260927, "003": 20260928, "004": 20260929, "005": 20260930},
}
assert regression_garch_config["cases"]["25"]["dataset_seeds"] == REGRESSION_GARCH_DATASET_SEEDS[25]
assert regression_garch_config["cases"]["50"]["dataset_seeds"] == REGRESSION_GARCH_DATASET_SEEDS[50]

EXPECTED_CANONICAL_HASHES = {
    25: "07a016de0093074e65d29f3a2a87fb5e9f4af0f38fe0f50ce10fa3e9fad295cd",
    50: "fa821d80909ca019ad41fdb3938f240df1b8e91e965bac1a05525c4186180b65",
}

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for p_reg in (25, 50):
    canonical_path = Path("data") / f"regression_garch_p{p_reg}_dataset_001.npz"
    actual_hash = file_sha256(canonical_path)
    if actual_hash != EXPECTED_CANONICAL_HASHES[p_reg]:
        raise RuntimeError(
            f"Canonical p={p_reg} dataset SHA-256 mismatch: {actual_hash}"
        )
    print(f"p={p_reg}, d={p_reg + 4}: {canonical_path}, SHA256={actual_hash}")
    print("fixed seeds:", REGRESSION_GARCH_DATASET_SEEDS[p_reg])
"""),
    markdown("""### 7.2 Create or verify the five datasets per dimension

This guarded cell performs data preparation only; it does not run MAP optimisation or MCMC. Leave the flag `False` when reproducing results from existing datasets. When enabled, the driver loads and verifies every existing archive, creates only missing datasets 002--005, prints filename/id/seed/SHA-256, and refuses to regenerate dataset 001 or overwrite any archive.
"""),
    code("""PREPARE_MISSING_REGRESSION_GARCH_DATASETS = False

if PREPARE_MISSING_REGRESSION_GARCH_DATASETS:
    subprocess.run(
        [sys.executable, "run_regression_garch_stress_test.py", "--prepare-datasets"],
        check=True,
    )
else:
    print("Dataset preparation skipped. Set the explicit flag to True when required.")
"""),
    markdown("""### 7.3 Cache-aware full-data and $b=0$ subsampling MCMC

The production driver processes datasets sequentially and immediately checkpoints dataset verification, MAP/Hessian, pilot MCMC, memory-mapped Taylor arrays, tuning, both production chains, timings, and posterior summaries under `regression_garch_outputs/p{p}/dataset_{id}/`. Rerunning resumes from valid dataset-hash-bound stages. Large arrays are released between datasets.

The production convention is 12,000 MCMC updates, 2,000 burn-in updates, and 10,000 retained draws. Proposal multipliers are the already validated zero-jitter values: 0.20 for $p=25$ and 0.75 for $p=50$. The five runs are computational robustness/timing replicates, not coverage replicates.
"""),
    code("""RUN_REGRESSION_GARCH_PRODUCTION = False

if RUN_REGRESSION_GARCH_PRODUCTION:
    # The driver itself enforces sequential dataset processing and cache validation.
    subprocess.run([sys.executable, "run_regression_garch_stress_test.py"], check=True)
else:
    print(
        "Production inference skipped. Set RUN_REGRESSION_GARCH_PRODUCTION=True "
        "only for the intended manual overnight run."
    )
"""),
    markdown(r"""### 7.4 Five-dataset computational and posterior summaries

Each row reports the requested timing, tuning, realized-compute, and marginal posterior-agreement diagnostics. The median and range of $R_{\mathrm{CPU}}=\mathrm{CPU}_{\mathrm{full}}/\mathrm{CPU}_{\mathrm{sub}}$ are reported separately for $p=25$ ($d=29$) and $p=50$ ($d=54$).
"""),
    code("""regression_garch_summary_rows = []
missing_regression_garch_summaries = []
for p_reg in (25, 50):
    for dataset_id in ("001", "002", "003", "004", "005"):
        summary_path = (
            Path("regression_garch_outputs") / f"p{p_reg}" /
            f"dataset_{dataset_id}" / "summary.json"
        )
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as source:
                regression_garch_summary_rows.append(json.load(source))
        else:
            missing_regression_garch_summaries.append(str(summary_path))

if missing_regression_garch_summaries:
    print("Regression-GARCH production summaries are not yet complete:")
    for missing_path in missing_regression_garch_summaries:
        print(" -", missing_path)

if regression_garch_summary_rows:
    regression_garch_summary = pd.DataFrame(regression_garch_summary_rows).sort_values(["p", "dataset_id"])
    display(regression_garch_summary[[
        "p", "total_parameter_dimension", "dataset_id", "seed",
        "full_data_acceptance_rate", "subsampling_acceptance_rate",
        "full_data_CPU_seconds", "subsampling_CPU_seconds", "R_CPU",
        "tuned_c", "tuned_gamma", "tuned_m", "expected_u_max",
        "expected_u_max_over_T", "realised_average_u_max",
        "marginal_95_intervals_overlapping", "marginal_95_intervals_total",
        "marginal_95_intervals_overlap_proportion",
        "maximum_standardised_posterior_mean_difference",
        "median_standardised_posterior_mean_difference",
        "minimum_posterior_SD_ratio", "maximum_posterior_SD_ratio",
    ]])

    speed_summary = (
        regression_garch_summary.groupby(["p", "total_parameter_dimension"])["R_CPU"]
        .agg(R_CPU_values=lambda values: list(values), median_R_CPU="median", min_R_CPU="min", max_R_CPU="max")
        .reset_index()
    )
    speed_summary["K_stress"] = np.floor(speed_summary["median_R_CPU"] + 0.5).astype(int)
    display(speed_summary)
"""),
    code("""# Dataset 001 was predesignated for detailed marginal diagnostics.
for p_reg in (25, 50):
    overlay_path = (
        Path("regression_garch_outputs") / f"p{p_reg}" /
        "dataset_001" / "selected_marginal_overlays.png"
    )
    if overlay_path.exists():
        display(Markdown(f"#### $p={p_reg}$ regressors ($d={p_reg + 4}$)"))
        display(Image(filename=str(overlay_path)))
    else:
        print(f"Detailed p={p_reg}, d={p_reg + 4} dataset-001 plot is not yet available: {overlay_path}")
"""),
    markdown(r"""### 7.5 Deterministic DC-BATS resource matching and R export

Only after all five Python summaries exist, the exporter computes the median $R_{\mathrm{CPU}}$ separately by dimension and defines $K_{\mathrm{stress}}$ by ordinary nearest-integer rounding of that median. For positive speed-ups this is implemented as $\lfloor\operatorname{median}(R_{\mathrm{CPU}})+0.5\rfloor$. The value of $T$ plays no role in selecting or rounding $K_{\mathrm{stress}}$.

For each dimension, DC-BATS is prepared on canonical dataset 001 only for the three pre-specified settings $K=10$, $K=20$, and $K=K_{\mathrm{stress}}$. If the stress setting equals 10 or 20, its standard result is reused and both roles are recorded. No setting is selected or removed using DC-BATS timing or posterior accuracy. For arbitrary $K$, all $T$ observations are assigned exactly once to contiguous shards whose sizes differ by at most one: if $q=\lfloor T/K\rfloor$ and $r=T-Kq$, the first $r$ shards have size $q+1$ and the remainder have size $q$. Shard $k$ uses likelihood power $T/n_k$, which reduces exactly to $K$ for equal shard sizes. The once-applied prior and coordinate-wise Wasserstein barycentre are unchanged.

The export contains the exact canonical dataset-001 observations and presample values, its original NPZ SHA-256, Python MAP initial values, and the 10,000 retained full-data and subsampling draws. It never resimulates data.
"""),
    code("""EXPORT_REGRESSION_GARCH_FOR_DC_BATS = False

if EXPORT_REGRESSION_GARCH_FOR_DC_BATS:
    subprocess.run(
        [sys.executable, "export_regression_garch_dcbats.py", "--include-python-results"],
        check=True,
    )
else:
    print(
        "R export/resource selection skipped. Enable only after both five-dataset "
        "Python summaries are complete."
    )
"""),
    code("""resource_config_path = Path("regression_garch_outputs") / "dcbats_resource_config.csv"
if resource_config_path.exists():
    regression_garch_dc_resources = pd.read_csv(resource_config_path)
    display(regression_garch_dc_resources)
    for row in regression_garch_dc_resources.itertuples(index=False):
        print(
            "Run later (not from this notebook cell): "
            f"Rscript run_regression_garch_dc_bats.R --p={int(row.p)} --K={int(row.K)}"
        )
else:
    print(
        "DC-BATS resource configuration is unavailable. The R runner will refuse "
        "to run until five-dataset median R_CPU values have been exported."
    )
"""),
    markdown(r"""### 7.6 Regression-GARCH Student-$t$ DC-BATS comparison on canonical dataset 001

The separate R implementation is `run_regression_garch_dc_bats.R` with Stan model `regression_garch_student_t.stan`. For each contiguous shard of size $n_k$, it multiplies the likelihood by $T/n_k$ and includes the matching prior once. When $K$ divides $T$, this power is exactly $K$. All regression residuals, including the preceding residual at a shard boundary, are recomputed using the current draw of the full $p$-vector of regression coefficients. As in the preserved Section-6 simulation design, the preceding true simulated conditional variance is used as an explicitly recorded oracle boundary value.

Shard posteriors use four CmdStan chains with 5,000 warmup and 5,000 sampling iterations per chain. Completed shards are cached separately. Marginal shard posteriors are combined using the same coordinate-wise one-dimensional Wasserstein quantile barycentre as Section 6. The primary ideal-parallel runtime is the maximum shard elapsed time plus extraction, barycentre, and comparison time; Stan compilation is excluded. For each $p$, the cells below read all distinct settings among $K=10$, $K=20$, and $K=K_{\mathrm{stress}}$, and skip with an explicit message until genuine R outputs exist.
"""),
    code("""regression_garch_dc_results = {}
resource_config_path = Path("regression_garch_outputs") / "dcbats_resource_config.csv"

if not resource_config_path.exists():
    print("DC-BATS evaluation skipped: deterministic K values are not yet available.")
else:
    resource_table = pd.read_csv(resource_config_path)
    for row in resource_table.itertuples(index=False):
        p_reg, K = int(row.p), int(row.K)
        result_dir = (
            Path("results") / "DC_BATS_regression_garch" / f"p{p_reg}" /
            "dataset_001" / f"K_{K}"
        )
        tag = f"regression_garch_p{p_reg}_dc_K{K}"
        required = {
            "bary": result_dir / f"bary_draws_{tag}.csv",
            "intervals": result_dir / f"results_intervals_{tag}.csv",
            "agreement": result_dir / f"inferential_agreement_{tag}.csv",
            "timing": result_dir / f"timing_{tag}.csv",
            "shards": result_dir / f"shard_configuration_{tag}.csv",
            "metadata": result_dir / "metadata.json",
            "complete": result_dir / "COMPLETED.json",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            print(f"DC-BATS p={p_reg}, d={p_reg + 4}, K={K} not yet available; skipped.")
            for path in missing:
                print(" -", path)
            continue
        regression_garch_dc_results[(p_reg, K)] = {
            "K": K,
            "K_roles": str(row.K_roles).split("|"),
            "bary": pd.read_csv(required["bary"]),
            "intervals": pd.read_csv(required["intervals"]),
            "agreement": pd.read_csv(required["agreement"]),
            "timing": pd.read_csv(required["timing"]),
            "shards": pd.read_csv(required["shards"]),
            "metadata": json.loads(required["metadata"].read_text(encoding="utf-8")),
        }
        result = regression_garch_dc_results[(p_reg, K)]
        roles_text = ", ".join(result["K_roles"])
        display(Markdown(f"#### DC-BATS: $p={p_reg}$, $d={p_reg + 4}$, $K={K}$ ({roles_text})"))
        display(result["timing"])
        display(result["shards"])
        display(result["intervals"])
        display(result["agreement"])
"""),
    code("""# Selected dataset-001 marginals for our two methods and every distinct DC-BATS setting.
for p_reg in sorted({key[0] for key in regression_garch_dc_results}):
    dc_settings = [
        result for (case_p, _), result in regression_garch_dc_results.items()
        if case_p == p_reg
    ]
    resource_order = {
        int(row.K): order
        for order, row in enumerate(
            resource_table[resource_table["p"] == p_reg].itertuples(index=False)
        )
    }
    dc_settings.sort(key=lambda result: resource_order[result["K"]])

    python_dir = Path("regression_garch_outputs") / f"p{p_reg}" / "dataset_001"
    with np.load(python_dir / "full_data_chain.npz", allow_pickle=False) as stored:
        full_draws_rg = stored["retained_theta"].copy()
    with np.load(python_dir / "subsampling_chain.npz", allow_pickle=False) as stored:
        subsampling_draws_rg = stored["retained_theta"].copy()

    names_rg = [f"beta_reg_{j}" for j in range(1, p_reg + 1)] + ["omega", "alpha", "beta_garch", "nu"]
    dc_draws_rg = [result["bary"][names_rg].to_numpy() for result in dc_settings]
    dc_labels_rg = []
    for result in dc_settings:
        role_suffix = ", CPU-matched" if "cpu_matched" in result["K_roles"] else ""
        dc_labels_rg.append(f"DC-BATS K={result['K']}{role_suffix}")

    selected = sorted(set([0, p_reg // 2, p_reg - 1, p_reg, p_reg + 1, p_reg + 2, p_reg + 3]))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for axis, index in zip(axes.flat, selected):
        sample_sets = [full_draws_rg[:, index], subsampling_draws_rg[:, index]] + [
            draws[:, index] for draws in dc_draws_rg
        ]
        pooled = np.concatenate(sample_sets)
        lower, upper = np.quantile(pooled, [0.0025, 0.9975])
        grid = np.linspace(lower, upper, 400)
        labels_rg = ["Full-data MCMC", "b=0 subsampling MCMC"] + dc_labels_rg
        colors_rg = ["black", "#0072B2", "#D55E00", "#009E73", "#CC79A7"][:len(sample_sets)]
        linestyles_rg = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))][:len(sample_sets)]
        for values, label, color, linestyle in zip(sample_sets, labels_rg, colors_rg, linestyles_rg):
            axis.plot(grid, sps.gaussian_kde(values)(grid), label=label, color=color, linestyle=linestyle)
        axis.set_title(names_rg[index])
        axis.set_yticks([])
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    K_values_text = ", ".join(str(result["K"]) for result in dc_settings)
    fig.suptitle(f"Canonical dataset 001: p={p_reg}, d={p_reg + 4}; DC-BATS K={K_values_text}")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    plt.show()

    del full_draws_rg, subsampling_draws_rg, dc_draws_rg
    gc.collect()
"""),
]


def main():
    parser = argparse.ArgumentParser(
        description="Build the unexecuted Version-3 reproduction notebook."
    )
    parser.add_argument(
        "--replace-generated",
        action="store_true",
        help="Replace the generated Version-3 target after validating the safety-copy source.",
    )
    arguments = parser.parse_args()

    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "The safety-copy source notebook changed; inspect it before rebuilding Version 3."
        )
    if TARGET.exists() and not arguments.replace_generated:
        raise FileExistsError(f"Refusing to overwrite existing Version-3 notebook: {TARGET}")
    with SOURCE.open(encoding="utf-8") as source:
        notebook = json.load(source)
    result = copy.deepcopy(notebook)

    intro = "".join(result["cells"][1]["source"])
    intro = intro.replace(
        "- **Section 5** contains the comparison with divide-and-conquer (Ou et al., 2025) presented in the supplement to the paper.\n"
        "- **Section 6** contains the code used to generate the remaining figures in the paper that are not produced in the sections above.",
        "- **Section 6** contains the preserved low-dimensional comparison with divide-and-conquer (Ou et al., 2025) presented in the supplement to the paper.\n"
        "- **Section 7** contains the Version-3 regression-GARCH dimensional stress test and its extended DC-BATS comparison.\n"
        "- **Section 8** contains the code used to generate the remaining figures in the paper that are not produced in the sections above.",
    )
    result["cells"][1]["source"] = source_lines(intro)

    dc_heading = "".join(result["cells"][255]["source"])
    if not dc_heading.startswith("## 5."):
        raise RuntimeError("Expected DC-BATS heading was not found at source cell 255.")
    result["cells"][255]["source"] = source_lines(dc_heading.replace("## 5.", "## 6.", 1))

    later_heading = "".join(result["cells"][261]["source"])
    if not later_heading.startswith("## 6."):
        raise RuntimeError("Expected later-section heading was not found at source cell 261.")
    result["cells"][261]["source"] = source_lines(later_heading.replace("## 6.", "## 8.", 1))

    insertion_index = 261
    result["cells"][insertion_index:insertion_index] = SECTION_7_CELLS
    with TARGET.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=1, ensure_ascii=False)
        output.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
