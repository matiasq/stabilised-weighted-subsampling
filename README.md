<p align="center">
  <img src="tpd_weights_readme.png" width="700"/>
</p>

# Stabilised weighted data subsampling

This repository contains the code accompanying Version 3 of the arXiv manuscript
“Stabilised weighted data subsampling for accelerated Bayesian inference in
models with recursive likelihoods.”

Quiroz, M., Bhaskaran, A., Wang, Z., and Goodwin, T. (2026).  
[arXiv:2605.13397](https://arxiv.org/abs/2605.13397)

## Main analysis

`stabilised_weighted_data_subsampling.ipynb` is the main reproduction notebook.
It contains the subsampling variational Bayes and MCMC analyses, the SG-MCMC and
DC-BATS comparisons, and the regression-GARCH dimensional experiment added for
Version 3. The notebook should be run cell by cell in order.

Section 7 is self-contained. It generates the ten regression-GARCH datasets from
their fixed seeds, verifies the two dataset-001 reference hashes, runs the
restart-safe full-data and subsampling analyses, calculates the five
computational ratios and their median, determines `K_stress`, and exports the
dataset-001 inputs required by the R comparison. The cases have `p=25` and
`p=50` regressors, corresponding to total parameter dimensions `d=29` and
`d=54`.

The production calculations are expensive. Completed stages are cached under
`regression_garch_outputs/`, so interrupted Section-7 cells can be rerun without
repeating valid stages.

## SG-MCMC and KSD comparison

`sg_mcmc_runs.ipynb` generates the SG-MCMC chains and kernelised Stein
discrepancy (KSD) results. It uses the comparison inputs in `data/`, including
the full-data and subsampling reference draws. The notebook records the small
compatibility changes required for the external
[`sgmcmc_ssm`](https://github.com/aicherc/sgmcmc_ssm_code/) package.

Run this notebook before the SG-MCMC result-loading cells in the main notebook.

## DC-BATS comparisons

The original GARCH(1,1) comparison is implemented by `run_dc_bats_runs.R` and
`garch11_normal.stan`. Its required input CSV files are tracked under `data/`.
The final settings are:

```bash
Rscript run_dc_bats_runs.R --K=10
Rscript run_dc_bats_runs.R --K=20
Rscript run_dc_bats_runs.R --K=76
```

The regression-GARCH Student-t comparison is implemented by
`run_regression_garch_dc_bats.R` and `regression_garch_student_t.stan`. First
run Section 7 of the main notebook through the DC-BATS export cells. Then run:

```bash
Rscript run_regression_garch_dc_bats.R --p=25 --K=10
Rscript run_regression_garch_dc_bats.R --p=25 --K=20
Rscript run_regression_garch_dc_bats.R --p=25 --K=37
Rscript run_regression_garch_dc_bats.R --p=50 --K=10
Rscript run_regression_garch_dc_bats.R --p=50 --K=20
Rscript run_regression_garch_dc_bats.R --p=50 --K=36
```

Finally, rerun the Section-7 result-loading and plotting cells to include the R
posterior draws in the comparison figures.

Both DC-BATS implementations use contiguous shards, shard-specific likelihood
powers and the coordinate-wise Wasserstein barycentre described in the paper.

## Data and generated results

The repository tracks the small input datasets needed by the notebooks and R
comparisons. In particular:

- the Apple and Dow Jones return series support the main applications;
- `data/garch_tau2_0p01pct.npz` supports the SG-MCMC example;
- `data/full_data_run_tau2_0p01.npz` and
  `data/subsampling_run_tau2_0p01.npz` contain its reference posterior draws;
- `data/garch11_with_sigma2.csv`, `data/garch11_init_values.csv`, and
  `data/full_truth.csv` are the inputs for the original DC-BATS comparison.

Regression-GARCH datasets, posterior chains, Hessians, checkpoints, R exports
and DC-BATS outputs are generated locally and ignored by Git. The principal
generated paths are:

```text
data/regression_garch_*_dataset_*.npz
regression_garch_outputs/
results/
```

## Software

The final regression-GARCH run used Python 3.12.12, NumPy 2.4.1 and SciPy
1.17.0. The notebooks also use pandas, Matplotlib, ArviZ, Autograd, h5py,
numdifftools and Jupyter.

The R comparisons use `cmdstanr`, `posterior`, `data.table`, `jsonlite`,
`digest`, and `ggplot2`, together with a working CmdStan installation.

## Authors

- `stabilised_weighted_data_subsampling.ipynb`: Matias Quiroz, Aishwarya Bhaskaran
- `sg_mcmc_runs.ipynb`: Thomas Goodwin, Matias Quiroz
- `run_dc_bats_runs.R`: Zixuan Wang, Matias Quiroz

## License

This repository is released under the MIT License.
