# Stabilised weighted subsampling

This repository contains code and datasets to reproduce the results in:

Quiroz, Bhaskaran, Wang and Goodwin (2026).  
*Stabilised weighted data subsampling for accelerated inference in models with recursive likelihoods.*

## Overview

The repository implements subsampling variational Bayes and subsampling Markov chain Monte Carlo methods for models with recursively defined likelihoods. It also includes comparisons with stochastic gradient MCMC (SG-MCMC) and divide-and-conquer methods for temporally dependent data.

## Structure

- `stabilised_weighted_data_subsampling.ipynb`: Main notebook used to reproduce the figures and tables in the paper.
- `sg_mcmc_runs.ipynb`: Notebook used to generate results for the SG-MCMC comparison.
- `run_dc_bats_runs.R`: Script used to generate results for the divide-and-conquer comparison.
- `garch11_normal.stan`: Stan model used in the DC-BATS implementation.
- Data files: Included to allow full reproducibility.

## Instructions

1. Run `sg_mcmc_runs.ipynb` and `run_dc_bats_runs.R` to generate results for the competing methods.
2. Run `stabilised_weighted_data_subsampling.ipynb` to reproduce the results in the paper.

The notebook should be executed **cell by cell in order**, as later sections depend on objects created earlier.

## Notes

- Some parts of the code are computationally intensive, particularly the full-data MCMC experiments.
- The notebook uses a caching mechanism: results are saved to disk and reused if available on disk.
- The code is intended for reproducibility, but can be adapted with care to related settings.

## Requirements

The code relies on standard Python and R scientific computing libraries.

- Python: `numpy`, `scipy`, `pandas`, `matplotlib`, `arviz`
- R: `rstan`, `cmdstanr`, and standard supporting packages

Users may need to install additional dependencies depending on their environment.

## Related code

The implementations used for comparison are based on the following publicly available repositories:

- SG-MCMC for state space models: https://github.com/aicherc/sgmcmc_ssm_code/
- DC-BATS (divide-and-conquer for Bayesian time series): https://github.com/astfalckl/dcbats/

## Authors

Contributions by file:

- `stabilised_weighted_data_subsampling.ipynb`: Matias Quiroz, Aishwarya Bhaskaran  
- `sg_mcmc_runs.ipynb`: Thomas Goodwin, Matias Quiroz  
- `run_dc_bats_runs.R`: Zixuan Wang, Matias Quiroz  

See individual files for additional details.

## License

This repository is released under the MIT License.
