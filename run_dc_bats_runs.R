# ------------------------------------------------------------------------------
# Run DC-BATS-style GARCH(1,1) with normal errors
# Matching our priors, parameterisation, recursion initialisation, 
# and full-data benchmark
#
# Description:
# This script runs a divide-and-conquer Bayesian approach for the GARCH(1,1)
# model with normal errors, following the DC-BATS framework. The implementation 
# is adapted to ensure consistency with the parameterisation, priors, and 
# initialisation used in our subsampling MCMC experiments, enabling a direct 
# comparison. The Stan code for the model is provided in garch11_normal.stan.
#
# The results produced by this script are read by the main code
# stabilised_weighted_data_subsampling.ipynb to generate the figures in the paper.
#
# The code should be run for K = 10, 20, 80 separately.
#
# Authors:
# - Zixuan Wang
# - Matias Quiroz
#
# Code adaptation:
# Core components adapted from Lachlan Astfalck's dcbats repository:
# https://github.com/astfalckl/dcbats/
#
# Last updated:
# 2026-05-03
# ------------------------------------------------------------------------------

library(cmdstanr)
library(posterior)
library(data.table)
library(ggplot2)

# ------------------------------------------------------------------------------
# Run settings
# ------------------------------------------------------------------------------

K <- 10   # We run this for 10, 20, 80. Results are saved for each K

n_chains <- 4
n_warmup <- 5000
n_sampling <- 5000
bary_M <- 10000 # Matches our 10 K post burn-in iteration

results_dir <- file.path("results", "DC_BATS", paste0("K_", K))
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------

data_file <- "data/garch11_with_sigma2.csv"
init_file <- "data/garch11_init_values.csv"
full_truth_file <- "data/full_truth.csv"
stan_file <- "garch11_normal.stan"

params <- c("mu", "omega", "alpha1", "beta1")

# MAP estimates from full-data
MAP <- list(
  mu = 5.96262719e-04,
  omega = 7.33360843e-02,
  alpha1 = 1.62675479e-01,
  beta1 = 7.82775666e-01
)

# Read starting values saved from Python
init_dt <- fread(init_file)

y_start_global <- as.numeric(init_dt$y_start[1])
sigma2_start_global <- as.numeric(init_dt$sigma2_start[1])

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

fit_chunk <- function(
    y_chunk,
    K,
    y_start,
    sigma2_start,
    MAP,
    chains = n_chains,
    iter_warmup = n_warmup,
    iter_sampling = n_sampling,
    seed = 123
) {
  data_list <- list(
    T = length(y_chunk),
    y = as.numeric(y_chunk),
    power = K,
    y_start = y_start,
    sigma2_start = sigma2_start
  )
  
  init_fun <- function() {
    MAP
  }
  
  mod$sample(
    data = data_list,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    seed = seed,
    init = init_fun,
    refresh = 200
  )
}

get_draws <- function(fit, vars) {
  as_draws_df(fit$draws(variables = vars))
}

ci_from_draws <- function(draw_df, vars, level = 0.95) {
  alpha <- (1 - level) / 2
  
  rbindlist(lapply(vars, function(v) {
    qs <- quantile(draw_df[[v]], probs = c(alpha, 1 - alpha), names = FALSE)
    
    data.table(
      parameter = v,
      lower = qs[1],
      upper = qs[2]
    )
  }))
}

wasserstein_barycenter_1d <- function(
    draw_list,
    vars,
    probs = seq(0.001, 0.999, by = 0.001)
) {
  out <- vector("list", length = length(vars))
  names(out) <- vars
  
  for (p in vars) {
    qmat <- sapply(draw_list, function(d) {
      quantile(d[[p]], probs = probs, names = FALSE)
    })
    
    out[[p]] <- data.frame(
      prob = probs,
      qbar = rowMeans(qmat)
    )
  }
  
  out
}

sample_barycenter_draws <- function(wbary_obj, M = bary_M) {
  u <- sort(runif(M))
  out <- data.frame(.draw = seq_len(M))
  
  for (p in names(wbary_obj)) {
    out[[p]] <- approx(
      x = wbary_obj[[p]]$prob,
      y = wbary_obj[[p]]$qbar,
      xout = u,
      rule = 2
    )$y
  }
  
  out
}

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

Y_dt <- fread(data_file)

Y <- as.numeric(Y_dt$y)
sigma2_true <- as.numeric(Y_dt$sigma2)

stopifnot(length(Y) == length(sigma2_true))

T_full <- length(Y)

# ------------------------------------------------------------------------------
# Split into contiguous chunks with correct boundary values
# ------------------------------------------------------------------------------

n_chunk <- T_full %/% K
T_used <- n_chunk * K

if (T_used < T_full) {
  message("Dropping the last ", T_full - T_used, " observations so chunks have equal size.")
  Y <- Y[seq_len(T_used)]
  sigma2_true <- sigma2_true[seq_len(T_used)]
}

chunks <- lapply(seq_len(K), function(k) {
  start_idx <- (k - 1) * n_chunk + 1
  end_idx <- k * n_chunk
  
  if (k == 1) {
    y_start_k <- y_start_global
    sigma2_start_k <- sigma2_start_global
  } else {
    y_start_k <- Y[start_idx - 1]
    sigma2_start_k <- sigma2_true[start_idx - 1]
  }
  
  list(
    y = Y[start_idx:end_idx],
    sigma2 = sigma2_true[start_idx:end_idx],
    y_start = y_start_k,
    sigma2_start = sigma2_start_k,
    start_idx = start_idx,
    end_idx = end_idx
  )
})

# ------------------------------------------------------------------------------
# Compile Stan model
# ------------------------------------------------------------------------------

mod <- cmdstan_model(stan_file)

# ------------------------------------------------------------------------------
# Fit powered shard posteriors and time them
# ------------------------------------------------------------------------------

total_start <- proc.time()

chunk_times <- numeric(length(chunks))

chunk_fits <- lapply(seq_along(chunks), function(k) {
  cat("\n===== Running chunk", k, "of", length(chunks), "=====\n")
  cat("Indices:", chunks[[k]]$start_idx, "to", chunks[[k]]$end_idx, "\n")
  cat("y_start:", chunks[[k]]$y_start, "\n")
  cat("sigma2_start:", chunks[[k]]$sigma2_start, "\n")
  flush.console()
  
  t0 <- proc.time()
  
  fit_k <- fit_chunk(
    y_chunk = chunks[[k]]$y,
    K = K,
    y_start = chunks[[k]]$y_start,
    sigma2_start = chunks[[k]]$sigma2_start,
    MAP = MAP,
    seed = 100 + k
  )
  
  chunk_times[k] <<- (proc.time() - t0)[["elapsed"]]
  
  fit_k
})

chunk_elapsed_total <- sum(chunk_times)
chunk_elapsed_mean <- mean(chunk_times)
chunk_elapsed_max <- max(chunk_times)

# ------------------------------------------------------------------------------
# HMC diagnostics across all chunks
# ------------------------------------------------------------------------------

hmc_diag_dt <- rbindlist(lapply(seq_along(chunk_fits), function(k) {
  fit_k <- chunk_fits[[k]]
  
  summ <- as.data.table(fit_k$summary(variables = params))
  summ[, chunk := k]
  
  sampler_diag <- fit_k$sampler_diagnostics()
  
  n_divergent <- sum(sampler_diag[, , "divergent__"])
  n_max_treedepth <- sum(sampler_diag[, , "treedepth__"] >= 10)
  
  data.table(
    chunk = k,
    max_rhat = max(summ$rhat, na.rm = TRUE),
    min_ess_bulk = min(summ$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(summ$ess_tail, na.rm = TRUE),
    n_divergent = n_divergent,
    n_max_treedepth = n_max_treedepth
  )
}))

print(hmc_diag_dt)

if (any(hmc_diag_dt$n_divergent > 0)) {
  warning("At least one chunk has divergent transitions.")
}

if (any(hmc_diag_dt$n_max_treedepth > 0)) {
  warning("At least one chunk hit the maximum treedepth.")
}

if (any(hmc_diag_dt$max_rhat > 1.01, na.rm = TRUE)) {
  warning("At least one chunk has max R-hat > 1.01.")
}

if (any(hmc_diag_dt$min_ess_bulk < 100, na.rm = TRUE)) {
  warning("At least one chunk has bulk ESS < 100.")
}

# ------------------------------------------------------------------------------
# Extract shard posterior draws
# ------------------------------------------------------------------------------

extract_start <- proc.time()

chunk_draws <- lapply(chunk_fits, get_draws, vars = params)

chunk_draws_dt <- rbindlist(lapply(seq_along(chunk_draws), function(k) {
  dt <- as.data.table(chunk_draws[[k]])
  dt[, chunk := k]
  dt
}))

extract_elapsed <- (proc.time() - extract_start)[["elapsed"]]

# ------------------------------------------------------------------------------
# Wasserstein barycentre aggregation
# ------------------------------------------------------------------------------

agg_start <- proc.time()

wbary <- wasserstein_barycenter_1d(
  draw_list = chunk_draws,
  vars = params
)

set.seed(123)

bary_draws <- sample_barycenter_draws(
  wbary_obj = wbary,
  M = bary_M
)

agg_elapsed <- (proc.time() - agg_start)[["elapsed"]]

# ------------------------------------------------------------------------------
# Compare with full-data posterior
# ------------------------------------------------------------------------------

comparison_start <- proc.time()

full_draws <- fread(full_truth_file)

ci_full <- ci_from_draws(full_draws, params)
ci_dc <- ci_from_draws(bary_draws, params)

result <- merge(
  ci_dc,
  ci_full,
  by = "parameter",
  suffixes = c("_dc", "_full")
)

comparison_elapsed <- (proc.time() - comparison_start)[["elapsed"]]

print(result)

total_elapsed <- (proc.time() - total_start)[["elapsed"]]

# Idealised parallel timing:
# - sequential observed time is total_elapsed
# - ideal shard fitting time is approximated by max chunk time
# - alternative average shard timing is total chunk time / K
ideal_parallel_elapsed_max <- chunk_elapsed_max + extract_elapsed + agg_elapsed + comparison_elapsed
ideal_parallel_elapsed_mean <- chunk_elapsed_mean + extract_elapsed + agg_elapsed + comparison_elapsed

timing_dt <- data.table(
  K = K,
  n_chunk = n_chunk,
  T_used = T_used,
  n_chains = n_chains,
  n_warmup = n_warmup,
  n_sampling = n_sampling,
  bary_M = bary_M,
  chunk_elapsed_total = chunk_elapsed_total,
  chunk_elapsed_mean = chunk_elapsed_mean,
  chunk_elapsed_max = chunk_elapsed_max,
  extract_elapsed = extract_elapsed,
  aggregation_elapsed = agg_elapsed,
  comparison_elapsed = comparison_elapsed,
  total_elapsed_sequential = total_elapsed,
  ideal_parallel_elapsed_mean = ideal_parallel_elapsed_mean,
  ideal_parallel_elapsed_max = ideal_parallel_elapsed_max
)

print(timing_dt)

chunk_timing_dt <- data.table(
  K = K,
  chunk = seq_along(chunk_times),
  elapsed = chunk_times,
  start_idx = vapply(chunks, function(x) as.integer(x$start_idx), integer(1)),
  end_idx = vapply(chunks, function(x) as.integer(x$end_idx), integer(1))
)

# ------------------------------------------------------------------------------
# KDE plot
# ------------------------------------------------------------------------------

full_dt <- as.data.table(full_draws)
dc_dt <- as.data.table(bary_draws)

full_long <- melt(
  full_dt[, params, with = FALSE],
  measure.vars = params,
  variable.name = "parameter",
  value.name = "value"
)
full_long[, method := "Full-data MCMC"]

dc_long <- melt(
  dc_dt[, params, with = FALSE],
  measure.vars = params,
  variable.name = "parameter",
  value.name = "value"
)
dc_long[, method := "DC-BATS"]

plot_dt <- rbind(full_long, dc_long)

param_labs <- c(
  mu = "mu",
  omega = "omega",
  alpha1 = "alpha[1]",
  beta1 = "beta[1]"
)

p <- ggplot(plot_dt, aes(x = value, linetype = method)) +
  geom_density(linewidth = 0.8) +
  facet_wrap(
    ~ parameter,
    scales = "free",
    ncol = 2,
    labeller = labeller(parameter = as_labeller(param_labs, label_parsed))
  ) +
  labs(
    x = NULL,
    y = "Density",
    linetype = NULL
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "white"),
    panel.grid.minor = element_blank()
  )

print(p)

ggsave(
  filename = file.path(results_dir, paste0("kde_full_vs_dcbats_garch11_K", K, ".pdf")),
  plot = p,
  width = 8,
  height = 6
)

# ------------------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------------------

fwrite(result, file.path(results_dir, paste0("results_intervals_garch11_dc_K", K, ".csv")))
fwrite(timing_dt, file.path(results_dir, paste0("timing_garch11_dc_K", K, ".csv")))
fwrite(chunk_timing_dt, file.path(results_dir, paste0("chunk_timing_garch11_dc_K", K, ".csv")))
fwrite(hmc_diag_dt, file.path(results_dir, paste0("hmc_diagnostics_garch11_dc_K", K, ".csv")))
fwrite(as.data.table(full_draws), file.path(results_dir, paste0("full_truth_used_garch11_dc_K", K, ".csv")))
fwrite(as.data.table(bary_draws), file.path(results_dir, paste0("bary_draws_garch11_dc_K", K, ".csv")))
fwrite(chunk_draws_dt, file.path(results_dir, paste0("chunk_draws_garch11_dc_K", K, ".csv")))


saveRDS(
  list(
    K = K,
    n_chunk = n_chunk,
    T_used = T_used,
    n_chains = n_chains,
    n_warmup = n_warmup,
    n_sampling = n_sampling,
    bary_M = bary_M,
    y_start_global = y_start_global,
    sigma2_start_global = sigma2_start_global,
    MAP = MAP,
    chunks = lapply(chunks, function(x) {
      list(
        start_idx = x$start_idx,
        end_idx = x$end_idx,
        y_start = x$y_start,
        sigma2_start = x$sigma2_start
      )
    }),
    hmc_diagnostics = hmc_diag_dt,
    wbary = wbary,
    ci = result,
    timing = timing_dt,
    chunk_timing = chunk_timing_dt
  ),
  file.path(results_dir, paste0("dc_bats_garch11_results_K", K, ".rds"))
)
