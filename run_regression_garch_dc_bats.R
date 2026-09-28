# ------------------------------------------------------------------------------
# Restart-safe DC-BATS extension for regression-GARCH(1,1), Student-t errors.
#
# This is deliberately separate from run_dc_bats_runs.R.  It uses canonical
# dataset 001 exported by export_regression_garch_dcbats.py and refuses to run
# until the five-dataset Python median R_CPU has deterministically selected K.
# No data are simulated in R.
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
  library(data.table)
  library(jsonlite)
  library(digest)
})

parse_run_arguments <- function(arguments) {
  p_matches <- grep("^--p=", arguments, value = TRUE)
  K_matches <- grep("^--K=", arguments, value = TRUE)
  if (length(p_matches) != 1L || length(K_matches) != 1L) {
    stop("Supply exactly one dimension and setting as --p=25|50 --K=<integer>.")
  }
  p <- as.integer(sub("^--p=", "", p_matches))
  K <- as.integer(sub("^--K=", "", K_matches))
  if (!(p %in% c(25L, 50L))) stop("--p must be 25 or 50.")
  if (is.na(K) || K < 1L) stop("--K must be a positive integer.")
  list(p = p, K = K)
}

atomic_fwrite <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  fwrite(value, temporary)
  if (!file.rename(temporary, path)) stop("Atomic rename failed for ", path)
}

atomic_save_rds <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  saveRDS(value, temporary)
  if (!file.rename(temporary, path)) stop("Atomic rename failed for ", path)
}

atomic_write_json <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  write_json(value, temporary, pretty = TRUE, auto_unbox = TRUE, digits = NA)
  if (!file.rename(temporary, path)) stop("Atomic rename failed for ", path)
}

select_k_stress <- function(median_R_CPU) {
  if (!is.finite(median_R_CPU) || median_R_CPU <= 0) {
    stop("median_R_CPU must be a positive finite value.")
  }
  K_stress <- as.integer(floor(median_R_CPU + 0.5))
  if (K_stress < 1L) stop("Rounded K_stress is zero, so no valid shard configuration exists.")
  K_stress
}

parameter_names <- function(P) {
  c(paste0("beta_reg_", seq_len(P)), "omega", "alpha", "beta_garch", "nu")
}

canonicalise_draw_names <- function(draws, P) {
  draws <- as.data.table(draws)
  bracket_names <- paste0("beta_reg[", seq_len(P), "]")
  expected <- c(bracket_names, "omega", "alpha", "beta_garch", "nu")
  missing <- setdiff(expected, names(draws))
  if (length(missing)) stop("Stan draws are missing: ", paste(missing, collapse = ", "))
  draws <- draws[, ..expected]
  setnames(draws, bracket_names, paste0("beta_reg_", seq_len(P)))
  draws
}

ci_from_draws <- function(draws, variables, method, level = 0.95) {
  alpha_level <- (1 - level) / 2
  rbindlist(lapply(variables, function(variable) {
    quantiles <- quantile(draws[[variable]], c(alpha_level, 0.5, 1 - alpha_level), names = FALSE)
    data.table(
      method = method, parameter = variable,
      mean = mean(draws[[variable]]), sd = sd(draws[[variable]]),
      lower = quantiles[1], median = quantiles[2], upper = quantiles[3]
    )
  }))
}

agreement_summary <- function(reference, candidate, variables, reference_name, candidate_name) {
  rbindlist(lapply(variables, function(variable) {
    ref <- reference[[variable]]
    cand <- candidate[[variable]]
    ref_q <- quantile(ref, c(0.025, 0.975), names = FALSE)
    cand_q <- quantile(cand, c(0.025, 0.975), names = FALSE)
    data.table(
      reference = reference_name,
      candidate = candidate_name,
      parameter = variable,
      absolute_standardised_mean_difference = abs(mean(cand) - mean(ref)) / sd(ref),
      posterior_sd_ratio = sd(cand) / sd(ref),
      marginal_95_interval_overlap = max(ref_q[1], cand_q[1]) <= min(ref_q[2], cand_q[2])
    )
  }))
}

wasserstein_barycenter_1d <- function(draw_list, variables, probs = seq(0.001, 0.999, by = 0.001)) {
  result <- vector("list", length(variables))
  names(result) <- variables
  for (variable in variables) {
    quantile_matrix <- sapply(draw_list, function(draws) {
      quantile(draws[[variable]], probs = probs, names = FALSE)
    })
    result[[variable]] <- data.frame(prob = probs, qbar = rowMeans(quantile_matrix))
  }
  result
}

sample_barycenter_draws <- function(barycenter, M) {
  probabilities <- sort(runif(M))
  result <- data.table(.draw = seq_len(M))
  for (variable in names(barycenter)) {
    result[[variable]] <- approx(
      barycenter[[variable]]$prob,
      barycenter[[variable]]$qbar,
      xout = probabilities,
      rule = 2
    )$y
  }
  result
}

run_arguments <- parse_run_arguments(commandArgs(trailingOnly = TRUE))
P <- run_arguments$p
K <- run_arguments$K
D <- P + 4L
dataset_id <- "001"
partition_schema <- "balanced_all_observations_v1"

expected_hashes <- c(
  `25` = "07a016de0093074e65d29f3a2a87fb5e9f4af0f38fe0f50ce10fa3e9fad295cd",
  `50` = "fa821d80909ca019ad41fdb3938f240df1b8e91e965bac1a05525c4186180b65"
)

n_chains <- 4L
n_warmup <- 5000L
n_sampling <- 5000L
bary_M <- 10000L

resource_file <- file.path("regression_garch_outputs", "dcbats_resource_config.csv")
if (!file.exists(resource_file)) {
  stop(
    "DC-BATS resource configuration is absent. Complete all five Python datasets and run ",
    "export_regression_garch_dcbats.py --include-python-results first."
  )
}
resource_all <- fread(resource_file)
resources_p <- resource_all[p == P]
if (!all(c(10L, 20L) %in% resources_p$K)) {
  stop("Resource configuration must retain standard K=10 and K=20 for p=", P)
}
if (length(unique(resources_p$median_R_CPU)) != 1L) {
  stop("Resource rows disagree about median_R_CPU for p=", P)
}
expected_K_stress <- select_k_stress(resources_p$median_R_CPU[1])
if (length(unique(resources_p$K_stress)) != 1L || resources_p$K_stress[1] != expected_K_stress) {
  stop("Stored K_stress does not equal ordinary nearest-integer rounding of median_R_CPU.")
}
expected_distinct_K <- unique(c(10L, 20L, expected_K_stress))
if (!setequal(resources_p$K, expected_distinct_K)) {
  stop("Resource configuration does not contain exactly the distinct K=10, K=20, K_stress settings.")
}
resource <- resources_p[K == run_arguments$K]
if (nrow(resource) != 1L) {
  stop("Requested (p,K) is not one of the pre-specified resource configurations.")
}
T_total <- as.integer(resource$T_total)
if (K > T_total) stop("K cannot exceed the number of observations.")
R_CPU_columns <- paste0("R_CPU_", seq_len(5))
if (!all(R_CPU_columns %in% names(resource))) {
  stop("Resource configuration must include all five R_CPU values.")
}
R_CPU_values <- unname(unlist(resource[1, ..R_CPU_columns], use.names = FALSE))
if (abs(median(R_CPU_values) - resource$median_R_CPU) > 1e-12) {
  stop("Resource median_R_CPU is inconsistent with its five R_CPU values.")
}
roles <- strsplit(resource$K_roles, "\\|", fixed = FALSE)[[1]]
if (K == 10L && !("standard_10" %in% roles)) stop("K=10 row lacks standard_10 role.")
if (K == 20L && !("standard_20" %in% roles)) stop("K=20 row lacks standard_20 role.")
if ((K == expected_K_stress) != ("cpu_matched" %in% roles)) {
  stop("cpu_matched role is inconsistent with K_stress.")
}

export_dir <- file.path(
  "regression_garch_outputs", "dcbats_exports", paste0("p", P), "dataset_001"
)
metadata_file <- file.path(export_dir, "metadata.json")
required_inputs <- file.path(
  export_dir,
  c(
    "X.csv", "series.csv", "presample.csv", "true_parameters.csv",
    "map_initial_values.csv", "full_data_draws.csv", "subsampling_draws.csv",
    "python_results_metadata.json"
  )
)
missing_inputs <- required_inputs[!file.exists(required_inputs)]
if (!file.exists(metadata_file) || length(missing_inputs)) {
  stop("R export is incomplete: ", paste(c(metadata_file[!file.exists(metadata_file)], missing_inputs), collapse = ", "))
}

metadata <- read_json(metadata_file, simplifyVector = TRUE)
if (metadata$p != P || metadata$total_parameter_dimension != D || metadata$dataset_id != dataset_id) {
  stop("Export metadata does not identify the requested experiment.")
}
if (metadata$source_npz_sha256 != expected_hashes[[as.character(P)]]) {
  stop("Export metadata contains the wrong canonical NPZ SHA-256.")
}
source_npz <- metadata$source_npz
if (!file.exists(source_npz)) stop("Canonical source NPZ is missing: ", source_npz)
actual_hash <- digest(source_npz, algo = "sha256", serialize = FALSE, file = TRUE)
if (actual_hash != metadata$source_npz_sha256) stop("Canonical source NPZ SHA-256 verification failed.")

X <- as.matrix(fread(file.path(export_dir, "X.csv")))
series <- fread(file.path(export_dir, "series.csv"))
presample <- fread(file.path(export_dir, "presample.csv"))
map_values <- fread(file.path(export_dir, "map_initial_values.csv"))
full_draws <- fread(file.path(export_dir, "full_data_draws.csv"))
subsampling_draws <- fread(file.path(export_dir, "subsampling_draws.csv"))
variables <- parameter_names(P)

if (!identical(dim(X), c(T_total, P))) stop("Exported X has the wrong dimensions.")
if (nrow(series) != T_total || !all(c("y", "sigma2_true") %in% names(series))) {
  stop("Exported response/variance series is invalid.")
}
if (!all(variables %in% names(map_values)) || !all(variables %in% names(full_draws)) ||
    !all(variables %in% names(subsampling_draws))) {
  stop("Python parameter exports have inconsistent columns.")
}

Y <- as.numeric(series$y)
sigma2_true <- as.numeric(series$sigma2_true)
q <- T_total %/% K
r <- T_total - K * q
shard_sizes <- c(rep(q + 1L, r), rep(q, K - r))
shard_likelihood_powers <- T_total / shard_sizes
shard_end_indices <- cumsum(shard_sizes)
shard_start_indices <- c(1L, head(shard_end_indices, -1L) + 1L)
T_used <- sum(shard_sizes)
dropped_observations <- T_total - T_used
if (
  length(shard_sizes) != K || T_used != T_total || dropped_observations != 0L ||
  max(shard_sizes) - min(shard_sizes) > 1L ||
  !identical(unname(shard_start_indices), c(1L, head(shard_end_indices, -1L) + 1L)) ||
  shard_end_indices[K] != T_total
) {
  stop("Balanced all-observation shard construction failed.")
}
if (any(abs(shard_likelihood_powers * shard_sizes - T_total) > 1e-10)) {
  stop("Shard-specific likelihood powers are inconsistent with T_total / n_k.")
}
if (r == 0L && any(abs(shard_likelihood_powers - K) > 1e-12)) {
  stop("Equal-size shards must reduce exactly to likelihood power K.")
}
configured_shard_sizes <- as.integer(fromJSON(resource$shard_sizes_json))
configured_likelihood_powers <- as.numeric(fromJSON(resource$shard_likelihood_powers_json))
if (
  resource$T_used != T_total || resource$dropped_observations != 0L ||
  resource$base_shard_size_q != q || resource$larger_shard_count_r != r ||
  !identical(configured_shard_sizes, as.integer(shard_sizes)) ||
  length(configured_likelihood_powers) != K ||
  any(abs(configured_likelihood_powers - shard_likelihood_powers) > 1e-12)
) {
  stop("Resource configuration does not match the balanced all-observation partition.")
}
chunks <- lapply(seq_len(K), function(k) {
  start_idx <- shard_start_indices[k]
  end_idx <- shard_end_indices[k]
  if (k == 1L) {
    y_start <- as.numeric(presample$y_start[1])
    X_start <- unname(unlist(
      presample[1, paste0("x_start_", seq_len(P)), with = FALSE],
      use.names = FALSE
    ))
    sigma2_start <- as.numeric(presample$sigma2_start[1])
  } else {
    # The residual y_{s-1} - X_{s-1} beta is recomputed inside Stan per draw.
    y_start <- Y[start_idx - 1L]
    X_start <- X[start_idx - 1L, ]
    # Oracle simulated boundary variance, preserving the Section-6 convention.
    sigma2_start <- sigma2_true[start_idx - 1L]
  }
  list(
    y = Y[start_idx:end_idx], X = X[start_idx:end_idx, , drop = FALSE],
    y_start = y_start, X_start = X_start, sigma2_start = sigma2_start,
    start_idx = start_idx, end_idx = end_idx, n_k = shard_sizes[k],
    likelihood_power = shard_likelihood_powers[k]
  )
})

shard_configuration <- data.table(
  shard = seq_len(K), start_idx = shard_start_indices, end_idx = shard_end_indices,
  n_k = shard_sizes, likelihood_power = shard_likelihood_powers
)

results_dir <- file.path(
  "results", "DC_BATS_regression_garch", paste0("p", P), "dataset_001", paste0("K_", K)
)
chunks_dir <- file.path(results_dir, "chunks")
dir.create(chunks_dir, recursive = TRUE, showWarnings = FALSE)
completion_file <- file.path(results_dir, "COMPLETED.json")
if (file.exists(completion_file)) {
  completed <- read_json(completion_file, simplifyVector = TRUE)
  if (
    completed$source_npz_sha256 == actual_hash && completed$K == K &&
    completed$p == P && setequal(completed$K_roles, roles) &&
    abs(completed$median_R_CPU - resource$median_R_CPU) <= 1e-12 &&
    identical(completed$partition_schema, partition_schema) &&
    identical(as.integer(completed$T_used), T_total) &&
    identical(as.integer(completed$dropped_observations), 0L)
  ) {
    message("Valid completed result already exists; nothing to rerun: ", results_dir)
    quit(save = "no", status = 0L)
  }
  stop("Completion marker exists but does not match the requested experiment.")
}

stan_file <- "regression_garch_student_t.stan"
if (!file.exists(stan_file)) stop("Missing Stan model: ", stan_file)
mod <- cmdstan_model(stan_file)

init_row <- map_values[1, ..variables]
init_list <- list(
  beta_reg = unname(unlist(
    init_row[, paste0("beta_reg_", seq_len(P)), with = FALSE],
    use.names = FALSE
  )),
  omega = as.numeric(init_row$omega),
  alpha = as.numeric(init_row$alpha),
  beta_garch = as.numeric(init_row$beta_garch),
  nu_minus_two = as.numeric(init_row$nu) - 2
)
if (init_list$alpha + init_list$beta_garch >= 1 || init_list$nu_minus_two <= 0) {
  stop("Exported MAP initial values violate the model constraints.")
}

fit_chunk <- function(chunk, k) {
  mod$sample(
    data = list(
      T = length(chunk$y), P = P, X = chunk$X, y = chunk$y,
      power = chunk$likelihood_power, X_start = chunk$X_start, y_start = chunk$y_start,
      sigma2_start = chunk$sigma2_start
    ),
    chains = n_chains,
    parallel_chains = n_chains,
    iter_warmup = n_warmup,
    iter_sampling = n_sampling,
    seed = 20267000L + P * 100L + k,
    init = function() init_list,
    refresh = 200
  )
}

chunk_draw_list <- vector("list", K)
chunk_timing <- vector("list", K)
chunk_diagnostics <- vector("list", K)

for (k in seq_len(K)) {
  prefix <- file.path(chunks_dir, sprintf("chunk_%03d", k))
  draws_path <- paste0(prefix, "_draws.csv")
  timing_path <- paste0(prefix, "_timing.csv")
  diagnostics_path <- paste0(prefix, "_diagnostics.csv")
  marker_path <- paste0(prefix, "_complete.rds")

  valid_cache <- FALSE
  cache_products <- c(draws_path, timing_path, diagnostics_path)
  if (!file.exists(marker_path) && all(file.exists(cache_products))) {
    cached_draws <- fread(draws_path)
    cached_timing <- fread(timing_path)
    cached_diagnostics <- fread(diagnostics_path)
    recoverable <- nrow(cached_draws) == n_chains * n_sampling &&
      all(variables %in% names(cached_draws)) && nrow(cached_timing) == 1L &&
      "draw_extraction_elapsed" %in% names(cached_timing) &&
      nrow(cached_diagnostics) == 1L && cached_timing$p == P &&
      cached_timing$K == K && cached_timing$start_idx == chunks[[k]]$start_idx &&
      cached_timing$end_idx == chunks[[k]]$end_idx &&
      cached_timing$n_k == chunks[[k]]$n_k &&
      abs(cached_timing$likelihood_power - chunks[[k]]$likelihood_power) <= 1e-12 &&
      cached_diagnostics$p == P && cached_diagnostics$K == K
    if (recoverable) {
      atomic_save_rds(
        list(
          source_npz_sha256 = actual_hash, p = P, K = K,
          start_idx = chunks[[k]]$start_idx, end_idx = chunks[[k]]$end_idx,
          n_k = chunks[[k]]$n_k, likelihood_power = chunks[[k]]$likelihood_power,
          partition_schema = partition_schema,
          n_draws = nrow(cached_draws),
          completed_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
          marker_recovered_after_interruption = TRUE
        ),
        marker_path
      )
    }
    rm(cached_draws, cached_timing, cached_diagnostics)
  }
  if (file.exists(marker_path) && file.exists(draws_path) && file.exists(timing_path) && file.exists(diagnostics_path)) {
    marker <- readRDS(marker_path)
    valid_cache <- identical(marker$source_npz_sha256, actual_hash) &&
      identical(marker$p, P) && identical(marker$K, K) &&
      identical(marker$start_idx, chunks[[k]]$start_idx) && identical(marker$end_idx, chunks[[k]]$end_idx)
    valid_cache <- valid_cache && identical(marker$n_k, chunks[[k]]$n_k) &&
      isTRUE(all.equal(marker$likelihood_power, chunks[[k]]$likelihood_power, tolerance = 1e-12)) &&
      identical(marker$partition_schema, partition_schema)
  }

  if (valid_cache) {
    message("Loading completed shard ", k, " of ", K)
    chunk_draw_list[[k]] <- fread(draws_path)
    chunk_timing[[k]] <- fread(timing_path)
    chunk_diagnostics[[k]] <- fread(diagnostics_path)
    next
  }
  if (any(file.exists(c(cache_products, marker_path)))) {
    interrupted_dir <- file.path(chunks_dir, "interrupted")
    dir.create(interrupted_dir, recursive = TRUE, showWarnings = FALSE)
    stamp <- format(Sys.time(), "%Y%m%dT%H%M%S", tz = "UTC")
    existing <- c(cache_products, marker_path)[file.exists(c(cache_products, marker_path))]
    archived <- file.path(interrupted_dir, paste0(basename(existing), ".", stamp))
    if (!all(file.rename(existing, archived))) {
      stop("Could not archive an incomplete shard cache: ", prefix)
    }
    message("Archived incomplete shard cache before rerun: ", prefix)
  }

  message("Running shard ", k, " of ", K, ": indices ", chunks[[k]]$start_idx, "--", chunks[[k]]$end_idx)
  started <- proc.time()
  fit <- fit_chunk(chunks[[k]], k)
  fit_elapsed <- (proc.time() - started)[["elapsed"]]

  extraction_started <- proc.time()
  draws <- canonicalise_draw_names(
    as_draws_df(fit$draws(variables = c("beta_reg", "omega", "alpha", "beta_garch", "nu"))), P
  )
  draw_extraction_elapsed <- (proc.time() - extraction_started)[["elapsed"]]
  summary <- as.data.table(fit$summary(variables = c("beta_reg", "omega", "alpha", "beta_garch", "nu")))
  sampler_diagnostics <- fit$sampler_diagnostics()
  diagnostics <- data.table(
    p = P, d = D, K = K, chunk = k,
    max_rhat = max(summary$rhat, na.rm = TRUE),
    min_ess_bulk = min(summary$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(summary$ess_tail, na.rm = TRUE),
    n_divergent = sum(sampler_diagnostics[, , "divergent__"]),
    n_max_treedepth = sum(sampler_diagnostics[, , "treedepth__"] >= 10)
  )
  timing <- data.table(
    p = P, d = D, K = K, chunk = k, elapsed = fit_elapsed,
    draw_extraction_elapsed = draw_extraction_elapsed,
    start_idx = chunks[[k]]$start_idx, end_idx = chunks[[k]]$end_idx,
    n_k = chunks[[k]]$n_k, likelihood_power = chunks[[k]]$likelihood_power
  )
  atomic_fwrite(draws, draws_path)
  atomic_fwrite(timing, timing_path)
  atomic_fwrite(diagnostics, diagnostics_path)
  atomic_save_rds(
    list(
      source_npz_sha256 = actual_hash, p = P, K = K,
      start_idx = chunks[[k]]$start_idx, end_idx = chunks[[k]]$end_idx,
      n_k = chunks[[k]]$n_k, likelihood_power = chunks[[k]]$likelihood_power,
      partition_schema = partition_schema,
      n_draws = nrow(draws), completed_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
    ),
    marker_path
  )
  chunk_draw_list[[k]] <- draws
  chunk_timing[[k]] <- timing
  chunk_diagnostics[[k]] <- diagnostics
  rm(fit, draws, summary, sampler_diagnostics, diagnostics, timing)
  gc()
}

cache_reload_started <- proc.time()
# Re-read all persisted draws so fresh and resumed runs use the identical path.
chunk_draw_list <- lapply(seq_len(K), function(k) {
  fread(file.path(chunks_dir, sprintf("chunk_%03d_draws.csv", k)))
})
chunk_draws <- rbindlist(lapply(seq_len(K), function(k) {
  result <- copy(chunk_draw_list[[k]])
  result[, chunk := k]
  result
}))
cache_reload_elapsed <- (proc.time() - cache_reload_started)[["elapsed"]]
chunk_timing_dt <- rbindlist(chunk_timing)
hmc_diagnostics_dt <- rbindlist(chunk_diagnostics)
extract_elapsed <- sum(chunk_timing_dt$draw_extraction_elapsed)
if (any(hmc_diagnostics_dt$n_divergent > 0)) warning("At least one shard has divergent transitions.")
if (any(hmc_diagnostics_dt$n_max_treedepth > 0)) warning("At least one shard hit maximum treedepth.")
if (any(hmc_diagnostics_dt$max_rhat > 1.01, na.rm = TRUE)) warning("At least one shard has max R-hat > 1.01.")
if (any(hmc_diagnostics_dt$min_ess_bulk < 100, na.rm = TRUE)) warning("At least one shard has bulk ESS < 100.")

aggregation_started <- proc.time()
barycenter <- wasserstein_barycenter_1d(chunk_draw_list, variables)
set.seed(123)
bary_draws <- sample_barycenter_draws(barycenter, bary_M)
aggregation_elapsed <- (proc.time() - aggregation_started)[["elapsed"]]

comparison_started <- proc.time()
intervals <- rbindlist(list(
  ci_from_draws(full_draws, variables, "Full-data MCMC"),
  ci_from_draws(subsampling_draws, variables, "b=0 subsampling MCMC"),
  ci_from_draws(bary_draws, variables, "DC-BATS")
))
agreement <- rbindlist(list(
  agreement_summary(full_draws, bary_draws, variables, "Full-data MCMC", "DC-BATS"),
  agreement_summary(subsampling_draws, bary_draws, variables, "b=0 subsampling MCMC", "DC-BATS")
))
comparison_elapsed <- (proc.time() - comparison_started)[["elapsed"]]

chunk_elapsed_total <- sum(chunk_timing_dt$elapsed)
chunk_elapsed_mean <- mean(chunk_timing_dt$elapsed)
chunk_elapsed_max <- max(chunk_timing_dt$elapsed)
serial_elapsed <- extract_elapsed + aggregation_elapsed + comparison_elapsed
timing <- data.table(
  p = P, d = D, dataset_id = dataset_id,
  source_npz_sha256 = actual_hash, K = K, K_roles = resource$K_roles,
  T_total = T_total, T_used = T_used, dropped_observations = dropped_observations,
  base_shard_size_q = q, larger_shard_count_r = r,
  minimum_shard_size = min(shard_sizes), maximum_shard_size = max(shard_sizes),
  shard_sizes_json = toJSON(as.integer(shard_sizes), auto_unbox = TRUE),
  shard_likelihood_powers_json = toJSON(shard_likelihood_powers, auto_unbox = TRUE, digits = NA),
  median_R_CPU_used_to_select_K = resource$median_R_CPU,
  n_chains = n_chains, n_warmup = n_warmup, n_sampling = n_sampling, bary_M = bary_M,
  chunk_elapsed_total = chunk_elapsed_total,
  chunk_elapsed_mean = chunk_elapsed_mean,
  chunk_elapsed_max = chunk_elapsed_max,
  extract_elapsed = extract_elapsed,
  cache_reload_elapsed_not_included = cache_reload_elapsed,
  aggregation_elapsed = aggregation_elapsed,
  comparison_elapsed = comparison_elapsed,
  total_elapsed_sequential = chunk_elapsed_total + serial_elapsed,
  ideal_parallel_elapsed_mean = chunk_elapsed_mean + serial_elapsed,
  ideal_parallel_elapsed_max = chunk_elapsed_max + serial_elapsed,
  stan_compilation_included = FALSE,
  aggregation_and_comparison_included = TRUE
)

file_tag <- paste0("regression_garch_p", P, "_dc_K", K)
atomic_fwrite(bary_draws, file.path(results_dir, paste0("bary_draws_", file_tag, ".csv")))
atomic_fwrite(chunk_draws, file.path(results_dir, paste0("chunk_draws_", file_tag, ".csv")))
atomic_fwrite(chunk_timing_dt, file.path(results_dir, paste0("chunk_timing_", file_tag, ".csv")))
atomic_fwrite(shard_configuration, file.path(results_dir, paste0("shard_configuration_", file_tag, ".csv")))
atomic_fwrite(hmc_diagnostics_dt, file.path(results_dir, paste0("hmc_diagnostics_", file_tag, ".csv")))
atomic_fwrite(full_draws[, ..variables], file.path(results_dir, paste0("full_truth_used_", file_tag, ".csv")))
atomic_fwrite(subsampling_draws[, ..variables], file.path(results_dir, paste0("subsampling_truth_used_", file_tag, ".csv")))
atomic_fwrite(intervals, file.path(results_dir, paste0("results_intervals_", file_tag, ".csv")))
atomic_fwrite(agreement, file.path(results_dir, paste0("inferential_agreement_", file_tag, ".csv")))
atomic_fwrite(timing, file.path(results_dir, paste0("timing_", file_tag, ".csv")))

result_metadata <- list(
  schema_version = 2L,
  p = P,
  total_parameter_dimension = D,
  canonical_dataset_id = dataset_id,
  source_npz = source_npz,
  source_npz_sha256 = actual_hash,
  K = K,
  K_roles = roles,
  is_standard_10 = "standard_10" %in% roles,
  is_standard_20 = "standard_20" %in% roles,
  is_cpu_matched = "cpu_matched" %in% roles,
  K_stress = expected_K_stress,
  median_R_CPU = resource$median_R_CPU,
  R_CPU_values = R_CPU_values,
  partition_schema = partition_schema,
  T_total = T_total,
  T_used = T_used,
  dropped_observations = dropped_observations,
  base_shard_size_q = q,
  larger_shard_count_r = r,
  shard_sizes = as.integer(shard_sizes),
  shard_likelihood_powers = as.numeric(shard_likelihood_powers),
  model = "no-intercept regression-GARCH(1,1), standardized Student-t errors",
  student_t_specification = "epsilon has variance one; Stan scale sqrt(sigma2 * (nu - 2) / nu)",
  priors = list(
    beta_reg = "independent N(0,1)", omega = "HalfNormal(1)",
    alpha = "HalfNormal(0.2)", beta_garch = "HalfNormal(0.8)",
    nu = "nu - 2 ~ Gamma(shape=2, rate=1)", stationarity = "alpha + beta_garch < 1"
  ),
  shard_posterior = "shard k likelihood powered by T_total / n_k; prior included once and unpowered",
  shard_partition = paste(
    "All T_total observations are assigned exactly once to K contiguous shards;",
    "r=T_total mod K shards have size floor(T_total/K)+1 and the others have size floor(T_total/K)."
  ),
  shard_initialisation = paste(
    "Oracle true simulated sigma2 at the preceding time point; preceding residual",
    "y_{s-1} - X_{s-1} beta is recomputed for each draw."
  ),
  combination = "coordinate-wise one-dimensional Wasserstein quantile barycentre",
  mcmc = list(chains = n_chains, warmup_per_chain = n_warmup, sampling_per_chain = n_sampling),
  timing = list(
    clock = "elapsed wall time from proc.time()",
    ideal_parallel = "max (primary) or mean shard fit time plus extraction, aggregation, and comparison",
    stan_compilation_included = FALSE,
    barycentre_and_comparison_included = TRUE
  ),
  resource_matching = list(
    median_R_CPU = resource$median_R_CPU,
    R_CPU_values = R_CPU_values,
    K_stress = expected_K_stress,
    K_roles = roles,
    rule = resource$rule,
    K_selected_from_DC_BATS_timing = FALSE,
    K_selected_from_DC_BATS_accuracy = FALSE,
    selected_before_DC_BATS_inferential_results = TRUE
  )
)
atomic_write_json(result_metadata, file.path(results_dir, "metadata.json"))
atomic_save_rds(
  list(
    metadata = result_metadata, timing = timing, chunk_timing = chunk_timing_dt,
    shard_configuration = shard_configuration,
    hmc_diagnostics = hmc_diagnostics_dt, intervals = intervals,
    inferential_agreement = agreement, barycenter = barycenter
  ),
  file.path(results_dir, paste0("dc_bats_", file_tag, "_results.rds"))
)
atomic_write_json(
  list(
    p = P, K = K, K_roles = roles, median_R_CPU = resource$median_R_CPU,
    partition_schema = partition_schema, T_total = T_total, T_used = T_used,
    dropped_observations = dropped_observations,
    shard_sizes = as.integer(shard_sizes),
    shard_likelihood_powers = as.numeric(shard_likelihood_powers),
    source_npz_sha256 = actual_hash,
    completed_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
  ),
  completion_file
)

print(timing)
print(intervals)
