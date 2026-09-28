data {
  int<lower=1> T;
  int<lower=1> P;
  matrix[T, P] X;
  vector[T] y;

  // DC-BATS powers the shard likelihood but includes the prior once.
  real<lower=0> power;

  // The preceding residual is recomputed from beta at every posterior draw.
  row_vector[P] X_start;
  real y_start;

  // Oracle simulated boundary variance, matching the Section-6 design.
  real<lower=0> sigma2_start;
}

parameters {
  vector[P] beta_reg;
  real<lower=0> omega;
  real<lower=0> alpha;
  real<lower=0> beta_garch;
  real<lower=0> nu_minus_two;
}

transformed parameters {
  real<lower=2> nu = nu_minus_two + 2;
}

model {
  vector[T] sigma2;
  real z_previous;

  if (alpha + beta_garch >= 1) {
    reject("Stationarity violated: alpha + beta_garch >= 1");
  }

  // Priors exactly matching the validated Python Section-7 model.
  beta_reg ~ normal(0, 1);
  omega ~ normal(0, 1);
  alpha ~ normal(0, 0.2);
  beta_garch ~ normal(0, 0.8);
  nu_minus_two ~ gamma(2, 1);

  z_previous = y_start - X_start * beta_reg;
  sigma2[1] = omega + alpha * square(z_previous) + beta_garch * sigma2_start;

  // Stan's Student-t scale is adjusted so sigma2 is the innovation variance.
  target += power * student_t_lpdf(
    y[1] | nu, X[1] * beta_reg, sqrt(sigma2[1] * (nu - 2) / nu)
  );

  for (t in 2:T) {
    z_previous = y[t - 1] - X[t - 1] * beta_reg;
    sigma2[t] = omega + alpha * square(z_previous) + beta_garch * sigma2[t - 1];
    target += power * student_t_lpdf(
      y[t] | nu, X[t] * beta_reg, sqrt(sigma2[t] * (nu - 2) / nu)
    );
  }
}
