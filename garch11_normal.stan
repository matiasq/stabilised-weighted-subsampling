data {
  int<lower=1> T;
  vector[T] y;

  // DC-BATS likelihood power. Use power = K for each chunk.
  real<lower=0> power;

  // Known recursion initialisation, matching our likelihood code.
  real y_start;
  real<lower=0> sigma2_start;
}

parameters {
  real mu;
  real<lower=0> omega;
  real<lower=0> alpha1;
  real<lower=0> beta1;
}

model {
  vector[T] sigma2;

  // Stationarity restriction imposed through the prior support.
  if (alpha1 + beta1 >= 1) {
    reject("Stationarity violated: alpha1 + beta1 >= 1");
  }

  // Priors used in Quiroz et al. (2026)
  // mu ~ N(0, 10^2)
  // omega ~ HalfNormal(1)
  // alpha1 ~ HalfNormal(0.2)
  // beta1 ~ HalfNormal(0.8)
  mu     ~ normal(0, 10);
  omega  ~ normal(0, 1);
  alpha1 ~ normal(0, 0.2);
  beta1  ~ normal(0, 0.8);

  // First observation, using supplied pre-sample values.
  sigma2[1] =
    omega
    + alpha1 * square(y_start - mu)
    + beta1 * sigma2_start;

  target += power * normal_lpdf(y[1] | mu, sqrt(sigma2[1]));

  // Remaining observations.
  for (t in 2:T) {
    sigma2[t] =
      omega
      + alpha1 * square(y[t - 1] - mu)
      + beta1 * sigma2[t - 1];

    target += power * normal_lpdf(y[t] | mu, sqrt(sigma2[t]));
  }
}
