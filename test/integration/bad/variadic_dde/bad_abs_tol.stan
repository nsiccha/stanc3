functions {
  vector delayed_turnover(real t, vector y, matrix y_lag, vector k) {
    vector[2] dydt;
    dydt[1] = k[1] - k[2] * y_lag[1, 1];
    dydt[2] = k[2] * y_lag[1, 1] - k[3] * y[2];
    return dydt;
  }
}
data {
  vector[2] y0;
  real t0;
  array[4] real ts;
  vector[2] tau;
  array[2] real atol;
  array[4] vector[2] y;
}
parameters {
  vector<lower=0>[3] k;
  real<lower=0> sigma;
}
transformed parameters {
  array[4] vector[2] y_hat = dde_rk45_tol(delayed_turnover, y0, t0, ts, tau, 1e-6, atol, 1000, k);
}
model {
  for (t in 1 : 4) {
    y[t] ~ normal(y_hat[t], sigma);
  }
}
