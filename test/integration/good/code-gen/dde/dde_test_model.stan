functions {
  // y_lag[i, j] is the i-th state at time t - tau[j]
  vector delayed_feedback(real t, vector y, matrix y_lag, real r, real K) {
    return r * y .* (1 - y_lag[ : , 1] / K);
  }
}
data {
  int<lower=1> T;
  vector[1] y0;
  real t0;
  array[T] real ts;
  vector<lower=0>[1] tau_data;
  array[T] vector[1] y;
}
transformed data {
  array[T] vector[1] y_data = dde_rk45(delayed_feedback, y0, t0, ts, tau_data,
                                       1.0, 10.0);
}
parameters {
  real<lower=0> r;
  real<lower=0> K;
  vector<lower=0>[1] tau;
  real<lower=0> sigma;
}
transformed parameters {
  array[T] vector[1] y_hat = dde_rk45(delayed_feedback, y0, t0, ts, tau, r, K);
  array[T] vector[1] y_hat_tol = dde_rk45_tol(delayed_feedback, y0, t0, ts,
                                              tau, 1e-8, 1e-8, 10000, r, K);
}
model {
  for (t in 1 : T) {
    y[t] ~ normal(y_hat[t], sigma);
  }
}
