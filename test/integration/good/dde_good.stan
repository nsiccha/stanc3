functions {
  // y_lag[i, j] is the i-th state at time t - tau[j]
  vector delayed_turnover(real t, vector y, matrix y_lag, vector k,
                          array[] real x) {
    vector[2] dydt;
    dydt[1] = k[1] - k[2] * y_lag[1, 1];
    dydt[2] = k[2] * y_lag[1, 1] - k[3] * y[2] / (x[1] + y_lag[2, 2]);
    return dydt;
  }
  vector delayed_decay(real t, vector y, matrix y_lag) {
    return -y_lag[ : , 1];
  }
}
data {
  int<lower=1> T;
  vector[2] y0;
  real t0;
  array[T] real ts;
  vector<lower=0>[2] tau_data;
  array[1] real x;
  array[T] vector[2] y;
}
transformed data {
  array[T] vector[2] y_td = dde_rk45(delayed_decay, y0, t0, ts, tau_data);
  array[T] vector[2] y_td_tol = dde_rk45_tol(delayed_turnover, y0, t0, ts,
                                             tau_data, 1e-6, 1e-6, 1000,
                                             [1, 2, 3]', x);
}
parameters {
  vector<lower=0>[3] k;
  vector<lower=0>[2] tau;
  vector[2] y0_var;
  real<lower=0> sigma;
}
transformed parameters {
  array[T] vector[2] y_hat;
  {
    y_hat = dde_rk45(delayed_turnover, y0, t0, ts, tau_data, k, x);
  }
  {
    y_hat = dde_rk45(delayed_turnover, y0_var, t0, ts, tau, k, x);
  }
  {
    y_hat = dde_rk45_tol(delayed_turnover, y0, t0, ts, tau, 1e-6, 1e-6, 1000,
                         k, x);
  }
  {
    y_hat = dde_rk45_tol(delayed_turnover, y0_var, t0, ts, tau_data, 1e-6,
                         1e-6, 1000, k, x);
  }
  {
    y_hat = dde_rk45(delayed_decay, y0_var, t0, ts, tau);
  }
}
model {
  for (t in 1 : T) {
    y[t] ~ normal(y_hat[t], sigma);
  }
}
generated quantities {
  array[T] vector[2] y_gq = dde_rk45(delayed_turnover, y0, t0, ts, tau, k, x);
}
