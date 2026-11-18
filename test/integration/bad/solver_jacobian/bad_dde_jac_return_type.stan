functions {
  vector delayed(real t, vector y, matrix y_lag, vector k) {
    return -k .* y_lag[ : , 1];
  }
  array[] vector delayed_jac(real t, vector y, matrix y_lag, vector k) {
    return {-k, -k};
  }
}
data {
  vector[2] y0;
  real t0;
  array[4] real ts;
  vector[1] tau;
}
parameters {
  vector<lower=0>[2] k;
}
transformed parameters {
  array[4] vector[2] y_hat = dde_rk45_tol(delayed, delayed_jac, y0, t0, ts, tau, 1e-6, 1e-6, 1000, k);
}
