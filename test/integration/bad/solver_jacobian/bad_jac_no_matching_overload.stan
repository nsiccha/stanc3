functions {
  vector decay(real t, vector y, vector k) {
    return -k .* y;
  }
  matrix decay_jac(real t, vector y) {
    return rep_matrix(0, 2, 2);
  }
  matrix decay_jac(real t, vector y, vector k, real u) {
    return rep_matrix(0, 2, 5);
  }
}
data {
  vector[2] y0;
  real t0;
  array[4] real ts;
}
parameters {
  vector<lower=0>[2] k;
}
transformed parameters {
  array[4] vector[2] y_hat = ode_rk45_tol(decay, decay_jac, y0, t0, ts, 1e-6, 1e-6, 1000, k);
}
