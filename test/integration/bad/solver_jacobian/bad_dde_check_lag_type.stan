functions {
  vector delayed(real t, vector y, matrix y_lag, vector k) {
    return -k .* y_lag[ : , 1];
  }
  matrix delayed_jac(real t, vector y, matrix y_lag, vector k) {
    return rep_matrix(0, 2, 6);
  }
}
data {
  vector[2] y0;
  real t0;
}
parameters {
  vector<lower=0>[2] k;
}
transformed parameters {
  // y_lag is a matrix with one column per delay
  real err = dde_jacobian_error(delayed, delayed_jac, t0, y0, y0, k);
}
